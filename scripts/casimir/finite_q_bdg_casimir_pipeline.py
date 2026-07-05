#!/usr/bin/env python3
"""Finite-q BdG Casimir main production pipeline v1."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.constants import KB  # noqa: E402
from lno327.electrodynamics.reflection import (  # noqa: E402
    sigma_tilde_xy_to_te_tm_reflection_matrix,
    vacuum_kappa,
)
from lno327.electrodynamics.conventions import (  # noqa: E402
    bilayer_sheet_conductivity_convention_metadata,
    model_response_to_sheet_conductivity,
    sheet_conductivity_to_reflection_dimensionless,
    spatial_response_to_bilayer_sheet_conductivity_model,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions, finite_q_bdg_response_from_ansatz  # noqa: E402
from lno327.workflows.finite_q_quadrature import FiniteQQuadratureOptions, finite_q_quadrature_points  # noqa: E402
from lno327.casimir.lifshitz_integrand import lifshitz_integrand_metadata, trace_log_point  # noqa: E402
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE  # noqa: E402
from lno327.models.lno327_four_orbital.collective import build_pairing_ansatz  # noqa: E402
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes  # noqa: E402
from lno327.numerics.matsubara import bosonic_matsubara_energy_eV  # noqa: E402
from lno327.response.config import KuboConfig  # noqa: E402

try:  # pragma: no cover - exercised when tqdm is available on a server.
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


PIPELINE_VERSION = "finite_q_bdg_casimir_pipeline_v1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "casimir" / "finite_q_bdg_pipeline"
FULL_RESPONSE_SOURCE = "amplitude_phase_schur"
CONFIG_HASH_LEN = 16


@dataclass(frozen=True)
class PlateReflectionTask:
    task_key: str
    config_hash: str
    task_index: int
    task_shard_index: int
    task_shard_count: int
    pairing: str
    plate_theta_deg: float
    n: int
    q_index: int
    phi_index: int
    omega_eV: float
    Q_nm_inv: float
    phi_rad: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp_path(path)
    try:
        tmp.write_text(text, encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_many_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in _load_jsonl(path):
            key = str(row.get("task_key") or row.get("energy_point_key") or row.get("roundtrip_key") or len(seen))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _filter_config_hash(rows: list[dict[str, Any]], config_hash: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("config_hash") or row.get("run_config_hash")) == str(config_hash)]


def _completed_task_keys(path: Path) -> set[str]:
    return {str(row["task_key"]) for row in _load_jsonl(path) if row.get("status") == "completed"}


def _git_commit_hash() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def matsubara_omega_eV(n: int, temperature_K: float) -> float:
    return float(bosonic_matsubara_energy_eV(int(n), float(temperature_K)))


def matsubara_weight(n: int) -> float:
    return 0.5 if int(n) == 0 else 1.0


def q_phi_measure(Q_m_inv: float, delta_Q_m_inv: float, delta_phi: float) -> float:
    return float(Q_m_inv) * float(delta_Q_m_inv) * float(delta_phi) / ((2.0 * np.pi) ** 2)


def rotate_xy_tensor(matrix: np.ndarray, theta_rad: float) -> np.ndarray:
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return rotation @ np.asarray(matrix, dtype=complex) @ rotation.T


def rotate_xy_vector(vector: np.ndarray, theta_rad: float) -> np.ndarray:
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return rotation @ np.asarray(vector, dtype=float)


def _ward_residuals(response: np.ndarray, omega_eV: float, q_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qx, qy = float(q_model[0]), float(q_model[1])
    matrix = np.asarray(response, dtype=complex)
    left = 1j * float(omega_eV) * matrix[0, :] + qx * matrix[1, :] + qy * matrix[2, :]
    right = 1j * float(omega_eV) * matrix[:, 0] - qx * matrix[:, 1] - qy * matrix[:, 2]
    return left, right


def _pairing_ansatz_name(pairing: str) -> str:
    return "onsite_s" if pairing == "normal" else pairing


def _pairing_delta(pairing: str, delta0_eV: float) -> float:
    return 0.0 if pairing == "normal" else float(delta0_eV)


def _complex_matrix_payload(matrix: np.ndarray) -> dict[str, list[list[float]]]:
    array = np.asarray(matrix, dtype=complex)
    return {"real": np.real(array).tolist(), "imag": np.imag(array).tolist()}


def _complex_matrix_from_payload(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray(payload["real"], dtype=float) + 1j * np.asarray(payload["imag"], dtype=float)


def _quadrature_options(config: dict[str, Any]) -> FiniteQQuadratureOptions:
    return FiniteQQuadratureOptions(
        integration_strategy=str(config["integration_strategy"]),
        coarse_grid=int(config["coarse_grid"]),
        adaptive_level=int(config["adaptive_level"]),
        gauss_order=int(config["gauss_order"]),
        fermi_window_eV=float(config["fermi_window_eV"]),
        q_specific_adaptive_grid=bool(config["q_specific_adaptive_grid"]),
        fermi_level_eV=float(config.get("fermi_level_eV", 0.0)),
    )


def _compute_full_response(
    pairing: str,
    omega_eV: float,
    q_model: np.ndarray,
    *,
    temperature_K: float,
    delta0_eV: float,
    eta_eV: float,
    quadrature_options: FiniteQQuadratureOptions,
) -> tuple[np.ndarray, dict[str, Any]]:
    ansatz = build_pairing_ansatz(_pairing_ansatz_name(pairing), phase_vertex="bond_endpoint_gauge")
    params = PairingAmplitudes(delta0_eV=_pairing_delta(pairing, delta0_eV))
    points, weights, quadrature_meta = finite_q_quadrature_points(np.asarray(q_model, dtype=float), quadrature_options)
    config = KuboConfig.from_kelvin(
        omega_eV=float(omega_eV),
        temperature_K=float(temperature_K),
        eta_eV=float(eta_eV),
        output_si=False,
    )
    response = finite_q_bdg_response_from_ansatz(
        ansatz,
        float(omega_eV),
        np.asarray(q_model, dtype=float),
        points,
        weights,
        config,
        params,
        FiniteQEngineOptions(
            current_vertex="peierls",
            collective_mode="amplitude_phase",
            collective_counterterm="goldstone_gap_equation",
            include_phase_phase_direct=True,
        ),
    )
    full_response = np.asarray(response.amplitude_phase_schur, dtype=complex)
    left, right = _ward_residuals(full_response, omega_eV, q_model)
    metadata = {
        "left_ward_residual_norm": float(np.linalg.norm(left)),
        "right_ward_residual_norm": float(np.linalg.norm(right)),
        "max_ward_residual_norm": float(max(np.linalg.norm(left), np.linalg.norm(right))),
        "quadrature": quadrature_meta,
        "num_quadrature_points": int(quadrature_meta["num_quadrature_points"]),
        "num_cells_total": int(quadrature_meta["num_cells_total"]),
        "num_cells_refined": int(quadrature_meta["num_cells_refined"]),
    }
    return full_response, metadata


def _response_to_reflection(
    full_response: np.ndarray,
    omega_eV: float,
    q_model_xy: np.ndarray,
    *,
    theta_rad: float,
    lattice_a_x_m: float,
    lattice_a_y_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    sigma_model = spatial_response_to_bilayer_sheet_conductivity_model(full_response, omega_eV)
    sigma_sheet = model_response_to_sheet_conductivity(sigma_model)
    sigma_tilde = sheet_conductivity_to_reflection_dimensionless(sigma_sheet)
    sigma_tilde_xy = rotate_xy_tensor(sigma_tilde.tensor.matrix(), theta_rad)
    reflection = sigma_tilde_xy_to_te_tm_reflection_matrix(
        sigma_tilde_xy,
        float(q_model_xy[0]),
        float(q_model_xy[1]),
        float(omega_eV),
        float(lattice_a_x_m),
        float(lattice_a_y_m),
        allow_q_zero=False,
    )
    metadata = {
        "sigma_model_norm": float(np.linalg.norm(sigma_model)),
        "sigma_sheet_norm": float(np.linalg.norm(sigma_sheet.tensor.matrix())),
        "sigma_tilde_norm": float(np.linalg.norm(sigma_tilde.tensor.matrix())),
        "sigma_tilde_xy_norm": float(np.linalg.norm(reflection["sigma_tilde_xy_matrix"])),
        "sigma_tilde_LT_norm": float(np.linalg.norm(reflection["sigma_tilde_LT_matrix"])),
        "reflection_norm": float(np.linalg.norm(reflection["reflection_TE_TM"])),
        "reflection_TE_TM_norm": float(np.linalg.norm(reflection["reflection_TE_TM"])),
        "Q_x_m_inv": float(reflection["Q_x_m_inv"]),
        "Q_y_m_inv": float(reflection["Q_y_m_inv"]),
        "Q_m_inv": float(reflection["Q_m_inv"]),
        "xi_si_s_inv": float(reflection["xi_si_s_inv"]),
        "kappa_m_inv": float(reflection["kappa_m_inv"]),
        "reflection_condition_status": "computed",
    }
    return np.asarray(reflection["reflection_TE_TM"], dtype=complex), metadata


def _reflection_for_pairing(
    pairing: str,
    omega_eV: float,
    q_model_lab: np.ndarray,
    *,
    plate_theta_rad: float,
    temperature_K: float,
    delta0_eV: float,
    eta_eV: float,
    quadrature_options: FiniteQQuadratureOptions,
    lattice_a_x_m: float,
    lattice_a_y_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    q_crystal = rotate_xy_vector(q_model_lab, -plate_theta_rad)
    response, ward = _compute_full_response(
        pairing,
        omega_eV,
        q_crystal,
        temperature_K=temperature_K,
        delta0_eV=delta0_eV,
        eta_eV=eta_eV,
        quadrature_options=quadrature_options,
    )
    reflection, meta = _response_to_reflection(
        response,
        omega_eV,
        q_model_lab,
        theta_rad=plate_theta_rad,
        lattice_a_x_m=lattice_a_x_m,
        lattice_a_y_m=lattice_a_y_m,
    )
    meta.update(ward)
    meta["q_crystal_x"] = float(q_crystal[0])
    meta["q_crystal_y"] = float(q_crystal[1])
    return reflection, meta


def _n0_stability(norms: list[float]) -> tuple[float, str]:
    if not norms:
        return 0.0, "not_applicable"
    variation = float(max(norms) - min(norms))
    scale = max(max(norms), 1e-300)
    relative = variation / scale
    if relative < 1e-3:
        return variation, "stable"
    if relative < 5e-2:
        return variation, "monitor"
    return variation, "unstable"


def _extrapolated_reflection_for_pairing(
    pairing: str,
    omega_eps_list: list[float],
    q_model_lab: np.ndarray,
    *,
    plate_theta_rad: float,
    temperature_K: float,
    delta0_eV: float,
    eta_eV: float,
    quadrature_options: FiniteQQuadratureOptions,
    lattice_a_x_m: float,
    lattice_a_y_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrices = []
    metas = []
    for omega in omega_eps_list:
        reflection, meta = _reflection_for_pairing(
            pairing,
            omega,
            q_model_lab,
            plate_theta_rad=plate_theta_rad,
            temperature_K=temperature_K,
            delta0_eV=delta0_eV,
            eta_eV=eta_eV,
            quadrature_options=quadrature_options,
            lattice_a_x_m=lattice_a_x_m,
            lattice_a_y_m=lattice_a_y_m,
        )
        matrices.append(reflection)
        metas.append(meta)
    x = np.asarray(omega_eps_list, dtype=float)
    y = np.asarray(matrices, dtype=complex).reshape(len(matrices), 4)
    coefficients = np.vstack([np.polyfit(x, y[:, i], 1) for i in range(4)])
    fitted = np.vstack([np.polyval(coefficients[i], x) for i in range(4)]).T
    extrapolated = coefficients[:, 1].reshape(2, 2)
    fit_residual_norm = float(np.linalg.norm(fitted - y))
    norms = [float(np.linalg.norm(matrix)) for matrix in matrices]
    variation, stability = _n0_stability(norms)
    meta = dict(metas[-1])
    meta.update(
        {
            "n0_status": "extrapolated",
            "n0_policy": "extrapolate",
            "n0_extrapolation_method": "linear_in_omega_eV",
            "n0_extrapolation_omega_eV": [float(v) for v in omega_eps_list],
            "n0_extrapolation_order": 1,
            "n0_reflection_norms": norms,
            "n0_reflection_norm_variation": variation,
            "n0_fit_residual_norm": fit_residual_norm,
            "n0_extrapolation_residual_norm": fit_residual_norm,
            "n0_stability_status": stability,
            "reflection_norm": float(np.linalg.norm(extrapolated)),
            "reflection_TE_TM_norm": float(np.linalg.norm(extrapolated)),
        }
    )
    return extrapolated, meta


def _q_values(args: argparse.Namespace) -> np.ndarray:
    return (np.arange(int(args.q_num), dtype=float) + 0.5) * float(args.q_max_nm_inv) / float(args.q_num)


def _phi_values(args: argparse.Namespace) -> np.ndarray:
    return (np.arange(int(args.phi_num), dtype=float) + 0.5) * 2.0 * np.pi / float(args.phi_num)


def _plate_theta_values(args: argparse.Namespace) -> list[float]:
    values = {0.0}
    values.update(float(v) for v in args.angles_deg)
    return sorted(values)


def _plate_task_key(
    config_hash: str,
    pairing: str,
    plate_theta: float,
    n: int,
    q_index: int,
    phi_index: int,
    n0_policy: str,
) -> str:
    return (
        f"{config_hash}|plate_reflection|pairing={pairing}|plate_theta_deg={plate_theta:.12g}|"
        f"n={n}|Q_index={q_index}|phi_index={phi_index}|n0_policy={n0_policy}"
    )


def _roundtrip_key(
    config_hash: str,
    pairing: str,
    theta: float,
    n: int,
    q_index: int,
    phi_index: int,
    n0_policy: str,
) -> str:
    return (
        f"{config_hash}|roundtrip|pairing={pairing}|theta_deg={theta:.12g}|"
        f"n={n}|Q_index={q_index}|phi_index={phi_index}|n0_policy={n0_policy}"
    )


def _energy_point_key(
    config_hash: str,
    pairing: str,
    distance: float,
    theta: float,
    n: int,
    q_index: int,
    phi_index: int,
) -> str:
    return (
        f"{config_hash}|energy_point|pairing={pairing}|distance_nm={distance:.12g}|theta_deg={theta:.12g}|"
        f"n={n}|Q_index={q_index}|phi_index={phi_index}"
    )


def build_plate_reflection_tasks(args: argparse.Namespace, config_hash: str) -> list[PlateReflectionTask]:
    q_values = _q_values(args)
    phi_values = _phi_values(args)
    tasks = []
    index = 0
    for pairing in args.pairings:
        for plate_theta in _plate_theta_values(args):
            for n in range(int(args.n_max) + 1):
                omega = matsubara_omega_eV(n, float(args.temperature_K))
                for q_index, q_value in enumerate(q_values):
                    for phi_index, phi in enumerate(phi_values):
                        key = _plate_task_key(config_hash, pairing, plate_theta, n, q_index, phi_index, args.n0_policy)
                        if index % int(args.task_shard_count) == int(args.task_shard_index):
                            tasks.append(
                                PlateReflectionTask(
                                    task_key=key,
                                    config_hash=config_hash,
                                    task_index=index,
                                    task_shard_index=int(args.task_shard_index),
                                    task_shard_count=int(args.task_shard_count),
                                    pairing=str(pairing),
                                    plate_theta_deg=float(plate_theta),
                                    n=int(n),
                                    q_index=int(q_index),
                                    phi_index=int(phi_index),
                                    omega_eV=float(omega),
                                    Q_nm_inv=float(q_value),
                                    phi_rad=float(phi),
                                )
                            )
                        index += 1
    return tasks


def _count_roundtrip_tasks(args: argparse.Namespace) -> int:
    return int(len(args.pairings) * len(args.angles_deg) * (int(args.n_max) + 1) * int(args.q_num) * int(args.phi_num))


def _count_energy_points(args: argparse.Namespace) -> int:
    return int(_count_roundtrip_tasks(args) * len(args.distances_nm))


def _current_shard_roundtrip_count(args: argparse.Namespace) -> int:
    count = 0
    index = 0
    for _pairing in args.pairings:
        for _theta in args.angles_deg:
            for _n in range(int(args.n_max) + 1):
                for _q_index in range(int(args.q_num)):
                    for _phi_index in range(int(args.phi_num)):
                        if index % int(args.task_shard_count) == int(args.task_shard_index):
                            count += 1
                        index += 1
    return count


def _setup_output(output_dir: Path, args: argparse.Namespace) -> dict[str, Path]:
    suffix = f"shard_{int(args.task_shard_index)}_of_{int(args.task_shard_count)}"
    paths = {
        "root": output_dir,
        "data": output_dir / "data",
        "figures": output_dir / "figures",
        "logs": output_dir / "logs",
        "run_config": output_dir / "run_config.json",
        "reflection_shard": output_dir / f"reflection_results.{suffix}.jsonl",
        "energy_shard": output_dir / f"energy_point_results.{suffix}.jsonl",
        "failed_shard": output_dir / f"failed_points.{suffix}.jsonl",
        "reflection_results": output_dir / "reflection_results.jsonl",
        "point_results": output_dir / "point_results.jsonl",
        "failed_points": output_dir / "failed_points.jsonl",
        "summary": output_dir / "summary.json",
        "status": output_dir / "status.json",
        "run_status": output_dir / "run_status.json",
    }
    for key in ("root", "data", "figures", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def _configure_logging(paths: dict[str, Path]) -> logging.Logger:
    logger = logging.getLogger("finite_q_bdg_casimir_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    run_handler = logging.FileHandler(paths["logs"] / "run.log")
    run_handler.setFormatter(formatter)
    error_handler = logging.FileHandler(paths["logs"] / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(run_handler)
    logger.addHandler(error_handler)
    return logger


def _run_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pipeline_name": "finite_q_bdg_casimir_pipeline",
        "pipeline_version": PIPELINE_VERSION,
        "pairings": list(args.pairings),
        "temperature_K": float(args.temperature_K),
        "delta0_eV": float(args.delta0_eV),
        "eta_eV": float(args.eta_eV),
        "q_max_nm_inv": float(args.q_max_nm_inv),
        "q_num": int(args.q_num),
        "phi_num": int(args.phi_num),
        "n_max": int(args.n_max),
        "n0_policy": args.n0_policy,
        "integration_strategy": args.integration_strategy,
        "coarse_grid": int(args.coarse_grid),
        "adaptive_level": int(args.adaptive_level),
        "gauss_order": int(args.gauss_order),
        "fermi_window_eV": float(args.fermi_window_eV),
        "q_specific_adaptive_grid": bool(args.q_specific_adaptive_grid),
        "fermi_level_eV": 0.0,
        "lattice_a_x_m": float(LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m),
        "lattice_a_y_m": float(LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m),
        "material_boundary_model": "bilayer_normalized_2D_sheet",
        "response_contract": {
            "phase_vertex": "bond_endpoint_gauge",
            "current_vertex": "peierls",
            "collective_mode": "amplitude_phase",
            "collective_counterterm": "goldstone_gap_equation",
            "include_phase_phase_direct": True,
            "full_response_source": FULL_RESPONSE_SOURCE,
        },
        "unit_contract": "bilayer_sheet_conductivity_convention_metadata",
        "reflection_contract": "sigma_tilde_xy_to_te_tm_reflection_matrix",
        "casimir_contract": "trace_log_point_external_grid_sum",
    }


def _config_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:CONFIG_HASH_LEN]


def _run_config_with_hash(args: argparse.Namespace) -> dict[str, Any]:
    payload = _run_config_payload(args)
    config_hash = _config_hash(payload)
    return {
        "config_hash": config_hash,
        "run_config_hash": config_hash,
        "serialized_payload_sha256_prefix": config_hash,
        "payload": payload,
        "created_at": _now_iso(),
        "git_commit": _git_commit_hash(),
        "distance_grid_nm_not_in_heavy_task_key": [float(v) for v in args.distances_nm],
        "angle_grid_deg": [float(v) for v in args.angles_deg],
        "valid_for_formal_casimir_claim": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_run_config_hash_matches(
    path: Path,
    current: dict[str, Any],
    *,
    allow_mismatch: bool,
) -> None:
    existing = _read_json(path)
    if existing.get("config_hash") != current.get("config_hash") and not allow_mismatch:
        raise ValueError(
            "Existing run_config hash differs from current config. Use a new output directory or disable --resume / "
            "pass --allow-config-mismatch intentionally."
        )


def _create_run_config_if_missing(path: Path, current: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp_path(path)
    try:
        tmp.write_text(json.dumps(current, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _check_or_write_run_config(paths: dict[str, Path], current: dict[str, Any], *, resume: bool, allow_mismatch: bool) -> None:
    path = paths["run_config"]
    if resume:
        if not path.exists():
            _create_run_config_if_missing(path, current)
        _ensure_run_config_hash_matches(path, current, allow_mismatch=allow_mismatch)
        return
    _atomic_write_json(path, current)


def _task_config(run_config: dict[str, Any]) -> dict[str, Any]:
    payload = dict(run_config["payload"])
    payload["config_hash"] = run_config["config_hash"]
    return payload


def _single_plate_reflection_task(task: PlateReflectionTask, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    plate_theta_rad = np.deg2rad(float(task.plate_theta_deg))
    phi = float(task.phi_rad)
    Q_m_inv = float(task.Q_nm_inv) * 1e9
    q_model_lab = np.array(
        [
            Q_m_inv * np.cos(phi) * float(config["lattice_a_x_m"]),
            Q_m_inv * np.sin(phi) * float(config["lattice_a_y_m"]),
        ],
        dtype=float,
    )
    base_row: dict[str, Any] = {
        **asdict(task),
        "run_config_hash": str(config["config_hash"]),
        "plate_theta_rad": float(plate_theta_rad),
        "phi_deg": float(np.rad2deg(phi)),
        "Q_m_inv": Q_m_inv,
        "q_model_lab_x": float(q_model_lab[0]),
        "q_model_lab_y": float(q_model_lab[1]),
        "temperature_K": float(config["temperature_K"]),
        "delta0_eV": float(config["delta0_eV"]),
        "eta_eV": float(config["eta_eV"]),
        "integration_strategy": config["integration_strategy"],
        "coarse_grid": int(config["coarse_grid"]),
        "adaptive_level": int(config["adaptive_level"]),
        "gauss_order": int(config["gauss_order"]),
        "fermi_window_eV": float(config["fermi_window_eV"]),
        "q_specific_adaptive_grid": bool(config["q_specific_adaptive_grid"]),
        "full_response_source": FULL_RESPONSE_SOURCE,
        "n0_policy": config["n0_policy"],
        "valid_for_formal_casimir_claim": False,
    }
    if task.n == 0 and config["n0_policy"] == "skip":
        return {
            **base_row,
            "status": "skipped_n0",
            "n0_status": "skipped",
            "complete_matsubara_sum": False,
            "complete_except_n0": True,
            "not_final_full_matsubara_result": True,
            "runtime_seconds": float(time.perf_counter() - started),
        }

    quadrature_options = _quadrature_options(config)
    if task.n == 0:
        omega_1 = matsubara_omega_eV(1, float(config["temperature_K"]))
        omega_eps_list = [omega_1 / 8.0, omega_1 / 16.0, omega_1 / 32.0]
        reflection, meta = _extrapolated_reflection_for_pairing(
            task.pairing,
            omega_eps_list,
            q_model_lab,
            plate_theta_rad=plate_theta_rad,
            temperature_K=float(config["temperature_K"]),
            delta0_eV=float(config["delta0_eV"]),
            eta_eV=float(config["eta_eV"]),
            quadrature_options=quadrature_options,
            lattice_a_x_m=float(config["lattice_a_x_m"]),
            lattice_a_y_m=float(config["lattice_a_y_m"]),
        )
        xi_si = 0.0
        kappa = float(vacuum_kappa(Q_m_inv, 1e-300))
        omega_used = float(omega_eps_list[-1])
    else:
        reflection, meta = _reflection_for_pairing(
            task.pairing,
            float(task.omega_eV),
            q_model_lab,
            plate_theta_rad=plate_theta_rad,
            temperature_K=float(config["temperature_K"]),
            delta0_eV=float(config["delta0_eV"]),
            eta_eV=float(config["eta_eV"]),
            quadrature_options=quadrature_options,
            lattice_a_x_m=float(config["lattice_a_x_m"]),
            lattice_a_y_m=float(config["lattice_a_y_m"]),
        )
        xi_si = float(meta["xi_si_s_inv"])
        kappa = float(meta["kappa_m_inv"])
        omega_used = float(task.omega_eV)
        meta.setdefault("n0_status", "not_n0")
        meta.setdefault("n0_policy", config["n0_policy"])

    row = {
        **base_row,
        "omega_eV_for_reflection": omega_used,
        "xi_si_s_inv": xi_si,
        "kappa_m_inv": kappa,
        "reflection_TE_TM": _complex_matrix_payload(reflection),
        "status": "completed",
        "runtime_seconds": float(time.perf_counter() - started),
    }
    row.update(meta)
    return row


def _plate_lookup(reflection_rows: list[dict[str, Any]]) -> dict[tuple[str, str, float, int, int, int], dict[str, Any]]:
    lookup = {}
    for row in reflection_rows:
        if row.get("status") != "completed":
            continue
        key = (
            str(row["config_hash"]),
            str(row["pairing"]),
            round(float(row["plate_theta_deg"]), 12),
            int(row["n"]),
            int(row["q_index"]),
            int(row["phi_index"]),
        )
        lookup[key] = row
    return lookup


def _roundtrip_rows(args: argparse.Namespace, config_hash: str, reflection_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = _plate_lookup(reflection_rows)
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    delta_q = float(args.q_max_nm_inv) * 1e9 / float(args.q_num)
    delta_phi = 2.0 * np.pi / float(args.phi_num)
    roundtrip_index = 0
    for pairing in args.pairings:
        for theta in args.angles_deg:
            for n in range(int(args.n_max) + 1):
                for q_index, q_value in enumerate(_q_values(args)):
                    for phi_index, phi in enumerate(_phi_values(args)):
                        key = _roundtrip_key(config_hash, pairing, float(theta), n, q_index, phi_index, args.n0_policy)
                        if roundtrip_index % int(args.task_shard_count) != int(args.task_shard_index):
                            roundtrip_index += 1
                            continue
                        left_key = (config_hash, str(pairing), 0.0, int(n), int(q_index), int(phi_index))
                        right_key = (config_hash, str(pairing), round(float(theta), 12), int(n), int(q_index), int(phi_index))
                        left = lookup.get(left_key)
                        right = lookup.get(right_key)
                        if left is None or right is None:
                            failed.append(
                                {
                                    "roundtrip_key": key,
                                    "config_hash": config_hash,
                                    "pairing": pairing,
                                    "theta_deg": float(theta),
                                    "n": int(n),
                                    "q_index": int(q_index),
                                    "phi_index": int(phi_index),
                                    "status": "missing_plate_reflection",
                                    "missing_left": left is None,
                                    "missing_right": right is None,
                                    "valid_for_formal_casimir_claim": False,
                                }
                            )
                            roundtrip_index += 1
                            continue
                        if left.get("n0_status") == "skipped" or right.get("n0_status") == "skipped":
                            roundtrip_index += 1
                            continue
                        R_left = _complex_matrix_from_payload(left["reflection_TE_TM"])
                        R_right = _complex_matrix_from_payload(right["reflection_TE_TM"])
                        kappa = max(float(left["kappa_m_inv"]), float(right["kappa_m_inv"]))
                        Q_m_inv = float(q_value) * 1e9
                        measure = q_phi_measure(Q_m_inv, delta_q, delta_phi)
                        for distance in args.distances_nm:
                            point = trace_log_point(R_left, R_right, kappa, float(distance) * 1e-9)
                            logdet = complex(point["logdet_integrand"])
                            contribution = KB * float(args.temperature_K) * matsubara_weight(n) * measure * logdet
                            rows.append(
                                {
                                    "energy_point_key": _energy_point_key(
                                        config_hash, pairing, float(distance), float(theta), n, q_index, phi_index
                                    ),
                                    "roundtrip_key": key,
                                    "config_hash": config_hash,
                                    "run_config_hash": config_hash,
                                    "pairing": pairing,
                                    "distance_nm": float(distance),
                                    "distance_m": float(distance) * 1e-9,
                                    "theta_deg": float(theta),
                                    "theta_rad": float(np.deg2rad(float(theta))),
                                    "n": int(n),
                                    "q_index": int(q_index),
                                    "phi_index": int(phi_index),
                                    "omega_eV": float(matsubara_omega_eV(n, float(args.temperature_K))),
                                    "Q_nm_inv": float(q_value),
                                    "Q_m_inv": Q_m_inv,
                                    "phi_rad": float(phi),
                                    "phi_deg": float(np.rad2deg(float(phi))),
                                    "kappa_m_inv": kappa,
                                    "round_trip_factor": float(point["round_trip_factor"]),
                                    "logdet_real": float(np.real(logdet)),
                                    "logdet_imag": float(np.imag(logdet)),
                                    "abs_logdet_imag": float(abs(np.imag(logdet))),
                                    "energy_contribution_real_J_m2": float(np.real(contribution)),
                                    "energy_contribution_imag_J_m2": float(np.imag(contribution)),
                                    "max_ward_residual_norm": float(
                                        max(left.get("max_ward_residual_norm", 0.0), right.get("max_ward_residual_norm", 0.0))
                                    ),
                                    "left_ward_residual_norm": float(
                                        max(left.get("left_ward_residual_norm", 0.0), right.get("left_ward_residual_norm", 0.0))
                                    ),
                                    "right_ward_residual_norm": float(
                                        max(right.get("right_ward_residual_norm", 0.0), left.get("right_ward_residual_norm", 0.0))
                                    ),
                                    "left_plate_task_key": left["task_key"],
                                    "right_plate_task_key": right["task_key"],
                                    "n0_policy": args.n0_policy,
                                    "n0_status": left.get("n0_status", right.get("n0_status", "not_n0")),
                                    "status": "completed",
                                    "distance_expanded_from_cached_reflection": True,
                                    "valid_for_formal_casimir_claim": False,
                                }
                            )
                        roundtrip_index += 1
    return rows, failed


def _shard_glob(paths: dict[str, Path], prefix: str) -> list[Path]:
    return sorted(paths["root"].glob(f"{prefix}.shard_*_of_*.jsonl"))


def _records_to_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp_path(path)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, path)


def _aggregate(rows: list[dict[str, Any]], paths: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    grouped: dict[tuple[str, float, float], float] = {}
    for row in completed:
        key = (row["pairing"], float(row["distance_nm"]), float(row["theta_deg"]))
        grouped[key] = grouped.get(key, 0.0) + float(row["energy_contribution_real_J_m2"])
    energy_grid = [
        {
            "pairing": pairing,
            "distance_nm": distance,
            "theta_deg": theta,
            "energy_per_area_J_m2": energy,
        }
        for (pairing, distance, theta), energy in sorted(grouped.items())
    ]
    pressure_rows = []
    for pairing in sorted({row["pairing"] for row in energy_grid}):
        for theta in sorted({row["theta_deg"] for row in energy_grid if row["pairing"] == pairing}):
            values = sorted(
                [row for row in energy_grid if row["pairing"] == pairing and row["theta_deg"] == theta],
                key=lambda row: row["distance_nm"],
            )
            if len(values) < 2:
                continue
            d = np.asarray([row["distance_nm"] * 1e-9 for row in values], dtype=float)
            e = np.asarray([row["energy_per_area_J_m2"] for row in values], dtype=float)
            pressure = -np.gradient(e, d)
            for row, p in zip(values, pressure, strict=True):
                pressure_rows.append({**row, "pressure_N_m2": float(p)})
    torque_rows = []
    for pairing in sorted({row["pairing"] for row in energy_grid}):
        for distance in sorted({row["distance_nm"] for row in energy_grid if row["pairing"] == pairing}):
            values = sorted(
                [row for row in energy_grid if row["pairing"] == pairing and row["distance_nm"] == distance],
                key=lambda row: row["theta_deg"],
            )
            if len(values) < 2:
                continue
            theta = np.deg2rad([row["theta_deg"] for row in values])
            e = np.asarray([row["energy_per_area_J_m2"] for row in values], dtype=float)
            torque = -np.gradient(e, theta)
            avg = float(np.mean(e))
            for row, t in zip(values, torque, strict=True):
                torque_rows.append(
                    {
                        **row,
                        "torque_per_area_J_m2_rad": float(t),
                        "anisotropic_energy_J_m2": float(row["energy_per_area_J_m2"] - avg),
                    }
                )
    matsubara = []
    by_n: dict[int, float] = {}
    for row in completed:
        by_n[int(row["n"])] = by_n.get(int(row["n"]), 0.0) + float(row["energy_contribution_real_J_m2"])
    for n, value in sorted(by_n.items()):
        matsubara.append({"n": n, "energy_contribution_J_m2": value})
    q_contrib = []
    by_q: dict[float, float] = {}
    for row in completed:
        by_q[float(row["Q_nm_inv"])] = by_q.get(float(row["Q_nm_inv"]), 0.0) + float(row["energy_contribution_real_J_m2"])
    for q, value in sorted(by_q.items()):
        q_contrib.append({"Q_nm_inv": q, "energy_contribution_J_m2": value})
    ward_rows = [
        {
            "pairing": row["pairing"],
            "distance_nm": row["distance_nm"],
            "theta_deg": row["theta_deg"],
            "n": row["n"],
            "Q_nm_inv": row["Q_nm_inv"],
            "phi_deg": row["phi_deg"],
            "max_ward_residual_norm": row["max_ward_residual_norm"],
            "left_ward_residual_norm": row["left_ward_residual_norm"],
            "right_ward_residual_norm": row["right_ward_residual_norm"],
        }
        for row in completed
    ]
    outputs = {
        "energy_grid": energy_grid,
        "pressure_rows": pressure_rows,
        "torque_rows": torque_rows,
        "matsubara": matsubara,
        "q_contrib": q_contrib,
        "ward_rows": ward_rows,
    }
    _records_to_csv(paths["data"] / "energy_distance_angle_grid.csv", energy_grid, ["pairing", "distance_nm", "theta_deg", "energy_per_area_J_m2"])
    _records_to_csv(paths["data"] / "energy_vs_distance.csv", [r for r in energy_grid if abs(float(r["theta_deg"])) < 1e-12], ["pairing", "distance_nm", "theta_deg", "energy_per_area_J_m2"])
    _records_to_csv(paths["data"] / "pairing_comparison.csv", [r for r in energy_grid if abs(float(r["theta_deg"])) < 1e-12], ["pairing", "distance_nm", "theta_deg", "energy_per_area_J_m2"])
    _records_to_csv(paths["data"] / "energy_vs_angle.csv", energy_grid, ["pairing", "distance_nm", "theta_deg", "energy_per_area_J_m2"])
    _records_to_csv(paths["data"] / "pressure_vs_distance.csv", pressure_rows, ["pairing", "distance_nm", "theta_deg", "energy_per_area_J_m2", "pressure_N_m2"])
    _records_to_csv(paths["data"] / "torque_vs_angle.csv", torque_rows, ["pairing", "distance_nm", "theta_deg", "energy_per_area_J_m2", "torque_per_area_J_m2_rad", "anisotropic_energy_J_m2"])
    _records_to_csv(paths["data"] / "matsubara_contribution.csv", matsubara, ["n", "energy_contribution_J_m2"])
    _records_to_csv(paths["data"] / "q_integrand_contribution.csv", q_contrib, ["Q_nm_inv", "energy_contribution_J_m2"])
    _records_to_csv(paths["data"] / "ward_residual_grid.csv", ward_rows, ["pairing", "distance_nm", "theta_deg", "n", "Q_nm_inv", "phi_deg", "max_ward_residual_norm", "left_ward_residual_norm", "right_ward_residual_norm"])
    return outputs


def _plot_outputs(aggregated: dict[str, list[dict[str, Any]]], paths: dict[str, Path], args: argparse.Namespace) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []

    def save(name: str) -> None:
        path = paths["figures"] / name
        tmp = _unique_tmp_path(path)
        plt.tight_layout()
        plt.savefig(tmp, dpi=180, format="png")
        plt.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, path)
        figures.append(_path_label(path))

    energy = aggregated["energy_grid"]
    pressure = aggregated["pressure_rows"]
    torque = aggregated["torque_rows"]
    mats = aggregated["matsubara"]
    q_contrib = aggregated["q_contrib"]
    ward = aggregated["ward_rows"]
    pairings = sorted({row["pairing"] for row in energy})
    if energy:
        theta0 = min({abs(float(row["theta_deg"])) for row in energy})
        if args.distance_scan:
            plt.figure(figsize=(6, 4))
            for pairing in pairings:
                rows = sorted([r for r in energy if r["pairing"] == pairing and abs(float(r["theta_deg"])) == theta0], key=lambda r: r["distance_nm"])
                if rows:
                    plt.plot([r["distance_nm"] for r in rows], [r["energy_per_area_J_m2"] for r in rows], marker="o", label=pairing)
            plt.xlabel("distance (nm)")
            plt.ylabel("energy per area (J/m^2)")
            plt.title("Finite-q BdG Casimir energy vs distance")
            plt.legend()
            save("casimir_energy_vs_distance.png")

            plt.figure(figsize=(6, 4))
            for pairing in pairings:
                rows = sorted([r for r in energy if r["pairing"] == pairing and abs(float(r["theta_deg"])) == theta0], key=lambda r: r["distance_nm"])
                if rows:
                    plt.loglog([r["distance_nm"] for r in rows], [abs(r["energy_per_area_J_m2"]) for r in rows], marker="o", label=pairing)
            plt.xlabel("distance (nm)")
            plt.ylabel("|energy per area| (J/m^2)")
            plt.title("Finite-q BdG Casimir energy magnitude")
            plt.legend()
            save("casimir_energy_vs_distance_loglog.png")

        if args.pairing_comparison:
            plt.figure(figsize=(6, 4))
            for pairing in pairings:
                rows = sorted([r for r in energy if r["pairing"] == pairing and abs(float(r["theta_deg"])) == theta0], key=lambda r: r["distance_nm"])
                if rows:
                    plt.plot([r["distance_nm"] for r in rows], [r["energy_per_area_J_m2"] for r in rows], marker="o", label=pairing)
            plt.xlabel("distance (nm)")
            plt.ylabel("energy per area (J/m^2)")
            plt.title("Pairing comparison at theta=0 deg")
            plt.legend()
            save("casimir_pairing_comparison_vs_distance.png")

        first_distance = min({float(row["distance_nm"]) for row in energy})
        if args.angle_scan:
            plt.figure(figsize=(6, 4))
            for pairing in pairings:
                rows = sorted([r for r in energy if r["pairing"] == pairing and float(r["distance_nm"]) == first_distance], key=lambda r: r["theta_deg"])
                if rows:
                    plt.plot([r["theta_deg"] for r in rows], [r["energy_per_area_J_m2"] for r in rows], marker="o", label=pairing)
            plt.xlabel("angle (deg)")
            plt.ylabel("energy per area (J/m^2)")
            plt.title(f"Finite-q BdG Casimir energy vs angle, d={first_distance:g} nm")
            plt.legend()
            save("casimir_energy_vs_angle.png")

            if torque:
                plt.figure(figsize=(6, 4))
                for pairing in sorted({row["pairing"] for row in torque}):
                    rows = sorted([r for r in torque if r["pairing"] == pairing and float(r["distance_nm"]) == first_distance], key=lambda r: r["theta_deg"])
                    if rows:
                        plt.plot([r["theta_deg"] for r in rows], [r["torque_per_area_J_m2_rad"] for r in rows], marker="o", label=pairing)
                plt.xlabel("angle (deg)")
                plt.ylabel("torque per area (J/(m^2 rad))")
                plt.title("Finite-difference Casimir torque")
                plt.legend()
                save("casimir_torque_vs_angle.png")

                plt.figure(figsize=(6, 4))
                for pairing in sorted({row["pairing"] for row in torque}):
                    rows = sorted([r for r in torque if r["pairing"] == pairing and float(r["distance_nm"]) == first_distance], key=lambda r: r["theta_deg"])
                    if rows:
                        plt.plot([r["theta_deg"] for r in rows], [r["anisotropic_energy_J_m2"] for r in rows], marker="o", label=pairing)
                plt.xlabel("angle (deg)")
                plt.ylabel("Delta energy per area (J/m^2)")
                plt.title("Angle-dependent Casimir energy")
                plt.legend()
                save("casimir_anisotropic_energy_vs_angle.png")

        if args.distance_scan and pressure:
            plt.figure(figsize=(6, 4))
            for pairing in sorted({row["pairing"] for row in pressure}):
                rows = sorted([r for r in pressure if r["pairing"] == pairing and abs(float(r["theta_deg"])) == theta0], key=lambda r: r["distance_nm"])
                if rows:
                    plt.plot([r["distance_nm"] for r in rows], [r["pressure_N_m2"] for r in rows], marker="o", label=pairing)
            plt.xlabel("distance (nm)")
            plt.ylabel("pressure (N/m^2)")
            plt.title("Finite-difference Casimir pressure")
            plt.legend()
            save("casimir_pressure_vs_distance.png")

        if args.heatmap_scan:
            heat_rows = [r for r in energy if r["pairing"] == pairings[-1]]
            if heat_rows:
                distances = sorted({r["distance_nm"] for r in heat_rows})
                angles = sorted({r["theta_deg"] for r in heat_rows})
                grid = np.full((len(distances), len(angles)), np.nan)
                for r in heat_rows:
                    grid[distances.index(r["distance_nm"]), angles.index(r["theta_deg"])] = r["energy_per_area_J_m2"]
                plt.figure(figsize=(6, 4))
                plt.imshow(grid, aspect="auto", origin="lower", extent=[min(angles), max(angles), min(distances), max(distances)])
                plt.xlabel("angle (deg)")
                plt.ylabel("distance (nm)")
                plt.colorbar(label="energy per area (J/m^2)")
                plt.title(f"Energy heatmap ({pairings[-1]})")
                save("casimir_energy_distance_angle_heatmap.png")

    if mats:
        plt.figure(figsize=(6, 4))
        plt.bar([r["n"] for r in mats], [r["energy_contribution_J_m2"] for r in mats])
        plt.xlabel("Matsubara index n")
        plt.ylabel("energy contribution (J/m^2)")
        plt.title("Matsubara contribution (n0 policy recorded in summary)")
        save("matsubara_contribution.png")
    if q_contrib:
        plt.figure(figsize=(6, 4))
        plt.plot([r["Q_nm_inv"] for r in q_contrib], [r["energy_contribution_J_m2"] for r in q_contrib], marker="o")
        plt.xlabel("Q (nm^-1)")
        plt.ylabel("energy contribution (J/m^2)")
        plt.title("Q-resolved contribution")
        save("q_integrand_contribution.png")
    if ward:
        plt.figure(figsize=(6, 4))
        plt.scatter([r["Q_nm_inv"] for r in ward], [r["max_ward_residual_norm"] for r in ward], s=8)
        plt.xlabel("Q (nm^-1)")
        plt.ylabel("max Ward residual norm")
        plt.yscale("log")
        plt.title("Ward residual metadata over Casimir grid")
        save("ward_residual_over_casimir_grid.png")
    return figures


def _contracts(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "response_contract": {
            "phase_vertex": "bond_endpoint_gauge",
            "current_vertex": "peierls",
            "collective_mode": "amplitude_phase",
            "collective_counterterm": "goldstone_gap_equation",
            "include_phase_phase_direct": True,
            "full_response_source": FULL_RESPONSE_SOURCE,
            "diagnostic_components_available": ["bare_bubble", "direct", "bare_total", "minus_schur", "plus_schur"],
        },
        "unit_contract": {
            **bilayer_sheet_conductivity_convention_metadata(),
            "sheet_conductivity_formula": "sigma_sheet_SI = (e^2 / hbar) * sigma_model",
            "reflection_dimensionless_formula": "sigma_tilde = sigma_sheet_SI / sigma0",
            "double_conversion_guarded_by": "SheetConductivityConversion.unit_stage",
        },
        "reflection_contract": {
            "adapter": "sigma_tilde_xy_to_te_tm_reflection_matrix",
            "formula": "R_TE_TM = [[R_TT, R_TL], [-R_LT, -R_LL]]",
        },
        "casimir_contract": {
            **lifshitz_integrand_metadata(),
            "formula": "F/A = k_B T sum_n' int Q dQ dphi/(2pi)^2 log det[I-exp(-2*kappa*d) R1 R2]",
            "n0_weight": 0.5,
            "n_ge_1_weight": 1.0,
            "logdet_real_enters_energy": True,
            "logdet_imag_metadata": True,
        },
        "material_boundary_model": "bilayer_normalized_2D_sheet",
        "not_bulk_3d": True,
        "not_finite_thickness_slab": True,
        "n0_policy": args.n0_policy,
        "valid_for_formal_casimir_claim": False,
        "not_final_material_conclusion": True,
        "ward_residual_recorded_not_gating": True,
        "main_production_pipeline_v1": True,
        "current_numerical_quality_metadata_retained": True,
    }


def _n0_summary(reflection_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    n0_rows = [row for row in reflection_rows if int(row.get("n", -1)) == 0]
    stable = [row for row in n0_rows if row.get("n0_stability_status") == "stable"]
    monitor = [row for row in n0_rows if row.get("n0_stability_status") == "monitor"]
    unstable = [row for row in n0_rows if row.get("n0_stability_status") == "unstable"]
    variations = [float(row.get("n0_reflection_norm_variation", 0.0)) for row in n0_rows]
    payload = {
        "n0_policy": args.n0_policy,
        "num_n0_points": int(len(n0_rows)),
        "num_n0_stable": int(len(stable)),
        "num_n0_monitor": int(len(monitor)),
        "num_n0_unstable": int(len(unstable)),
        "max_n0_reflection_norm_variation": float(max(variations, default=0.0)),
    }
    if args.n0_policy == "skip":
        payload.update(
            {
                "complete_matsubara_sum": False,
                "complete_except_n0": True,
                "not_final_full_matsubara_result": True,
            }
        )
    else:
        payload.update(
            {
                "complete_matsubara_sum": True,
                "n0_uses_reflection_extrapolation": True,
            }
        )
    return payload


def _summary_payload(
    args: argparse.Namespace,
    run_config: dict[str, Any],
    reflection_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    figures: list[str],
    data_files: list[str],
) -> dict[str, Any]:
    completed = [r for r in rows if r.get("status") == "completed"]
    num_points = len(rows) + len(failed)
    max_ward = max((float(r.get("max_ward_residual_norm", 0.0)) for r in completed), default=0.0)
    max_logdet_imag = max((float(r.get("abs_logdet_imag", 0.0)) for r in completed), default=0.0)
    return {
        "pipeline_name": "finite_q_bdg_casimir_pipeline",
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": run_config["config_hash"],
        "run_config_hash": run_config["run_config_hash"],
        "pipeline_complete": len(failed) == 0,
        "pipeline_status": "completed" if len(failed) == 0 else "completed_with_failures",
        **_contracts(args),
        "pairing_channels": list(args.pairings),
        "distance_grid_nm": [float(v) for v in args.distances_nm],
        "angle_grid_deg": [float(v) for v in args.angles_deg],
        "matsubara_grid": {"n_min": 0, "n_max": int(args.n_max)},
        "Q_grid": {"q_num": int(args.q_num), "q_max_nm_inv": float(args.q_max_nm_inv)},
        "phi_grid": {"phi_num": int(args.phi_num)},
        "integration_strategy": {
            "integration_strategy": args.integration_strategy,
            "coarse_grid": int(args.coarse_grid),
            "adaptive_level": None if args.integration_strategy == "uniform" else int(args.adaptive_level),
            "gauss_order": None if args.integration_strategy == "uniform" else int(args.gauss_order),
            "fermi_window_eV": None if args.integration_strategy == "uniform" else float(args.fermi_window_eV),
            "eta_eV": float(args.eta_eV),
            "q_specific_adaptive_grid": False if args.integration_strategy == "uniform" else bool(args.q_specific_adaptive_grid),
            "source_module": "src/lno327/workflows/finite_q_quadrature.py",
        },
        "distance_reuse_enabled": True,
        "num_plate_reflection_rows": int(len(reflection_rows)),
        "num_roundtrip_tasks": _count_roundtrip_tasks(args),
        "num_energy_points": _count_energy_points(args),
        "num_completed_points": int(len(completed)),
        "num_failed_points": int(len(failed)),
        "failed_fraction": float(len(failed) / max(num_points, 1)),
        "task_shard_index": int(args.task_shard_index),
        "task_shard_count": int(args.task_shard_count),
        "shard_specific_jsonl": True,
        "max_ward_residual": float(max_ward),
        "max_logdet_imag": float(max_logdet_imag),
        "n0_summary": _n0_summary(reflection_rows, args),
        "energy_per_area_outputs": ["data/energy_vs_distance.csv", "data/energy_vs_angle.csv", "data/energy_distance_angle_grid.csv"],
        "pressure_outputs": ["data/pressure_vs_distance.csv"],
        "torque_outputs": ["data/torque_vs_angle.csv"],
        "figure_files": figures,
        "data_files": data_files,
    }


def _run_status_payload(
    *,
    total_tasks: int,
    completed: int,
    failed: int,
    skipped: int,
    current_task: str | None,
    start_time: str,
    pipeline_status: str,
) -> dict[str, Any]:
    start_dt = datetime.fromisoformat(start_time)
    elapsed = (datetime.now(timezone.utc) - start_dt).total_seconds()
    return {
        "total_plate_reflection_tasks_current_shard": int(total_tasks),
        "completed_plate_reflection_tasks": int(completed),
        "failed_plate_reflection_tasks": int(failed),
        "skipped_existing_plate_reflection_tasks": int(skipped),
        "current_task": current_task,
        "start_time": start_time,
        "last_update_time": _now_iso(),
        "elapsed_seconds": float(elapsed),
        "pipeline_status": pipeline_status,
    }


def _progress(iterable: Any, total: int, desc: str) -> Any:
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def finalize_outputs(args: argparse.Namespace, paths: dict[str, Path], run_config: dict[str, Any]) -> dict[str, Any]:
    config_hash = str(run_config["config_hash"])
    reflection_rows = _filter_config_hash(_load_many_jsonl(_shard_glob(paths, "reflection_results")), config_hash)
    _atomic_write_jsonl(paths["reflection_results"], reflection_rows)
    energy_rows, roundtrip_failures = _roundtrip_rows(args, config_hash, reflection_rows)
    _atomic_write_jsonl(paths["energy_shard"], energy_rows)
    existing_failed = _filter_config_hash(_load_many_jsonl(_shard_glob(paths, "failed_points")), config_hash)
    all_failed = existing_failed + roundtrip_failures
    _atomic_write_jsonl(paths["failed_points"], all_failed)
    all_energy_rows = _filter_config_hash(_load_many_jsonl(_shard_glob(paths, "energy_point_results")), config_hash)
    _atomic_write_jsonl(paths["point_results"], all_energy_rows)
    aggregated = _aggregate(all_energy_rows, paths)
    figures = _plot_outputs(aggregated, paths, args)
    data_files = sorted(_path_label(path) for path in paths["data"].glob("*.csv"))
    summary = _summary_payload(args, run_config, reflection_rows, all_energy_rows, all_failed, figures, data_files)
    _atomic_write_json(paths["summary"], summary)
    _atomic_write_json(paths["status"], summary)
    return summary


def _dry_run_payload(args: argparse.Namespace, run_config: dict[str, Any], tasks: list[PlateReflectionTask]) -> dict[str, Any]:
    current_shard_roundtrip = _current_shard_roundtrip_count(args)
    return {
        "pipeline_name": "finite_q_bdg_casimir_pipeline",
        "dry_run": True,
        "config_hash": run_config["config_hash"],
        "output_dir": str(args.output_dir),
        "num_plate_reflection_tasks": int(len(args.pairings) * len(_plate_theta_values(args)) * (int(args.n_max) + 1) * int(args.q_num) * int(args.phi_num)),
        "num_roundtrip_tasks": _count_roundtrip_tasks(args),
        "num_energy_points": _count_energy_points(args),
        "num_distances": int(len(args.distances_nm)),
        "num_angles": int(len(args.angles_deg)),
        "num_pairings": int(len(args.pairings)),
        "n_max": int(args.n_max),
        "q_num": int(args.q_num),
        "phi_num": int(args.phi_num),
        "distance_reuse_enabled": True,
        "adaptive_quadrature_enabled": args.integration_strategy == "best_available_adaptive",
        "integration_strategy": args.integration_strategy,
        "shard_index": int(args.task_shard_index),
        "shard_count": int(args.task_shard_count),
        "current_shard_plate_reflection_tasks": int(len(tasks)),
        "current_shard_roundtrip_tasks": int(current_shard_roundtrip),
        "current_shard_energy_points": int(current_shard_roundtrip * len(args.distances_nm)),
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    paths = _setup_output(args.output_dir, args)
    run_config = _run_config_with_hash(args)
    tasks = build_plate_reflection_tasks(args, run_config["config_hash"])
    if args.dry_run:
        payload = _dry_run_payload(args, run_config, tasks)
        print(json.dumps(payload, indent=2, default=str))
        return payload
    _check_or_write_run_config(
        paths,
        run_config,
        resume=bool(args.resume),
        allow_mismatch=bool(args.allow_config_mismatch),
    )
    if args.plot_only:
        summary = finalize_outputs(args, paths, run_config)
        print(f"plot-only completed: figures={len(summary['figure_files'])}")
        return summary

    logger = _configure_logging(paths)
    start_time = _now_iso()
    logger.info("start pipeline git_commit=%s config_hash=%s args=%s", _git_commit_hash(), run_config["config_hash"], vars(args))
    completed_keys = _completed_task_keys(paths["reflection_shard"]) if args.resume else set()
    to_run = [task for task in tasks if task.task_key not in completed_keys]
    skipped = len(tasks) - len(to_run)
    failed_count = 0
    completed_count = len(completed_keys)
    config = _task_config(run_config)
    _atomic_write_json(
        paths["run_status"],
        _run_status_payload(
            total_tasks=len(tasks),
            completed=completed_count,
            failed=failed_count,
            skipped=skipped,
            current_task=None,
            start_time=start_time,
            pipeline_status="running",
        ),
    )

    def handle_failure(task: PlateReflectionTask, exc: BaseException) -> None:
        nonlocal failed_count
        failed_count += 1
        row = {
            **asdict(task),
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "valid_for_formal_casimir_claim": False,
        }
        _append_jsonl(paths["failed_shard"], row)
        logger.error("failed plate task=%s error=%s\n%s", task.task_key, exc, row["traceback"])

    if int(args.num_workers) > 1:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as executor:
            future_to_task = {executor.submit(_single_plate_reflection_task, task, config): task for task in to_run}
            for future in _progress(as_completed(future_to_task), len(future_to_task), "finite-q plate reflections"):
                task = future_to_task[future]
                try:
                    row = future.result()
                    _append_jsonl(paths["reflection_shard"], row)
                    completed_count += 1
                    logger.info("finished plate task=%s runtime=%s", task.task_key, row.get("runtime_seconds"))
                except Exception as exc:  # noqa: BLE001
                    handle_failure(task, exc)
                _atomic_write_json(
                    paths["run_status"],
                    _run_status_payload(
                        total_tasks=len(tasks),
                        completed=completed_count,
                        failed=failed_count,
                        skipped=skipped,
                        current_task=task.task_key,
                        start_time=start_time,
                        pipeline_status="running",
                    ),
                )
    else:
        for task in _progress(to_run, len(to_run), "finite-q plate reflections"):
            try:
                row = _single_plate_reflection_task(task, config)
                _append_jsonl(paths["reflection_shard"], row)
                completed_count += 1
                logger.info("finished plate task=%s runtime=%s", task.task_key, row.get("runtime_seconds"))
            except Exception as exc:  # noqa: BLE001
                handle_failure(task, exc)
            _atomic_write_json(
                paths["run_status"],
                _run_status_payload(
                    total_tasks=len(tasks),
                    completed=completed_count,
                    failed=failed_count,
                    skipped=skipped,
                    current_task=task.task_key,
                    start_time=start_time,
                    pipeline_status="running",
                ),
            )
    summary = finalize_outputs(args, paths, run_config)
    _atomic_write_json(
        paths["run_status"],
        _run_status_payload(
            total_tasks=len(tasks),
            completed=completed_count,
            failed=failed_count,
            skipped=skipped,
            current_task=None,
            start_time=start_time,
            pipeline_status=summary["pipeline_status"],
        ),
    )
    logger.info("end pipeline status=%s completed=%s failed=%s", summary["pipeline_status"], completed_count, failed_count)
    return summary


def run_self_check(output_dir: Path) -> dict[str, Any]:
    zero = np.zeros((2, 2), dtype=complex)
    logdet_zero = trace_log_point(zero, zero, 1.0e7, 50e-9)["logdet_integrand"]
    weak = np.eye(2, dtype=complex) * 1e-8
    logdet_weak = trace_log_point(weak, weak, 1.0e7, 50e-9)["logdet_integrand"]
    isotropic = np.eye(2, dtype=complex) * 0.1
    energy_angles = [
        trace_log_point(isotropic, rotate_xy_tensor(isotropic, theta), 1.0e7, 50e-9)["logdet_integrand"].real
        for theta in (0.0, 0.5, 1.0)
    ]
    payload = {
        "zero_sigma_logdet_zero": abs(logdet_zero) < 1e-14,
        "weak_conductivity_tends_to_zero": abs(logdet_weak) < 1e-12,
        "identical_isotropic_sheets_torque_zero": float(np.ptp(energy_angles)) < 1e-14,
        "no_nan_inf": bool(np.isfinite(logdet_zero.real) and np.isfinite(logdet_weak.real)),
        "valid_for_formal_casimir_claim": False,
    }
    paths = _setup_output(output_dir, argparse.Namespace(task_shard_index=0, task_shard_count=1))
    _atomic_write_json(paths["root"] / "self_check.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finite-q BdG Casimir main production pipeline v1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairings", nargs="+", choices=("normal", "spm", "dwave"), default=["normal", "spm", "dwave"])
    parser.add_argument("--distances-nm", nargs="+", type=float, default=[20, 30, 50, 80, 100, 150, 200])
    parser.add_argument("--angles-deg", nargs="+", type=float, default=[0, 15, 30, 45, 60, 75, 90])
    parser.add_argument("--n-max", type=int, default=16)
    parser.add_argument("--q-num", type=int, default=24)
    parser.add_argument("--q-max-nm-inv", type=float, default=0.05)
    parser.add_argument("--q-max-m-inv", type=float)
    parser.add_argument("--phi-num", type=int, default=12)
    parser.add_argument("--temperature-K", type=float, default=30.0)
    parser.add_argument("--delta0-eV", type=float, default=0.04)
    parser.add_argument("--eta-eV", type=float, default=1e-10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--n0-policy", choices=("extrapolate", "skip"), default="extrapolate")
    parser.add_argument("--distance-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--angle-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heatmap-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pairing-comparison", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--integration-strategy", choices=("uniform", "best_available_adaptive"), default="best_available_adaptive")
    parser.add_argument("--coarse-grid", type=int, default=32)
    parser.add_argument("--adaptive-level", type=int, default=5)
    parser.add_argument("--gauss-order", type=int, default=5)
    parser.add_argument("--fermi-window-eV", type=float, default=0.12)
    parser.add_argument("--q-specific-adaptive-grid", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.q_max_m_inv is not None:
        args.q_max_nm_inv = float(args.q_max_m_inv) / 1e9
    if args.task_shard_count <= 0:
        raise ValueError("--task-shard-count must be positive")
    if not (0 <= args.task_shard_index < args.task_shard_count):
        raise ValueError("--task-shard-index must satisfy 0 <= index < count")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        payload = run_self_check(args.output_dir)
        print(json.dumps(payload, indent=2))
        checks = [
            "zero_sigma_logdet_zero",
            "weak_conductivity_tends_to_zero",
            "identical_isotropic_sheets_torque_zero",
            "no_nan_inf",
        ]
        return 0 if all(bool(payload[key]) for key in checks) else 1
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
