#!/usr/bin/env python3
"""Finite-q BdG Casimir main production pipeline v1."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.casimir_integrand import casimir_integrand_single_point  # noqa: E402
from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.constants import C0, KB  # noqa: E402
from lno327.finite_q_engine import FiniteQEngineOptions, finite_q_bdg_response_from_ansatz  # noqa: E402
from lno327.material_structure import LNO327_THIN_FILM_SLAO_IN_PLANE  # noqa: E402
from lno327.pairing import PairingAmplitudes, build_pairing_ansatz  # noqa: E402
from lno327.reflection_input import (  # noqa: E402
    sigma_tilde_xy_to_te_tm_reflection_matrix,
    vacuum_kappa,
)
from lno327.response_conventions import (  # noqa: E402
    bilayer_sheet_conductivity_convention_metadata,
    model_response_to_sheet_conductivity,
    sheet_conductivity_to_reflection_dimensionless,
    spatial_response_to_bilayer_sheet_conductivity_model,
)

try:  # pragma: no cover - exercised when tqdm is available on a server.
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


PIPELINE_VERSION = "finite_q_bdg_casimir_pipeline_v1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "casimir" / "finite_q_bdg_pipeline"
FIGURE_FILES = [
    "casimir_energy_vs_distance.png",
    "casimir_energy_vs_distance_loglog.png",
    "casimir_pressure_vs_distance.png",
    "casimir_energy_vs_angle.png",
    "casimir_torque_vs_angle.png",
    "casimir_energy_distance_angle_heatmap.png",
    "casimir_pairing_comparison_vs_distance.png",
    "casimir_anisotropic_energy_vs_angle.png",
    "matsubara_contribution.png",
    "q_integrand_contribution.png",
    "ward_residual_over_casimir_grid.png",
]


@dataclass(frozen=True)
class PipelineTask:
    task_key: str
    task_index: int
    task_shard_index: int
    task_shard_count: int
    pairing: str
    distance_nm: float
    theta_deg: float
    n: int
    q_index: int
    phi_index: int
    omega_eV: float
    Q_nm_inv: float
    phi_rad: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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


def _integration_points(coarse_grid: int) -> tuple[np.ndarray, np.ndarray]:
    points = uniform_bz_mesh(int(coarse_grid))
    return points, k_weights(points)


def _pairing_ansatz_name(pairing: str) -> str:
    return "onsite_s" if pairing == "normal" else pairing


def _pairing_delta(pairing: str, delta0_eV: float) -> float:
    return 0.0 if pairing == "normal" else float(delta0_eV)


def _compute_full_response(
    pairing: str,
    omega_eV: float,
    q_model: np.ndarray,
    *,
    temperature_K: float,
    delta0_eV: float,
    eta_eV: float,
    coarse_grid: int,
) -> tuple[np.ndarray, dict[str, float]]:
    ansatz = build_pairing_ansatz(_pairing_ansatz_name(pairing), phase_vertex="bond_endpoint_gauge")
    params = PairingAmplitudes(delta0_eV=_pairing_delta(pairing, delta0_eV))
    points, weights = _integration_points(coarse_grid)
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
    coarse_grid: int,
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
        coarse_grid=coarse_grid,
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
    return reflection, meta


def _extrapolated_reflection_for_pairing(
    pairing: str,
    omega_eps_list: list[float],
    q_model_lab: np.ndarray,
    *,
    plate_theta_rad: float,
    temperature_K: float,
    delta0_eV: float,
    eta_eV: float,
    coarse_grid: int,
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
            coarse_grid=coarse_grid,
            lattice_a_x_m=lattice_a_x_m,
            lattice_a_y_m=lattice_a_y_m,
        )
        matrices.append(reflection)
        metas.append(meta)
    x = np.asarray(omega_eps_list, dtype=float)
    y = np.asarray(matrices, dtype=complex).reshape(len(matrices), 4)
    coefficients = np.vstack([np.polyfit(x, y[:, i], 1) for i in range(4)])
    extrapolated = coefficients[:, 1].reshape(2, 2)
    meta = dict(metas[-1])
    meta.update(
        {
            "n0_status": "extrapolated",
            "n0_extrapolation_omega_eV": [float(v) for v in omega_eps_list],
            "reflection_norm": float(np.linalg.norm(extrapolated)),
            "reflection_TE_TM_norm": float(np.linalg.norm(extrapolated)),
        }
    )
    return extrapolated, meta


def _single_task(task: PipelineTask, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    distance_m = float(task.distance_nm) * 1e-9
    theta_rad = np.deg2rad(float(task.theta_deg))
    phi = float(task.phi_rad)
    Q_m_inv = float(task.Q_nm_inv) * 1e9
    q_model_lab = np.array(
        [
            Q_m_inv * np.cos(phi) * float(config["lattice_a_x_m"]),
            Q_m_inv * np.sin(phi) * float(config["lattice_a_y_m"]),
        ],
        dtype=float,
    )
    omega_eV = float(task.omega_eV)
    n0_status = "not_n0"
    if task.n == 0 and config["n0_policy"] == "skip":
        return {
            **asdict(task),
            "distance_m": distance_m,
            "theta_rad": theta_rad,
            "phi_deg": float(np.rad2deg(phi)),
            "q_model_x": float(q_model_lab[0]),
            "q_model_y": float(q_model_lab[1]),
            "temperature_K": float(config["temperature_K"]),
            "delta0_eV": float(config["delta0_eV"]),
            "integration_strategy": config["integration_strategy"],
            "full_response_source": "amplitude_phase_schur",
            "status": "skipped_n0",
            "n0_status": "skipped",
            "runtime_seconds": float(time.perf_counter() - started),
        }
    if task.n == 0:
        omega_1 = matsubara_omega_eV(1, float(config["temperature_K"]))
        omega_eps_list = [omega_1 / 8.0, omega_1 / 16.0, omega_1 / 32.0]
        left, left_meta = _extrapolated_reflection_for_pairing(
            task.pairing,
            omega_eps_list,
            q_model_lab,
            plate_theta_rad=0.0,
            temperature_K=float(config["temperature_K"]),
            delta0_eV=float(config["delta0_eV"]),
            eta_eV=float(config["eta_eV"]),
            coarse_grid=int(config["coarse_grid"]),
            lattice_a_x_m=float(config["lattice_a_x_m"]),
            lattice_a_y_m=float(config["lattice_a_y_m"]),
        )
        right, right_meta = _extrapolated_reflection_for_pairing(
            task.pairing,
            omega_eps_list,
            q_model_lab,
            plate_theta_rad=theta_rad,
            temperature_K=float(config["temperature_K"]),
            delta0_eV=float(config["delta0_eV"]),
            eta_eV=float(config["eta_eV"]),
            coarse_grid=int(config["coarse_grid"]),
            lattice_a_x_m=float(config["lattice_a_x_m"]),
            lattice_a_y_m=float(config["lattice_a_y_m"]),
        )
        xi_si = 0.0
        kappa = float(vacuum_kappa(Q_m_inv, 1e-300))
        n0_status = "extrapolated"
        omega_eV = omega_eps_list[-1]
    else:
        left, left_meta = _reflection_for_pairing(
            task.pairing,
            omega_eV,
            q_model_lab,
            plate_theta_rad=0.0,
            temperature_K=float(config["temperature_K"]),
            delta0_eV=float(config["delta0_eV"]),
            eta_eV=float(config["eta_eV"]),
            coarse_grid=int(config["coarse_grid"]),
            lattice_a_x_m=float(config["lattice_a_x_m"]),
            lattice_a_y_m=float(config["lattice_a_y_m"]),
        )
        right, right_meta = _reflection_for_pairing(
            task.pairing,
            omega_eV,
            q_model_lab,
            plate_theta_rad=theta_rad,
            temperature_K=float(config["temperature_K"]),
            delta0_eV=float(config["delta0_eV"]),
            eta_eV=float(config["eta_eV"]),
            coarse_grid=int(config["coarse_grid"]),
            lattice_a_x_m=float(config["lattice_a_x_m"]),
            lattice_a_y_m=float(config["lattice_a_y_m"]),
        )
        xi_si = float(left_meta["xi_si_s_inv"])
        kappa = float(left_meta["kappa_m_inv"])
    logdet = casimir_integrand_single_point(left, right, kappa, distance_m)["logdet_integrand"]
    delta_q = float(config["q_max_nm_inv"]) * 1e9 / float(config["q_num"])
    delta_phi = 2.0 * np.pi / float(config["phi_num"])
    measure = q_phi_measure(Q_m_inv, delta_q, delta_phi)
    contribution = (
        KB
        * float(config["temperature_K"])
        * matsubara_weight(task.n)
        * measure
        * complex(logdet)
    )
    row = {
        **asdict(task),
        "distance_m": distance_m,
        "theta_rad": theta_rad,
        "phi_deg": float(np.rad2deg(phi)),
        "Q_m_inv": Q_m_inv,
        "q_model_x": float(q_model_lab[0]),
        "q_model_y": float(q_model_lab[1]),
        "temperature_K": float(config["temperature_K"]),
        "delta0_eV": float(config["delta0_eV"]),
        "integration_strategy": config["integration_strategy"],
        "coarse_grid": int(config["coarse_grid"]),
        "adaptive_level": int(config["adaptive_level"]),
        "gauss_order": int(config["gauss_order"]),
        "fermi_window_eV": float(config["fermi_window_eV"]),
        "eta_eV": float(config["eta_eV"]),
        "q_specific_adaptive_grid": bool(config["q_specific_adaptive_grid"]),
        "full_response_source": "amplitude_phase_schur",
        "max_ward_residual_norm": float(max(left_meta["max_ward_residual_norm"], right_meta["max_ward_residual_norm"])),
        "left_ward_residual_norm": float(max(left_meta["left_ward_residual_norm"], right_meta["left_ward_residual_norm"])),
        "right_ward_residual_norm": float(max(left_meta["right_ward_residual_norm"], right_meta["right_ward_residual_norm"])),
        "sigma_model_norm": float(max(left_meta["sigma_model_norm"], right_meta["sigma_model_norm"])),
        "sigma_sheet_norm": float(max(left_meta["sigma_sheet_norm"], right_meta["sigma_sheet_norm"])),
        "sigma_tilde_norm": float(max(left_meta["sigma_tilde_norm"], right_meta["sigma_tilde_norm"])),
        "reflection_norm": float(max(left_meta["reflection_norm"], right_meta["reflection_norm"])),
        "kappa_m_inv": kappa,
        "xi_si_s_inv": xi_si,
        "sigma_tilde_xy_norm": float(max(left_meta["sigma_tilde_xy_norm"], right_meta["sigma_tilde_xy_norm"])),
        "sigma_tilde_LT_norm": float(max(left_meta["sigma_tilde_LT_norm"], right_meta["sigma_tilde_LT_norm"])),
        "reflection_TE_TM_norm": float(max(left_meta["reflection_TE_TM_norm"], right_meta["reflection_TE_TM_norm"])),
        "reflection_condition_status": "computed",
        "logdet_real": float(np.real(logdet)),
        "logdet_imag": float(np.imag(logdet)),
        "abs_logdet_imag": float(abs(np.imag(logdet))),
        "energy_contribution_real_J_m2": float(np.real(contribution)),
        "energy_contribution_imag_J_m2": float(np.imag(contribution)),
        "n0_policy": config["n0_policy"],
        "n0_status": n0_status,
        "status": "completed",
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return row


def _task_key(pairing: str, distance: float, theta: float, n: int, q_index: int, phi_index: int, n0_policy: str) -> str:
    return (
        f"pairing={pairing}|distance_nm={distance:.12g}|theta_deg={theta:.12g}|"
        f"n={n}|Q_index={q_index}|phi_index={phi_index}|n0_policy={n0_policy}"
    )


def build_tasks(args: argparse.Namespace) -> list[PipelineTask]:
    q_values = (np.arange(int(args.q_num), dtype=float) + 0.5) * float(args.q_max_nm_inv) / float(args.q_num)
    phi_values = (np.arange(int(args.phi_num), dtype=float) + 0.5) * 2.0 * np.pi / float(args.phi_num)
    tasks = []
    index = 0
    for pairing in args.pairings:
        for distance in args.distances_nm:
            for theta in args.angles_deg:
                for n in range(int(args.n_max) + 1):
                    omega = matsubara_omega_eV(n, float(args.temperature_K))
                    for q_index, q_value in enumerate(q_values):
                        for phi_index, phi in enumerate(phi_values):
                            key = _task_key(pairing, distance, theta, n, q_index, phi_index, args.n0_policy)
                            if index % int(args.task_shard_count) == int(args.task_shard_index):
                                tasks.append(
                                    PipelineTask(
                                        task_key=key,
                                        task_index=index,
                                        task_shard_index=int(args.task_shard_index),
                                        task_shard_count=int(args.task_shard_count),
                                        pairing=str(pairing),
                                        distance_nm=float(distance),
                                        theta_deg=float(theta),
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


def _setup_output(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "data": output_dir / "data",
        "figures": output_dir / "figures",
        "logs": output_dir / "logs",
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
        "total_tasks": int(total_tasks),
        "completed_tasks": int(completed),
        "failed_tasks": int(failed),
        "skipped_existing_tasks": int(skipped),
        "current_task": current_task,
        "start_time": start_time,
        "last_update_time": _now_iso(),
        "elapsed_seconds": float(elapsed),
        "pipeline_status": pipeline_status,
    }


def _records_to_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
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
                torque_rows.append({**row, "torque_per_area_J_m2_rad": float(t), "anisotropic_energy_J_m2": float(row["energy_per_area_J_m2"] - avg)})
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


def _plot_outputs(aggregated: dict[str, list[dict[str, Any]]], paths: dict[str, Path]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []

    def save(name: str) -> None:
        path = paths["figures"] / name
        tmp = path.with_name(path.name + ".tmp")
        plt.tight_layout()
        plt.savefig(tmp, dpi=180, format="png")
        plt.close()
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

        if pressure:
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
            "full_response_source": "amplitude_phase_schur",
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


def _summary_payload(args: argparse.Namespace, rows: list[dict[str, Any]], failed: list[dict[str, Any]], figures: list[str], data_files: list[str]) -> dict[str, Any]:
    completed = [r for r in rows if r.get("status") == "completed"]
    num_points = len(rows) + len(failed)
    max_ward = max((float(r.get("max_ward_residual_norm", 0.0)) for r in completed), default=0.0)
    max_logdet_imag = max((float(r.get("abs_logdet_imag", 0.0)) for r in completed), default=0.0)
    return {
        "pipeline_name": "finite_q_bdg_casimir_pipeline",
        "pipeline_version": PIPELINE_VERSION,
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
            "adaptive_level": int(args.adaptive_level),
            "gauss_order": int(args.gauss_order),
            "fermi_window_eV": float(args.fermi_window_eV),
            "eta_eV": float(args.eta_eV),
            "q_specific_adaptive_grid": bool(args.q_specific_adaptive_grid),
        },
        "num_points": int(num_points),
        "num_completed_points": int(len(completed)),
        "num_failed_points": int(len(failed)),
        "failed_fraction": float(len(failed) / max(num_points, 1)),
        "max_ward_residual": float(max_ward),
        "max_logdet_imag": float(max_logdet_imag),
        "energy_per_area_outputs": ["data/energy_vs_distance.csv", "data/energy_vs_angle.csv", "data/energy_distance_angle_grid.csv"],
        "pressure_outputs": ["data/pressure_vs_distance.csv"],
        "torque_outputs": ["data/torque_vs_angle.csv"],
        "figure_files": figures,
        "data_files": data_files,
    }


def finalize_outputs(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    rows = _load_jsonl(paths["point_results"])
    failed = _load_jsonl(paths["failed_points"])
    aggregated = _aggregate(rows, paths)
    figures = _plot_outputs(aggregated, paths)
    data_files = sorted(_path_label(path) for path in paths["data"].glob("*.csv"))
    summary = _summary_payload(args, rows, failed, figures, data_files)
    _atomic_write_json(paths["summary"], summary)
    _atomic_write_json(paths["status"], summary)
    return summary


def _progress(iterable: Any, total: int, desc: str) -> Any:
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    paths = _setup_output(args.output_dir)
    logger = _configure_logging(paths)
    tasks = build_tasks(args)
    if args.dry_run:
        payload = {
            "pipeline_name": "finite_q_bdg_casimir_pipeline",
            "dry_run": True,
            "total_tasks": len(tasks),
            "output_dir": str(args.output_dir),
            "grid_spec": vars(args),
        }
        print(json.dumps(payload, indent=2, default=str))
        return payload
    if args.plot_only:
        summary = finalize_outputs(args, paths)
        print(f"plot-only completed: figures={len(summary['figure_files'])}")
        return summary

    start_time = _now_iso()
    logger.info("start pipeline git_commit=%s args=%s", _git_commit_hash(), vars(args))
    completed_keys = _completed_task_keys(paths["point_results"]) if args.resume else set()
    to_run = [task for task in tasks if task.task_key not in completed_keys]
    skipped = len(tasks) - len(to_run)
    failed_count = 0
    completed_count = len(completed_keys)
    config = {
        "temperature_K": float(args.temperature_K),
        "delta0_eV": float(args.delta0_eV),
        "eta_eV": float(args.eta_eV),
        "coarse_grid": int(args.coarse_grid),
        "adaptive_level": int(args.adaptive_level),
        "gauss_order": int(args.gauss_order),
        "fermi_window_eV": float(args.fermi_window_eV),
        "q_specific_adaptive_grid": bool(args.q_specific_adaptive_grid),
        "integration_strategy": args.integration_strategy,
        "q_num": int(args.q_num),
        "q_max_nm_inv": float(args.q_max_nm_inv),
        "phi_num": int(args.phi_num),
        "n0_policy": args.n0_policy,
        "lattice_a_x_m": float(LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m),
        "lattice_a_y_m": float(LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m),
    }
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

    def handle_failure(task: PipelineTask, exc: BaseException) -> None:
        nonlocal failed_count
        failed_count += 1
        row = {
            **asdict(task),
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "valid_for_formal_casimir_claim": False,
        }
        _append_jsonl(paths["failed_points"], row)
        logger.error("failed task=%s error=%s\n%s", task.task_key, exc, row["traceback"])

    if int(args.num_workers) > 1:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as executor:
            future_to_task = {executor.submit(_single_task, task, config): task for task in to_run}
            for future in _progress(as_completed(future_to_task), len(future_to_task), "finite-q BdG Casimir"):
                task = future_to_task[future]
                try:
                    row = future.result()
                    _append_jsonl(paths["point_results"], row)
                    completed_count += 1
                    logger.info("finished task=%s runtime=%s", task.task_key, row.get("runtime_seconds"))
                except Exception as exc:  # noqa: BLE001
                    handle_failure(task, exc)
                _atomic_write_json(paths["run_status"], _run_status_payload(total_tasks=len(tasks), completed=completed_count, failed=failed_count, skipped=skipped, current_task=task.task_key, start_time=start_time, pipeline_status="running"))
    else:
        for task in _progress(to_run, len(to_run), "finite-q BdG Casimir"):
            try:
                row = _single_task(task, config)
                _append_jsonl(paths["point_results"], row)
                completed_count += 1
                logger.info("finished task=%s runtime=%s", task.task_key, row.get("runtime_seconds"))
            except Exception as exc:  # noqa: BLE001
                handle_failure(task, exc)
            _atomic_write_json(paths["run_status"], _run_status_payload(total_tasks=len(tasks), completed=completed_count, failed=failed_count, skipped=skipped, current_task=task.task_key, start_time=start_time, pipeline_status="running"))
    summary = finalize_outputs(args, paths)
    _atomic_write_json(paths["run_status"], _run_status_payload(total_tasks=len(tasks), completed=completed_count, failed=failed_count, skipped=skipped, current_task=None, start_time=start_time, pipeline_status=summary["pipeline_status"]))
    logger.info("end pipeline status=%s completed=%s failed=%s", summary["pipeline_status"], completed_count, failed_count)
    return summary


def run_self_check(output_dir: Path) -> dict[str, Any]:
    zero = np.zeros((2, 2), dtype=complex)
    logdet_zero = casimir_integrand_single_point(zero, zero, 1.0e7, 50e-9)["logdet_integrand"]
    weak = np.eye(2, dtype=complex) * 1e-8
    logdet_weak = casimir_integrand_single_point(weak, weak, 1.0e7, 50e-9)["logdet_integrand"]
    isotropic = np.eye(2, dtype=complex) * 0.1
    energy_angles = [
        casimir_integrand_single_point(isotropic, rotate_xy_tensor(isotropic, theta), 1.0e7, 50e-9)["logdet_integrand"].real
        for theta in (0.0, 0.5, 1.0)
    ]
    payload = {
        "zero_sigma_logdet_zero": abs(logdet_zero) < 1e-14,
        "weak_conductivity_tends_to_zero": abs(logdet_weak) < 1e-12,
        "identical_isotropic_sheets_torque_zero": float(np.ptp(energy_angles)) < 1e-14,
        "no_nan_inf": bool(np.isfinite(logdet_zero.real) and np.isfinite(logdet_weak.real)),
        "valid_for_formal_casimir_claim": False,
    }
    paths = _setup_output(output_dir)
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
    parser.add_argument("--integration-strategy", default="best_available_adaptive")
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
