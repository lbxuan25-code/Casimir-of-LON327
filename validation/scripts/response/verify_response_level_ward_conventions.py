#!/usr/bin/env python3
"""Historical diagnostic / convention scanner for normal-state Pi_mu_nu.

This script is diagnostic-only.  It compares self-consistent response-level
convention choices without changing the default Ward response implementation.
It is not finite-q conductivity and not a reflection/Casimir input.
It is retained to reproduce Stage 4.4 convention scans; it is not the main
response implementation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh  # noqa: E402
from validation.lib.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402
from lno327.ward_response import normal_density_current_response_imag_axis  # noqa: E402

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "response_level_ward_conventions"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "response_level_ward_conventions"
SUMMARY_NAME = "summary.md"
EPS = 1e-300

DEFAULT_MATSUBARA_N_LIST = (1, 2, 4)
DEFAULT_TEMPERATURE_K = 30.0
DEFAULT_Q_LIST = (0.001, 0.005, 0.01, 0.05, 0.1)
DEFAULT_Q_ANGLE_LIST = (0.0, np.pi / 4.0, np.pi / 2.0)
DEFAULT_NK_LIST = (8, 12, 16)
DEFAULT_DEGENERACY_TOL_EV = 1e-10

QUICK_MATSUBARA_N_LIST = (1,)
QUICK_Q_LIST = (0.001, 0.01, 0.1)
QUICK_Q_ANGLE_LIST = (0.0, np.pi / 4.0)
QUICK_NK_LIST = (8,)

COMPONENT_LABELS = ("0", "x", "y")
SECTOR_LABELS = ("density", "spatial", "full_component")

CONVENTION_CASES = (
    {
        "case_label": "current_code_phys_q_plus",
        "current_vertex_multiplier": 1.0,
        "contact_sign_convention": "plus",
        "ward_q_sign": 1.0,
        "meaning": "Current diagnostic baseline: code-like plus-q residual, not a final convention claim.",
    },
    {
        "case_label": "hamiltonian_vertex_q_minus_contact_plus",
        "current_vertex_multiplier": 1.0,
        "contact_sign_convention": "plus",
        "ward_q_sign": -1.0,
        "meaning": "Hamiltonian derivative vertex convention: Gamma_i^H with Q_H=(iOmega,-qx,-qy).",
    },
    {
        "case_label": "physical_current_q_plus_contact_minus",
        "current_vertex_multiplier": -1.0,
        "contact_sign_convention": "minus",
        "ward_q_sign": 1.0,
        "meaning": "Physical-current convention: -Gamma_i^H with Q_phys=(iOmega,+qx,+qy).",
    },
    {
        "case_label": "physical_current_q_plus_contact_plus",
        "current_vertex_multiplier": -1.0,
        "contact_sign_convention": "plus",
        "ward_q_sign": 1.0,
        "meaning": "Physical-current control with contact plus.",
    },
    {
        "case_label": "hamiltonian_vertex_q_minus_contact_minus",
        "current_vertex_multiplier": 1.0,
        "contact_sign_convention": "minus",
        "ward_q_sign": -1.0,
        "meaning": "Hamiltonian-vertex control with contact minus.",
    },
)

COMPACT_COLUMNS = (
    "case_label",
    "current_vertex_multiplier",
    "contact_scheme",
    "contact_sign_convention",
    "ward_q_sign",
    "matsubara_n",
    "omega_eV",
    "nk",
    "q_model",
    "q_angle",
    "qx_model",
    "qy_model",
    "side",
    "component",
    "sector",
    "longitudinal_or_transverse",
    "residual_real",
    "residual_imag",
    "residual_abs",
    "component_scale",
    "closure_ratio",
    "dominant_term",
    "left_right_partner_ratio",
    "response_computed",
    "conductivity_computed",
    "casimir_computed",
    "normal_state_only",
    "not_final_finite_q_conductivity",
    "not_final_casimir_conclusion",
    "diagnosis",
)

EXPANDED_COLUMNS = COMPACT_COLUMNS + (
    "Pi_00",
    "Pi_0x",
    "Pi_0y",
    "Pi_x0",
    "Pi_xx",
    "Pi_xy",
    "Pi_y0",
    "Pi_yx",
    "Pi_yy",
)


def _angle_expression(value: str) -> float:
    normalized = re.sub(r"(?<=\d)\s*pi\b", "*pi", value)
    node = ast.parse(normalized, mode="eval")
    allowed_binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
    }
    allowed_unary = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}

    def evaluate(item: ast.AST) -> float:
        if isinstance(item, ast.Expression):
            return evaluate(item.body)
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return float(item.value)
        if isinstance(item, ast.Name) and item.id == "pi":
            return float(np.pi)
        if isinstance(item, ast.BinOp) and type(item.op) in allowed_binary:
            return float(allowed_binary[type(item.op)](evaluate(item.left), evaluate(item.right)))
        if isinstance(item, ast.UnaryOp) and type(item.op) in allowed_unary:
            return float(allowed_unary[type(item.op)](evaluate(item.operand)))
        raise argparse.ArgumentTypeError(f"unsupported angle expression: {value}")

    return evaluate(node)


def _case_by_label() -> dict[str, dict[str, object]]:
    return {str(case["case_label"]): dict(case) for case in CONVENTION_CASES}


def _validate_case_labels(case_labels: list[str]) -> None:
    allowed = _case_by_label()
    unknown = [label for label in case_labels if label not in allowed]
    if unknown:
        raise ValueError(f"unknown convention case labels: {unknown}")


def _transform_bubble(response_none: np.ndarray, current_vertex_multiplier: float) -> np.ndarray:
    transformed = np.array(response_none, dtype=complex, copy=True)
    for mu in range(3):
        for nu in range(3):
            spatial_count = int(mu > 0) + int(nu > 0)
            transformed[mu, nu] *= current_vertex_multiplier**spatial_count
    return transformed


def _contact_only_spatial(response_none: np.ndarray, response_contact_plus: np.ndarray) -> np.ndarray:
    contact = np.zeros((3, 3), dtype=complex)
    contact[1:, 1:] = response_contact_plus[1:, 1:] - response_none[1:, 1:]
    return contact


def _compose_case_response(
    response_none: np.ndarray,
    response_contact_plus: np.ndarray,
    *,
    current_vertex_multiplier: float,
    contact_sign_convention: str,
) -> np.ndarray:
    response = _transform_bubble(response_none, current_vertex_multiplier)
    contact_sign = 1.0 if contact_sign_convention == "plus" else -1.0
    response += contact_sign * _contact_only_spatial(response_none, response_contact_plus)
    return response


def _side_terms(response: np.ndarray, omega_eV: float, qx: float, qy: float, ward_q_sign: float, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if side == "left":
        density = 1j * omega_eV * response[0, :]
        spatial = ward_q_sign * (qx * response[1, :] + qy * response[2, :])
    elif side == "right":
        density = 1j * omega_eV * response[:, 0]
        spatial = ward_q_sign * (response[:, 1] * qx + response[:, 2] * qy)
    else:
        raise ValueError("side must be left or right")
    return density, spatial, density + spatial


def _component_kind(component: str, qx: float, qy: float) -> str:
    if component == "0":
        return "density_component"
    q_norm = float(np.hypot(qx, qy))
    if q_norm <= EPS:
        return "spatial_component_q_zero"
    unit_value = qx / q_norm if component == "x" else qy / q_norm
    if abs(unit_value) > 0.9238795325:
        return "mostly_longitudinal"
    if abs(unit_value) < 0.3826834324:
        return "mostly_transverse"
    return "mixed_longitudinal_transverse"


def _diagnosis(case_label: str, sector: str, closure_ratio: float) -> str:
    if closure_ratio < 1e-6:
        return f"{case_label}_{sector}_residual_small"
    if closure_ratio < 1e-3:
        return f"{case_label}_{sector}_residual_moderate"
    return f"{case_label}_{sector}_warning_not_closed"


def _row_for_residual(
    *,
    case: dict[str, object],
    matsubara_n: int,
    omega_eV: float,
    nk: int,
    q_model: float,
    q_angle: float,
    qx: float,
    qy: float,
    side: str,
    component_index: int,
    sector: str,
    residual: complex,
    density_term: complex,
    spatial_term: complex,
    full_term: complex,
    partner_abs: float,
    response: np.ndarray,
) -> dict[str, object]:
    component = COMPONENT_LABELS[component_index]
    density_abs = abs(density_term)
    spatial_abs = abs(spatial_term)
    full_abs = abs(full_term)
    component_scale = max(density_abs, spatial_abs, float(np.linalg.norm(response)), EPS)
    residual_abs = abs(residual)
    dominant_term = "density" if density_abs >= spatial_abs else "spatial"
    return {
        "case_label": str(case["case_label"]),
        "current_vertex_multiplier": float(case["current_vertex_multiplier"]),
        "contact_scheme": "finite_q_peierls",
        "contact_sign_convention": str(case["contact_sign_convention"]),
        "ward_q_sign": float(case["ward_q_sign"]),
        "matsubara_n": int(matsubara_n),
        "omega_eV": float(omega_eV),
        "nk": int(nk),
        "q_model": float(q_model),
        "q_angle": float(q_angle),
        "qx_model": float(qx),
        "qy_model": float(qy),
        "side": side,
        "component": component,
        "sector": sector,
        "longitudinal_or_transverse": _component_kind(component, qx, qy),
        "residual_real": float(np.real(residual)),
        "residual_imag": float(np.imag(residual)),
        "residual_abs": float(residual_abs),
        "component_scale": float(component_scale),
        "closure_ratio": float(residual_abs / component_scale),
        "dominant_term": dominant_term,
        "left_right_partner_ratio": float(residual_abs / max(partner_abs, EPS)),
        "response_computed": True,
        "conductivity_computed": False,
        "casimir_computed": False,
        "normal_state_only": True,
        "not_final_finite_q_conductivity": True,
        "not_final_casimir_conclusion": True,
        "diagnosis": _diagnosis(str(case["case_label"]), sector, residual_abs / component_scale),
        "Pi_00": complex(response[0, 0]),
        "Pi_0x": complex(response[0, 1]),
        "Pi_0y": complex(response[0, 2]),
        "Pi_x0": complex(response[1, 0]),
        "Pi_xx": complex(response[1, 1]),
        "Pi_xy": complex(response[1, 2]),
        "Pi_y0": complex(response[2, 0]),
        "Pi_yx": complex(response[2, 1]),
        "Pi_yy": complex(response[2, 2]),
    }


def run_verification(
    *,
    matsubara_n_list: list[int],
    temperature_K: float,
    q_list: list[float],
    q_angle_list: list[float],
    nk_list: list[int],
    degeneracy_tol_eV: float,
    case_labels: list[str],
) -> dict[str, np.ndarray]:
    if any(n < 1 for n in matsubara_n_list):
        raise ValueError("matsubara-n-list must contain only n >= 1")
    if any(q < 0.0 for q in q_list):
        raise ValueError("q-list values must be non-negative")
    if any(nk <= 0 for nk in nk_list):
        raise ValueError("nk-list values must be positive")
    if temperature_K < 0.0:
        raise ValueError("temperature must be non-negative")
    if degeneracy_tol_eV <= 0.0:
        raise ValueError("degeneracy-tol must be positive")
    _validate_case_labels(case_labels)
    cases = [_case_by_label()[label] for label in case_labels]

    rows: list[dict[str, object]] = []
    for nk in nk_list:
        mesh = uniform_bz_mesh(nk)
        weights = k_weights(mesh)
        for matsubara_n in matsubara_n_list:
            omega_eV = bosonic_matsubara_energy_eV(matsubara_n, temperature_K)
            config = KuboConfig.from_kelvin(
                omega_eV=omega_eV,
                temperature_K=temperature_K,
                eta_eV=degeneracy_tol_eV,
                output_si=False,
            )
            for q_model in q_list:
                for q_angle in q_angle_list:
                    qx = float(q_model * np.cos(q_angle))
                    qy = float(q_model * np.sin(q_angle))
                    q_vector = np.array([qx, qy], dtype=float)
                    response_none = normal_density_current_response_imag_axis(
                        mesh,
                        config,
                        q_vector,
                        weights,
                        vertex_scheme="peierls",
                        contact_scheme="none",
                        contact_sign_convention="plus",
                    )
                    response_contact_plus = normal_density_current_response_imag_axis(
                        mesh,
                        config,
                        q_vector,
                        weights,
                        vertex_scheme="peierls",
                        contact_scheme="finite_q_peierls",
                        contact_sign_convention="plus",
                    )
                    for case in cases:
                        response = _compose_case_response(
                            response_none,
                            response_contact_plus,
                            current_vertex_multiplier=float(case["current_vertex_multiplier"]),
                            contact_sign_convention=str(case["contact_sign_convention"]),
                        )
                        side_data = {}
                        for side in ("left", "right"):
                            density, spatial, full = _side_terms(
                                response,
                                omega_eV,
                                qx,
                                qy,
                                float(case["ward_q_sign"]),
                                side,
                            )
                            side_data[side] = (density, spatial, full)
                        for side in ("left", "right"):
                            density, spatial, full = side_data[side]
                            partner_side = "right" if side == "left" else "left"
                            partner_full = side_data[partner_side][2]
                            for component_index in range(3):
                                for sector, residual in (
                                    ("density", density[component_index]),
                                    ("spatial", spatial[component_index]),
                                    ("full_component", full[component_index]),
                                ):
                                    rows.append(
                                        _row_for_residual(
                                            case=case,
                                            matsubara_n=matsubara_n,
                                            omega_eV=omega_eV,
                                            nk=nk,
                                            q_model=q_model,
                                            q_angle=q_angle,
                                            qx=qx,
                                            qy=qy,
                                            side=side,
                                            component_index=component_index,
                                            sector=sector,
                                            residual=complex(residual),
                                            density_term=complex(density[component_index]),
                                            spatial_term=complex(spatial[component_index]),
                                            full_term=complex(full[component_index]),
                                            partner_abs=float(abs(partner_full[component_index])),
                                            response=response,
                                        )
                                    )
    return {column: np.array([row[column] for row in rows]) for column in EXPANDED_COLUMNS}


def _write_csv(path: Path, data: dict[str, np.ndarray], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(data["case_label"])
    if row_count == 0:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    for column in columns:
        if len(data[column]) != row_count:
            raise RuntimeError(
                f"CSV column length mismatch for {column}: {len(data[column])} != {row_count}"
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for index in range(row_count):
            writer.writerow({column: data[column][index] for column in columns})


def _write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(data)
    np.savez(path, **payload)


def _output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path]:
    compact_csv = output_prefix.with_name(f"{output_prefix.name}_compact").with_suffix(".csv")
    expanded_csv = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".csv")
    expanded_npz = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".npz")
    result_root = output_prefix.parent.parent if output_prefix.parent.name == "data" else output_prefix.parent
    figure_dir = result_root / "figures"
    summary_path = result_root / SUMMARY_NAME
    return compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path


def _mask_case(data: dict[str, np.ndarray], case_label: str) -> np.ndarray:
    return data["case_label"] == case_label


def _mask_case_summary_kind(data: dict[str, np.ndarray], case_label: str, summary_kind: str) -> np.ndarray:
    full = _mask_case(data, case_label) & (data["sector"] == "full_component")
    if summary_kind == "full":
        return full
    if summary_kind == "density":
        return full & (data["component"] == "0")
    if summary_kind == "spatial":
        return full & (data["component"] != "0")
    raise ValueError("summary_kind must be full, density, or spatial")


def _max_residual(data: dict[str, np.ndarray], case_label: str, summary_kind: str) -> float:
    mask = _mask_case_summary_kind(data, case_label, summary_kind)
    if not np.any(mask):
        return float("nan")
    return float(np.max(data["residual_abs"][mask].astype(float)))


def _q_case_summary_kind_max(data: dict[str, np.ndarray], case_label: str, summary_kind: str, q_model: float) -> float:
    mask = _mask_case_summary_kind(data, case_label, summary_kind) & np.isclose(
        data["q_model"].astype(float),
        q_model,
    )
    if not np.any(mask):
        return float("nan")
    return float(np.max(data["residual_abs"][mask].astype(float)))


def _small_q_alpha(data: dict[str, np.ndarray], case_label: str, summary_kind: str) -> float:
    q_values = np.array([0.001, 0.005, 0.01], dtype=float)
    y_values = np.array(
        [_q_case_summary_kind_max(data, case_label, summary_kind, q_value) for q_value in q_values],
        dtype=float,
    )
    valid = np.isfinite(y_values) & (y_values > 0.0)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(q_values[valid]), np.log(y_values[valid]), 1)
    return float(slope)


def _best_case(data: dict[str, np.ndarray], summary_kind: str) -> str:
    labels = sorted(set(str(label) for label in data["case_label"]))
    return min(labels, key=lambda label: _max_residual(data, label, summary_kind))


def _case_labels_in_order(data: dict[str, np.ndarray]) -> list[str]:
    available = set(str(label) for label in data["case_label"])
    return [str(case["case_label"]) for case in CONVENTION_CASES if str(case["case_label"]) in available]


def _plot_case_q_lines(
    data: dict[str, np.ndarray],
    figure_dir: Path,
    *,
    summary_kind: str,
    filename: str,
    ylabel: str,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    q_values = np.array(sorted(set(float(q) for q in data["q_model"])), dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label in _case_labels_in_order(data):
        y_values = [_q_case_summary_kind_max(data, label, summary_kind, q_model) for q_model in q_values]
        ax.loglog(q_values, np.maximum(y_values, EPS), marker="o", label=label)
    ax.set(xlabel="q_model", ylabel=ylabel, title=title)
    style_publication_axis(ax)
    ax.legend(fontsize=7)
    path = figure_dir / filename
    save_publication_figure(fig, path)
    plt.close(fig)
    return path


def _plot_alpha_bars(
    data: dict[str, np.ndarray],
    figure_dir: Path,
    *,
    summary_kind: str,
    filename: str,
    ylabel: str,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    labels = _case_labels_in_order(data)
    alphas = [_small_q_alpha(data, label, summary_kind) for label in labels]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(np.arange(len(labels)), alphas)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set(ylabel=ylabel, title=title)
    style_publication_axis(ax)
    path = figure_dir / filename
    save_publication_figure(fig, path)
    plt.close(fig)
    return path


def _plot_outputs(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_case_q_lines(
            data,
            figure_dir,
            summary_kind="spatial",
            filename="spatial_residual_vs_q_by_convention.png",
            ylabel="max spatial residual_abs",
            title="Spatial Ward residual by convention",
        ),
        _plot_case_q_lines(
            data,
            figure_dir,
            summary_kind="full",
            filename="full_residual_vs_q_by_convention.png",
            ylabel="max full residual_abs",
            title="Full Ward residual by convention",
        ),
        _plot_alpha_bars(
            data,
            figure_dir,
            summary_kind="spatial",
            filename="spatial_alpha_by_convention.png",
            ylabel="small-q spatial alpha",
            title="Small-q spatial residual scaling",
        ),
        _plot_alpha_bars(
            data,
            figure_dir,
            summary_kind="density",
            filename="density_alpha_by_convention.png",
            ylabel="small-q density alpha",
            title="Small-q density residual scaling",
        ),
    ]


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _summary_lines(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    compact_csv: Path,
    expanded_csv: Path,
    expanded_npz: Path,
    figure_paths: list[Path],
) -> list[str]:
    labels = _case_labels_in_order(data)
    full_max = {label: _max_residual(data, label, "full") for label in labels}
    density_max = {label: _max_residual(data, label, "density") for label in labels}
    spatial_max = {label: _max_residual(data, label, "spatial") for label in labels}
    full_alpha = {label: _small_q_alpha(data, label, "full") for label in labels}
    density_alpha = {label: _small_q_alpha(data, label, "density") for label in labels}
    spatial_alpha = {label: _small_q_alpha(data, label, "spatial") for label in labels}
    best_spatial = min(labels, key=lambda label: spatial_max[label])
    best_alpha = max(labels, key=lambda label: spatial_alpha[label] if np.isfinite(spatial_alpha[label]) else -np.inf)
    improves_to_alpha2 = np.isfinite(spatial_alpha[best_alpha]) and spatial_alpha[best_alpha] >= 1.8

    if best_spatial == "hamiltonian_vertex_q_minus_contact_plus":
        interpretation = (
            "Hamiltonian-vertex convention is the best spatial-residual case in this diagnostic; "
            "the current response is closest to the Gamma_i^H / Q_H convention."
        )
    elif best_spatial == "physical_current_q_plus_contact_minus":
        interpretation = (
            "Physical-current convention with contact minus is the best spatial-residual case in this diagnostic; "
            "physical current sign and contact minus should be treated as the next implementation candidate, "
            "subject to the analytic convention."
        )
    else:
        interpretation = (
            f"The best spatial-residual case is {best_spatial}; the two main self-consistent conventions do not "
            "uniquely close the Ward residual.  Remaining suspects include Kubo bubble sign, equal-time/commutator "
            "term, density vertex sign, denominator convention, or response index ordering."
        )

    lines = [
        "# Response-level Ward convention verification",
        "",
        "This is Stage 4.4 response-level Ward convention verification.",
        "It compares Hamiltonian-vertex and physical-current convention cases for normal-state Pi_mu_nu.",
        "It is not final finite-q conductivity, not reflection/Casimir input, and not a material conclusion.",
        "The best residual case must not be promoted to final physics until it is consistent with the analytic convention.",
        "",
        f"run_command = `{command}`",
        f"quick_mode={bool(args.quick)}",
        f"expanded_data_written={bool(args.write_expanded_data)}",
        "response_computed=True",
        "conductivity_computed=False",
        "casimir_computed=False",
        "normal_state_only=True",
        "not_final_finite_q_conductivity=True",
        "not_final_casimir_conclusion=True",
        "",
        "## Convention cases",
    ]
    for case in CONVENTION_CASES:
        label = str(case["case_label"])
        if label in labels:
            lines.append(
                f"- {label}: current_vertex_multiplier={case['current_vertex_multiplier']}, "
                f"contact_sign_convention={case['contact_sign_convention']}, "
                f"ward_q_sign={case['ward_q_sign']}. {case['meaning']}"
            )
    lines.extend(
        [
            "",
            "All cases use vertex_scheme=peierls and contact_scheme=finite_q_peierls.",
            "The script constructs contact-only response by subtracting contact_scheme=none from finite_q_peierls plus.",
            "",
            "## Parameter grid",
            f"- matsubara_n_list = {' '.join(str(int(n)) for n in args.matsubara_n_list)}",
            f"- temperature_K = {_fmt(float(args.temperature))}",
            f"- q_list = {' '.join(_fmt(float(q)) for q in args.q_list)}",
            f"- q_angle_list = {' '.join(_fmt(float(a)) for a in args.q_angle_list)}",
            f"- nk_list = {' '.join(str(int(nk)) for nk in args.nk_list)}",
            f"- degeneracy_tol_eV = {_fmt(float(args.degeneracy_tol))}",
            "",
            "## Max residuals by case",
        ]
    )
    for label in labels:
        lines.extend(
            [
                f"- {label}: max full residual_abs = {_fmt(full_max[label])}",
                f"- {label}: max density residual_abs = {_fmt(density_max[label])}",
                f"- {label}: max spatial residual_abs = {_fmt(spatial_max[label])}",
            ]
        )
    lines.append("")
    lines.append("## Small-q alpha from q_model = 0.001, 0.005, 0.01")
    for label in labels:
        lines.extend(
            [
                f"- {label}: full alpha = {_fmt(full_alpha[label])}",
                f"- {label}: density alpha = {_fmt(density_alpha[label])}",
                f"- {label}: spatial alpha = {_fmt(spatial_alpha[label])}",
            ]
        )
    lines.extend(
        [
            "",
            "## q_model max residual trend",
        ]
    )
    q_values = sorted(set(float(q) for q in data["q_model"]))
    for label in labels:
        spatial_trend = ", ".join(
            f"q={_fmt(q)}:{_fmt(_q_case_summary_kind_max(data, label, 'spatial', q))}" for q in q_values
        )
        full_trend = ", ".join(
            f"q={_fmt(q)}:{_fmt(_q_case_summary_kind_max(data, label, 'full', q))}" for q in q_values
        )
        lines.append(f"- {label} spatial: {spatial_trend}")
        lines.append(f"- {label} full: {full_trend}")
    lines.extend(
        [
            "",
            "## Interpretation",
            f"- Best case for max spatial residual_abs: {best_spatial}.",
            f"- Best case for spatial small-q alpha: {best_alpha} with alpha = {_fmt(spatial_alpha[best_alpha])}.",
            (
                "- A case raises the spatial alpha from about 1 toward 2 or higher."
                if improves_to_alpha2
                else "- No case raises the spatial alpha close to 2 on this grid; the spatial residual remains effectively O(q)-like."
            ),
            f"- {interpretation}",
            (
                "- If both Hamiltonian and physical-current conventions remain unclosed, the next checks should include "
                "Kubo bubble sign, equal-time/commutator term, density vertex sign, denominator convention, and response index order."
            ),
            "",
            "## Output files",
            f"- compact_csv = `{_display_path(compact_csv)}`",
            f"- expanded_csv = `{_display_path(expanded_csv)}`",
            f"- expanded_npz = `{_display_path(expanded_npz)}`",
            "Expanded CSV/NPZ are written only when --write-expanded-data is passed.",
            "",
            "## Figures",
        ]
    )
    lines.extend(f"- `{_display_path(path)}`" for path in figure_paths)
    lines.extend(
        [
            "",
            "## Explicit boundary",
            "This stage is convention verification only.",
            "It is not final finite-q conductivity.",
            "It is not reflection/Casimir input.",
            "It is not a material conclusion.",
            "Do not directly treat the best residual case as the final physical implementation without analytic convention closure.",
        ]
    )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matsubara-n-list", nargs="+", type=int, default=list(DEFAULT_MATSUBARA_N_LIST))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--q-list", nargs="+", type=float, default=list(DEFAULT_Q_LIST))
    parser.add_argument("--q-angle-list", nargs="+", type=_angle_expression, default=list(DEFAULT_Q_ANGLE_LIST))
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--degeneracy-tol-eV", dest="degeneracy_tol", type=float, default=DEFAULT_DEGENERACY_TOL_EV)
    parser.add_argument(
        "--case-labels",
        nargs="+",
        default=[str(case["case_label"]) for case in CONVENTION_CASES],
        help="Convention case labels to run.",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--write-expanded-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.quick:
        args.matsubara_n_list = list(QUICK_MATSUBARA_N_LIST)
        args.q_list = list(QUICK_Q_LIST)
        args.q_angle_list = list(QUICK_Q_ANGLE_LIST)
        args.nk_list = list(QUICK_NK_LIST)
    data = run_verification(
        matsubara_n_list=[int(value) for value in args.matsubara_n_list],
        temperature_K=float(args.temperature),
        q_list=[float(value) for value in args.q_list],
        q_angle_list=[float(value) for value in args.q_angle_list],
        nk_list=[int(value) for value in args.nk_list],
        degeneracy_tol_eV=float(args.degeneracy_tol),
        case_labels=[str(value) for value in args.case_labels],
    )
    compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path = _output_paths(Path(args.output_prefix))
    _write_csv(compact_csv, data, COMPACT_COLUMNS)
    if args.write_expanded_data:
        _write_csv(expanded_csv, data, EXPANDED_COLUMNS)
        _write_npz(expanded_npz, data)
    figure_paths = _plot_outputs(data, figure_dir)
    command = "python " + " ".join(shlex.quote(value) for value in sys.argv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            _summary_lines(
                data,
                args,
                command,
                compact_csv,
                expanded_csv,
                expanded_npz,
                figure_paths,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {compact_csv}")
    print(f"Wrote {summary_path}")
    for path in figure_paths:
        print(f"Wrote {path}")
    if args.write_expanded_data:
        print(f"Wrote {expanded_csv}")
        print(f"Wrote {expanded_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
