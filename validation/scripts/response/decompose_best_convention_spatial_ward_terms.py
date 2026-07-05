#!/usr/bin/env python3
"""Historical diagnostic / convention scanner for spatial Ward residual terms.

This is a normal-state diagnostic only.  It does not change the Ward response
implementation and it is not finite-q conductivity, reflection, or Casimir
input.  It is retained to reproduce Stage 4.5 convention decompositions; it is
not the main response implementation.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh  # noqa: E402
from validation.lib.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402
from lno327.response.normal_density_current import normal_density_current_response_imag_axis  # noqa: E402

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "best_convention_spatial_ward_terms"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "best_convention_spatial_ward_terms"
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

CASE_LIBRARY = {
    "physical_current_q_plus_contact_minus": {
        "current_vertex_multiplier": -1.0,
        "contact_sign_convention": "minus",
        "ward_q_sign": 1.0,
        "meaning": "physical current j_i=-delta H/delta A_i with Q_phys and contact minus",
    },
    "hamiltonian_vertex_q_minus_contact_minus": {
        "current_vertex_multiplier": 1.0,
        "contact_sign_convention": "minus",
        "ward_q_sign": -1.0,
        "meaning": "Hamiltonian derivative vertex Gamma_i^H with Q_H and contact minus",
    },
    "current_code_phys_q_plus": {
        "current_vertex_multiplier": 1.0,
        "contact_sign_convention": "plus",
        "ward_q_sign": 1.0,
        "meaning": "historical code-like plus-q baseline",
    },
    "physical_current_q_plus_contact_plus": {
        "current_vertex_multiplier": -1.0,
        "contact_sign_convention": "plus",
        "ward_q_sign": 1.0,
        "meaning": "physical current control with contact plus",
    },
    "hamiltonian_vertex_q_minus_contact_plus": {
        "current_vertex_multiplier": 1.0,
        "contact_sign_convention": "plus",
        "ward_q_sign": -1.0,
        "meaning": "Hamiltonian-vertex control with contact plus",
    },
}
DEFAULT_CASE_LABELS = (
    "physical_current_q_plus_contact_minus",
    "hamiltonian_vertex_q_minus_contact_minus",
)

COMPACT_COLUMNS = (
    "case_label",
    "current_vertex_multiplier",
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
    "longitudinal_or_transverse",
    "term_iomega_real",
    "term_iomega_imag",
    "term_iomega_abs",
    "term_bubble_real",
    "term_bubble_imag",
    "term_bubble_abs",
    "term_contact_real",
    "term_contact_imag",
    "term_contact_abs",
    "term_spatial_total_real",
    "term_spatial_total_imag",
    "term_spatial_total_abs",
    "density_bubble_partial_real",
    "density_bubble_partial_imag",
    "density_bubble_partial_abs",
    "residual_real",
    "residual_imag",
    "residual_abs",
    "bubble_contact_cancellation_ratio",
    "density_spatial_cancellation_ratio",
    "contact_to_bubble_abs_ratio",
    "contact_to_residual_abs_ratio",
    "dominant_leftover_term",
    "left_right_partner_ratio",
    "response_computed",
    "conductivity_computed",
    "casimir_computed",
    "normal_state_only",
    "not_final_finite_q_conductivity",
    "not_final_casimir_conclusion",
    "diagnosis",
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


def _contact_case(response_none: np.ndarray, response_contact_plus: np.ndarray, contact_sign_convention: str) -> np.ndarray:
    sign = 1.0 if contact_sign_convention == "plus" else -1.0
    return sign * _contact_only_spatial(response_none, response_contact_plus)


def _component_kind(component: str, qx: float, qy: float) -> str:
    q_norm = float(np.hypot(qx, qy))
    if q_norm <= EPS:
        return "spatial_component_q_zero"
    unit_value = qx / q_norm if component == "x" else qy / q_norm
    if abs(unit_value) > 0.9238795325:
        return "mostly_longitudinal"
    if abs(unit_value) < 0.3826834324:
        return "mostly_transverse"
    return "mixed_longitudinal_transverse"


def _dominant_leftover(
    term_iomega_abs: float,
    term_bubble_abs: float,
    term_contact_abs: float,
    density_bubble_partial_abs: float,
    residual_abs: float,
) -> str:
    if density_bubble_partial_abs > 0.75 * residual_abs and density_bubble_partial_abs > term_contact_abs:
        return "density_bubble_mismatch"
    if term_contact_abs > 2.0 * max(term_bubble_abs, EPS) and residual_abs > density_bubble_partial_abs:
        return "contact_overcorrects"
    max_term = max(term_iomega_abs, term_bubble_abs, term_contact_abs)
    if term_contact_abs == max_term and term_contact_abs > 0.5 * residual_abs:
        return "contact_dominant"
    if term_bubble_abs == max_term and term_bubble_abs > 0.5 * residual_abs:
        return "bubble_dominant"
    if term_iomega_abs == max_term and term_iomega_abs > 0.5 * residual_abs:
        return "density_current_dominant"
    return "mixed_unresolved"


def _diagnosis(residual_abs: float, density_spatial_ratio: float, dominant: str) -> str:
    if residual_abs < 1e-10:
        return "spatial_residual_near_machine_precision"
    if density_spatial_ratio < 1e-2:
        return f"strong_density_spatial_cancellation_{dominant}"
    if density_spatial_ratio < 0.2:
        return f"partial_density_spatial_cancellation_{dominant}"
    return f"warning_spatial_leftover_{dominant}"


def _complex_parts(prefix: str, value: complex) -> dict[str, float]:
    return {
        f"{prefix}_real": float(np.real(value)),
        f"{prefix}_imag": float(np.imag(value)),
        f"{prefix}_abs": float(abs(value)),
    }


def _row(
    *,
    case_label: str,
    case: dict[str, float | str],
    matsubara_n: int,
    omega_eV: float,
    nk: int,
    q_model: float,
    q_angle: float,
    qx: float,
    qy: float,
    side: str,
    component_index: int,
    term_iomega: complex,
    term_bubble: complex,
    term_contact: complex,
    partner_residual_abs: float,
) -> dict[str, object]:
    term_spatial_total = term_bubble + term_contact
    density_bubble_partial = term_iomega + term_bubble
    residual = density_bubble_partial + term_contact
    bubble_contact_ratio = abs(term_spatial_total) / (abs(term_bubble) + abs(term_contact) + EPS)
    density_spatial_ratio = abs(residual) / (abs(term_iomega) + abs(term_spatial_total) + EPS)
    contact_to_bubble = abs(term_contact) / (abs(term_bubble) + EPS)
    contact_to_residual = abs(term_contact) / (abs(residual) + EPS)
    dominant = _dominant_leftover(
        abs(term_iomega),
        abs(term_bubble),
        abs(term_contact),
        abs(density_bubble_partial),
        abs(residual),
    )
    component = "x" if component_index == 1 else "y"
    row: dict[str, object] = {
        "case_label": case_label,
        "current_vertex_multiplier": float(case["current_vertex_multiplier"]),
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
        "longitudinal_or_transverse": _component_kind(component, qx, qy),
        "bubble_contact_cancellation_ratio": float(bubble_contact_ratio),
        "density_spatial_cancellation_ratio": float(density_spatial_ratio),
        "contact_to_bubble_abs_ratio": float(contact_to_bubble),
        "contact_to_residual_abs_ratio": float(contact_to_residual),
        "dominant_leftover_term": dominant,
        "left_right_partner_ratio": float(abs(residual) / max(partner_residual_abs, EPS)),
        "response_computed": True,
        "conductivity_computed": False,
        "casimir_computed": False,
        "normal_state_only": True,
        "not_final_finite_q_conductivity": True,
        "not_final_casimir_conclusion": True,
        "diagnosis": _diagnosis(abs(residual), density_spatial_ratio, dominant),
    }
    row.update(_complex_parts("term_iomega", term_iomega))
    row.update(_complex_parts("term_bubble", term_bubble))
    row.update(_complex_parts("term_contact", term_contact))
    row.update(_complex_parts("term_spatial_total", term_spatial_total))
    row.update(_complex_parts("density_bubble_partial", density_bubble_partial))
    row.update(_complex_parts("residual", residual))
    return row


def _terms_for_side(
    bubble_case: np.ndarray,
    contact_case: np.ndarray,
    *,
    omega_eV: float,
    qx: float,
    qy: float,
    ward_q_sign: float,
    side: str,
    component_index: int,
) -> tuple[complex, complex, complex]:
    if side == "left":
        term_iomega = 1j * omega_eV * bubble_case[0, component_index]
        term_bubble = ward_q_sign * (
            qx * bubble_case[1, component_index] + qy * bubble_case[2, component_index]
        )
        term_contact = ward_q_sign * (
            qx * contact_case[1, component_index] + qy * contact_case[2, component_index]
        )
    elif side == "right":
        term_iomega = 1j * omega_eV * bubble_case[component_index, 0]
        term_bubble = ward_q_sign * (
            bubble_case[component_index, 1] * qx + bubble_case[component_index, 2] * qy
        )
        term_contact = ward_q_sign * (
            contact_case[component_index, 1] * qx + contact_case[component_index, 2] * qy
        )
    else:
        raise ValueError("side must be left or right")
    return complex(term_iomega), complex(term_bubble), complex(term_contact)


def run_decomposition(
    *,
    matsubara_n_list: list[int],
    temperature_K: float,
    q_list: list[float],
    q_angle_list: list[float],
    nk_list: list[int],
    degeneracy_tol_eV: float,
    case_labels: list[str],
) -> dict[str, np.ndarray]:
    if any(label not in CASE_LIBRARY for label in case_labels):
        raise ValueError("unknown case label")
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
                    for case_label in case_labels:
                        case = CASE_LIBRARY[case_label]
                        bubble_case = _transform_bubble(
                            response_none,
                            float(case["current_vertex_multiplier"]),
                        )
                        contact_case = _contact_case(
                            response_none,
                            response_contact_plus,
                            str(case["contact_sign_convention"]),
                        )
                        terms: dict[tuple[str, int], tuple[complex, complex, complex]] = {}
                        residual_abs: dict[tuple[str, int], float] = {}
                        for side in ("left", "right"):
                            for component_index in (1, 2):
                                current_terms = _terms_for_side(
                                    bubble_case,
                                    contact_case,
                                    omega_eV=omega_eV,
                                    qx=qx,
                                    qy=qy,
                                    ward_q_sign=float(case["ward_q_sign"]),
                                    side=side,
                                    component_index=component_index,
                                )
                                terms[(side, component_index)] = current_terms
                                residual_abs[(side, component_index)] = float(abs(sum(current_terms)))
                        for side in ("left", "right"):
                            partner_side = "right" if side == "left" else "left"
                            for component_index in (1, 2):
                                rows.append(
                                    _row(
                                        case_label=case_label,
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
                                        term_iomega=terms[(side, component_index)][0],
                                        term_bubble=terms[(side, component_index)][1],
                                        term_contact=terms[(side, component_index)][2],
                                        partner_residual_abs=residual_abs[(partner_side, component_index)],
                                    )
                                )
    return {column: np.array([row[column] for row in rows]) for column in COMPACT_COLUMNS}


def _write_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(data["case_label"])
    if row_count == 0:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPACT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for index in range(row_count):
            writer.writerow({column: data[column][index] for column in COMPACT_COLUMNS})


def _output_paths(output_prefix: Path) -> tuple[Path, Path, Path]:
    compact_csv = output_prefix.with_name(f"{output_prefix.name}_compact").with_suffix(".csv")
    result_root = output_prefix.parent.parent if output_prefix.parent.name == "data" else output_prefix.parent
    figure_dir = result_root / "figures"
    summary_path = result_root / SUMMARY_NAME
    return compact_csv, figure_dir, summary_path


def _mask_case(data: dict[str, np.ndarray], case_label: str) -> np.ndarray:
    return data["case_label"] == case_label


def _q_case_max(data: dict[str, np.ndarray], case_label: str, field: str, q_model: float) -> float:
    mask = _mask_case(data, case_label) & np.isclose(data["q_model"].astype(float), q_model)
    if not np.any(mask):
        return float("nan")
    return float(np.max(data[field][mask].astype(float)))


def _case_max(data: dict[str, np.ndarray], case_label: str, field: str) -> float:
    mask = _mask_case(data, case_label)
    if not np.any(mask):
        return float("nan")
    return float(np.max(data[field][mask].astype(float)))


def _small_q_alpha(data: dict[str, np.ndarray], case_label: str, field: str) -> float:
    q_values = np.array([0.001, 0.005, 0.01], dtype=float)
    y_values = np.array([_q_case_max(data, case_label, field, q) for q in q_values], dtype=float)
    valid = np.isfinite(y_values) & (y_values > 0.0)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(q_values[valid]), np.log(y_values[valid]), 1)
    return float(slope)


def _case_labels_in_order(data: dict[str, np.ndarray]) -> list[str]:
    available = set(str(label) for label in data["case_label"])
    return [label for label in CASE_LIBRARY if label in available]


def _best_case(data: dict[str, np.ndarray]) -> str:
    labels = _case_labels_in_order(data)
    return min(labels, key=lambda label: _case_max(data, label, "residual_abs"))


def _max_by_filter(data: dict[str, np.ndarray], field: str, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.max(data[field][mask].astype(float)))


def _plot_outputs(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    q_values = np.array(sorted(set(float(q) for q in data["q_model"])), dtype=float)
    labels = _case_labels_in_order(data)
    best = _best_case(data)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for field, label in (
        ("term_iomega_abs", "iOmega density-current"),
        ("term_bubble_abs", "spatial bubble"),
        ("term_contact_abs", "spatial contact"),
        ("density_bubble_partial_abs", "density + bubble"),
        ("residual_abs", "residual"),
    ):
        ax.loglog(q_values, [_q_case_max(data, best, field, q) for q in q_values], marker="o", label=label)
    ax.set(xlabel="q_model", ylabel="max abs term", title=f"Best case term decomposition: {best}")
    style_publication_axis(ax)
    ax.legend(fontsize=8)
    path = figure_dir / "best_case_terms_vs_q.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label in labels:
        ax.loglog(q_values, [_q_case_max(data, label, "residual_abs", q) for q in q_values], marker="o", label=label)
    ax.set(xlabel="q_model", ylabel="max residual_abs", title="Spatial Ward residual by case")
    style_publication_axis(ax)
    ax.legend(fontsize=8)
    path = figure_dir / "residual_vs_q_by_case.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for field, label in (
        ("density_spatial_cancellation_ratio", "density/spatial"),
        ("bubble_contact_cancellation_ratio", "bubble/contact"),
    ):
        ax.semilogy(q_values, [_q_case_max(data, best, field, q) for q in q_values], marker="o", label=label)
    ax.set(xlabel="q_model", ylabel="max cancellation ratio", title=f"Cancellation ratios: {best}")
    style_publication_axis(ax)
    ax.legend(fontsize=8)
    path = figure_dir / "cancellation_ratios_vs_q.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    kinds = ["mostly_longitudinal", "mixed_longitudinal_transverse", "mostly_transverse"]
    x = np.arange(len(kinds))
    width = 0.8 / max(len(labels), 1)
    for offset, label in enumerate(labels):
        values = []
        for kind in kinds:
            mask = _mask_case(data, label) & (data["longitudinal_or_transverse"] == kind)
            values.append(_max_by_filter(data, "residual_abs", mask))
        ax.bar(x + (offset - (len(labels) - 1) / 2) * width, values, width=width, label=label)
    ax.set_xticks(x, kinds, rotation=20, ha="right")
    ax.set(ylabel="max residual_abs", title="Longitudinal/transverse spatial residuals")
    style_publication_axis(ax)
    ax.legend(fontsize=8)
    path = figure_dir / "longitudinal_transverse_residuals.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    return paths


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _common_dominant(data: dict[str, np.ndarray], case_label: str) -> str:
    mask = _mask_case(data, case_label)
    values, counts = np.unique(data["dominant_leftover_term"][mask], return_counts=True)
    if len(values) == 0:
        return "unavailable"
    return str(values[int(np.argmax(counts))])


def _left_right_summary(data: dict[str, np.ndarray], case_label: str) -> tuple[float, str]:
    mask = _mask_case(data, case_label)
    values = data["left_right_partner_ratio"][mask].astype(float)
    max_deviation = float(np.max(np.abs(np.log(np.maximum(values, EPS)))))
    if max_deviation < 1e-6:
        text = "left/right residuals are nearly identical"
    elif max_deviation < 0.1:
        text = "left/right residuals are close"
    else:
        text = "left/right residuals differ; check response index order and conjugation"
    return max_deviation, text


def _summary_lines(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    compact_csv: Path,
    figure_paths: list[Path],
) -> list[str]:
    labels = _case_labels_in_order(data)
    best = _best_case(data)
    q_values = sorted(set(float(q) for q in data["q_model"]))
    best_dominant = _common_dominant(data, best)
    max_lr_deviation, lr_text = _left_right_summary(data, best)
    best_mask = _mask_case(data, best)
    long_max = _max_by_filter(
        data,
        "residual_abs",
        best_mask & (data["longitudinal_or_transverse"] == "mostly_longitudinal"),
    )
    trans_max = _max_by_filter(
        data,
        "residual_abs",
        best_mask & (data["longitudinal_or_transverse"] == "mostly_transverse"),
    )
    mixed_max = _max_by_filter(
        data,
        "residual_abs",
        best_mask & (data["longitudinal_or_transverse"] == "mixed_longitudinal_transverse"),
    )
    cancellation_text = (
        "contact partly reduces the residual coefficient but does not remove the O(q) leftover"
        if _case_max(data, best, "contact_to_bubble_abs_ratio") > 0.1
        else "contact is small compared with the bubble term, so leftover is more likely bubble/density convention"
    )
    if best_dominant == "density_bubble_mismatch":
        source_text = (
            "The leftover most resembles a density-current/bubble mismatch, so Kubo bubble sign, denominator, "
            "matrix-element order, or equal-time term should be checked next."
        )
    elif best_dominant in {"contact_dominant", "contact_overcorrects"}:
        source_text = (
            "The leftover is contact-sensitive, so contact normalization/factor/sign and equal-time terms "
            "should be checked next."
        )
    elif best_dominant == "bubble_dominant":
        source_text = "The leftover is bubble-dominant; inspect Kubo bubble sign and response index convention."
    else:
        source_text = (
            "The leftover is mixed; remaining suspects include equal-time/commutator term, response index order, "
            "and Kubo convention."
        )

    lines = [
        "# Best-convention spatial Ward term decomposition",
        "",
        "This is Stage 4.5 diagnostic output.",
        "It decomposes spatial Ward residual terms under the best Stage 4.4 convention candidates.",
        "It is not final finite-q conductivity, not reflection/Casimir input, and not a material conclusion.",
        "",
        f"run_command = `{command}`",
        f"quick_mode={bool(args.quick)}",
        "response_computed=True",
        "conductivity_computed=False",
        "casimir_computed=False",
        "normal_state_only=True",
        "not_final_finite_q_conductivity=True",
        "not_final_casimir_conclusion=True",
        "",
        "## Convention cases",
    ]
    for label in labels:
        case = CASE_LIBRARY[label]
        lines.append(
            f"- {label}: current_vertex_multiplier={case['current_vertex_multiplier']}, "
            f"contact_sign_convention={case['contact_sign_convention']}, ward_q_sign={case['ward_q_sign']}; "
            f"{case['meaning']}."
        )
    lines.extend(
        [
            "",
            "All cases use Peierls current vertices. The contact-only block is extracted as "
            "Pi(finite_q_peierls, plus) - Pi(none), then assigned the case contact sign.",
            "",
            "## Max residual and alpha by case",
        ]
    )
    for label in labels:
        lines.extend(
            [
                f"- {label}: max residual_abs = {_fmt(_case_max(data, label, 'residual_abs'))}",
                f"- {label}: small-q residual alpha = {_fmt(_small_q_alpha(data, label, 'residual_abs'))}",
                f"- {label}: term_iomega_abs alpha = {_fmt(_small_q_alpha(data, label, 'term_iomega_abs'))}",
                f"- {label}: term_bubble_abs alpha = {_fmt(_small_q_alpha(data, label, 'term_bubble_abs'))}",
                f"- {label}: term_contact_abs alpha = {_fmt(_small_q_alpha(data, label, 'term_contact_abs'))}",
            ]
        )
    lines.extend(
        [
            "",
            f"## Best case q trend: {best}",
        ]
    )
    for q_value in q_values:
        lines.append(
            "- q_model="
            f"{_fmt(q_value)}: "
            f"max term_iomega_abs={_fmt(_q_case_max(data, best, 'term_iomega_abs', q_value))}, "
            f"max term_bubble_abs={_fmt(_q_case_max(data, best, 'term_bubble_abs', q_value))}, "
            f"max term_contact_abs={_fmt(_q_case_max(data, best, 'term_contact_abs', q_value))}, "
            f"max density_bubble_partial_abs={_fmt(_q_case_max(data, best, 'density_bubble_partial_abs', q_value))}, "
            f"max residual_abs={_fmt(_q_case_max(data, best, 'residual_abs', q_value))}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            f"- Best case by max residual_abs: {best}.",
            f"- Dominant leftover classification in the best case: {best_dominant}.",
            f"- Contact role: {cancellation_text}.",
            f"- Source diagnosis: {source_text}",
            f"- Longitudinal max residual_abs = {_fmt(long_max)}.",
            f"- Transverse max residual_abs = {_fmt(trans_max)}.",
            f"- Mixed-angle max residual_abs = {_fmt(mixed_max)}.",
            (
                "- Longitudinal residual is larger than transverse."
                if long_max >= trans_max
                else "- Transverse residual is larger than longitudinal."
            ),
            f"- Left/right comparison: {lr_text}; max log-ratio deviation = {_fmt(max_lr_deviation)}.",
            "",
            "## Next-step rules",
            "- If density_bubble_partial is already small but residual is contact-dominated, check contact factor/sign.",
            "- If density_bubble_partial is O(q), check Kubo bubble sign, denominator, and matrix-element order.",
            "- If contact cancels part of the residual but not enough, check equal-time/commutator term or contact normalization.",
            "- If left/right differ strongly, check response index order and Hermitian conjugation.",
            "",
            "## Output files",
            f"- compact_csv = `{_display_path(compact_csv)}`",
            "",
            "## Figures",
        ]
    )
    lines.extend(f"- `{_display_path(path)}`" for path in figure_paths)
    lines.extend(
        [
            "",
            "## Explicit boundary",
            "This stage is diagnostic only.",
            "It is not final finite-q conductivity.",
            "It is not reflection/Casimir input.",
            "It is not a material conclusion.",
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
    parser.add_argument("--case-labels", nargs="+", default=list(DEFAULT_CASE_LABELS))
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
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
    data = run_decomposition(
        matsubara_n_list=[int(value) for value in args.matsubara_n_list],
        temperature_K=float(args.temperature),
        q_list=[float(value) for value in args.q_list],
        q_angle_list=[float(value) for value in args.q_angle_list],
        nk_list=[int(value) for value in args.nk_list],
        degeneracy_tol_eV=float(args.degeneracy_tol),
        case_labels=[str(value) for value in args.case_labels],
    )
    compact_csv, figure_dir, summary_path = _output_paths(Path(args.output_prefix))
    _write_csv(compact_csv, data)
    figure_paths = _plot_outputs(data, figure_dir)
    command = "python " + " ".join(shlex.quote(value) for value in sys.argv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(_summary_lines(data, args, command, compact_csv, figure_paths)) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {compact_csv}")
    print(f"Wrote {summary_path}")
    for path in figure_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
