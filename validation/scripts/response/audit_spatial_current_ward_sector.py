#!/usr/bin/env python3
"""Historical diagnostic / convention scanner for the spatial-current Ward sector.

This script audits Stage 4 diagnostic residuals.  It is not the main response
implementation, not finite-q conductivity, and not a reflection/Casimir input.
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

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "spatial_current_ward_sector"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "spatial_current_ward_sector"
SUMMARY_NAME = "summary.md"
EPS = 1e-300

DEFAULT_MATSUBARA_N_LIST = (1, 2, 4)
DEFAULT_TEMPERATURE_K = 30.0
DEFAULT_Q_LIST = (0.001, 0.005, 0.01, 0.05, 0.1)
DEFAULT_Q_ANGLE_LIST = (0.0, np.pi / 4.0, np.pi / 2.0)
DEFAULT_NK_LIST = (8, 12, 16)
DEFAULT_DEGENERACY_TOL_EV = 1e-10
DEFAULT_COMBOS = (
    "peierls:none:not_applicable",
    "peierls:q0_mass_diagnostic:plus",
    "peierls:q0_mass_diagnostic:minus",
    "peierls:finite_q_peierls:plus",
    "peierls:finite_q_peierls:minus",
    "midpoint:none:not_applicable",
)

QUICK_MATSUBARA_N_LIST = (1,)
QUICK_Q_LIST = (0.001, 0.01, 0.1)
QUICK_Q_ANGLE_LIST = (0.0, np.pi / 4.0)
QUICK_NK_LIST = (8,)
SPATIAL_COMPONENTS = ("x", "y")

COMPACT_COLUMNS = (
    "vertex_scheme",
    "current_vertex_sign_convention",
    "contact_scheme",
    "contact_sign_convention",
    "density_vertex_scheme",
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
    "Pi_0x_real",
    "Pi_0x_imag",
    "Pi_0y_real",
    "Pi_0y_imag",
    "Pi_x0_real",
    "Pi_x0_imag",
    "Pi_y0_real",
    "Pi_y0_imag",
    "Pi_xx_real",
    "Pi_xx_imag",
    "Pi_xy_real",
    "Pi_xy_imag",
    "Pi_yx_real",
    "Pi_yx_imag",
    "Pi_yy_real",
    "Pi_yy_imag",
    "term_iomega_real",
    "term_iomega_imag",
    "term_qx_real",
    "term_qx_imag",
    "term_qy_real",
    "term_qy_imag",
    "residual_real",
    "residual_imag",
    "residual_abs",
    "component_scale",
    "closure_ratio",
    "paramagnetic_spatial_residual_abs",
    "contact_shift_estimate_abs",
    "contact_effect_ratio",
    "dominant_term",
    "contact_sensitive",
    "left_right_partner_abs_difference",
    "left_right_partner_ratio",
    "response_computed",
    "conductivity_computed",
    "casimir_computed",
    "normal_state_only",
    "bdg_computed",
    "not_final_finite_q_conductivity",
    "not_final_casimir_conclusion",
    "diagnosis",
)

EXPANDED_COLUMNS = COMPACT_COLUMNS + (
    "Pi_00_real",
    "Pi_00_imag",
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


def _parse_combo(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("combo must have form vertex:contact:sign")
    vertex_scheme, contact_scheme, sign = parts
    if vertex_scheme not in {"midpoint", "peierls"}:
        raise argparse.ArgumentTypeError("vertex scheme must be midpoint or peierls")
    if contact_scheme not in {"none", "q0_mass_diagnostic", "finite_q_peierls"}:
        raise argparse.ArgumentTypeError("contact scheme must be none, q0_mass_diagnostic, or finite_q_peierls")
    if contact_scheme == "none" and sign != "not_applicable":
        raise argparse.ArgumentTypeError("none contact must use not_applicable sign")
    if contact_scheme != "none" and sign not in {"plus", "minus"}:
        raise argparse.ArgumentTypeError("contact sign must be plus or minus")
    return vertex_scheme, contact_scheme, sign


def _combo_label(combo: tuple[str, str, str]) -> str:
    vertex_scheme, contact_scheme, sign = combo
    if contact_scheme == "none":
        return f"{vertex_scheme}+none"
    return f"{vertex_scheme}+{contact_scheme}+{sign}"


def _complex_parts(prefix: str, value: complex) -> dict[str, float]:
    return {
        f"{prefix}_real": float(np.real(value)),
        f"{prefix}_imag": float(np.imag(value)),
    }


def _dominant_term(term_iomega: complex, term_qx: complex, term_qy: complex) -> str:
    values = {"iomega": abs(term_iomega), "qx": abs(term_qx), "qy": abs(term_qy)}
    return max(values, key=values.__getitem__)


def _orientation(q_angle: float, component: str) -> str:
    if np.isclose(q_angle, 0.0):
        return "longitudinal" if component == "x" else "transverse"
    if np.isclose(q_angle, np.pi / 2.0):
        return "longitudinal" if component == "y" else "transverse"
    return "mixed"


def _spatial_terms(matrix: np.ndarray, omega_eV: float, qx: float, qy: float, side: str, component: str) -> tuple[complex, complex, complex, complex]:
    if side == "left" and component == "x":
        term_iomega = 1j * omega_eV * matrix[0, 1]
        term_qx = qx * matrix[1, 1]
        term_qy = qy * matrix[2, 1]
    elif side == "left" and component == "y":
        term_iomega = 1j * omega_eV * matrix[0, 2]
        term_qx = qx * matrix[1, 2]
        term_qy = qy * matrix[2, 2]
    elif side == "right" and component == "x":
        term_iomega = 1j * omega_eV * matrix[1, 0]
        term_qx = matrix[1, 1] * qx
        term_qy = matrix[1, 2] * qy
    elif side == "right" and component == "y":
        term_iomega = 1j * omega_eV * matrix[2, 0]
        term_qx = matrix[2, 1] * qx
        term_qy = matrix[2, 2] * qy
    else:
        raise ValueError("invalid side/component")
    residual = term_iomega + term_qx + term_qy
    return term_iomega, term_qx, term_qy, residual


def _base_row(
    *,
    combo: tuple[str, str, str],
    matsubara_n: int,
    omega_eV: float,
    nk: int,
    q_model: float,
    q_angle: float,
    qx: float,
    qy: float,
    side: str,
    component: str,
    matrix: np.ndarray,
) -> dict[str, object]:
    vertex_scheme, contact_scheme, contact_sign = combo
    term_iomega, term_qx, term_qy, residual = _spatial_terms(matrix, omega_eV, qx, qy, side, component)
    component_scale = abs(term_iomega) + abs(term_qx) + abs(term_qy) + EPS
    row: dict[str, object] = {
        "vertex_scheme": vertex_scheme,
        "current_vertex_sign_convention": "plus" if vertex_scheme == "peierls" else "not_applicable",
        "contact_scheme": contact_scheme,
        "contact_sign_convention": contact_sign,
        "density_vertex_scheme": "identity_4_orbitals_shared_in_plane_position",
        "matsubara_n": int(matsubara_n),
        "omega_eV": float(omega_eV),
        "nk": int(nk),
        "q_model": float(q_model),
        "q_angle": float(q_angle),
        "qx_model": float(qx),
        "qy_model": float(qy),
        "side": side,
        "component": component,
        "longitudinal_or_transverse": _orientation(q_angle, component),
        **_complex_parts("Pi_00", complex(matrix[0, 0])),
        **_complex_parts("Pi_0x", complex(matrix[0, 1])),
        **_complex_parts("Pi_0y", complex(matrix[0, 2])),
        **_complex_parts("Pi_x0", complex(matrix[1, 0])),
        **_complex_parts("Pi_y0", complex(matrix[2, 0])),
        **_complex_parts("Pi_xx", complex(matrix[1, 1])),
        **_complex_parts("Pi_xy", complex(matrix[1, 2])),
        **_complex_parts("Pi_yx", complex(matrix[2, 1])),
        **_complex_parts("Pi_yy", complex(matrix[2, 2])),
        **_complex_parts("term_iomega", term_iomega),
        **_complex_parts("term_qx", term_qx),
        **_complex_parts("term_qy", term_qy),
        **_complex_parts("residual", residual),
        "residual_abs": float(abs(residual)),
        "component_scale": float(component_scale),
        "closure_ratio": float(abs(residual) / component_scale),
        "paramagnetic_spatial_residual_abs": float(abs(residual)),
        "contact_shift_estimate_abs": 0.0,
        "contact_effect_ratio": 0.0,
        "dominant_term": _dominant_term(term_iomega, term_qx, term_qy),
        "contact_sensitive": False,
        "left_right_partner_abs_difference": 0.0,
        "left_right_partner_ratio": 1.0,
        "response_computed": True,
        "conductivity_computed": False,
        "casimir_computed": False,
        "normal_state_only": True,
        "bdg_computed": False,
        "not_final_finite_q_conductivity": True,
        "not_final_casimir_conclusion": True,
        "diagnosis": "",
    }
    return row


def _diagnosis(row: dict[str, object]) -> str:
    if float(row["closure_ratio"]) < 1e-3:
        return "residual_small"
    if float(row["left_right_partner_ratio"]) > 1.1:
        return "left_right_spatial_asymmetry_warning"
    if bool(row["contact_sensitive"]):
        effect = float(row["contact_effect_ratio"])
        if effect > 0.01:
            return "finite_q_contact_improves_but_does_not_close"
        if effect < -0.01:
            return "finite_q_contact_worsens"
        return "spatial_residual_contact_sensitive"
    return "spatial_residual_not_contact_sensitive"


def _mark_comparisons(rows: list[dict[str, object]]) -> None:
    by_key: dict[tuple[int, int, float, float, str, str, str], dict[tuple[str, str, str], dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["matsubara_n"]),
            int(row["nk"]),
            float(row["q_model"]),
            float(row["q_angle"]),
            str(row["side"]),
            str(row["component"]),
            str(row["vertex_scheme"]),
        )
        combo = (str(row["vertex_scheme"]), str(row["contact_scheme"]), str(row["contact_sign_convention"]))
        by_key.setdefault(key, {})[combo] = row

    for group in by_key.values():
        none_candidates = [combo for combo in group if combo[1] == "none"]
        if not none_candidates:
            continue
        none_row = group[none_candidates[0]]
        none_abs = float(none_row["residual_abs"])
        for combo, row in group.items():
            row["paramagnetic_spatial_residual_abs"] = none_abs
            row["contact_shift_estimate_abs"] = abs(float(row["residual_abs"]) - none_abs)
            row["contact_effect_ratio"] = (none_abs - float(row["residual_abs"])) / max(none_abs, EPS)
        values = [float(row["residual_abs"]) for row in group.values()]
        sensitive = (max(values) - min(values)) / max(none_abs, EPS) > 0.01
        for row in group.values():
            row["contact_sensitive"] = sensitive

    partner_key: dict[tuple[int, int, float, float, str, str, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = (
            int(row["matsubara_n"]),
            int(row["nk"]),
            float(row["q_model"]),
            float(row["q_angle"]),
            str(row["vertex_scheme"]),
            str(row["contact_scheme"]),
            str(row["contact_sign_convention"]),
            str(row["component"]),
            "right" if row["side"] == "left" else "left",
        )
        partner_key[key] = row
    for row in rows:
        lookup = (
            int(row["matsubara_n"]),
            int(row["nk"]),
            float(row["q_model"]),
            float(row["q_angle"]),
            str(row["vertex_scheme"]),
            str(row["contact_scheme"]),
            str(row["contact_sign_convention"]),
            str(row["component"]),
            str(row["side"]),
        )
        partner = partner_key.get(lookup)
        if partner is not None:
            own = float(row["residual_abs"])
            other = float(partner["residual_abs"])
            row["left_right_partner_abs_difference"] = abs(own - other)
            row["left_right_partner_ratio"] = max(own, other) / max(min(own, other), EPS)
        row["diagnosis"] = _diagnosis(row)


def run_audit(
    *,
    matsubara_n_list: list[int],
    temperature_K: float,
    q_list: list[float],
    q_angle_list: list[float],
    nk_list: list[int],
    degeneracy_tol_eV: float,
    combos: list[tuple[str, str, str]],
) -> dict[str, np.ndarray]:
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
                    for combo in combos:
                        vertex_scheme, contact_scheme, contact_sign = combo
                        response_contact_sign = "plus" if contact_sign == "not_applicable" else contact_sign
                        matrix = normal_density_current_response_imag_axis(
                            mesh,
                            config,
                            np.array([qx, qy], dtype=float),
                            weights,
                            vertex_scheme=vertex_scheme,
                            contact_scheme=contact_scheme,
                            contact_sign_convention=response_contact_sign,
                        )
                        for side in ("left", "right"):
                            for component in SPATIAL_COMPONENTS:
                                rows.append(
                                    _base_row(
                                        combo=combo,
                                        matsubara_n=matsubara_n,
                                        omega_eV=omega_eV,
                                        nk=nk,
                                        q_model=float(q_model),
                                        q_angle=float(q_angle),
                                        qx=qx,
                                        qy=qy,
                                        side=side,
                                        component=component,
                                        matrix=matrix,
                                    )
                                )
    _mark_comparisons(rows)
    return {column: np.array([row[column] for row in rows]) for column in EXPANDED_COLUMNS}


def _write_csv(path: Path, data: dict[str, np.ndarray], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for index in range(len(data["q_model"])):
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


def _combo_mask(data: dict[str, np.ndarray], combo: tuple[str, str, str]) -> np.ndarray:
    return (
        (data["vertex_scheme"] == combo[0])
        & (data["contact_scheme"] == combo[1])
        & (data["contact_sign_convention"] == combo[2])
    )


def _max_for_mask(data: dict[str, np.ndarray], field: str, mask: np.ndarray) -> float:
    return float(np.max(data[field][mask].astype(float)))


def _fit_alpha(data: dict[str, np.ndarray], combo: tuple[str, str, str]) -> float:
    values = []
    q_values = []
    base = _combo_mask(data, combo)
    for q_model in (0.001, 0.005, 0.01):
        mask = base & np.isclose(data["q_model"].astype(float), q_model)
        if np.any(mask):
            value = _max_for_mask(data, "residual_abs", mask)
            if value > 0.0:
                q_values.append(q_model)
                values.append(value)
    if len(values) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(q_values), np.log(values), deg=1)
    return float(slope)


def _fmt(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.6g}"


def _plot_outputs(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    selected = [
        ("peierls", "none", "not_applicable"),
        ("peierls", "q0_mass_diagnostic", "plus"),
        ("peierls", "finite_q_peierls", "plus"),
        ("peierls", "finite_q_peierls", "minus"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for combo in selected:
        if not np.any(_combo_mask(data, combo)):
            continue
        q_values = sorted(set(float(q) for q in data["q_model"][_combo_mask(data, combo)]))
        values = []
        for q_model in q_values:
            mask = _combo_mask(data, combo) & np.isclose(data["q_model"].astype(float), q_model)
            values.append(_max_for_mask(data, "residual_abs", mask))
        ax.loglog(q_values, np.maximum(values, EPS), marker="o", label=_combo_label(combo))
    ax.set(xlabel="q_model", ylabel="max spatial residual_abs", title="Spatial-current Ward residual")
    style_publication_axis(ax)
    path = figure_dir / "spatial_residual_vs_q.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    labels = [f"{side}:{component}" for side in ("left", "right") for component in SPATIAL_COMPONENTS]
    x = np.arange(len(labels))
    for combo in selected:
        if not np.any(_combo_mask(data, combo)):
            continue
        values = []
        for side in ("left", "right"):
            for component in SPATIAL_COMPONENTS:
                mask = _combo_mask(data, combo) & (data["side"] == side) & (data["component"] == component)
                values.append(_max_for_mask(data, "closure_ratio", mask))
        ax.semilogy(x, np.maximum(values, EPS), marker="o", label=_combo_label(combo))
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set(ylabel="max closure_ratio", title="Spatial closure ratio by component")
    style_publication_axis(ax)
    path = figure_dir / "spatial_closure_ratio_by_component.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for combo in (("peierls", "q0_mass_diagnostic", "plus"), ("peierls", "finite_q_peierls", "plus")):
        if not np.any(_combo_mask(data, combo)):
            continue
        q_values = sorted(set(float(q) for q in data["q_model"][_combo_mask(data, combo)]))
        values = []
        for q_model in q_values:
            mask = _combo_mask(data, combo) & np.isclose(data["q_model"].astype(float), q_model)
            values.append(_max_for_mask(data, "contact_effect_ratio", mask))
        ax.semilogx(q_values, values, marker="o", label=_combo_label(combo))
    ax.set(xlabel="q_model", ylabel="max contact_effect_ratio", title="Spatial contact effect")
    style_publication_axis(ax)
    path = figure_dir / "contact_effect_ratio_vs_q.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    return paths


def _row_of_max(data: dict[str, np.ndarray]) -> dict[str, object]:
    index = int(np.argmax(data["residual_abs"].astype(float)))
    return {column: data[column][index] for column in COMPACT_COLUMNS}


def _best_contact_summary(data: dict[str, np.ndarray], contact_scheme: str) -> tuple[str, float]:
    candidates = [("peierls", contact_scheme, "plus"), ("peierls", contact_scheme, "minus")]
    available = [combo for combo in candidates if np.any(_combo_mask(data, combo))]
    best = min(available, key=lambda combo: _max_for_mask(data, "residual_abs", _combo_mask(data, combo)))
    return best[2], _max_for_mask(data, "residual_abs", _combo_mask(data, best))


def _orientation_effect(data: dict[str, np.ndarray], combo: tuple[str, str, str]) -> tuple[str, float]:
    best_label = "unavailable"
    best_value = -np.inf
    for label in ("longitudinal", "transverse", "mixed"):
        mask = _combo_mask(data, combo) & (data["longitudinal_or_transverse"] == label)
        if np.any(mask):
            value = _max_for_mask(data, "contact_effect_ratio", mask)
            if value > best_value:
                best_label = label
                best_value = value
    return best_label, float(best_value)


def _summary_lines(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    combos: list[tuple[str, str, str]],
    compact_csv: Path,
    expanded_csv: Path,
    expanded_npz: Path,
    figure_paths: list[Path],
) -> list[str]:
    max_row = _row_of_max(data)
    max_combo = (str(max_row["vertex_scheme"]), str(max_row["contact_scheme"]), str(max_row["contact_sign_convention"]))
    peierls_none = ("peierls", "none", "not_applicable")
    finite_plus = ("peierls", "finite_q_peierls", "plus")
    q0_plus = ("peierls", "q0_mass_diagnostic", "plus")
    finite_alpha = _fit_alpha(data, finite_plus)
    none_alpha = _fit_alpha(data, peierls_none)
    q0_alpha = _fit_alpha(data, q0_plus)
    q0_best_sign, q0_best_value = _best_contact_summary(data, "q0_mass_diagnostic")
    finite_best_sign, finite_best_value = _best_contact_summary(data, "finite_q_peierls")
    none_value = _max_for_mask(data, "residual_abs", _combo_mask(data, peierls_none))
    finite_over_q0 = q0_best_value / max(finite_best_value, EPS)
    finite_effect_orientation, finite_effect_value = _orientation_effect(data, finite_plus)
    left_max = _max_for_mask(data, "residual_abs", data["side"] == "left")
    right_max = _max_for_mask(data, "residual_abs", data["side"] == "right")
    left_right_ratio = max(left_max, right_max) / max(min(left_max, right_max), EPS)
    scaling_lines = []
    for combo in combos:
        alpha = _fit_alpha(data, combo)
        note = "approximately O(q), spatial sector is the full Ward O(q) source" if abs(alpha - 1.0) < 0.35 else "not a clean O(q) scaling"
        scaling_lines.append(f"- {_combo_label(combo)}: alpha={_fmt(alpha)} ({note})")
    q0_vs_none = none_value / max(q0_best_value, EPS)
    finite_vs_none = none_value / max(finite_best_value, EPS)
    if finite_vs_none < 1.0 / 1.01:
        next_hint = (
            "finite_q_peierls contact worsens the max spatial residual relative to no-contact; prioritize "
            "response-level equal-time/contact convention and sign/normalization before using this as closure."
        )
    elif finite_best_sign == "minus":
        next_hint = "finite_q_peierls minus is better; prioritize checking contact sign convention."
    elif abs(finite_alpha - 1.0) < 0.35:
        next_hint = (
            "spatial residual remains O(q) with contact; prioritize response-level Ward derivation and "
            "paramagnetic-diamagnetic equal-time cancellation/Kubo convention."
        )
    else:
        next_hint = "spatial residual is not cleanly O(q); inspect component/orientation-specific conventions."
    return [
        "# Spatial-current Ward sector audit",
        "",
        "This is a spatial-current Ward-sector audit.",
        "It is not conductivity, not a reflection/Casimir input, and not a material conclusion.",
        "No Peierls current/contact vertex formula is modified here.",
        "",
        f"run_command = `{command}`",
        f"quick_mode={bool(args.quick)}",
        f"expanded_data_written={bool(args.write_expanded_data)}",
        "response_computed=True",
        "conductivity_computed=False",
        "casimir_computed=False",
        "normal_state_only=True",
        "bdg_computed=False",
        "not_final_finite_q_conductivity=True",
        "not_final_casimir_conclusion=True",
        "",
        "## Spatial residual maximum",
        f"- max combo = {_combo_label(max_combo)}",
        f"- side = {max_row['side']}",
        f"- component = {max_row['component']}",
        f"- orientation = {max_row['longitudinal_or_transverse']}",
        f"- matsubara_n = {int(max_row['matsubara_n'])}",
        f"- nk = {int(max_row['nk'])}",
        f"- q_model = {_fmt(float(max_row['q_model']))}",
        f"- q_angle = {_fmt(float(max_row['q_angle']))}",
        f"- max residual_abs = {_fmt(float(max_row['residual_abs']))}",
        f"- left/right residual max ratio = {_fmt(left_right_ratio)}",
        (
            "- left/right spatial residuals are close."
            if left_right_ratio < 1.1
            else "- left/right spatial residuals differ noticeably; check vertex order, complex conjugation, denominator, or Pi index convention."
        ),
        "",
        "## Small-q scaling",
        "- alpha is fitted from max residual_abs at q_model = 0.001, 0.005, 0.01.",
        *scaling_lines,
        "",
        "## Contact comparison",
        f"- peierls+none max residual_abs = {_fmt(none_value)}",
        f"- q0_mass_diagnostic best sign = {q0_best_sign}; max residual_abs = {_fmt(q0_best_value)}",
        f"- finite_q_peierls best sign = {finite_best_sign}; max residual_abs = {_fmt(finite_best_value)}",
        f"- peierls+none / q0_mass_diagnostic best residual factor = {_fmt(q0_vs_none)}",
        f"- peierls+none / finite_q_peierls best residual factor = {_fmt(finite_vs_none)}",
        f"- q0_best / finite_best residual factor = {_fmt(finite_over_q0)}",
        (
            "- finite_q_peierls is clearly better than q0_mass_diagnostic."
            if finite_over_q0 > 1.01
            else "- finite_q_peierls is not materially better than q0_mass_diagnostic at the 1% threshold."
        ),
        f"- strongest finite_q_peierls+plus contact_effect_ratio occurs in {finite_effect_orientation}; value = {_fmt(finite_effect_value)}",
        (
            "- Contact does not improve the max spatial residual even if some individual orientation has positive contact_effect_ratio."
            if finite_vs_none < 1.0 / 1.01
            else (
                "- Contact improves middle/large q but does not remove the small-q O(q) term; the form factor is not the O(q) closure issue."
                if abs(finite_alpha - 1.0) < 0.35
                else "- Contact changes do not show a simple O(q) closure pattern."
            )
        ),
        "",
        "## Next checks",
        f"- {next_hint}",
        "- If longitudinal alone is bad, inspect longitudinal current/contact and density-current relations.",
        "- If transverse is also bad, inspect the current-current bubble convention as a whole.",
        "",
        "## Output files",
        f"- compact CSV: {compact_csv}",
        f"- expanded_data_written={bool(args.write_expanded_data)}",
        *(
            [f"- expanded CSV: {expanded_csv}", f"- expanded NPZ: {expanded_npz}"]
            if args.write_expanded_data
            else ["- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally."]
        ),
        *(f"- figure: {path}" for path in figure_paths),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matsubara-n-list", nargs="+", type=int, default=list(DEFAULT_MATSUBARA_N_LIST))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--q-list", nargs="+", type=float, default=list(DEFAULT_Q_LIST))
    parser.add_argument("--q-angle-list", nargs="+", type=_angle_expression, default=list(DEFAULT_Q_ANGLE_LIST))
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--degeneracy-tol", type=float, default=DEFAULT_DEGENERACY_TOL_EV)
    parser.add_argument("--combos", nargs="+", type=_parse_combo, default=[_parse_combo(item) for item in DEFAULT_COMBOS])
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--write-expanded-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.matsubara_n_list = list(QUICK_MATSUBARA_N_LIST)
        args.q_list = list(QUICK_Q_LIST)
        args.q_angle_list = list(QUICK_Q_ANGLE_LIST)
        args.nk_list = list(QUICK_NK_LIST)
        args.temperature = DEFAULT_TEMPERATURE_K
    combos = list(args.combos)
    data = run_audit(
        matsubara_n_list=list(args.matsubara_n_list),
        temperature_K=float(args.temperature),
        q_list=list(args.q_list),
        q_angle_list=list(args.q_angle_list),
        nk_list=list(args.nk_list),
        degeneracy_tol_eV=float(args.degeneracy_tol),
        combos=combos,
    )
    compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path = _output_paths(args.output_prefix)
    _write_csv(compact_csv, data, COMPACT_COLUMNS)
    if args.write_expanded_data:
        _write_csv(expanded_csv, data, EXPANDED_COLUMNS)
        _write_npz(expanded_npz, data)
    figure_paths = _plot_outputs(data, figure_dir)
    command = "python " + " ".join(shlex.quote(value) for value in sys.argv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(_summary_lines(data, args, command, combos, compact_csv, expanded_csv, expanded_npz, figure_paths))
        + "\n",
        encoding="utf-8",
    )
    print(f"compact_csv_path = {compact_csv}")
    print(f"expanded_data_written = {bool(args.write_expanded_data)}")
    if args.write_expanded_data:
        print(f"expanded_csv_path = {expanded_csv}")
        print(f"expanded_npz_path = {expanded_npz}")
    print(f"summary_path = {summary_path}")
    print("figure_paths = " + ", ".join(str(path) for path in figure_paths))


if __name__ == "__main__":
    main()
