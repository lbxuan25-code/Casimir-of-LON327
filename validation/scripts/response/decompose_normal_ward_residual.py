#!/usr/bin/env python3
"""Decompose normal-state Ward residuals by side and component."""

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
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402
from lno327.ward_response import normal_density_current_response_imag_axis, ward_residuals  # noqa: E402

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "normal_ward_residual_decomposition"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "normal_ward_residual_decomposition"
SUMMARY_NAME = "summary.md"
EPS = 1e-300

DEFAULT_MATSUBARA_N_LIST = (1, 2, 4)
DEFAULT_TEMPERATURE_K = 30.0
DEFAULT_Q_LIST = (0.001, 0.005, 0.01, 0.05, 0.1)
DEFAULT_Q_ANGLE_LIST = (0.0, np.pi / 4.0, np.pi / 2.0)
DEFAULT_NK_LIST = (8, 12, 16)
DEFAULT_DEGENERACY_TOL_EV = 1e-10
DEFAULT_COMBOS = (
    "midpoint:none:not_applicable",
    "peierls:none:not_applicable",
    "peierls:q0_mass_diagnostic:plus",
    "peierls:finite_q_peierls:plus",
    "peierls:finite_q_peierls:minus",
)

QUICK_MATSUBARA_N_LIST = (1,)
QUICK_Q_LIST = (0.001, 0.01, 0.1)
QUICK_Q_ANGLE_LIST = (0.0, np.pi / 4.0)
QUICK_NK_LIST = (8,)

COMPONENTS = ("0", "x", "y")
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
    "global_response_norm",
    "global_norm_error",
    "dominant_term",
    "density_component",
    "spatial_component",
    "diamagnetic_contact_included",
    "normal_state_only",
    "bdg_computed",
    "response_computed",
    "conductivity_computed",
    "casimir_computed",
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


def _dominant_term(term_iomega: complex, term_qx: complex, term_qy: complex) -> str:
    values = {
        "iomega": abs(term_iomega),
        "qx": abs(term_qx),
        "qy": abs(term_qy),
    }
    return max(values, key=values.__getitem__)


def _diagnosis(side: str, component: str, contact_scheme: str, closure_ratio: float) -> str:
    if closure_ratio < 1e-3:
        return "residual_small"
    if component == "0":
        return "density_sector_large_residual"
    if contact_scheme != "none":
        return "contact_sensitive_spatial_residual"
    if side not in {"left", "right"}:
        return "left_right_asymmetry_warning"
    return "spatial_sector_large_residual"


def _decompose_side(
    matrix: np.ndarray,
    omega_eV: float,
    qx: float,
    qy: float,
    side: str,
    component_index: int,
) -> tuple[complex, complex, complex, complex]:
    if side == "left":
        term_iomega = 1j * omega_eV * matrix[0, component_index]
        term_qx = qx * matrix[1, component_index]
        term_qy = qy * matrix[2, component_index]
    elif side == "right":
        term_iomega = 1j * omega_eV * matrix[component_index, 0]
        term_qx = matrix[component_index, 1] * qx
        term_qy = matrix[component_index, 2] * qy
    else:
        raise ValueError("side must be left or right")
    residual = term_iomega + term_qx + term_qy
    return term_iomega, term_qx, term_qy, residual


def run_decomposition(
    *,
    matsubara_n_list: list[int],
    temperature_K: float,
    q_list: list[float],
    q_angle_list: list[float],
    nk_list: list[int],
    degeneracy_tol_eV: float,
    combos: list[tuple[str, str, str]],
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
            for combo in combos:
                vertex_scheme, contact_scheme, contact_sign = combo
                response_contact_sign = "plus" if contact_sign == "not_applicable" else contact_sign
                for q_model in q_list:
                    for q_angle in q_angle_list:
                        qx = float(q_model * np.cos(q_angle))
                        qy = float(q_model * np.sin(q_angle))
                        matrix = normal_density_current_response_imag_axis(
                            mesh,
                            config,
                            np.array([qx, qy], dtype=float),
                            weights,
                            vertex_scheme=vertex_scheme,
                            contact_scheme=contact_scheme,
                            contact_sign_convention=response_contact_sign,
                        )
                        left, right = ward_residuals(matrix, omega_eV, np.array([qx, qy], dtype=float))
                        side_residuals = {"left": left, "right": right}
                        global_response_norm = max(float(np.linalg.norm(matrix)), EPS)
                        pi_entries = {
                            "Pi_00": complex(matrix[0, 0]),
                            "Pi_0x": complex(matrix[0, 1]),
                            "Pi_0y": complex(matrix[0, 2]),
                            "Pi_x0": complex(matrix[1, 0]),
                            "Pi_xx": complex(matrix[1, 1]),
                            "Pi_xy": complex(matrix[1, 2]),
                            "Pi_y0": complex(matrix[2, 0]),
                            "Pi_yx": complex(matrix[2, 1]),
                            "Pi_yy": complex(matrix[2, 2]),
                        }
                        for side in ("left", "right"):
                            global_norm_error = float(np.linalg.norm(side_residuals[side]) / global_response_norm)
                            for component_index, component in enumerate(COMPONENTS):
                                term_iomega, term_qx, term_qy, residual = _decompose_side(
                                    matrix,
                                    omega_eV,
                                    qx,
                                    qy,
                                    side,
                                    component_index,
                                )
                                component_scale = abs(term_iomega) + abs(term_qx) + abs(term_qy) + EPS
                                closure_ratio = float(abs(residual) / component_scale)
                                rows.append(
                                    {
                                        "vertex_scheme": vertex_scheme,
                                        "current_vertex_sign_convention": (
                                            "plus" if vertex_scheme == "peierls" else "not_applicable"
                                        ),
                                        "contact_scheme": contact_scheme,
                                        "contact_sign_convention": contact_sign,
                                        "density_vertex_scheme": "identity_4_orbitals_shared_in_plane_position",
                                        "matsubara_n": int(matsubara_n),
                                        "omega_eV": float(omega_eV),
                                        "nk": int(nk),
                                        "q_model": float(q_model),
                                        "q_angle": float(q_angle),
                                        "qx_model": qx,
                                        "qy_model": qy,
                                        "side": side,
                                        "component": component,
                                        "term_iomega_real": float(np.real(term_iomega)),
                                        "term_iomega_imag": float(np.imag(term_iomega)),
                                        "term_qx_real": float(np.real(term_qx)),
                                        "term_qx_imag": float(np.imag(term_qx)),
                                        "term_qy_real": float(np.real(term_qy)),
                                        "term_qy_imag": float(np.imag(term_qy)),
                                        "residual_real": float(np.real(residual)),
                                        "residual_imag": float(np.imag(residual)),
                                        "residual_abs": float(abs(residual)),
                                        "component_scale": float(component_scale),
                                        "closure_ratio": closure_ratio,
                                        "global_response_norm": global_response_norm,
                                        "global_norm_error": global_norm_error,
                                        "dominant_term": _dominant_term(term_iomega, term_qx, term_qy),
                                        "density_component": component == "0",
                                        "spatial_component": component in {"x", "y"},
                                        "diamagnetic_contact_included": contact_scheme != "none",
                                        "normal_state_only": True,
                                        "bdg_computed": False,
                                        "response_computed": True,
                                        "conductivity_computed": False,
                                        "casimir_computed": False,
                                        "not_final_finite_q_conductivity": True,
                                        "not_final_casimir_conclusion": True,
                                        "diagnosis": _diagnosis(side, component, contact_scheme, closure_ratio),
                                        **pi_entries,
                                    }
                                )
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
    vertex_scheme, contact_scheme, sign = combo
    return (
        (data["vertex_scheme"] == vertex_scheme)
        & (data["contact_scheme"] == contact_scheme)
        & (data["contact_sign_convention"] == sign)
    )


def _component_mask(data: dict[str, np.ndarray], component: str) -> np.ndarray:
    return data["component"] == component


def _max_for_mask(data: dict[str, np.ndarray], field: str, mask: np.ndarray) -> float:
    return float(np.max(data[field][mask].astype(float)))


def _row_of_max(data: dict[str, np.ndarray], field: str) -> dict[str, object]:
    index = int(np.argmax(data[field].astype(float)))
    return {column: data[column][index] for column in COMPACT_COLUMNS}


def _fit_q_scaling(data: dict[str, np.ndarray], combo: tuple[str, str, str], field: str) -> float:
    q_values = [0.001, 0.005, 0.01]
    errors = []
    used_q = []
    base_mask = _combo_mask(data, combo)
    for q_model in q_values:
        mask = base_mask & np.isclose(data["q_model"].astype(float), q_model)
        if np.any(mask):
            value = _max_for_mask(data, field, mask)
            if value > 0.0:
                used_q.append(q_model)
                errors.append(value)
    if len(errors) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(used_q), np.log(errors), deg=1)
    return float(slope)


def _fmt(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.6g}"


def _side_component_max(data: dict[str, np.ndarray], combo: tuple[str, str, str], side: str, component: str) -> float:
    mask = _combo_mask(data, combo) & (data["side"] == side) & (data["component"] == component)
    return _max_for_mask(data, "closure_ratio", mask)


def _plot_outputs(data: dict[str, np.ndarray], combos: list[tuple[str, str, str]], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    x_labels = [f"{side}:{component}" for side in ("left", "right") for component in COMPONENTS]
    x = np.arange(len(x_labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for combo in combos:
        values = [
            _side_component_max(data, combo, side, component)
            for side in ("left", "right")
            for component in COMPONENTS
        ]
        ax.semilogy(x, np.maximum(values, EPS), marker="o", label=_combo_label(combo))
    ax.set_xticks(x, x_labels, rotation=30, ha="right")
    ax.set(ylabel="max component closure ratio", title="Ward residual closure by side/component")
    style_publication_axis(ax)
    path = figure_dir / "max_closure_ratio_by_component.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    selected = [
        combo
        for combo in (
            ("peierls", "none", "not_applicable"),
            ("peierls", "finite_q_peierls", "plus"),
            ("peierls", "finite_q_peierls", "minus"),
        )
        if np.any(_combo_mask(data, combo))
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for combo in selected:
        q_values = sorted(set(float(q) for q in data["q_model"][_combo_mask(data, combo)]))
        values = []
        for q_model in q_values:
            mask = _combo_mask(data, combo) & np.isclose(data["q_model"].astype(float), q_model)
            values.append(_max_for_mask(data, "residual_abs", mask))
        ax.loglog(q_values, np.maximum(values, EPS), marker="o", label=_combo_label(combo))
    ax.set(xlabel="q_model", ylabel="max residual_abs", title="Small-q Ward residual scaling")
    style_publication_axis(ax)
    path = figure_dir / "q_scaling_by_combo.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    return paths


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
    max_residual_row = _row_of_max(data, "residual_abs")
    max_closure_row = _row_of_max(data, "closure_ratio")
    density_max = _max_for_mask(data, "closure_ratio", data["component"] == "0")
    spatial_max = _max_for_mask(data, "closure_ratio", data["component"] != "0")
    density_abs_max = _max_for_mask(data, "residual_abs", data["component"] == "0")
    spatial_abs_max = _max_for_mask(data, "residual_abs", data["component"] != "0")
    left_max = _max_for_mask(data, "closure_ratio", data["side"] == "left")
    right_max = _max_for_mask(data, "closure_ratio", data["side"] == "right")
    left_right_ratio = max(left_max, right_max) / max(min(left_max, right_max), EPS)
    peierls_none = ("peierls", "none", "not_applicable")
    finite_plus = ("peierls", "finite_q_peierls", "plus")
    finite_minus = ("peierls", "finite_q_peierls", "minus")
    q0_plus = ("peierls", "q0_mass_diagnostic", "plus")

    component_comparison_lines = []
    if np.any(_combo_mask(data, peierls_none)) and np.any(_combo_mask(data, finite_plus)):
        for component in COMPONENTS:
            base = _max_for_mask(data, "closure_ratio", _combo_mask(data, peierls_none) & _component_mask(data, component))
            finite = _max_for_mask(data, "closure_ratio", _combo_mask(data, finite_plus) & _component_mask(data, component))
            component_comparison_lines.append(
                f"- component {component}: peierls+none / finite_q_peierls+plus closure factor = {_fmt(base / max(finite, EPS))}"
            )
    q0_vs_finite_line = "q0_mass_diagnostic versus finite_q_peierls comparison unavailable."
    if np.any(_combo_mask(data, q0_plus)) and np.any(_combo_mask(data, finite_plus)):
        q0_spatial = _max_for_mask(data, "closure_ratio", _combo_mask(data, q0_plus) & (data["component"] != "0"))
        finite_spatial = _max_for_mask(data, "closure_ratio", _combo_mask(data, finite_plus) & (data["component"] != "0"))
        q0_vs_finite_line = (
            "spatial closure q0_mass_diagnostic+plus / finite_q_peierls+plus factor = "
            f"{_fmt(q0_spatial / max(finite_spatial, EPS))}"
        )
    finite_minus_line = "finite_q_peierls+minus comparison unavailable."
    if np.any(_combo_mask(data, finite_plus)) and np.any(_combo_mask(data, finite_minus)):
        plus_max = _max_for_mask(data, "closure_ratio", _combo_mask(data, finite_plus))
        minus_max = _max_for_mask(data, "closure_ratio", _combo_mask(data, finite_minus))
        finite_minus_line = f"finite_q_peierls minus/plus max closure ratio factor = {_fmt(minus_max / max(plus_max, EPS))}"

    scaling_lines = []
    for combo in combos:
        alpha = _fit_q_scaling(data, combo, "residual_abs")
        if np.isnan(alpha):
            note = "insufficient small-q points"
        elif abs(alpha - 1.0) < 0.35:
            note = "approximately O(q), suggesting a response-level O(q) gap"
        elif abs(alpha - 2.0) < 0.5 or abs(alpha - 3.0) < 0.5:
            note = "closer to higher-order finite-q contact behavior"
        else:
            note = "not close to a simple q or q^2/q^3 scaling"
        scaling_lines.append(f"- {_combo_label(combo)}: alpha = {_fmt(alpha)} ({note})")

    return [
        "# Normal Ward residual decomposition",
        "",
        "This is a Ward decomposition diagnostic.",
        "It is not conductivity, not a reflection/Casimir input, and not a material conclusion.",
        "The rows decompose each left/right component into iOmega, qx, and qy terms using a local component scale.",
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
        "## Maxima",
        (
            "- max residual_abs: "
            f"combo={_combo_label((str(max_residual_row['vertex_scheme']), str(max_residual_row['contact_scheme']), str(max_residual_row['contact_sign_convention'])))}; "
            f"side={max_residual_row['side']}; component={max_residual_row['component']}; "
            f"q={_fmt(float(max_residual_row['q_model']))}; value={_fmt(float(max_residual_row['residual_abs']))}"
        ),
        (
            "- max closure_ratio: "
            f"combo={_combo_label((str(max_closure_row['vertex_scheme']), str(max_closure_row['contact_scheme']), str(max_closure_row['contact_sign_convention'])))}; "
            f"side={max_closure_row['side']}; component={max_closure_row['component']}; "
            f"q={_fmt(float(max_closure_row['q_model']))}; value={_fmt(float(max_closure_row['closure_ratio']))}"
        ),
        f"- density component max closure_ratio = {_fmt(density_max)}",
        f"- spatial component max closure_ratio = {_fmt(spatial_max)}",
        f"- density component max residual_abs = {_fmt(density_abs_max)}",
        f"- spatial component max residual_abs = {_fmt(spatial_abs_max)}",
        f"- left max closure_ratio = {_fmt(left_max)}",
        f"- right max closure_ratio = {_fmt(right_max)}",
        f"- left/right max ratio = {_fmt(left_right_ratio)}",
        (
            "- left/right structures are close."
            if left_right_ratio < 1.1
            else "- left/right structures are noticeably asymmetric; check vertex order, conjugation, denominator, or Pi index convention."
        ),
        "",
        "## Contact/component comparison",
        (
            "- density component is already comparable to or larger than spatial residuals."
            if density_abs_max > 0.5 * spatial_abs_max
            else "- density component closure_ratio can be large, but residual_abs is smaller than the dominant spatial residual."
        ),
        *component_comparison_lines,
        f"- {q0_vs_finite_line}",
        f"- {finite_minus_line}",
        "",
        "## Small-q scaling",
        "- alpha is fitted from max residual_abs at q_model = 0.001, 0.005, 0.01.",
        *scaling_lines,
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
    data = run_decomposition(
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
    figure_paths = _plot_outputs(data, combos, figure_dir)
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
