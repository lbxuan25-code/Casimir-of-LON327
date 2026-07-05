#!/usr/bin/env python3
"""Historical diagnostic / convention scanner for the density-current Ward sector.

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

from lno327 import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh  # noqa: E402
from validation.lib.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402
from lno327.response.normal_density_current import normal_density_current_response_imag_axis  # noqa: E402

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "density_current_ward_sector"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "density_current_ward_sector"
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
    "peierls:finite_q_peierls:plus",
    "peierls:finite_q_peierls:minus",
    "midpoint:none:not_applicable",
)

QUICK_MATSUBARA_N_LIST = (1,)
QUICK_Q_LIST = (0.001, 0.01, 0.1)
QUICK_Q_ANGLE_LIST = (0.0, np.pi / 4.0)
QUICK_NK_LIST = (8,)

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
    "Pi_00_real",
    "Pi_00_imag",
    "Pi_x0_real",
    "Pi_x0_imag",
    "Pi_y0_real",
    "Pi_y0_imag",
    "Pi_0x_real",
    "Pi_0x_imag",
    "Pi_0y_real",
    "Pi_0y_imag",
    "term_iomega_real",
    "term_iomega_imag",
    "left_term_qx_real",
    "left_term_qx_imag",
    "left_term_qy_real",
    "left_term_qy_imag",
    "left_residual_real",
    "left_residual_imag",
    "left_residual_abs",
    "left_scale",
    "left_density_closure_ratio",
    "right_term_qx_real",
    "right_term_qx_imag",
    "right_term_qy_real",
    "right_term_qy_imag",
    "right_residual_real",
    "right_residual_imag",
    "right_residual_abs",
    "right_scale",
    "right_density_closure_ratio",
    "left_right_abs_difference",
    "left_right_ratio",
    "contact_sensitive",
    "dominant_left_term",
    "dominant_right_term",
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
    "Pi_xx_real",
    "Pi_xx_imag",
    "Pi_xy_real",
    "Pi_xy_imag",
    "Pi_yx_real",
    "Pi_yx_imag",
    "Pi_yy_real",
    "Pi_yy_imag",
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
    values = {"iomega": abs(term_iomega), "qx": abs(term_qx), "qy": abs(term_qy)}
    return max(values, key=values.__getitem__)


def _diagnosis(left_ratio: float, right_ratio: float, contact_sensitive: bool, left_right_ratio: float) -> str:
    if contact_sensitive:
        return "density_residual_contact_sensitive_warning"
    if left_right_ratio > 1.1:
        return "left_right_density_asymmetry_warning"
    if max(left_ratio, right_ratio) < 1e-3:
        return "density_residual_small"
    return "density_sector_large_residual"


def _complex_parts(prefix: str, value: complex) -> dict[str, float]:
    return {
        f"{prefix}_real": float(np.real(value)),
        f"{prefix}_imag": float(np.imag(value)),
    }


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
    matrix: np.ndarray,
) -> dict[str, object]:
    vertex_scheme, contact_scheme, contact_sign = combo
    pi_00 = complex(matrix[0, 0])
    pi_0x = complex(matrix[0, 1])
    pi_0y = complex(matrix[0, 2])
    pi_x0 = complex(matrix[1, 0])
    pi_y0 = complex(matrix[2, 0])
    term_iomega = 1j * omega_eV * pi_00
    left_qx = qx * pi_x0
    left_qy = qy * pi_y0
    right_qx = pi_0x * qx
    right_qy = pi_0y * qy
    left_residual = term_iomega + left_qx + left_qy
    right_residual = term_iomega + right_qx + right_qy
    left_scale = abs(term_iomega) + abs(left_qx) + abs(left_qy) + EPS
    right_scale = abs(term_iomega) + abs(right_qx) + abs(right_qy) + EPS
    left_abs = float(abs(left_residual))
    right_abs = float(abs(right_residual))
    left_right_ratio = max(left_abs, right_abs) / max(min(left_abs, right_abs), EPS)
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
        **_complex_parts("Pi_00", pi_00),
        **_complex_parts("Pi_x0", pi_x0),
        **_complex_parts("Pi_y0", pi_y0),
        **_complex_parts("Pi_0x", pi_0x),
        **_complex_parts("Pi_0y", pi_0y),
        **_complex_parts("term_iomega", term_iomega),
        **_complex_parts("left_term_qx", left_qx),
        **_complex_parts("left_term_qy", left_qy),
        **_complex_parts("left_residual", left_residual),
        "left_residual_abs": left_abs,
        "left_scale": float(left_scale),
        "left_density_closure_ratio": float(left_abs / left_scale),
        **_complex_parts("right_term_qx", right_qx),
        **_complex_parts("right_term_qy", right_qy),
        **_complex_parts("right_residual", right_residual),
        "right_residual_abs": right_abs,
        "right_scale": float(right_scale),
        "right_density_closure_ratio": float(right_abs / right_scale),
        "left_right_abs_difference": float(abs(left_abs - right_abs)),
        "left_right_ratio": float(left_right_ratio),
        "contact_sensitive": False,
        "dominant_left_term": _dominant_term(term_iomega, left_qx, left_qy),
        "dominant_right_term": _dominant_term(term_iomega, right_qx, right_qy),
        "response_computed": True,
        "conductivity_computed": False,
        "casimir_computed": False,
        "normal_state_only": True,
        "bdg_computed": False,
        "not_final_finite_q_conductivity": True,
        "not_final_casimir_conclusion": True,
        "diagnosis": "",
        **_complex_parts("Pi_xx", complex(matrix[1, 1])),
        **_complex_parts("Pi_xy", complex(matrix[1, 2])),
        **_complex_parts("Pi_yx", complex(matrix[2, 1])),
        **_complex_parts("Pi_yy", complex(matrix[2, 2])),
    }
    return row


def _mark_contact_sensitivity(rows: list[dict[str, object]]) -> None:
    keyed: dict[tuple[int, int, float, float], dict[tuple[str, str, str], dict[str, object]]] = {}
    for row in rows:
        if row["vertex_scheme"] != "peierls":
            continue
        key = (
            int(row["matsubara_n"]),
            int(row["nk"]),
            float(row["q_model"]),
            float(row["q_angle"]),
        )
        combo = (str(row["vertex_scheme"]), str(row["contact_scheme"]), str(row["contact_sign_convention"]))
        keyed.setdefault(key, {})[combo] = row
    for group in keyed.values():
        watched = [
            ("peierls", "none", "not_applicable"),
            ("peierls", "q0_mass_diagnostic", "plus"),
            ("peierls", "finite_q_peierls", "plus"),
        ]
        if not all(combo in group for combo in watched):
            continue
        reference = max(float(group[watched[0]]["left_residual_abs"]), float(group[watched[0]]["right_residual_abs"]))
        values = [max(float(group[combo]["left_residual_abs"]), float(group[combo]["right_residual_abs"])) for combo in watched]
        sensitive = (max(values) - min(values)) / max(reference, EPS) > 0.01
        for combo in watched:
            group[combo]["contact_sensitive"] = sensitive
    for row in rows:
        row["diagnosis"] = _diagnosis(
            float(row["left_density_closure_ratio"]),
            float(row["right_density_closure_ratio"]),
            bool(row["contact_sensitive"]),
            float(row["left_right_ratio"]),
        )


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
    if any(n < 1 for n in matsubara_n_list):
        raise ValueError("matsubara-n-list must contain only n >= 1")
    if any(q < 0.0 for q in q_list):
        raise ValueError("q-list values must be non-negative")
    if any(nk <= 0 for nk in nk_list):
        raise ValueError("nk-list values must be positive")
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
                                matrix=matrix,
                            )
                        )
    _mark_contact_sensitivity(rows)
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


def _fit_alpha(data: dict[str, np.ndarray], combo: tuple[str, str, str], field: str) -> float:
    used_q = []
    values = []
    base = _combo_mask(data, combo)
    for q_model in (0.001, 0.005, 0.01):
        mask = base & np.isclose(data["q_model"].astype(float), q_model)
        if np.any(mask):
            value = _max_for_mask(data, field, mask)
            if value > 0.0:
                used_q.append(q_model)
                values.append(value)
    if len(values) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(used_q), np.log(values), deg=1)
    return float(slope)


def _fmt(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.6g}"


def _max_density_row(data: dict[str, np.ndarray]) -> tuple[str, dict[str, object]]:
    left = data["left_residual_abs"].astype(float)
    right = data["right_residual_abs"].astype(float)
    if float(np.max(left)) >= float(np.max(right)):
        side = "left"
        index = int(np.argmax(left))
    else:
        side = "right"
        index = int(np.argmax(right))
    return side, {column: data[column][index] for column in COMPACT_COLUMNS}


def _plot_outputs(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        ("peierls", "none", "not_applicable"),
        ("peierls", "finite_q_peierls", "plus"),
        ("peierls", "finite_q_peierls", "minus"),
    ]
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for combo in selected:
        if not np.any(_combo_mask(data, combo)):
            continue
        q_values = sorted(set(float(q) for q in data["q_model"][_combo_mask(data, combo)]))
        values = []
        for q_model in q_values:
            mask = _combo_mask(data, combo) & np.isclose(data["q_model"].astype(float), q_model)
            values.append(max(_max_for_mask(data, "left_residual_abs", mask), _max_for_mask(data, "right_residual_abs", mask)))
        ax.loglog(q_values, np.maximum(values, EPS), marker="o", label=_combo_label(combo))
    ax.set(xlabel="q_model", ylabel="max density residual_abs", title="Density-sector Ward residual")
    style_publication_axis(ax)
    path = figure_dir / "density_residual_vs_q.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for combo in selected:
        if not np.any(_combo_mask(data, combo)):
            continue
        q_values = sorted(set(float(q) for q in data["q_model"][_combo_mask(data, combo)]))
        values = []
        for q_model in q_values:
            mask = _combo_mask(data, combo) & np.isclose(data["q_model"].astype(float), q_model)
            values.append(
                max(
                    _max_for_mask(data, "left_density_closure_ratio", mask),
                    _max_for_mask(data, "right_density_closure_ratio", mask),
                )
            )
        ax.semilogx(q_values, values, marker="o", label=_combo_label(combo))
    ax.set(xlabel="q_model", ylabel="max density closure ratio", title="Density-sector closure ratio")
    style_publication_axis(ax)
    path = figure_dir / "density_closure_ratio_vs_q.png"
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
    max_side, max_row = _max_density_row(data)
    max_combo = (str(max_row["vertex_scheme"]), str(max_row["contact_scheme"]), str(max_row["contact_sign_convention"]))
    contact_sensitive = bool(np.any(data["contact_sensitive"].astype(bool)))
    peierls_none = ("peierls", "none", "not_applicable")
    finite_plus = ("peierls", "finite_q_peierls", "plus")
    finite_minus = ("peierls", "finite_q_peierls", "minus")
    none_vs_finite = "peierls+none versus finite_q_peierls+plus comparison unavailable."
    if np.any(_combo_mask(data, peierls_none)) and np.any(_combo_mask(data, finite_plus)):
        none_value = max(
            _max_for_mask(data, "left_residual_abs", _combo_mask(data, peierls_none)),
            _max_for_mask(data, "right_residual_abs", _combo_mask(data, peierls_none)),
        )
        finite_value = max(
            _max_for_mask(data, "left_residual_abs", _combo_mask(data, finite_plus)),
            _max_for_mask(data, "right_residual_abs", _combo_mask(data, finite_plus)),
        )
        none_vs_finite = f"peierls+none / peierls+finite_q_peierls+plus max density residual factor = {_fmt(none_value / max(finite_value, EPS))}"
    left_max = float(np.max(data["left_residual_abs"].astype(float)))
    right_max = float(np.max(data["right_residual_abs"].astype(float)))
    left_right_ratio = max(left_max, right_max) / max(min(left_max, right_max), EPS)
    scaling_lines = []
    left_alphas = []
    right_alphas = []
    for combo in combos:
        left_alpha = _fit_alpha(data, combo, "left_residual_abs")
        right_alpha = _fit_alpha(data, combo, "right_residual_abs")
        left_alphas.append(left_alpha)
        right_alphas.append(right_alpha)
        if abs(left_alpha - 1.0) < 0.35 and abs(right_alpha - 1.0) < 0.35:
            note = "approximately O(q), suggesting a density-sector response-level gap"
        elif abs(left_alpha - 2.0) < 0.5 and abs(right_alpha - 2.0) < 0.5:
            note = "approximately O(q^2), so this density sector is unlikely to be the dominant O(q) gap"
        else:
            note = "not a clean O(q) or O(q^2) scaling"
        scaling_lines.append(f"- {_combo_label(combo)}: left alpha={_fmt(left_alpha)}, right alpha={_fmt(right_alpha)} ({note})")
    finite_minus_line = "finite_q_peierls minus comparison unavailable."
    if np.any(_combo_mask(data, finite_plus)) and np.any(_combo_mask(data, finite_minus)):
        plus_value = max(
            _max_for_mask(data, "left_residual_abs", _combo_mask(data, finite_plus)),
            _max_for_mask(data, "right_residual_abs", _combo_mask(data, finite_plus)),
        )
        minus_value = max(
            _max_for_mask(data, "left_residual_abs", _combo_mask(data, finite_minus)),
            _max_for_mask(data, "right_residual_abs", _combo_mask(data, finite_minus)),
        )
        finite_minus_line = f"finite_q_peierls minus/plus density residual factor = {_fmt(minus_value / max(plus_value, EPS))}"
    finite_alphas = [value for value in (*left_alphas, *right_alphas) if not np.isnan(value)]
    density_o_q = bool(finite_alphas and np.median(finite_alphas) < 1.35)
    if density_o_q:
        likely_next = (
            "Density-sector scaling is close to O(q); next checks should prioritize density vertex Gamma0, "
            "equal-time/commutator terms, and Kubo denominator/vertex-order/complex-conjugation convention."
        )
    else:
        likely_next = (
            "Density-sector residual is contact-insensitive and closer to O(q^2), so the O(q) gap seen in the "
            "full decomposition is more likely tied to spatial current/equal-time or Kubo convention closure; "
            "still keep Gamma0 embedding as a later consistency check."
        )
    return [
        "# Density-current Ward sector audit",
        "",
        "This is a density-current Ward-sector audit.",
        "It is not conductivity, not a reflection/Casimir input, and not a material conclusion.",
        "Only the density residuals R_L[0] and R_R[0] are analyzed.",
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
        "## Density residual maxima",
        f"- max density residual side = {max_side}",
        f"- max combo = {_combo_label(max_combo)}",
        f"- matsubara_n = {int(max_row['matsubara_n'])}",
        f"- nk = {int(max_row['nk'])}",
        f"- q_model = {_fmt(float(max_row['q_model']))}",
        f"- q_angle = {_fmt(float(max_row['q_angle']))}",
        f"- max residual_abs = {_fmt(float(max_row[max_side + '_residual_abs']))}",
        f"- left/right residual max ratio = {_fmt(left_right_ratio)}",
        (
            "- left/right density residuals are close."
            if left_right_ratio < 1.1
            else "- left/right density residuals differ noticeably; check vertex order, conjugation, denominator, or Pi index convention."
        ),
        "",
        "## Contact sensitivity",
        f"- contact_sensitive_warning = {contact_sensitive}",
        (
            "- density residual is not contact-sensitive at the 1% threshold, as expected because contact enters only the spatial-spatial block."
            if not contact_sensitive
            else "- warning: density residual changes by more than 1% across contact schemes."
        ),
        f"- {none_vs_finite}",
        f"- {finite_minus_line}",
        "",
        "## Small-q scaling",
        "- alpha is fitted from q_model = 0.001, 0.005, 0.01.",
        *scaling_lines,
        "",
        "## Likely next checks",
        f"- {likely_next}",
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
