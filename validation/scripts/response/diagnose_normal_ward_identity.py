#!/usr/bin/env python3
"""Historical diagnostic / convention scanner for normal-state Ward residuals.

This script scans vertex/contact convention choices for Stage 4 diagnostics.
It is not the main response implementation, not finite-q conductivity, and not
a reflection/Casimir input.
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
from lno327.ward_response import normal_density_current_response_imag_axis, ward_errors  # noqa: E402

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "normal_ward_identity"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "normal_ward_identity"
SUMMARY_NAME = "summary.md"
EPS = 1e-300

DEFAULT_MATSUBARA_N_LIST = (1, 2, 4)
DEFAULT_TEMPERATURE_K = 30.0
DEFAULT_Q_LIST = (0.001, 0.005, 0.01, 0.05, 0.1)
DEFAULT_Q_ANGLE_LIST = (0.0, np.pi / 4.0, np.pi / 2.0)
DEFAULT_NK_LIST = (8, 12, 16)
DEFAULT_DEGENERACY_TOL_EV = 1e-10
DEFAULT_VERTEX_SCHEMES = ("midpoint", "peierls")
DEFAULT_CONTACT_SCHEMES = ("none", "q0_mass_diagnostic", "finite_q_peierls")
DEFAULT_CONTACT_SIGN_CONVENTIONS = ("plus", "minus")

QUICK_MATSUBARA_N_LIST = (1,)
QUICK_Q_LIST = (0.001, 0.01, 0.1)
QUICK_Q_ANGLE_LIST = (0.0, np.pi / 4.0)
QUICK_NK_LIST = (8,)

COMPACT_COLUMNS = (
    "vertex_scheme",
    "current_vertex_sign_convention",
    "density_vertex_scheme",
    "contact_scheme",
    "contact_sign_convention",
    "matsubara_n",
    "omega_eV",
    "nk",
    "q_model",
    "q_angle",
    "qx_model",
    "qy_model",
    "left_ward_error",
    "right_ward_error",
    "max_ward_error",
    "density_current_included",
    "current_current_included",
    "diamagnetic_contact_included",
    "not_final_finite_q_contact",
    "normal_state_only",
    "bdg_computed",
    "conductivity_computed",
    "casimir_computed",
    "not_final_casimir_conclusion",
    "not_final_finite_q_conductivity",
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


def _diagnosis(max_error: float, vertex_scheme: str, contact_scheme: str, contact_sign_convention: str) -> str:
    prefix = f"{vertex_scheme}_{contact_scheme}_{contact_sign_convention}_"
    if max_error < 1e-6:
        return prefix + "prototype_residual_small"
    if max_error < 1e-3:
        return prefix + "prototype_residual_moderate"
    return prefix + "warning_large_ward_residual_contact_or_vertex_not_closed"


def run_diagnostic(
    *,
    matsubara_n_list: list[int],
    temperature_K: float,
    q_list: list[float],
    q_angle_list: list[float],
    nk_list: list[int],
    degeneracy_tol_eV: float,
    vertex_schemes: list[str],
    contact_schemes: list[str],
    contact_sign_conventions: list[str],
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
    for scheme in vertex_schemes:
        if scheme not in {"midpoint", "peierls"}:
            raise ValueError("vertex schemes must be midpoint or peierls")
    for scheme in contact_schemes:
        if scheme not in {"none", "q0_mass_diagnostic", "finite_q_peierls"}:
            raise ValueError("contact schemes must be none, q0_mass_diagnostic, or finite_q_peierls")
    for convention in contact_sign_conventions:
        if convention not in {"plus", "minus"}:
            raise ValueError("contact sign conventions must be plus or minus")

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
            for vertex_scheme in vertex_schemes:
                for contact_scheme in contact_schemes:
                    sign_conventions = ("not_applicable",) if contact_scheme == "none" else tuple(contact_sign_conventions)
                    for contact_sign_convention in sign_conventions:
                        response_contact_sign = "plus" if contact_sign_convention == "not_applicable" else contact_sign_convention
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
                                left_error, right_error, max_error = ward_errors(
                                    matrix,
                                    omega_eV,
                                    np.array([qx, qy], dtype=float),
                                )
                                rows.append(
                                    {
                                        "vertex_scheme": vertex_scheme,
                                        "current_vertex_sign_convention": (
                                            "plus" if vertex_scheme == "peierls" else "not_applicable"
                                        ),
                                        "density_vertex_scheme": "identity_4_orbitals_shared_in_plane_position",
                                        "contact_scheme": contact_scheme,
                                        "contact_sign_convention": contact_sign_convention,
                                        "matsubara_n": int(matsubara_n),
                                        "omega_eV": float(omega_eV),
                                        "nk": int(nk),
                                        "q_model": float(q_model),
                                        "q_angle": float(q_angle),
                                        "qx_model": qx,
                                        "qy_model": qy,
                                        "left_ward_error": left_error,
                                        "right_ward_error": right_error,
                                        "max_ward_error": max_error,
                                        "density_current_included": True,
                                        "current_current_included": True,
                                        "diamagnetic_contact_included": contact_scheme != "none",
                                        "not_final_finite_q_contact": True,
                                        "normal_state_only": True,
                                        "bdg_computed": False,
                                        "conductivity_computed": False,
                                        "casimir_computed": False,
                                        "not_final_casimir_conclusion": True,
                                        "not_final_finite_q_conductivity": True,
                                        "diagnosis": _diagnosis(
                                            max_error,
                                            vertex_scheme,
                                            contact_scheme,
                                            contact_sign_convention,
                                        ),
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
                                )

    return {column: np.array([row[column] for row in rows]) for column in EXPANDED_COLUMNS}


def _write_csv(path: Path, data: dict[str, np.ndarray], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for index in range(len(data["matsubara_n"])):
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


def _scheme_q_max(data: dict[str, np.ndarray], field: str, scheme: str, q_model: float) -> float:
    mask = (data["vertex_scheme"] == scheme) & np.isclose(data["q_model"].astype(float), q_model)
    return float(np.max(data[field][mask].astype(float)))


def _combo_mask(data: dict[str, np.ndarray], combo: tuple[str, str, str]) -> np.ndarray:
    vertex_scheme, contact_scheme, contact_sign_convention = combo
    return (
        (data["vertex_scheme"] == vertex_scheme)
        & (data["contact_scheme"] == contact_scheme)
        & (data["contact_sign_convention"] == contact_sign_convention)
    )


def _combo_label(combo: tuple[str, str, str]) -> str:
    vertex_scheme, contact_scheme, contact_sign_convention = combo
    if contact_scheme == "none":
        return f"{vertex_scheme} + none"
    return f"{vertex_scheme} + {contact_scheme} + {contact_sign_convention}"


def _combo_q_max(data: dict[str, np.ndarray], field: str, combo: tuple[str, str, str], q_model: float) -> float:
    mask = _combo_mask(data, combo) & np.isclose(data["q_model"].astype(float), q_model)
    return float(np.max(data[field][mask].astype(float)))


def _combo_field_max(data: dict[str, np.ndarray], field: str, combo: tuple[str, str, str]) -> float:
    mask = _combo_mask(data, combo)
    return float(np.max(data[field][mask].astype(float)))


def _combo_field_max_for_q_limit(
    data: dict[str, np.ndarray],
    field: str,
    combo: tuple[str, str, str],
    q_max: float,
) -> float:
    mask = _combo_mask(data, combo) & (data["q_model"].astype(float) <= q_max)
    return float(np.max(data[field][mask].astype(float)))


def _best_combo_for_contact(
    data: dict[str, np.ndarray],
    contact_scheme: str,
    *,
    q_max: float | None = None,
) -> tuple[str, str, str] | None:
    candidates = [
        ("peierls", contact_scheme, "plus"),
        ("peierls", contact_scheme, "minus"),
    ]
    available = [combo for combo in candidates if np.any(_combo_mask(data, combo))]
    if not available:
        return None
    if q_max is None:
        return min(available, key=lambda combo: _combo_field_max(data, "max_ward_error", combo))
    return min(available, key=lambda combo: _combo_field_max_for_q_limit(data, "max_ward_error", combo, q_max))


def _plot_outputs(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    q_values = np.array(sorted(set(float(q) for q in data["q_model"])))

    schemes = sorted(set(str(item) for item in data["vertex_scheme"]))
    combos = sorted(
        set(
            zip(
                (str(item) for item in data["vertex_scheme"]),
                (str(item) for item in data["contact_scheme"]),
                (str(item) for item in data["contact_sign_convention"]),
                strict=True,
            )
        )
    )

    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scheme in schemes:
        max_error = [_scheme_q_max(data, "max_ward_error", scheme, q_model) for q_model in q_values]
        ax.semilogy(q_values, np.maximum(max_error, EPS), marker="o", label=scheme)
    ax.set(xlabel="q_model", ylabel="max Ward error", title="Normal-state Ward residual by vertex scheme")
    style_publication_axis(ax)
    path = figure_dir / "ward_error_vs_q.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scheme in schemes:
        left_error = [_scheme_q_max(data, "left_ward_error", scheme, q_model) for q_model in q_values]
        right_error = [_scheme_q_max(data, "right_ward_error", scheme, q_model) for q_model in q_values]
        ax.semilogy(q_values, np.maximum(left_error, EPS), marker="o", label=f"{scheme} left")
        ax.semilogy(q_values, np.maximum(right_error, EPS), marker="s", linestyle="--", label=f"{scheme} right")
    ax.set(xlabel="q_model", ylabel="Ward error", title="Left/right Ward prototype residuals")
    style_publication_axis(ax)
    path = figure_dir / "left_right_ward_error_vs_q.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scheme in schemes:
        max_error = [_scheme_q_max(data, "max_ward_error", scheme, q_model) for q_model in q_values]
        ax.semilogy(q_values, np.maximum(max_error, EPS), marker="o", label=scheme)
    ax.set(xlabel="q_model", ylabel="max Ward error", title="Midpoint vs Peierls Ward residual")
    style_publication_axis(ax)
    path = figure_dir / "ward_error_vs_q_by_vertex_scheme.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    best_q0 = _best_combo_for_contact(data, "q0_mass_diagnostic")
    best_finite = _best_combo_for_contact(data, "finite_q_peierls")
    priority = [
        ("midpoint", "none", "not_applicable"),
        ("peierls", "none", "not_applicable"),
    ]
    for combo in (best_q0, best_finite):
        if combo is not None:
            priority.append(combo)
    plotted = [combo for combo in priority if combo in combos]
    for combo in plotted:
        max_error = [_combo_q_max(data, "max_ward_error", combo, q_model) for q_model in q_values]
        ax.semilogy(q_values, np.maximum(max_error, EPS), marker="o", label=_combo_label(combo))
    ax.set(xlabel="q_model", ylabel="max Ward error", title="Ward residual by contact scheme")
    style_publication_axis(ax)
    path = figure_dir / "ward_error_vs_q_by_contact_scheme.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    return paths


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _summary_lines(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    compact_csv: Path,
    expanded_csv: Path,
    expanded_npz: Path,
    figure_paths: list[Path],
) -> list[str]:
    combos = sorted(
        set(
            zip(
                (str(item) for item in data["vertex_scheme"]),
                (str(item) for item in data["contact_scheme"]),
                (str(item) for item in data["contact_sign_convention"]),
                strict=True,
            )
        )
    )
    priority = [
        ("midpoint", "none", "not_applicable"),
        ("peierls", "none", "not_applicable"),
        ("peierls", "q0_mass_diagnostic", "plus"),
        ("peierls", "q0_mass_diagnostic", "minus"),
        ("peierls", "finite_q_peierls", "plus"),
        ("peierls", "finite_q_peierls", "minus"),
    ]
    ordered_combos = [combo for combo in priority if combo in combos]
    ordered_combos.extend(combo for combo in combos if combo not in ordered_combos)
    combo_summary: dict[tuple[str, str, str], dict[str, float]] = {}
    for combo in ordered_combos:
        combo_summary[combo] = {
            "left": _combo_field_max(data, "left_ward_error", combo),
            "right": _combo_field_max(data, "right_ward_error", combo),
            "max": _combo_field_max(data, "max_ward_error", combo),
            "small_q_max": _combo_field_max_for_q_limit(data, "max_ward_error", combo, 0.01),
        }

    improvement_text = "Peierls current vertex comparison unavailable."
    midpoint_none = ("midpoint", "none", "not_applicable")
    peierls_none = ("peierls", "none", "not_applicable")
    peierls_q0_plus = ("peierls", "q0_mass_diagnostic", "plus")
    peierls_q0_minus = ("peierls", "q0_mass_diagnostic", "minus")
    peierls_finite_plus = ("peierls", "finite_q_peierls", "plus")
    peierls_finite_minus = ("peierls", "finite_q_peierls", "minus")
    if midpoint_none in combo_summary and peierls_none in combo_summary:
        midpoint = combo_summary[midpoint_none]["max"]
        peierls = combo_summary[peierls_none]["max"]
        ratio = midpoint / max(peierls, EPS)
        if ratio > 1.01:
            improvement_text = f"Peierls current vertex lowers the max Ward residual by a factor of {_fmt(ratio)}."
        else:
            improvement_text = (
                f"Peierls current vertex does not materially lower the max Ward residual in this prototype "
                f"(midpoint/Peierls factor = {_fmt(ratio)}); possible reasons include the missing contact term "
                "or remaining vertex/contact closure gaps."
            )
    contact_text = "q0_mass_diagnostic comparison unavailable."
    contact_small_q_text = "q0_mass_diagnostic small-q comparison unavailable."
    finite_contact_text = "finite_q_peierls contact comparison unavailable."
    finite_vs_q0_text = "finite_q_peierls versus q0_mass_diagnostic comparison unavailable."
    finite_small_q_text = "finite_q_peierls small-q comparison unavailable."
    finite_mid_q_text = "finite_q_peierls mid/large-q comparison unavailable."
    if peierls_none in combo_summary and (peierls_q0_plus in combo_summary or peierls_q0_minus in combo_summary):
        candidates = [combo for combo in (peierls_q0_plus, peierls_q0_minus) if combo in combo_summary]
        best_full = min(candidates, key=lambda combo: combo_summary[combo]["max"])
        best_small = min(candidates, key=lambda combo: combo_summary[combo]["small_q_max"])
        full_ratio = combo_summary[peierls_none]["max"] / max(combo_summary[best_full]["max"], EPS)
        small_ratio = combo_summary[peierls_none]["small_q_max"] / max(combo_summary[best_small]["small_q_max"], EPS)
        if full_ratio > 1.01:
            contact_text = (
                f"q0_mass_diagnostic lowers the full-grid Peierls Ward residual most for "
                f"{best_full[2]} sign by a factor of {_fmt(full_ratio)}."
            )
        else:
            contact_text = (
                "q0_mass_diagnostic does not materially lower the full-grid Peierls Ward residual; "
                "this may reflect that q=0 mass contact is only a small-q approximation, or that a "
                "finite-q Peierls contact/bubble-level closure is still missing."
            )
        if small_ratio > 1.01:
            contact_small_q_text = (
                f"For q_model <= 0.01, q0_mass_diagnostic improves the Peierls residual most for "
                f"{best_small[2]} sign by a factor of {_fmt(small_ratio)}."
            )
        else:
            contact_small_q_text = (
                "For q_model <= 0.01, q0_mass_diagnostic does not materially improve the Peierls residual; "
                "a complete finite-q Peierls contact may still be required."
            )
    if peierls_none in combo_summary and (
        peierls_finite_plus in combo_summary or peierls_finite_minus in combo_summary
    ):
        finite_candidates = [
            combo for combo in (peierls_finite_plus, peierls_finite_minus) if combo in combo_summary
        ]
        best_finite_full = min(finite_candidates, key=lambda combo: combo_summary[combo]["max"])
        best_finite_small = min(finite_candidates, key=lambda combo: combo_summary[combo]["small_q_max"])
        finite_full_ratio = combo_summary[peierls_none]["max"] / max(combo_summary[best_finite_full]["max"], EPS)
        finite_small_ratio = combo_summary[peierls_none]["small_q_max"] / max(
            combo_summary[best_finite_small]["small_q_max"],
            EPS,
        )
        finite_contact_text = (
            f"finite_q_peierls contact is best with {best_finite_full[2]} sign on the full grid; "
            f"Peierls-none / finite_q_peierls factor = {_fmt(finite_full_ratio)}."
        )
        if finite_small_ratio > 1.01:
            finite_small_q_text = (
                f"For q_model <= 0.01, finite_q_peierls improves the Peierls residual most for "
                f"{best_finite_small[2]} sign by a factor of {_fmt(finite_small_ratio)}."
            )
        else:
            finite_small_q_text = (
                "For q_model <= 0.01, finite_q_peierls does not materially improve the Peierls residual."
            )
        mid_q_mask = data["q_model"].astype(float) >= 0.05
        if np.any(mid_q_mask):
            finite_mid_values = {}
            none_mid = float(np.max(data["max_ward_error"][_combo_mask(data, peierls_none) & mid_q_mask].astype(float)))
            for combo in finite_candidates:
                finite_mid_values[combo] = float(
                    np.max(data["max_ward_error"][_combo_mask(data, combo) & mid_q_mask].astype(float))
                )
            best_finite_mid = min(finite_mid_values, key=finite_mid_values.__getitem__)
            finite_mid_ratio = none_mid / max(finite_mid_values[best_finite_mid], EPS)
            if finite_mid_ratio > 1.01:
                finite_mid_q_text = (
                    f"For q_model >= 0.05, finite_q_peierls improves most for {best_finite_mid[2]} sign "
                    f"by a factor of {_fmt(finite_mid_ratio)}."
                )
            else:
                finite_mid_q_text = (
                    "For q_model >= 0.05, finite_q_peierls does not materially improve the Peierls residual."
                )
        q0_candidates = [combo for combo in (peierls_q0_plus, peierls_q0_minus) if combo in combo_summary]
        if q0_candidates:
            best_q0_full = min(q0_candidates, key=lambda combo: combo_summary[combo]["max"])
            q0_best = combo_summary[best_q0_full]["max"]
            finite_best = combo_summary[best_finite_full]["max"]
            if finite_best < q0_best / 1.01:
                finite_vs_q0_text = (
                    f"finite_q_peierls lowers the full-grid residual more than q0_mass_diagnostic "
                    f"({best_finite_full[2]} vs {best_q0_full[2]} sign)."
                )
            elif q0_best < finite_best / 1.01:
                finite_vs_q0_text = (
                    f"q0_mass_diagnostic lowers the full-grid residual more than finite_q_peierls "
                    f"({best_q0_full[2]} vs {best_finite_full[2]} sign)."
                )
            else:
                finite_vs_q0_text = (
                    "finite_q_peierls and q0_mass_diagnostic give similar full-grid Ward residuals "
                    "within a 1% comparison threshold."
                )
    lines = [
        "# Normal-state Pi_mu_nu Ward identity prototype",
        "",
        "This is a normal-state Pi_mu_nu Ward diagnostic.",
        "It compares midpoint velocity, Peierls current vertex, q=0 mass contact, and finite-q Peierls contact schemes.",
        "It is not conductivity and not a reflection/Casimir input.",
        "The finite_q_peierls contact is connected to this Ward diagnostic, but this is still not final conductivity.",
        "Response-level sign, normalization, equal-time term, and density-vertex conventions still need final closure.",
        (
            "Large Ward residuals may reflect finite-q vertex/contact-term closure gaps, "
            "not a material conclusion."
        ),
        "",
        f"run_command = `{command}`",
        f"quick_mode={bool(args.quick)}",
        f"expanded_data_written={bool(args.write_expanded_data)}",
        "density_current_included=True",
        "current_current_included=True",
        "diamagnetic_contact_included=True only for contact_scheme=q0_mass_diagnostic or finite_q_peierls",
        "contact_scheme=none or q0_mass_diagnostic or finite_q_peierls",
        "not_final_finite_q_contact=True",
        "normal_state_only=True",
        "bdg_computed=False",
        "conductivity_computed=False",
        "casimir_computed=False",
        "not_final_casimir_conclusion=True",
        "not_final_finite_q_conductivity=True",
        "",
        "## Parameter grid",
        f"- vertex_schemes = {' '.join(args.vertex_schemes)}",
        f"- contact_schemes = {' '.join(args.contact_schemes)}",
        f"- contact_sign_conventions = {' '.join(args.contact_sign_conventions)}",
        f"- matsubara_n_list = {' '.join(str(int(n)) for n in args.matsubara_n_list)}",
        f"- temperature_K = {_fmt(float(args.temperature))}",
        f"- q_list = {' '.join(_fmt(float(q)) for q in args.q_list)}",
        f"- q_angle_list = {' '.join(_fmt(float(a)) for a in args.q_angle_list)}",
        f"- nk_list = {' '.join(str(int(nk)) for nk in args.nk_list)}",
        f"- degeneracy_tol_eV = {_fmt(float(args.degeneracy_tol))}",
        "",
        "## Ward residual summary by vertex/contact scheme",
    ]
    for combo in ordered_combos:
        lines.extend(
            [
                f"- {_combo_label(combo)}: max left Ward error = {_fmt(combo_summary[combo]['left'])}",
                f"- {_combo_label(combo)}: max right Ward error = {_fmt(combo_summary[combo]['right'])}",
                f"- {_combo_label(combo)}: max Ward error = {_fmt(combo_summary[combo]['max'])}",
                f"- {_combo_label(combo)}: max Ward error for q_model <= 0.01 = {_fmt(combo_summary[combo]['small_q_max'])}",
            ]
        )
    lines.extend(
        [
        "",
        "## q_model max-error trend",
    ]
    )
    for combo in ordered_combos:
        trend = []
        for q_model in sorted(set(float(q) for q in data["q_model"])):
            trend.append(f"q={_fmt(q_model)}:{_fmt(_combo_q_max(data, 'max_ward_error', combo, q_model))}")
        lines.append(f"- {_combo_label(combo)}: " + ", ".join(trend))
    lines.extend(
        [
        "",
        "## Comparison",
        f"- {improvement_text}",
        f"- {contact_text}",
        f"- {contact_small_q_text}",
        f"- {finite_contact_text}",
        f"- {finite_vs_q0_text}",
        f"- {finite_small_q_text}",
        f"- {finite_mid_q_text}",
        (
            "- If finite_q_peierls does not close the Ward residual, likely causes include contact sign/normalization "
            "still being inconsistent at response level, missing equal-time or density-vertex pieces, or the need for "
            "a stricter response-level Ward derivation."
        ),
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
    )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matsubara-n-list", nargs="+", type=int, default=list(DEFAULT_MATSUBARA_N_LIST))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--q-list", nargs="+", type=float, default=list(DEFAULT_Q_LIST))
    parser.add_argument("--q-angle-list", nargs="+", type=_angle_expression, default=list(DEFAULT_Q_ANGLE_LIST))
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--degeneracy-tol", type=float, default=DEFAULT_DEGENERACY_TOL_EV)
    parser.add_argument("--vertex-schemes", nargs="+", choices=("midpoint", "peierls"), default=list(DEFAULT_VERTEX_SCHEMES))
    parser.add_argument(
        "--contact-schemes",
        nargs="+",
        choices=("none", "q0_mass_diagnostic", "finite_q_peierls"),
        default=list(DEFAULT_CONTACT_SCHEMES),
    )
    parser.add_argument(
        "--contact-sign-conventions",
        nargs="+",
        choices=("plus", "minus"),
        default=list(DEFAULT_CONTACT_SIGN_CONVENTIONS),
    )
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
    data = run_diagnostic(
        matsubara_n_list=list(args.matsubara_n_list),
        temperature_K=float(args.temperature),
        q_list=list(args.q_list),
        q_angle_list=list(args.q_angle_list),
        nk_list=list(args.nk_list),
        degeneracy_tol_eV=float(args.degeneracy_tol),
        vertex_schemes=list(args.vertex_schemes),
        contact_schemes=list(args.contact_schemes),
        contact_sign_conventions=list(args.contact_sign_conventions),
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
        "\n".join(_summary_lines(data, args, command, compact_csv, expanded_csv, expanded_npz, figure_paths)) + "\n",
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
