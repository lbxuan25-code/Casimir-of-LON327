#!/usr/bin/env python3
"""Stage 4.19 multi-parameter corrected Ward robustness scan.

Diagnostic-only.  This script scans the corrected full-response Ward residual
over temperature, Matsubara index, q direction/scale, adaptive level, Gauss
order, and Fermi-window width.  It does not modify the main response formula,
bubble prefactor sign, direct contact, source/observable split, conductivity,
reflection, or Casimir code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV  # noqa: E402
from lno327.ward_response import physical_ward_residuals  # noqa: E402

from stage4_15_fermi_window_adaptive_quadrature import build_adaptive_cells, quadrature_points_for_cells  # noqa: E402
from stage4_16_full_response_adaptive_ward_diagnostic import (  # noqa: E402
    integrate_physical_components_on_points,
    project_spatial_components,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "ward_identity"
JSON_OUTPUT = OUTPUT_DIR / "stage4_19_multi_parameter_ward_robustness_scan.json"
MD_OUTPUT = OUTPUT_DIR / "stage4_19_multi_parameter_ward_robustness_scan.md"

TEMPERATURES_K = (30.0, 100.0, 300.0)
MATSUBARA_INDICES = (1, 2, 4)
Q_CASES = {
    "qx": np.array([0.02, 0.0], dtype=float),
    "qy": np.array([0.0, 0.02], dtype=float),
    "q_diag_pos": np.array([0.02, 0.013], dtype=float),
    "q_diag_neg": np.array([0.02, -0.013], dtype=float),
}
Q_SCALES = (1.0, 0.5, 0.25, 0.125)
ADAPTIVE_LEVELS = (3, 4)
GAUSS_ORDERS = (3, 5)
FERMI_WINDOWS_EV = (0.03, 0.05, 0.08)
COARSE_GRID = 32
ETA_EV = 1e-10
OUTPUT_SI = False
CLOSURE_THRESHOLD = 1e-6
MONITOR_THRESHOLD = 1e-5


def to_jsonable(value: Any) -> Any:
    if isinstance(value, complex | np.complexfloating):
        raise TypeError("complex values must be split before JSON serialization")
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _config(temperature_K: float, matsubara_index: int) -> KuboConfig:
    omega_eV = bosonic_matsubara_energy_eV(int(matsubara_index), float(temperature_K))
    return KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=float(temperature_K),
        eta_eV=ETA_EV,
        output_si=OUTPUT_SI,
    )


def _case_status(max_corrected_norm: float) -> str:
    if max_corrected_norm < CLOSURE_THRESHOLD:
        return "CLOSED"
    if max_corrected_norm < MONITOR_THRESHOLD:
        return "ACCEPTABLE_BUT_MONITOR"
    return "NOT_CLOSED"


def _residual_row(
    *,
    temperature_K: float,
    matsubara_index: int,
    q_case: str,
    q_scale: float,
    q: np.ndarray,
    adaptive_level: int,
    gauss_order: int,
    fermi_window_eV: float,
    coarse_grid: int,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    config = _config(temperature_K, matsubara_index)
    cells, refined_count, _flagged_base = build_adaptive_cells(
        q,
        coarse_grid=int(coarse_grid),
        refinement_level=int(adaptive_level),
        fermi_window_eV=float(fermi_window_eV),
        fermi_level_eV=config.fermi_level_eV,
    )
    points, weights = quadrature_points_for_cells(cells, int(gauss_order))
    components = integrate_physical_components_on_points(points, weights, q, config)
    left, right = physical_ward_residuals(components["total"], config.omega_eV, q)
    left_long, left_trans = project_spatial_components(q, left[1:])
    right_long, right_trans = project_spatial_components(q, right[1:])
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    max_norm = max(left_norm, right_norm)
    return {
        "temperature_K": float(temperature_K),
        "matsubara_index": int(matsubara_index),
        "omega_eV": float(config.omega_eV),
        "q_case": str(q_case),
        "q_scale": float(q_scale),
        "q_model": [float(q[0]), float(q[1])],
        "adaptive_level": int(adaptive_level),
        "gauss_order": int(gauss_order),
        "fermi_window_eV": float(fermi_window_eV),
        "left_norm": left_norm,
        "right_norm": right_norm,
        "max_corrected_norm": float(max_norm),
        "left_density_source_abs": float(abs(left[0])),
        "left_spatial_source_norm": float(np.linalg.norm(left[1:])),
        "right_density_observable_abs": float(abs(right[0])),
        "right_spatial_observable_norm": float(np.linalg.norm(right[1:])),
        "left_longitudinal_abs": float(abs(left_long)),
        "left_transverse_abs": float(abs(left_trans)),
        "right_longitudinal_abs": float(abs(right_long)),
        "right_transverse_abs": float(abs(right_trans)),
        "num_cells_total": int(len(cells)),
        "num_cells_refined": int(refined_count),
        "num_quadrature_points": int(len(points)),
        "runtime_seconds": float(time.perf_counter() - start_time),
        "status": _case_status(max_norm),
    }


def _strip_vectors(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "temperature_K",
        "matsubara_index",
        "omega_eV",
        "q_case",
        "q_scale",
        "q_model",
        "adaptive_level",
        "gauss_order",
        "fermi_window_eV",
        "left_norm",
        "right_norm",
        "max_corrected_norm",
        "left_density_source_abs",
        "left_spatial_source_norm",
        "right_density_observable_abs",
        "right_spatial_observable_norm",
        "left_longitudinal_abs",
        "left_transverse_abs",
        "right_longitudinal_abs",
        "right_transverse_abs",
        "num_cells_total",
        "num_cells_refined",
        "num_quadrature_points",
        "runtime_seconds",
        "status",
    )
    return {key: row[key] for key in keys}


def _top_rows(rows: list[dict[str, Any]], key: str, n: int = 10) -> list[dict[str, Any]]:
    return [_strip_vectors(row) for row in sorted(rows, key=lambda item: float(item[key]), reverse=True)[:n]]


def _summary_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.array([float(row["max_corrected_norm"]) for row in rows], dtype=float)
    worst = max(rows, key=lambda item: float(item["max_corrected_norm"])) if rows else {}
    return {
        "num_total_cases": int(len(rows)),
        "num_closed": int(sum(1 for row in rows if row["status"] == "CLOSED")),
        "num_acceptable_but_monitor": int(sum(1 for row in rows if row["status"] == "ACCEPTABLE_BUT_MONITOR")),
        "num_not_closed": int(sum(1 for row in rows if row["status"] == "NOT_CLOSED")),
        "max_corrected_norm_global": float(np.max(values)) if len(values) else float("nan"),
        "median_corrected_norm": float(np.median(values)) if len(values) else float("nan"),
        "p95_corrected_norm": float(np.percentile(values, 95)) if len(values) else float("nan"),
        "worst_case_parameters": _strip_vectors(worst) if worst else {},
    }


def _dominant_failure_channel(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    target_rows = [row for row in rows if row["status"] != "CLOSED"] or rows
    metrics = {
        "left_density_source": max(float(row["left_density_source_abs"]) for row in target_rows),
        "left_spatial_source": max(float(row["left_spatial_source_norm"]) for row in target_rows),
        "right_density_observable": max(float(row["right_density_observable_abs"]) for row in target_rows),
        "right_spatial_observable": max(float(row["right_spatial_observable_norm"]) for row in target_rows),
        "left_longitudinal": max(float(row["left_longitudinal_abs"]) for row in target_rows),
        "left_transverse": max(float(row["left_transverse_abs"]) for row in target_rows),
        "right_longitudinal": max(float(row["right_longitudinal_abs"]) for row in target_rows),
        "right_transverse": max(float(row["right_transverse_abs"]) for row in target_rows),
    }
    return max(metrics, key=metrics.get)


def _likely_issue(rows: list[dict[str, Any]], robustness_status: str) -> str:
    if robustness_status == "ROBUSTLY_CLOSED":
        return "CORRECTED_WARD_CONVENTION_AND_ADAPTIVE_QUADRATURE_ROBUST"
    if robustness_status == "MOSTLY_CLOSED_WITH_MINOR_OUTLIERS":
        return "MINOR_NUMERICAL_OUTLIERS_REQUIRE_TARGETED_REFINEMENT"
    failures = [row for row in rows if row["status"] == "NOT_CLOSED"]
    if not failures:
        return "MINOR_NUMERICAL_OUTLIERS_REQUIRE_TARGETED_REFINEMENT"
    min_level = min(int(item["adaptive_level"]) for item in rows)
    max_level = max(int(item["adaptive_level"]) for item in rows)
    min_window = min(float(item["fermi_window_eV"]) for item in rows)
    if max_level > min_level and all(int(row["adaptive_level"]) == min_level for row in failures):
        return "ADAPTIVE_QUADRATURE_NEEDS_TARGETED_REFINEMENT"
    if any(float(row["fermi_window_eV"]) == min_window for row in failures):
        return "ADAPTIVE_QUADRATURE_NEEDS_TARGETED_REFINEMENT"
    low_temp_or_high_mats = sum(
        1
        for row in failures
        if float(row["temperature_K"]) == min(float(item["temperature_K"]) for item in rows)
        or int(row["matsubara_index"]) == max(int(item["matsubara_index"]) for item in rows)
        or float(row["q_scale"]) == min(float(item["q_scale"]) for item in rows)
    )
    direction_counts = {case: sum(1 for row in failures if row["q_case"] == case) for case in {row["q_case"] for row in rows}}
    if direction_counts and max(direction_counts.values()) >= max(2, int(0.6 * len(failures))):
        return "Q_DIRECTION_ROUTING_OR_QUADRATURE_ANISOTROPY"
    if low_temp_or_high_mats >= max(2, int(0.5 * len(failures))):
        return "ADAPTIVE_QUADRATURE_NEEDS_TARGETED_REFINEMENT"
    left_fail = max(float(row["left_norm"]) for row in failures)
    right_fail = max(float(row["right_norm"]) for row in failures)
    if left_fail >= MONITOR_THRESHOLD and right_fail >= MONITOR_THRESHOLD:
        return "POSSIBLE_REMAINING_RESPONSE_ROUTING_ISSUE"
    return "ADAPTIVE_QUADRATURE_NEEDS_TARGETED_REFINEMENT"


def _diagnostic_status(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    total = int(summary["num_total_cases"])
    closed = int(summary["num_closed"])
    acceptable = int(summary["num_acceptable_but_monitor"])
    not_closed = int(summary["num_not_closed"])
    if total > 0 and closed == total:
        robustness = "ROBUSTLY_CLOSED"
    elif total > 0 and not_closed == 0 and closed / total >= 0.95 and acceptable > 0:
        robustness = "MOSTLY_CLOSED_WITH_MINOR_OUTLIERS"
    elif not_closed > 0:
        robustness = "ROBUSTNESS_FAILURE"
    else:
        robustness = "MOSTLY_CLOSED_WITH_MINOR_OUTLIERS"
    likely = _likely_issue(rows, robustness)
    if robustness == "ROBUSTLY_CLOSED":
        next_step = "Next: enter response-to-conductivity validation as an independent stage; do not treat this as Casimir-ready."
    elif robustness == "MOSTLY_CLOSED_WITH_MINOR_OUTLIERS":
        next_step = "Next: rerun targeted outliers with higher refinement before response-to-conductivity validation."
    else:
        next_step = "Next: diagnose the worst-case parameter cluster before any downstream conductivity/reflection/Casimir use."
    return {
        "robustness_status": robustness,
        "closure_threshold": CLOSURE_THRESHOLD,
        "monitor_threshold": MONITOR_THRESHOLD,
        "dominant_failure_channel": _dominant_failure_channel(rows),
        "likely_issue": likely,
        "next_step": next_step,
    }


def run_scan(
    *,
    temperatures_K: list[float] | tuple[float, ...] = TEMPERATURES_K,
    matsubara_indices: list[int] | tuple[int, ...] = MATSUBARA_INDICES,
    q_cases: dict[str, np.ndarray] | None = None,
    q_scales: list[float] | tuple[float, ...] = Q_SCALES,
    adaptive_levels: list[int] | tuple[int, ...] = ADAPTIVE_LEVELS,
    gauss_orders: list[int] | tuple[int, ...] = GAUSS_ORDERS,
    fermi_windows_eV: list[float] | tuple[float, ...] = FERMI_WINDOWS_EV,
    coarse_grid: int = COARSE_GRID,
) -> dict[str, Any]:
    q_case_map = Q_CASES if q_cases is None else q_cases
    rows: list[dict[str, Any]] = []
    for temperature_K in temperatures_K:
        for matsubara_index in matsubara_indices:
            for q_case, q_base in q_case_map.items():
                q_base_array = np.asarray(q_base, dtype=float)
                for q_scale in q_scales:
                    q = float(q_scale) * q_base_array
                    for adaptive_level in adaptive_levels:
                        for gauss_order in gauss_orders:
                            for fermi_window_eV in fermi_windows_eV:
                                rows.append(
                                    _residual_row(
                                        temperature_K=float(temperature_K),
                                        matsubara_index=int(matsubara_index),
                                        q_case=str(q_case),
                                        q_scale=float(q_scale),
                                        q=q,
                                        adaptive_level=int(adaptive_level),
                                        gauss_order=int(gauss_order),
                                        fermi_window_eV=float(fermi_window_eV),
                                        coarse_grid=int(coarse_grid),
                                    )
                                )
    summary = _summary_statistics(rows)
    worst_cases = {
        "top_10_largest_max_corrected_norm": _top_rows(rows, "max_corrected_norm"),
        "top_10_largest_left_norm": _top_rows(rows, "left_norm"),
        "top_10_largest_right_norm": _top_rows(rows, "right_norm"),
    }
    return {
        "stage": "Stage 4.19",
        "purpose": "Multi-parameter robustness scan for corrected full-response Ward validation",
        "config": {
            "scan_mode": "cartesian",
            "temperatures_K": [float(item) for item in temperatures_K],
            "matsubara_indices": [int(item) for item in matsubara_indices],
            "q_cases": {name: [float(value[0]), float(value[1])] for name, value in q_case_map.items()},
            "q_scales": [float(item) for item in q_scales],
            "adaptive_levels": [int(item) for item in adaptive_levels],
            "gauss_orders": [int(item) for item in gauss_orders],
            "fermi_windows_eV": [float(item) for item in fermi_windows_eV],
            "coarse_grid": int(coarse_grid),
            "eta_eV": ETA_EV,
            "output_si": OUTPUT_SI,
            "right_residual_convention": "iOmega Pi[mu,0] - qx Pi[mu,x] - qy Pi[mu,y]",
        },
        "scan_results": rows,
        "summary_statistics": summary,
        "worst_cases": worst_cases,
        "diagnostic_status": _diagnostic_status(rows, summary),
        "boundary": {
            "no_main_response_change": True,
            "no_bubble_sign_change": True,
            "no_direct_contact_change": True,
            "no_source_observable_change": True,
            "no_residual_tuning": True,
            "no_fitted_contact": True,
            "no_E_ET_added": True,
            "no_conductivity_reflection_casimir": True,
            "not_casimir_ready_claim": True,
        },
    }


def _unique_parameter_sets(parameter_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for item in parameter_sets:
        q = np.asarray(item["q"], dtype=float)
        key = (
            float(item["temperature_K"]),
            int(item["matsubara_index"]),
            str(item["q_case"]),
            float(item["q_scale"]),
            float(q[0]),
            float(q[1]),
            int(item["adaptive_level"]),
            int(item["gauss_order"]),
            float(item["fermi_window_eV"]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def representative_parameter_sets(
    *,
    temperatures_K: list[float] | tuple[float, ...] = TEMPERATURES_K,
    matsubara_indices: list[int] | tuple[int, ...] = MATSUBARA_INDICES,
    q_cases: dict[str, np.ndarray] | None = None,
    q_scales: list[float] | tuple[float, ...] = Q_SCALES,
    adaptive_levels: list[int] | tuple[int, ...] = ADAPTIVE_LEVELS,
    gauss_orders: list[int] | tuple[int, ...] = GAUSS_ORDERS,
    fermi_windows_eV: list[float] | tuple[float, ...] = FERMI_WINDOWS_EV,
) -> list[dict[str, Any]]:
    """Return a tractable one-factor robustness grid covering every axis value."""

    q_case_map = Q_CASES if q_cases is None else q_cases
    baseline = {
        "temperature_K": float(temperatures_K[0]),
        "matsubara_index": int(matsubara_indices[0]),
        "q_case": "q_diag_pos" if "q_diag_pos" in q_case_map else next(iter(q_case_map)),
        "q_scale": float(q_scales[0]),
        "adaptive_level": int(min(adaptive_levels)),
        "gauss_order": int(gauss_orders[0]),
        "fermi_window_eV": 0.05 if 0.05 in fermi_windows_eV else float(fermi_windows_eV[0]),
    }
    parameter_sets: list[dict[str, Any]] = []

    def add_case(**updates: Any) -> None:
        item = dict(baseline)
        item.update(updates)
        q_base = np.asarray(q_case_map[str(item["q_case"])], dtype=float)
        item["q"] = float(item["q_scale"]) * q_base
        parameter_sets.append(item)

    for q_case in q_case_map:
        for q_scale in q_scales:
            add_case(q_case=q_case, q_scale=float(q_scale))
    for temperature_K in temperatures_K:
        add_case(temperature_K=float(temperature_K))
    for matsubara_index in matsubara_indices:
        add_case(matsubara_index=int(matsubara_index))
    for adaptive_level in adaptive_levels:
        add_case(adaptive_level=int(adaptive_level))
    for gauss_order in gauss_orders:
        add_case(gauss_order=int(gauss_order))
    for fermi_window_eV in fermi_windows_eV:
        add_case(fermi_window_eV=float(fermi_window_eV))
    return _unique_parameter_sets(parameter_sets)


def run_representative_scan(
    *,
    temperatures_K: list[float] | tuple[float, ...] = TEMPERATURES_K,
    matsubara_indices: list[int] | tuple[int, ...] = MATSUBARA_INDICES,
    q_cases: dict[str, np.ndarray] | None = None,
    q_scales: list[float] | tuple[float, ...] = Q_SCALES,
    adaptive_levels: list[int] | tuple[int, ...] = ADAPTIVE_LEVELS,
    gauss_orders: list[int] | tuple[int, ...] = GAUSS_ORDERS,
    fermi_windows_eV: list[float] | tuple[float, ...] = FERMI_WINDOWS_EV,
    coarse_grid: int = COARSE_GRID,
) -> dict[str, Any]:
    q_case_map = Q_CASES if q_cases is None else q_cases
    parameter_sets = representative_parameter_sets(
        temperatures_K=temperatures_K,
        matsubara_indices=matsubara_indices,
        q_cases=q_case_map,
        q_scales=q_scales,
        adaptive_levels=adaptive_levels,
        gauss_orders=gauss_orders,
        fermi_windows_eV=fermi_windows_eV,
    )
    rows = [
        _residual_row(
            temperature_K=float(item["temperature_K"]),
            matsubara_index=int(item["matsubara_index"]),
            q_case=str(item["q_case"]),
            q_scale=float(item["q_scale"]),
            q=np.asarray(item["q"], dtype=float),
            adaptive_level=int(item["adaptive_level"]),
            gauss_order=int(item["gauss_order"]),
            fermi_window_eV=float(item["fermi_window_eV"]),
            coarse_grid=int(coarse_grid),
        )
        for item in parameter_sets
    ]
    summary = _summary_statistics(rows)
    worst_cases = {
        "top_10_largest_max_corrected_norm": _top_rows(rows, "max_corrected_norm"),
        "top_10_largest_left_norm": _top_rows(rows, "left_norm"),
        "top_10_largest_right_norm": _top_rows(rows, "right_norm"),
    }
    return {
        "stage": "Stage 4.19",
        "purpose": "Multi-parameter robustness scan for corrected full-response Ward validation",
        "config": {
            "scan_mode": "representative_default",
            "cartesian_full_available_with": "--cartesian-full",
            "temperatures_K": [float(item) for item in temperatures_K],
            "matsubara_indices": [int(item) for item in matsubara_indices],
            "q_cases": {name: [float(value[0]), float(value[1])] for name, value in q_case_map.items()},
            "q_scales": [float(item) for item in q_scales],
            "adaptive_levels": [int(item) for item in adaptive_levels],
            "gauss_orders": [int(item) for item in gauss_orders],
            "fermi_windows_eV": [float(item) for item in fermi_windows_eV],
            "coarse_grid": int(coarse_grid),
            "eta_eV": ETA_EV,
            "output_si": OUTPUT_SI,
            "right_residual_convention": "iOmega Pi[mu,0] - qx Pi[mu,x] - qy Pi[mu,y]",
        },
        "scan_results": rows,
        "summary_statistics": summary,
        "worst_cases": worst_cases,
        "diagnostic_status": _diagnostic_status(rows, summary),
        "boundary": {
            "no_main_response_change": True,
            "no_bubble_sign_change": True,
            "no_direct_contact_change": True,
            "no_source_observable_change": True,
            "no_residual_tuning": True,
            "no_fitted_contact": True,
            "no_E_ET_added": True,
            "no_conductivity_reflection_casimir": True,
            "not_casimir_ready_claim": True,
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.6e}"


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _group_status_table(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> str:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        grouped.setdefault(key, []).append(row)
    table_rows = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        table_rows.append(
            tuple(str(item) for item in key)
            + (
                len(group_rows),
                sum(1 for row in group_rows if row["status"] == "CLOSED"),
                sum(1 for row in group_rows if row["status"] == "ACCEPTABLE_BUT_MONITOR"),
                sum(1 for row in group_rows if row["status"] == "NOT_CLOSED"),
                _fmt(max(float(row["max_corrected_norm"]) for row in group_rows)),
            )
        )
    return _table(group_keys + ("cases", "closed", "monitor", "not_closed", "max_norm"), table_rows)


def render_markdown(data: dict[str, Any]) -> str:
    rows = data["scan_results"]
    stats = data["summary_statistics"]
    status = data["diagnostic_status"]
    worst_rows = [
        (
            row["temperature_K"],
            row["matsubara_index"],
            row["q_case"],
            _fmt(float(row["q_scale"])),
            row["adaptive_level"],
            row["gauss_order"],
            _fmt(float(row["fermi_window_eV"])),
            _fmt(float(row["max_corrected_norm"])),
            row["status"],
        )
        for row in data["worst_cases"]["top_10_largest_max_corrected_norm"]
    ]
    return "\n\n".join(
        [
            "# Stage 4.19 Multi-parameter Ward robustness scan",
            "## Boundary\n\n"
            "- no main response change\n"
            "- no bubble sign change\n"
            "- no direct contact change\n"
            "- no source/observable change\n"
            "- no residual tuning\n"
            "- no fitted contact\n"
            "- no E_ET added\n"
            "- no conductivity / reflection / Casimir\n"
            "- no Casimir-ready claim",
            "## Corrected Ward residual convention\n\n"
            "$$R_L[\\nu]=i\\Omega\\Pi_{0\\nu}+q_x\\Pi_{x\\nu}+q_y\\Pi_{y\\nu},$$\n\n"
            "$$R_R[\\mu]=i\\Omega\\Pi_{\\mu0}-q_x\\Pi_{\\mu x}-q_y\\Pi_{\\mu y}.$$",
            "## Scan grid\n\n"
            + _table(
                ("quantity", "values"),
                [
                    ("scan_mode", data["config"]["scan_mode"]),
                    ("temperatures_K", data["config"]["temperatures_K"]),
                    ("matsubara_indices", data["config"]["matsubara_indices"]),
                    ("q_cases", list(data["config"]["q_cases"].keys())),
                    ("q_scales", data["config"]["q_scales"]),
                    ("adaptive_levels", data["config"]["adaptive_levels"]),
                    ("gauss_orders", data["config"]["gauss_orders"]),
                    ("fermi_windows_eV", data["config"]["fermi_windows_eV"]),
                    ("coarse_grid", data["config"]["coarse_grid"]),
                ],
            ),
            "## Summary statistics\n\n"
            + _table(
                ("quantity", "value"),
                [
                    ("num_total_cases", stats["num_total_cases"]),
                    ("num_closed", stats["num_closed"]),
                    ("num_acceptable_but_monitor", stats["num_acceptable_but_monitor"]),
                    ("num_not_closed", stats["num_not_closed"]),
                    ("max_corrected_norm_global", _fmt(float(stats["max_corrected_norm_global"]))),
                    ("median_corrected_norm", _fmt(float(stats["median_corrected_norm"]))),
                    ("p95_corrected_norm", _fmt(float(stats["p95_corrected_norm"]))),
                ],
            ),
            "## Closure table by temperature and Matsubara index\n\n"
            + _group_status_table(rows, ("temperature_K", "matsubara_index")),
            "## Closure table by q direction\n\n" + _group_status_table(rows, ("q_case",)),
            "## Adaptive level / Gauss order / Fermi window comparison\n\n"
            + _group_status_table(rows, ("adaptive_level", "gauss_order", "fermi_window_eV")),
            "## Worst cases\n\n"
            + _table(
                (
                    "T",
                    "n",
                    "q_case",
                    "q_scale",
                    "level",
                    "order",
                    "window",
                    "max_norm",
                    "status",
                ),
                worst_rows,
            ),
            "## Diagnostic decision\n\n"
            + _table(
                ("quantity", "status"),
                [
                    ("robustness_status", status["robustness_status"]),
                    ("closure_threshold", _fmt(float(status["closure_threshold"]))),
                    ("monitor_threshold", _fmt(float(status["monitor_threshold"]))),
                    ("dominant_failure_channel", status["dominant_failure_channel"]),
                    ("likely_issue", status["likely_issue"]),
                ],
            )
            + "\n\nStage 4.13 fixed the bubble sign. Stage 4.15 addressed the $C-K$ quadrature issue. Stage 4.17/4.18 fixed the right Ward diagnostic convention. Passing this scan means the normal-state response Ward validation is robust over this diagnostic grid; it is not conductivity, reflection, or Casimir completion.",
            "## Next step\n\n"
            + status["next_step"]
            + "\n\nIf a future robustness scan passes, the next stage may enter response-to-conductivity validation as an independent check.",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run the lightweight diagnostic grid used by tests.")
    parser.add_argument(
        "--cartesian-full",
        action="store_true",
        help="Run the full Cartesian product of all default parameter lists; this is much slower.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        data = run_scan(
            temperatures_K=[30.0],
            matsubara_indices=[1],
            q_cases={"q_diag_pos": np.array([0.02, 0.013], dtype=float)},
            q_scales=[1.0, 0.5],
            adaptive_levels=[1],
            gauss_orders=[2],
            fermi_windows_eV=[0.05],
            coarse_grid=8,
        )
    elif args.cartesian_full:
        data = run_scan()
    else:
        data = run_representative_scan()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MD_OUTPUT.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {MD_OUTPUT}")


if __name__ == "__main__":
    main()
