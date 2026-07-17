"""Helpers for fixed-rule complete-periodic d-wave reference scans."""
from __future__ import annotations

import json
import os
import resource
import time
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.workflows.dwave_periodic_shift_ensemble import merge_shift_components_before_schur
from validation.lib.dwave_global_extrapolation import (
    relative_difference,
    static_power_law_fits,
    summarize_fit_ensemble,
)
from validation.lib.dwave_shift_batch import ShiftBatchConfig, evaluate_one_shift, postprocess_merged
from validation.lib.dwave_shift_spatial import shift_rule


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def config_from_task(task: Mapping[str, Any]) -> ShiftBatchConfig:
    return ShiftBatchConfig(
        base_nk=int(task["nk"]), qx=float(task["qx"]), qy=float(task["qy"]),
        temperature_K=float(task["temperature_K"]), delta0_eV=float(task["delta0_eV"]),
        eta_eV=float(task["eta_eV"]), ward_tolerance=float(task["ward_tolerance"]),
        ward_absolute_tolerance=float(task["ward_absolute_tolerance"]),
        condition_max=float(task["condition_max"]),
        raw_longitudinal_ceiling=float(task["raw_longitudinal_ceiling"]),
        longitudinal_tolerance=float(task["longitudinal_tolerance"]),
        mixing_tolerance=float(task["mixing_tolerance"]),
        reality_tolerance=float(task["reality_tolerance"]),
        passivity_tolerance=float(task["passivity_tolerance"]),
        separation_nm=float(task["separation_nm"]),
    )


def run_ensemble_task(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one rule at one nk, merging primitives and RHS before one Schur."""
    started, cpu_started = time.perf_counter(), time.process_time()
    config = config_from_task(task)
    rule = str(task["rule"])
    shifts, weights = shift_rule(rule)
    first = evaluate_one_shift(config, 0, shifts[0])
    template = first.pop("workspace")
    results = [first]
    for index in range(1, len(shifts)):
        results.append(evaluate_one_shift(config, index, shifts[index]))
    components, rhs = merge_shift_components_before_schur(
        [item["components"] for item in results],
        [item["rhs"] for item in results], weights, template, omega_eV=0.0,
    )
    processed = postprocess_merged(components, rhs, config)
    return {
        "rule": rule, "nk": int(config.base_nk), "num_shifts": int(len(shifts)),
        "num_points_per_shift": int(config.base_nk) ** 2,
        "num_quadrature_points": int(config.base_nk) ** 2 * int(len(shifts)),
        "shift_json": json.dumps(shifts.tolist()),
        "weight_json": json.dumps(np.asarray(weights, dtype=float).tolist()),
        "qx": float(config.qx), "qy": float(config.qy),
        "temperature_K": float(config.temperature_K),
        "delta0_eV": float(config.delta0_eV), "eta_eV": float(config.eta_eV),
        **processed,
        "total_wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_mb": _peak_rss_mb(), "pid": os.getpid(),
    }


def annotate_drift(rows: list[dict[str, Any]]) -> None:
    previous = None
    for row in rows:
        for field, output in (
            ("chi_bar", "relative_chi_to_previous"),
            ("dbar_t", "relative_dbar_to_previous"),
            ("raw_longitudinal", "relative_raw_longitudinal_to_previous"),
        ):
            row[output] = (
                float("nan") if previous is None
                else relative_difference(row[field], previous[field])
            )
        previous = row


def fit_primary(
    rows: list[dict[str, Any]], powers: Sequence[int], tails: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nks = [int(row["nk"]) for row in rows]
    fit_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for field in ("chi_bar", "dbar_t"):
        fits = static_power_law_fits(
            nks, [float(row[field]) for row in rows],
            powers=tuple(int(v) for v in powers), tail_sizes=tuple(int(v) for v in tails),
        )
        for fit in fits:
            fit["field"] = field
            fit_rows.append(fit)
        value = summarize_fit_ensemble(fits)
        summaries[field] = {
            "estimate": value.estimate, "minimum": value.minimum, "maximum": value.maximum,
            "relative_spread": value.relative_spread, "best_model": value.best_model,
            "best_tail_points": value.best_tail_points,
            "best_normalized_rms": value.best_normalized_rms,
            "num_accepted_models": value.num_accepted_models,
        }
    return fit_rows, summaries


def cross_rule_metrics(primary: Mapping[str, Any], secondary: Mapping[str, Any]) -> dict[str, float]:
    return {
        "relative_chi_cross_rule": relative_difference(primary["chi_bar"], secondary["chi_bar"]),
        "relative_dbar_cross_rule": relative_difference(primary["dbar_t"], secondary["dbar_t"]),
        "relative_raw_longitudinal_cross_rule": relative_difference(
            primary["raw_longitudinal"], secondary["raw_longitudinal"]
        ),
    }


def reference_status(
    primary_rows: list[dict[str, Any]], secondary: Mapping[str, Any],
    fits: Mapping[str, Mapping[str, Any]], *, screening_drift: float,
    screening_cross: float, drift: float, fit_spread: float, cross_rule: float,
) -> dict[str, Any]:
    final = primary_rows[-1]
    cross = cross_rule_metrics(final, secondary)
    final_drift = max(float(final["relative_chi_to_previous"]), float(final["relative_dbar_to_previous"]))
    spread = max(float(fits["chi_bar"]["relative_spread"]), float(fits["dbar_t"]["relative_spread"]))
    cross_max = max(cross["relative_chi_cross_rule"], cross["relative_dbar_cross_rule"])
    tail = primary_rows[-min(3, len(primary_rows)):]
    ward_ok = all(
        bool(row["ward_passed"]) and str(row["schur_inverse_method"]) == "inv"
        and float(row["ward_primitive_mixed_ratio_max"]) < 1.0
        and float(row["ward_effective_mixed_ratio_max"]) < 1.0
        for row in [*tail, secondary]
    )
    promising = bool(ward_ok and final_drift <= screening_drift and cross_max <= screening_cross)
    converged = bool(
        ward_ok and final_drift <= drift and spread <= fit_spread and cross_max <= cross_rule
    )
    eligible = bool(
        converged and bool(final["projection_eligible"]) and bool(secondary["projection_eligible"])
    )
    return {
        **cross, "final_step_relative_drift_max": final_drift,
        "fit_relative_spread_max": spread, "cross_rule_relative_difference_max": cross_max,
        "tail_ward_and_inverse_ok": ward_ok,
        "ensemble_screening_promising": promising,
        "numerical_reference_converged": converged,
        "primary_final_projection_eligible": bool(final["projection_eligible"]),
        "secondary_projection_eligible": bool(secondary["projection_eligible"]),
        "valid_for_casimir_input": eligible,
    }


__all__ = [
    "annotate_drift", "cross_rule_metrics", "fit_primary", "reference_status",
    "run_ensemble_task",
]
