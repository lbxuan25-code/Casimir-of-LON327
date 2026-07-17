"""Equal-total-point d-wave shift-allocation screening helpers."""
from __future__ import annotations

import json
import os
import resource
import time
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.workflows.dwave_periodic_shift_ensemble import (
    merge_shift_components_before_schur,
    nested_c4_antithetic_shifts,
)
from validation.lib.dwave_global_extrapolation import relative_difference
from validation.lib.dwave_shift_batch import (
    ShiftBatchConfig,
    evaluate_one_shift,
    postprocess_merged,
)


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def parse_allocations(values: Sequence[str]) -> list[tuple[int, int]]:
    """Parse ``SHIFT_COUNT:NK`` specifications and return them by shift count."""
    allocations: list[tuple[int, int]] = []
    for value in values:
        try:
            shifts_text, nk_text = str(value).split(":", maxsplit=1)
            shifts, nk = int(shifts_text), int(nk_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid allocation {value!r}; expected SHIFT_COUNT:NK") from exc
        if shifts <= 0 or shifts % 4 != 0:
            raise ValueError("shift counts must be positive multiples of four")
        if nk <= 0:
            raise ValueError("allocation nk values must be positive")
        allocations.append((shifts, nk))
    if len(allocations) < 2:
        raise ValueError("at least two distinct allocations are required")
    if len({item[0] for item in allocations}) != len(allocations):
        raise ValueError("allocation shift counts must be distinct")
    return sorted(allocations)


def _config(task: Mapping[str, Any]) -> ShiftBatchConfig:
    return ShiftBatchConfig(
        base_nk=int(task["nk"]),
        qx=float(task["qx"]),
        qy=float(task["qy"]),
        temperature_K=float(task["temperature_K"]),
        delta0_eV=float(task["delta0_eV"]),
        eta_eV=float(task["eta_eV"]),
        ward_tolerance=float(task["ward_tolerance"]),
        ward_absolute_tolerance=float(task["ward_absolute_tolerance"]),
        condition_max=float(task["condition_max"]),
        raw_longitudinal_ceiling=float(task["raw_longitudinal_ceiling"]),
        longitudinal_tolerance=float(task["longitudinal_tolerance"]),
        mixing_tolerance=float(task["mixing_tolerance"]),
        reality_tolerance=float(task["reality_tolerance"]),
        passivity_tolerance=float(task["passivity_tolerance"]),
        separation_nm=float(task["separation_nm"]),
    )


def run_budget_task(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one ``(num_shifts, base_nk)`` allocation before one Schur."""
    started, cpu_started = time.perf_counter(), time.process_time()
    config = _config(task)
    count = int(task["num_shifts"])
    shifts = nested_c4_antithetic_shifts(count)

    first = evaluate_one_shift(config, 0, shifts[0])
    template = first.pop("workspace")
    results = [first]
    for index in range(1, count):
        result = evaluate_one_shift(config, index, shifts[index])
        result.pop("workspace", None)
        results.append(result)

    weights = np.full(count, 1.0 / float(count), dtype=float)
    components, rhs = merge_shift_components_before_schur(
        [item["components"] for item in results],
        [item["rhs"] for item in results],
        weights,
        template,
        omega_eV=0.0,
    )
    processed = postprocess_merged(components, rhs, config)
    return {
        "num_shifts": count,
        "base_nk": int(config.base_nk),
        "num_points_per_shift": int(config.base_nk) ** 2,
        "num_quadrature_points": count * int(config.base_nk) ** 2,
        "shift_family": "nested_halton_bases_2_3_c4_antithetic",
        "shift_json": json.dumps(shifts.tolist()),
        "qx": float(config.qx),
        "qy": float(config.qy),
        "temperature_K": float(config.temperature_K),
        "delta0_eV": float(config.delta0_eV),
        "eta_eV": float(config.eta_eV),
        **processed,
        "total_wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_mb": _peak_rss_mb(),
        "pid": os.getpid(),
    }


def allocation_metrics(
    rows: Sequence[Mapping[str, Any]], *, agreement_tolerance: float
) -> dict[str, Any]:
    """Compare equal-budget allocations without promoting one to a reference."""
    ordered = sorted(rows, key=lambda row: int(row["num_shifts"]))
    if len(ordered) < 2:
        raise ValueError("allocation comparison requires at least two rows")

    transitions: list[dict[str, float | int]] = []
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        transitions.append(
            {
                "from_shifts": int(left["num_shifts"]),
                "to_shifts": int(right["num_shifts"]),
                "relative_chi": relative_difference(right["chi_bar"], left["chi_bar"]),
                "relative_dbar": relative_difference(right["dbar_t"], left["dbar_t"]),
                "relative_raw_longitudinal": relative_difference(
                    right["raw_longitudinal"], left["raw_longitudinal"]
                ),
            }
        )

    chi_values = np.asarray([float(row["chi_bar"]) for row in ordered])
    dbar_values = np.asarray([float(row["dbar_t"]) for row in ordered])
    chi_span = float((np.max(chi_values) - np.min(chi_values)) / max(abs(np.mean(chi_values)), 1e-30))
    dbar_span = float((np.max(dbar_values) - np.min(dbar_values)) / max(abs(np.mean(dbar_values)), 1e-30))
    physical_span = max(chi_span, dbar_span)

    transition_scales = [
        max(float(item["relative_chi"]), float(item["relative_dbar"]))
        for item in transitions
    ]
    decreasing_transition = bool(
        len(transition_scales) >= 2 and transition_scales[-1] < transition_scales[0]
    )
    raw_values = [float(row["raw_longitudinal"]) for row in ordered]
    raw_improves = bool(raw_values[-1] < raw_values[0])
    ward_ok = all(
        bool(row["ward_passed"])
        and str(row["schur_inverse_method"]) == "inv"
        and float(row["ward_primitive_mixed_ratio_max"]) < 1.0
        and float(row["ward_effective_mixed_ratio_max"]) < 1.0
        for row in ordered
    )

    if decreasing_transition and raw_improves:
        preference = "more_shifts"
    elif not decreasing_transition and raw_values[0] <= raw_values[-1]:
        preference = "higher_base_nk"
    else:
        preference = "inconclusive"

    return {
        "transitions": transitions,
        "relative_chi_span": chi_span,
        "relative_dbar_span": dbar_span,
        "physical_span_max": physical_span,
        "ward_and_inverse_ok": ward_ok,
        "equal_budget_agreement": bool(ward_ok and physical_span <= agreement_tolerance),
        "more_shifts_reduce_transition": decreasing_transition,
        "more_shifts_reduce_raw_longitudinal": raw_improves,
        "allocation_preference": preference,
        "production_reference_established": False,
    }


__all__ = [
    "allocation_metrics",
    "parse_allocations",
    "run_budget_task",
]
