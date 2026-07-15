"""Unified transverse-point sweet-spot search with universal convergence gates.

All pairings, momentum magnitudes/directions, and Matsubara indices use the same
physical and numerical acceptance policy. Physical closure remains a hard gate.
Numerically, a point may establish a working/audit pair either through consecutive
adjacent-N convergence or through a three-level oscillatory envelope. Every logdet
comparison checks a universal absolute tolerance first and falls back to the same
universal relative tolerance; no q- or frequency-specific exception exists.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from multiprocessing import get_all_start_methods
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from lno327.workflows.cpu_parallel import choose_cpu_parallel_plan, estimate_context_bytes
from validation.lib import transverse_point_sweet_spot_engine as _engine

DEFAULT_LOGDET_ATOL = 1e-6
ENVELOPE_LEVELS = 3


def _absolute_then_relative_pair(
    previous: float,
    current: float,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left = float(previous)
    right = float(current)
    if not np.isfinite(left) or not np.isfinite(right):
        return {
            "finite": False,
            "absolute": float("nan"),
            "relative": float("nan"),
            "absolute_tolerance": float(atol),
            "relative_tolerance": float(rtol),
            "absolute_passed": False,
            "relative_passed": False,
            "passed_by": "failed",
            "passed": False,
        }
    absolute = abs(right - left)
    scale = max(abs(left), abs(right))
    relative = absolute / max(scale, np.finfo(float).tiny)
    absolute_passed = bool(absolute <= float(atol))
    relative_passed = bool(relative <= float(rtol))
    passed_by = (
        "absolute"
        if absolute_passed
        else "relative"
        if relative_passed
        else "failed"
    )
    return {
        "finite": True,
        "absolute": absolute,
        "relative": relative,
        "scale": scale,
        "absolute_tolerance": float(atol),
        "relative_tolerance": float(rtol),
        "absolute_passed": absolute_passed,
        "relative_passed": relative_passed,
        "passed_by": passed_by,
        "passed": bool(absolute_passed or relative_passed),
    }


def _absolute_then_relative_spread(
    values: Sequence[float],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        return {
            "finite": False,
            "minimum": float("nan"),
            "maximum": float("nan"),
            "absolute": float("nan"),
            "relative": float("nan"),
            "absolute_tolerance": float(atol),
            "relative_tolerance": float(rtol),
            "absolute_passed": False,
            "relative_passed": False,
            "passed_by": "failed",
            "passed": False,
        }
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    absolute = maximum - minimum
    scale = float(np.max(np.abs(array)))
    relative = absolute / max(scale, np.finfo(float).tiny)
    absolute_passed = bool(absolute <= float(atol))
    relative_passed = bool(relative <= float(rtol))
    passed_by = (
        "absolute"
        if absolute_passed
        else "relative"
        if relative_passed
        else "failed"
    )
    return {
        "finite": True,
        "minimum": minimum,
        "maximum": maximum,
        "absolute": absolute,
        "relative": relative,
        "scale": scale,
        "absolute_tolerance": float(atol),
        "relative_tolerance": float(rtol),
        "absolute_passed": absolute_passed,
        "relative_passed": relative_passed,
        "passed_by": passed_by,
        "passed": bool(absolute_passed or relative_passed),
    }


def assess_frequency_level(
    *,
    current_by_shift: dict[str, dict[str, Any]],
    previous_by_shift: dict[str, dict[str, Any]] | None,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    ordered_labels = tuple(current_by_shift)
    current_logdets = [
        float(current_by_shift[label]["two_plate_logdet"])
        for label in ordered_labels
    ]
    hard_closure = all(
        bool(current_by_shift[label]["hard_physical_passed"])
        for label in ordered_labels
    )
    cross_shift = _absolute_then_relative_spread(
        current_logdets,
        rtol=rtol,
        atol=atol,
    )
    adjacent = None
    adjacent_passed = False
    if previous_by_shift is not None:
        adjacent = {
            label: _absolute_then_relative_pair(
                previous_by_shift[label]["two_plate_logdet"],
                current_by_shift[label]["two_plate_logdet"],
                rtol=rtol,
                atol=atol,
            )
            for label in ordered_labels
        }
        adjacent_passed = all(bool(row["passed"]) for row in adjacent.values())
    accepted = bool(
        previous_by_shift is not None
        and hard_closure
        and cross_shift["passed"]
        and adjacent_passed
    )
    return {
        "hard_physical_closure_across_shifts": hard_closure,
        "two_plate_logdet_cross_shift": cross_shift,
        "adjacent_N_by_shift": adjacent,
        "adjacent_N_all_shifts_passed": adjacent_passed,
        "accepted_transition": accepted,
    }


def assess_oscillatory_envelope(
    history: Sequence[dict[str, Any]],
    *,
    rtol: float,
    atol: float,
    levels: int = ENVELOPE_LEVELS,
) -> dict[str, Any]:
    count = int(levels)
    if count < 3:
        raise ValueError("oscillatory envelope requires at least three N levels")
    if len(history) < count:
        return {
            "available": False,
            "levels": count,
            "N_window": [],
            "hard_physical_closure": False,
            "cross_shift_all_levels_passed": False,
            "joint_logdet_envelope": None,
            "passed": False,
        }
    window = list(history[-count:])
    hard = all(bool(row["hard_physical_closure_across_shifts"]) for row in window)
    cross = all(bool(row["two_plate_logdet_cross_shift"]["passed"]) for row in window)
    values = [
        float(state["two_plate_logdet"])
        for row in window
        for state in row["shifts"].values()
    ]
    envelope = _absolute_then_relative_spread(values, rtol=rtol, atol=atol)
    return {
        "available": True,
        "levels": count,
        "N_window": [int(row["N"]) for row in window],
        "hard_physical_closure": hard,
        "cross_shift_all_levels_passed": cross,
        "joint_logdet_envelope": envelope,
        "passed": bool(hard and cross and envelope["passed"]),
    }


def _parse_args(argv: Sequence[str] | None):
    raw = list(sys.argv[1:] if argv is None else argv)
    explicit_atol = "--logdet-atol" in raw
    args = _engine._parse_args(raw)
    if not explicit_atol:
        args.logdet_atol = float(DEFAULT_LOGDET_ATOL)
    return args


def _build_payload(
    *,
    args,
    q_by_label: dict[str, np.ndarray],
    result_records: dict[tuple[str, str, int], dict[str, Any]],
    execution_levels: list[dict[str, Any]],
    observed_cache_bytes_per_point: float | None,
    run_complete: bool,
) -> dict[str, Any]:
    payload = _engine._build_payload(
        args=args,
        q_by_label=q_by_label,
        result_records=result_records,
        execution_levels=execution_levels,
        observed_cache_bytes_per_point=observed_cache_bytes_per_point,
        run_complete=run_complete,
    )
    payload["schema"] = "transverse-point-sweet-spot-v4"
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["convergence_policy"] = {
        "scope": "universal_for_all_pairings_q_directions_q_magnitudes_and_matsubara_indices",
        "q_or_frequency_specific_exceptions": False,
        "comparison_order": "absolute_first_then_relative_fallback",
        "absolute_tolerance": float(args.logdet_atol),
        "relative_tolerance": float(args.logdet_rtol),
        "strict_path": {
            "required_consecutive_accepted_transitions": int(
                args.required_consecutive_passes
            ),
            "requires_each_level_cross_shift_pass": True,
        },
        "oscillatory_envelope_path": {
            "levels": int(ENVELOPE_LEVELS),
            "joint_over_all_levels_and_shifts": True,
            "requires_all_hard_physical_gates": True,
            "requires_each_level_cross_shift_pass": True,
        },
        "absolute_tolerance_status": (
            "provisional_global_floor_pending_outer_integral_error_budget_calibration"
        ),
    }
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    q_by_label = {
        str(point["label"]): np.asarray(point["q_lab"], dtype=float)
        for point in args.q_points
    }
    result_records: dict[tuple[str, str, int], dict[str, Any]] = {}
    active: dict[tuple[str, str], set[int]] = {}
    consecutive: dict[tuple[str, str, int], int] = {}
    previous_states: dict[
        tuple[str, str, int],
        dict[str, dict[str, Any]],
    ] = {}

    for pairing_name in args.pairings:
        for label in q_by_label:
            active[(pairing_name, label)] = set(args.matsubara_indices)
            for n in args.matsubara_indices:
                key = (pairing_name, label, int(n))
                consecutive[key] = 0
                result_records[key] = {
                    "pairing": pairing_name,
                    "q_label": label,
                    "q_lab": q_by_label[label].tolist(),
                    "n": int(n),
                    "history": [],
                    "sweet_spot": {
                        "status": "not_established",
                        "establishment_mode": None,
                        "working_N": None,
                        "audit_N": None,
                    },
                }

    execution_levels: list[dict[str, Any]] = []
    observed_cache_bytes_per_point: float | None = None
    fork_supported = "fork" in get_all_start_methods()

    for n_grid in args.N_candidates:
        jobs = _engine._build_context_jobs(
            n_grid=int(n_grid),
            args=args,
            active=active,
            q_by_label=q_by_label,
        )
        if not jobs:
            break
        estimated_bytes = estimate_context_bytes(
            point_count=int(n_grid) ** 2,
            observed_bytes_per_point=observed_cache_bytes_per_point,
            safety_factor=float(args.memory_safety_factor),
            fallback_bytes_per_point=float(args.fallback_context_bytes_per_point),
        )
        plan = choose_cpu_parallel_plan(
            mode=str(args.parallel_mode),
            requested_workers=int(args.workers),
            context_count=len(jobs),
            max_q_tasks_per_context=_engine._max_q_tasks_per_context(jobs),
            total_flat_tasks=_engine._total_flat_tasks(jobs),
            estimated_context_bytes=estimated_bytes,
            memory_budget_gb=float(args.memory_budget_gb),
            max_context_workers=int(args.max_context_workers),
            q_parallel_supported=fork_supported,
        )
        level_started = perf_counter()
        records, waves = _engine._execute_level(jobs=jobs, plan=plan)

        for record in records.values():
            points = max(int(record["point_count"]), 1)
            measured = float(record["material_cache_array_bytes"]) / points
            observed_cache_bytes_per_point = (
                measured
                if observed_cache_bytes_per_point is None
                else max(observed_cache_bytes_per_point, measured)
            )

        level_record: dict[str, Any] = {
            "N": int(n_grid),
            "parallel_plan": plan.as_dict(),
            "waves": waves,
            "level_wall_seconds": float(perf_counter() - level_started),
            "pairings": {},
        }
        for pairing_name in args.pairings:
            shift_records = [
                records[(pairing_name, shift_index)]
                for shift_index in range(len(args.shifts))
                if (pairing_name, shift_index) in records
            ]
            if not shift_records:
                continue
            level_record["pairings"][pairing_name] = shift_records
            active_by_label = {
                label: tuple(sorted(active[(pairing_name, label)]))
                for label in q_by_label
            }
            resolved_now: list[tuple[str, int]] = []
            shift_labels = tuple(
                f"shift_{index}:{tuple(record['shift'])}"
                for index, record in enumerate(shift_records)
            )
            for label, indices in active_by_label.items():
                for n in indices:
                    key = (pairing_name, label, int(n))
                    current_by_shift = {
                        shift_label: record["points"][label][str(n)]
                        for shift_label, record in zip(
                            shift_labels,
                            shift_records,
                            strict=True,
                        )
                    }
                    assessment = assess_frequency_level(
                        current_by_shift=current_by_shift,
                        previous_by_shift=previous_states.get(key),
                        rtol=float(args.logdet_rtol),
                        atol=float(args.logdet_atol),
                    )
                    consecutive[key] = (
                        consecutive[key] + 1
                        if assessment["accepted_transition"]
                        else 0
                    )
                    history_row = {
                        "N": int(n_grid),
                        "shifts": current_by_shift,
                        **assessment,
                        "consecutive_accepted_transitions": int(consecutive[key]),
                    }
                    history = result_records[key]["history"]
                    history.append(history_row)
                    envelope = assess_oscillatory_envelope(
                        history,
                        rtol=float(args.logdet_rtol),
                        atol=float(args.logdet_atol),
                    )
                    history_row["oscillatory_envelope"] = envelope
                    previous_states[key] = current_by_shift

                    strict_ready = consecutive[key] >= int(
                        args.required_consecutive_passes
                    )
                    envelope_ready = bool(envelope["passed"])
                    if strict_ready or envelope_ready:
                        if len(history) < 2:
                            raise RuntimeError(
                                "established convergence lacks a previous N level"
                            )
                        mode = (
                            "strict_consecutive_adjacent"
                            if strict_ready
                            else "three_level_oscillatory_envelope"
                        )
                        result_records[key]["sweet_spot"] = {
                            "status": "established",
                            "establishment_mode": mode,
                            "working_N": int(history[-2]["N"]),
                            "audit_N": int(history[-1]["N"]),
                            "required_consecutive_passes": int(
                                args.required_consecutive_passes
                            ),
                            "envelope_levels": int(ENVELOPE_LEVELS),
                            "envelope_N_window": list(envelope["N_window"]),
                            "criterion": (
                                "universal hard physical closure and cross-shift "
                                "stability plus either consecutive adjacent-N "
                                "convergence or a three-level absolute-first, "
                                "relative-fallback oscillatory envelope"
                            ),
                        }
                        resolved_now.append((label, int(n)))
            for label, n in resolved_now:
                active[(pairing_name, label)].discard(n)

        execution_levels.append(level_record)
        _engine._atomic_write(
            args.output,
            _build_payload(
                args=args,
                q_by_label=q_by_label,
                result_records=result_records,
                execution_levels=execution_levels,
                observed_cache_bytes_per_point=observed_cache_bytes_per_point,
                run_complete=False,
            ),
        )
        if not any(active.values()):
            break

    payload = _build_payload(
        args=args,
        q_by_label=q_by_label,
        result_records=result_records,
        execution_levels=execution_levels,
        observed_cache_bytes_per_point=observed_cache_bytes_per_point,
        run_complete=True,
    )
    _engine._atomic_write(args.output, payload)

    summary = {
        "output": str(args.output),
        "schema": payload["schema"],
        "all_requested_sweet_spots_established": payload[
            "all_requested_sweet_spots_established"
        ],
        "convergence_policy": payload["convergence_policy"],
        "parallel_plans": [
            {
                "N": level["N"],
                "strategy": level["parallel_plan"]["strategy"],
                "total_worker_budget": level["parallel_plan"][
                    "total_worker_budget"
                ],
                "context_workers": level["parallel_plan"]["context_workers"],
                "q_workers": level["parallel_plan"]["q_workers"],
                "flat_workers": level["parallel_plan"]["flat_workers"],
                "total_flat_tasks": level["parallel_plan"]["total_flat_tasks"],
                "wave_count": level["parallel_plan"]["wave_count"],
                "reason": level["parallel_plan"]["reason"],
            }
            for level in execution_levels
        ],
        "points": [
            {
                "pairing": row["pairing"],
                "q_label": row["q_label"],
                "n": row["n"],
                **row["sweet_spot"],
                "evaluated_N": [item["N"] for item in row["history"]],
            }
            for row in payload["point_results"]
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
