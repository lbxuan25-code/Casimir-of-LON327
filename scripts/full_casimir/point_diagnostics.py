from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from ._diagnostic_io import finite_number, mapping, sequence

_ENVELOPE_LEVELS = 3


def _float_from_hex(value: Any, *, name: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an IEEE-754 hexadecimal string")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must decode to a finite float")
    return result


def _entry_identity(entry: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(entry.get("pairing")),
        int(entry.get("n")),
        str(entry.get("qx_hex")),
        str(entry.get("qy_hex")),
    )


def _shift_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "two_plate_logdet": finite_number(state.get("two_plate_logdet")),
        "hard_physical_passed": bool(state.get("hard_physical_passed")),
        "error": str(state.get("error", "")),
    }


def _history_row_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    shifts = mapping(row.get("shifts"))
    adjacent = row.get("adjacent_N_by_shift")
    adjacent_mapping = mapping(adjacent)
    cross = dict(mapping(row.get("two_plate_logdet_cross_shift")))
    envelope = dict(mapping(row.get("oscillatory_envelope")))
    failed_gates: list[str] = []
    if not bool(row.get("hard_physical_closure_across_shifts")):
        failed_gates.append("hard_physical_closure_across_shifts")
    if not bool(cross.get("passed")):
        failed_gates.append("two_plate_logdet_cross_shift")
    if adjacent is None:
        failed_gates.append("adjacent_N_unavailable")
    elif not bool(row.get("adjacent_N_all_shifts_passed")):
        failed_gates.append("adjacent_N_all_shifts")
    if envelope.get("available") and not bool(envelope.get("passed")):
        failed_gates.append("oscillatory_envelope")
    return {
        "N": int(row.get("N")),
        "shifts": {
            str(label): _shift_summary(mapping(state))
            for label, state in shifts.items()
        },
        "hard_physical_closure_across_shifts": bool(
            row.get("hard_physical_closure_across_shifts")
        ),
        "two_plate_logdet_cross_shift": cross,
        "adjacent_N_by_shift": {
            str(label): dict(mapping(state))
            for label, state in adjacent_mapping.items()
        }
        if adjacent is not None
        else None,
        "adjacent_N_all_shifts_passed": bool(
            row.get("adjacent_N_all_shifts_passed")
        ),
        "accepted_transition": bool(row.get("accepted_transition")),
        "consecutive_accepted_transitions": int(
            row.get("consecutive_accepted_transitions", 0)
        ),
        "oscillatory_envelope": envelope,
        "failed_gates": failed_gates,
    }


def _latest_point_blocker(
    history: Sequence[Mapping[str, Any]],
    *,
    required_consecutive_passes: int,
) -> dict[str, Any]:
    if not history:
        return {"classification": "no_history", "failed_gates": ["history_missing"]}
    latest = history[-1]
    cross = mapping(latest.get("two_plate_logdet_cross_shift"))
    envelope = mapping(latest.get("oscillatory_envelope"))
    consecutive = int(latest.get("consecutive_accepted_transitions", 0))
    if not bool(latest.get("hard_physical_closure_across_shifts")):
        classification = "hard_physical_closure_failed"
    elif not bool(cross.get("passed")):
        classification = "cross_shift_stability_failed"
    elif latest.get("adjacent_N_by_shift") is None:
        classification = "adjacent_N_not_available"
    elif not bool(latest.get("adjacent_N_all_shifts_passed")):
        classification = "adjacent_N_stability_failed"
    elif bool(latest.get("accepted_transition")) and consecutive < int(
        required_consecutive_passes
    ):
        classification = "additional_consecutive_pass_required"
    elif envelope.get("available") and not bool(envelope.get("passed")):
        classification = "oscillatory_envelope_failed"
    else:
        classification = "unidentified_additional_gate"
    adjacent = mapping(latest.get("adjacent_N_by_shift"))
    return {
        "classification": classification,
        "latest_N": int(latest.get("N")),
        "required_consecutive_passes": int(required_consecutive_passes),
        "consecutive_accepted_transitions": consecutive,
        "hard_physical_closure_across_shifts": bool(
            latest.get("hard_physical_closure_across_shifts")
        ),
        "cross_shift_passed": bool(cross.get("passed")),
        "cross_shift": dict(cross),
        "adjacent_N_all_shifts_passed": bool(
            latest.get("adjacent_N_all_shifts_passed")
        ),
        "adjacent_N_failures": {
            str(label): dict(mapping(record))
            for label, record in adjacent.items()
            if not bool(mapping(record).get("passed"))
        },
        "oscillatory_envelope": dict(envelope),
    }


def _comparison(left: float, right: float, *, rtol: float, atol: float) -> dict[str, Any]:
    finite = math.isfinite(left) and math.isfinite(right)
    if not finite:
        return {
            "finite": False,
            "absolute": math.nan,
            "relative": math.nan,
            "absolute_tolerance": float(atol),
            "relative_tolerance": float(rtol),
            "absolute_passed": False,
            "relative_passed": False,
            "passed_by": "failed",
            "passed": False,
        }
    absolute = abs(right - left)
    scale = max(abs(left), abs(right))
    relative = absolute / max(scale, float.fromhex("0x1.0p-1022"))
    absolute_passed = bool(absolute <= atol)
    relative_passed = bool(relative <= rtol)
    return {
        "finite": True,
        "absolute": absolute,
        "relative": relative,
        "scale": scale,
        "absolute_tolerance": float(atol),
        "relative_tolerance": float(rtol),
        "absolute_passed": absolute_passed,
        "relative_passed": relative_passed,
        "passed_by": "absolute" if absolute_passed else "relative" if relative_passed else "failed",
        "passed": bool(absolute_passed or relative_passed),
    }


def _spread(values: Sequence[float], *, rtol: float, atol: float) -> dict[str, Any]:
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        return {
            "finite": False,
            "minimum": math.nan,
            "maximum": math.nan,
            "absolute": math.nan,
            "relative": math.nan,
            "absolute_tolerance": float(atol),
            "relative_tolerance": float(rtol),
            "absolute_passed": False,
            "relative_passed": False,
            "passed_by": "failed",
            "passed": False,
        }
    minimum = min(values)
    maximum = max(values)
    absolute = maximum - minimum
    scale = max(abs(value) for value in values)
    relative = absolute / max(scale, float.fromhex("0x1.0p-1022"))
    absolute_passed = bool(absolute <= atol)
    relative_passed = bool(relative <= rtol)
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
        "passed_by": "absolute" if absolute_passed else "relative" if relative_passed else "failed",
        "passed": bool(absolute_passed or relative_passed),
    }


def replay_point_policy(
    point_result: Mapping[str, Any],
    *,
    logdet_rtol: float,
    logdet_atol: float,
    required_consecutive_passes: int,
) -> dict[str, Any]:
    """Re-evaluate one stored point history without running new microscopic work."""

    if logdet_rtol < 0.0 or logdet_atol < 0.0:
        raise ValueError("logdet tolerances must be non-negative")
    if required_consecutive_passes < 1:
        raise ValueError("required_consecutive_passes must be positive")
    history = [mapping(row) for row in sequence(point_result.get("history"))]
    previous_shifts: Mapping[str, Any] | None = None
    replayed: list[dict[str, Any]] = []
    consecutive = 0
    establishment: dict[str, Any] | None = None
    for source_row in history:
        shifts = mapping(source_row.get("shifts"))
        labels = tuple(str(label) for label in shifts)
        values = [float(mapping(shifts[label]).get("two_plate_logdet")) for label in labels]
        hard = bool(labels) and all(
            bool(mapping(shifts[label]).get("hard_physical_passed")) for label in labels
        )
        cross = _spread(values, rtol=logdet_rtol, atol=logdet_atol)
        adjacent: dict[str, Any] | None = None
        adjacent_all = False
        if previous_shifts is not None and tuple(previous_shifts) == labels:
            adjacent = {
                label: _comparison(
                    float(mapping(previous_shifts[label]).get("two_plate_logdet")),
                    float(mapping(shifts[label]).get("two_plate_logdet")),
                    rtol=logdet_rtol,
                    atol=logdet_atol,
                )
                for label in labels
            }
            adjacent_all = all(bool(record["passed"]) for record in adjacent.values())
        accepted = bool(previous_shifts is not None and hard and cross["passed"] and adjacent_all)
        consecutive = consecutive + 1 if accepted else 0
        row = {
            "N": int(source_row.get("N")),
            "hard_physical_closure_across_shifts": hard,
            "two_plate_logdet_cross_shift": cross,
            "adjacent_N_by_shift": adjacent,
            "adjacent_N_all_shifts_passed": adjacent_all,
            "accepted_transition": accepted,
            "consecutive_accepted_transitions": consecutive,
            "shifts": {label: _shift_summary(mapping(shifts[label])) for label in labels},
        }
        replayed.append(row)
        window = replayed[-_ENVELOPE_LEVELS:]
        if len(window) == _ENVELOPE_LEVELS:
            envelope_values = [
                float(mapping(state).get("two_plate_logdet"))
                for window_row in window
                for state in mapping(window_row.get("shifts")).values()
            ]
            envelope = {
                "available": True,
                "levels": _ENVELOPE_LEVELS,
                "N_window": [int(window_row["N"]) for window_row in window],
                "hard_physical_closure": all(
                    bool(window_row["hard_physical_closure_across_shifts"])
                    for window_row in window
                ),
                "cross_shift_all_levels_passed": all(
                    bool(mapping(window_row["two_plate_logdet_cross_shift"]).get("passed"))
                    for window_row in window
                ),
                "joint_logdet_envelope": _spread(
                    envelope_values, rtol=logdet_rtol, atol=logdet_atol
                ),
            }
            envelope["passed"] = bool(
                envelope["hard_physical_closure"]
                and envelope["cross_shift_all_levels_passed"]
                and mapping(envelope["joint_logdet_envelope"]).get("passed")
            )
        else:
            envelope = {
                "available": False,
                "levels": _ENVELOPE_LEVELS,
                "N_window": [],
                "hard_physical_closure": False,
                "cross_shift_all_levels_passed": False,
                "joint_logdet_envelope": None,
                "passed": False,
            }
        row["oscillatory_envelope"] = envelope
        strict_ready = consecutive >= int(required_consecutive_passes)
        envelope_ready = bool(envelope["passed"])
        if establishment is None and (strict_ready or envelope_ready):
            establishment = {
                "status": "established",
                "establishment_mode": (
                    "strict_consecutive_adjacent"
                    if strict_ready
                    else "three_level_oscillatory_envelope"
                ),
                "working_N": int(replayed[-2]["N"]),
                "audit_N": int(replayed[-1]["N"]),
                "accepted_history_levels": len(replayed),
                "latest_cross_shift": cross,
                "latest_adjacent_N_by_shift": adjacent,
                "latest_oscillatory_envelope": envelope,
            }
            break
        previous_shifts = shifts
    if establishment is None:
        establishment = {
            "status": "not_established",
            "establishment_mode": None,
            "working_N": None,
            "audit_N": None,
            "accepted_history_levels": len(replayed),
            "latest_cross_shift": (
                replayed[-1]["two_plate_logdet_cross_shift"] if replayed else None
            ),
            "latest_adjacent_N_by_shift": (
                replayed[-1]["adjacent_N_by_shift"] if replayed else None
            ),
            "latest_oscillatory_envelope": (
                replayed[-1]["oscillatory_envelope"] if replayed else None
            ),
        }
    evaluated = [int(row.get("N")) for row in history]
    used = evaluated[: int(establishment["accepted_history_levels"])]
    observed_proxy = sum(value * value for value in evaluated)
    used_proxy = sum(value * value for value in used)
    return {
        "policy": {
            "logdet_rtol": float(logdet_rtol),
            "logdet_atol": float(logdet_atol),
            "required_consecutive_passes": int(required_consecutive_passes),
            "hard_physical_gates_unchanged": True,
        },
        **establishment,
        "evaluated_N": evaluated,
        "used_N": used,
        "point_level_N2_work_proxy": used_proxy,
        "observed_point_level_N2_work_proxy": observed_proxy,
        "point_level_N2_work_proxy_saved": observed_proxy - used_proxy,
    }


def tolerance_replay_audit(
    cache: Mapping[str, Any],
    *,
    candidate_logdet_rtols: Sequence[float],
    logdet_atol: float | None = None,
    required_consecutive_passes: int | None = None,
) -> dict[str, Any]:
    """Replay one global acceptance policy over every stored point history."""

    point_policy = mapping(cache.get("point_policy"))
    source_rtol = float(point_policy.get("logdet_rtol", 0.0))
    source_atol = float(
        point_policy.get("logdet_atol", 0.0) if logdet_atol is None else logdet_atol
    )
    source_required = int(
        point_policy.get("required_consecutive_passes", 2)
        if required_consecutive_passes is None
        else required_consecutive_passes
    )
    candidates = tuple(float(value) for value in candidate_logdet_rtols)
    if not candidates or any(value < 0.0 or not math.isfinite(value) for value in candidates):
        raise ValueError("candidate_logdet_rtols must contain finite non-negative values")
    entries = [mapping(entry) for entry in sequence(cache.get("entries"))]
    policies: list[dict[str, Any]] = []
    for rtol in candidates:
        mode_counter: Counter[str] = Counter()
        audit_N_counter: Counter[int] = Counter()
        unresolved: list[list[Any]] = []
        hard_failures = 0
        total_proxy = 0
        observed_proxy = 0
        details: list[dict[str, Any]] = []
        for entry in entries:
            point = mapping(entry.get("point_result"))
            replay = replay_point_policy(
                point,
                logdet_rtol=rtol,
                logdet_atol=source_atol,
                required_consecutive_passes=source_required,
            )
            identity = list(_entry_identity(entry))
            status = str(replay["status"])
            if status == "established":
                mode_counter[str(replay["establishment_mode"])] += 1
                audit_N_counter[int(replay["audit_N"])] += 1
            else:
                unresolved.append(identity)
                history = [mapping(row) for row in sequence(point.get("history"))]
                if history and not bool(history[-1].get("hard_physical_closure_across_shifts")):
                    hard_failures += 1
            total_proxy += int(replay["point_level_N2_work_proxy"])
            observed_proxy += int(replay["observed_point_level_N2_work_proxy"])
            details.append(
                {
                    "identity": identity,
                    "status": status,
                    "establishment_mode": replay["establishment_mode"],
                    "working_N": replay["working_N"],
                    "audit_N": replay["audit_N"],
                    "used_N": replay["used_N"],
                    "point_level_N2_work_proxy": replay["point_level_N2_work_proxy"],
                    "point_level_N2_work_proxy_saved": replay[
                        "point_level_N2_work_proxy_saved"
                    ],
                }
            )
        saved = observed_proxy - total_proxy
        policies.append(
            {
                "logdet_rtol": rtol,
                "logdet_atol": source_atol,
                "required_consecutive_passes": source_required,
                "entry_count": len(entries),
                "established_count": len(entries) - len(unresolved),
                "unresolved_count": len(unresolved),
                "hard_physical_failure_count": hard_failures,
                "establishment_modes": dict(sorted(mode_counter.items())),
                "audit_N_histogram": {
                    str(key): value for key, value in sorted(audit_N_counter.items())
                },
                "unresolved_identities": unresolved,
                "point_level_N2_work_proxy": total_proxy,
                "observed_point_level_N2_work_proxy": observed_proxy,
                "point_level_N2_work_proxy_saved": saved,
                "point_level_N2_work_proxy_saved_fraction": (
                    0.0 if observed_proxy == 0 else saved / observed_proxy
                ),
                "details": details,
            }
        )
    return {
        "schema": "point-tolerance-replay-audit-v1",
        "source_policy": {
            "logdet_rtol": source_rtol,
            "logdet_atol": float(point_policy.get("logdet_atol", source_atol)),
            "required_consecutive_passes": int(
                point_policy.get("required_consecutive_passes", source_required)
            ),
        },
        "candidate_policies": policies,
        "work_proxy_note": (
            "The N^2 proxy is summed per point and deliberately is not reported as wall time; "
            "shared contexts, batching, and cache reuse make actual speedup smaller."
        ),
        "scientific_limitations": {
            "quadrature_weighted_energy_impact_available": False,
            "high_N_holdout_validation_available": False,
            "production_policy_change_authorized": False,
        },
    }


def point_cache_diagnostics(
    cache: Mapping[str, Any],
    *,
    source_dropped_identities: Sequence[tuple[str, int, str, str]] = (),
) -> dict[str, Any]:
    entries = sequence(cache.get("entries"))
    required = int(mapping(cache.get("point_policy")).get("required_consecutive_passes", 2))
    source_dropped = set(source_dropped_identities)
    status_counter: Counter[str] = Counter()
    matsubara_counter: Counter[int] = Counter()
    unresolved: list[dict[str, Any]] = []
    symmetry_groups: dict[tuple[int, str, str], list[str]] = defaultdict(list)
    for raw_entry in entries:
        entry = mapping(raw_entry)
        point = mapping(entry.get("point_result"))
        sweet = mapping(point.get("sweet_spot"))
        status = str(sweet.get("status", "missing"))
        status_counter[status] += 1
        n = int(entry.get("n"))
        matsubara_counter[n] += 1
        if status == "established":
            continue
        qx = _float_from_hex(entry.get("qx_hex"), name="qx_hex")
        qy = _float_from_hex(entry.get("qy_hex"), name="qy_hex")
        history = [mapping(row) for row in sequence(point.get("history"))]
        identity = _entry_identity(entry)
        components = sorted((abs(qx), abs(qy)))
        symmetry_key = (n, components[0].hex(), components[1].hex())
        symmetry_groups[symmetry_key].append(
            "|".join((identity[0], str(identity[1]), identity[2], identity[3]))
        )
        unresolved.append(
            {
                "identity": list(identity),
                "was_dropped_from_source_extension": identity in source_dropped,
                "q_model": [qx, qy],
                "q_radius": math.hypot(qx, qy),
                "q_angle_deg": math.degrees(math.atan2(qy, qx)),
                "symmetry_signature": list(symmetry_key),
                "sweet_spot": dict(sweet),
                "evaluated_N": [int(row.get("N")) for row in history],
                "latest_blocker": _latest_point_blocker(
                    history,
                    required_consecutive_passes=required,
                ),
                "history": [_history_row_summary(row) for row in history],
            }
        )
    return {
        "schema": str(cache.get("schema", "")),
        "entry_count": len(entries),
        "status_counts": dict(sorted(status_counter.items())),
        "matsubara_histogram": {
            str(key): value for key, value in sorted(matsubara_counter.items())
        },
        "unresolved_count": len(unresolved),
        "required_consecutive_passes": required,
        "unresolved_points": unresolved,
        "unresolved_symmetry_groups": [
            {
                "n": int(key[0]),
                "absolute_q_components_hex": [key[1], key[2]],
                "member_count": len(members),
                "members": sorted(members),
            }
            for key, members in sorted(symmetry_groups.items())
        ],
    }


__all__ = [
    "point_cache_diagnostics",
    "replay_point_policy",
    "tolerance_replay_audit",
]
