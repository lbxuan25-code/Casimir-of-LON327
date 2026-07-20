from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from ._diagnostic_io import finite_number, mapping, sequence


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
