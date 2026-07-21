"""Replay historical three-shift point histories under the frozen two-shift policy."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lno327.casimir.fixed_transverse_point_certification import (
    ENVELOPE_LEVELS,
    assess_frequency_level,
    assess_oscillatory_envelope,
)
from lno327.casimir.transverse_policy import FORMAL_TRANSVERSE_SHIFTS


def _shift_from_label(label: str) -> tuple[float, float] | None:
    _prefix, separator, suffix = str(label).partition(":")
    if not separator:
        return None
    try:
        value = ast.literal_eval(suffix)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _select_formal_states(states: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    parsed = {_shift_from_label(label): (label, value) for label, value in states.items()}
    if all(shift in parsed for shift in FORMAL_TRANSVERSE_SHIFTS):
        for shift in FORMAL_TRANSVERSE_SHIFTS:
            label, value = parsed[shift]
            selected[str(label)] = dict(value)
        return selected
    # Historical cache rows preserve shift insertion order.  This fallback is
    # accepted only when at least the original three states are present.
    if len(states) < 3:
        raise ValueError("history row lacks the two formal shifts and a historical audit shift")
    for label, value in list(states.items())[:2]:
        selected[str(label)] = dict(value)
    return selected


def replay_point_two_shift(
    point: Mapping[str, Any],
    *,
    rtol: float,
    atol: float,
    required_consecutive_passes: int,
) -> dict[str, Any]:
    replay_history: list[dict[str, Any]] = []
    previous: dict[str, dict[str, Any]] | None = None
    consecutive = 0
    established = False
    mode: str | None = None
    working_N: int | None = None
    audit_N: int | None = None

    for raw in point.get("history", ()):
        row = dict(raw)
        current = _select_formal_states(row.get("shifts", {}))
        assessment = assess_frequency_level(
            current_by_shift=current,
            previous_by_shift=previous,
            rtol=float(rtol),
            atol=float(atol),
        )
        consecutive = consecutive + 1 if assessment["accepted_transition"] else 0
        replay_row = {
            "N": int(row["N"]),
            "shifts": current,
            **assessment,
            "consecutive_accepted_transitions": int(consecutive),
        }
        replay_history.append(replay_row)
        envelope = assess_oscillatory_envelope(
            replay_history,
            rtol=float(rtol),
            atol=float(atol),
            levels=ENVELOPE_LEVELS,
        )
        replay_row["oscillatory_envelope"] = envelope
        previous = current
        strict_ready = consecutive >= int(required_consecutive_passes)
        envelope_ready = bool(envelope["passed"])
        if strict_ready or envelope_ready:
            established = True
            mode = (
                "strict_consecutive_adjacent"
                if strict_ready
                else "three_level_oscillatory_envelope"
            )
            if len(replay_history) < 2:
                raise RuntimeError("two-shift replay established without a previous N level")
            working_N = int(replay_history[-2]["N"])
            audit_N = int(replay_history[-1]["N"])
            break

    original = point.get("sweet_spot", {})
    original_established = original.get("status") == "established"
    return {
        "pairing": point.get("pairing"),
        "q_label": point.get("q_label"),
        "n": point.get("n"),
        "original_three_shift_established": bool(original_established),
        "two_shift_established": bool(established),
        "decision_matches": bool(original_established == established),
        "two_shift_establishment_mode": mode,
        "two_shift_working_N": working_N,
        "two_shift_audit_N": audit_N,
        "evaluated_N": [int(row["N"]) for row in replay_history],
    }


def _point_rows(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = payload.get("point_results")
    if isinstance(rows, list):
        return tuple(row for row in rows if isinstance(row, Mapping))
    entries = payload.get("entries")
    if isinstance(entries, list):
        return tuple(row for row in entries if isinstance(row, Mapping))
    raise ValueError("input contains neither point_results nor cache entries")


def replay_payload(
    payload: Mapping[str, Any],
    *,
    rtol: float,
    atol: float,
    required_consecutive_passes: int,
) -> dict[str, Any]:
    records = [
        replay_point_two_shift(
            point,
            rtol=rtol,
            atol=atol,
            required_consecutive_passes=required_consecutive_passes,
        )
        for point in _point_rows(payload)
    ]
    mismatches = [record for record in records if not record["decision_matches"]]
    return {
        "schema": "transverse-two-shift-replay-audit-v1",
        "formal_shifts": [list(value) for value in FORMAL_TRANSVERSE_SHIFTS],
        "point_count": len(records),
        "decision_match_count": len(records) - len(mismatches),
        "decision_mismatch_count": len(mismatches),
        "all_decisions_match": not mismatches,
        "mismatches": mismatches,
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logdet-rtol", type=float, default=2.0e-3)
    parser.add_argument("--logdet-atol", type=float, default=1.0e-6)
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = replay_payload(
        payload,
        rtol=args.logdet_rtol,
        atol=args.logdet_atol,
        required_consecutive_passes=args.required_consecutive_passes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"points: {report['point_count']}")
    print(f"decision mismatches: {report['decision_mismatch_count']}")
    print(f"written: {args.output}")
    return 0 if report["all_decisions_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
