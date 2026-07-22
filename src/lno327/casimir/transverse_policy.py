"""Frozen transverse-shift and adaptive-ladder execution contracts.

The production transverse certifier uses two independent shifted periodic grids.  A
third historical shift is retained only as an explicit conditional audit input; it
is never part of the routine production point cost.  Every numerical ladder is an
adaptive ceiling: once the corresponding formal certificate passes, higher levels
for that object are forbidden.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

TRANSVERSE_SHIFT_POLICY_ID = "full-casimir-two-shift-policy-v1"
LADDER_EARLY_STOP_CONTRACT = "full-casimir-adaptive-ceiling-early-stop-v1"

FORMAL_TRANSVERSE_SHIFTS: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),
    (0.25, 0.75),
)
CONDITIONAL_AUDIT_SHIFT: tuple[float, float] = (0.75, 0.25)


def normalize_shifts(values: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    shifts = tuple(tuple(float(component) for component in value) for value in values)
    if any(len(value) != 2 for value in shifts):
        raise ValueError("every transverse shift must contain two components")
    if len(shifts) < 2 or len(set(shifts)) != len(shifts):
        raise ValueError("formal transverse certification requires two unique shifts")
    return shifts


def transverse_policy_payload() -> dict[str, Any]:
    return {
        "policy_id": TRANSVERSE_SHIFT_POLICY_ID,
        "formal_shifts": [list(value) for value in FORMAL_TRANSVERSE_SHIFTS],
        "formal_shift_count": len(FORMAL_TRANSVERSE_SHIFTS),
        "conditional_audit_shift": list(CONDITIONAL_AUDIT_SHIFT),
        "conditional_audit_is_routine_production_work": False,
        "single_shift_production_acceptance_forbidden": True,
        "cross_shift_acceptance_required": True,
        "conditional_audit_uses": [
            "historical_three_shift_replay",
            "independent_holdout",
            "near_threshold_or_nonmonotone_diagnostic",
        ],
        "conditional_audit_cannot_rescue_failed_formal_shifts": True,
    }


def ladder_early_stop_policy_payload() -> dict[str, Any]:
    common = {
        "contract": LADDER_EARLY_STOP_CONTRACT,
        "candidate_values_are_adaptive_ceilings": True,
        "stop_immediately_after_formal_certificate": True,
        "higher_levels_after_certificate_forbidden": True,
        "resume_may_compute_only_missing_or_unresolved_work": True,
    }
    return {
        **common,
        "microscopic_N": {
            "scope": "independent_pairing_q_n_point",
            "resolved_points_removed_from_active_set": True,
        },
        "outer_Q": {
            "scope": "independent_matsubara_outer_integral",
            "analytic_certificate_attempted_at_every_cutoff": True,
        },
        "finite_outer_domain": {
            "advance_only_unresolved_error_direction": True,
            "repeat_passed_radial_or_angular_axis_forbidden": True,
        },
        "matsubara": {
            "scope": "complete_dyadic_blocks",
            "only_new_frequency_block_may_be_computed": True,
            "cached_lower_frequencies_must_be_reused": True,
        },
    }


def validate_formal_shift_policy(payload: Mapping[str, Any]) -> None:
    shifts = normalize_shifts(payload.get("formal_shifts", ()))
    if shifts != FORMAL_TRANSVERSE_SHIFTS:
        raise ValueError("formal transverse shifts do not match the frozen policy")
    audit = tuple(float(value) for value in payload.get("conditional_audit_shift", ()))
    if audit != CONDITIONAL_AUDIT_SHIFT:
        raise ValueError("conditional audit shift does not match the frozen policy")
    if audit in shifts:
        raise ValueError("conditional audit shift must be separate from formal shifts")


__all__ = [
    "CONDITIONAL_AUDIT_SHIFT",
    "FORMAL_TRANSVERSE_SHIFTS",
    "LADDER_EARLY_STOP_CONTRACT",
    "TRANSVERSE_SHIFT_POLICY_ID",
    "ladder_early_stop_policy_payload",
    "normalize_shifts",
    "transverse_policy_payload",
    "validate_formal_shift_policy",
]
