from __future__ import annotations

import math

import pytest

from scripts.full_casimir.budget_audit import (
    budget_fraction_sensitivity,
    conditional_analytic_outer_tail_bound,
    holdout_plan,
    weighted_microscopic_impact,
)
from scripts.full_casimir.candidate_replay_audit import production_equivalence_audit
from scripts.full_casimir.policy_audit import compare_policy_snapshots


def _point_result() -> dict:
    labels = ("s0", "s1", "s2")
    levels = [
        (128, (-1.0, -1.0000001, -0.9999999)),
        (192, (-1.0010, -1.0010001, -1.0009999)),
        (256, (-1.0015, -1.0015001, -1.0014999)),
    ]
    history = []
    for index, (N, values) in enumerate(levels):
        history.append(
            {
                "N": N,
                "shifts": {
                    label: {
                        "two_plate_logdet": value,
                        "hard_physical_passed": True,
                    }
                    for label, value in zip(labels, values, strict=True)
                },
                "hard_physical_closure_across_shifts": True,
                "two_plate_logdet_cross_shift": {"passed": True},
                "adjacent_N_by_shift": None if index == 0 else {},
                "adjacent_N_all_shifts_passed": index > 0,
                "accepted_transition": index > 0,
                "consecutive_accepted_transitions": index,
                "oscillatory_envelope": {
                    "available": index >= 2,
                    "passed": True,
                },
            }
        )
    return {
        "sweet_spot": {
            "status": "established",
            "working_N": 192,
            "audit_N": 256,
            "establishment_mode": "strict_consecutive_adjacent",
        },
        "history": history,
    }


def test_production_equivalence_is_pointwise() -> None:
    cache = {
        "entries": [
            {
                "pairing": "dwave",
                "n": 1,
                "qx_hex": float(0.1).hex(),
                "qy_hex": float(0.2).hex(),
                "point_result": _point_result(),
            }
        ]
    }
    result = production_equivalence_audit(
        cache,
        source_logdet_rtol=0.002,
        source_logdet_atol=0.0,
        required_consecutive_passes=2,
    )
    assert result["checked_point_count"] == 1
    assert result["equivalent"] is True
    assert result["mismatch_count"] == 0


def test_weighted_impact_keeps_signed_delta_and_absolute_bound_separate() -> None:
    identity = ("dwave", 1, float(0.1).hex(), float(0.2).hex())
    trace = [
        {
            "identity": list(identity),
            "signed_weight_J_m2_per_logdet": 2.0,
            "absolute_weight_J_m2_per_logdet": 2.0,
        }
    ]
    evidence = {
        identity: {
            "empirical_delta": -0.25,
            "local_absolute_uncertainty": 0.4,
            "point_level_N2_work_proxy_saved": 10,
        }
    }
    result = weighted_microscopic_impact(trace, evidence)
    assert result["missing_identity_count"] == 0
    assert result["total_signed_delta_J_m2"] == pytest.approx(-0.5)
    assert result["total_absolute_error_bound_J_m2"] == pytest.approx(0.8)


def test_holdout_plan_uses_new_levels_above_reference() -> None:
    identity = ("dwave", 1, "x", "y")
    evidence = {
        identity: {
            "reference_highest_valid_N": 1280,
            "local_absolute_uncertainty": 1e-6,
        }
    }
    impact = {
        "top_point_contributors": [
            {
                "identity": list(identity),
                "absolute_energy_error_bound_J_m2": 1e-13,
            }
        ]
    }
    result = holdout_plan(evidence, impact)
    levels = result["points"][0]["suggested_holdout_N"]
    assert levels[0] > 1280
    assert levels[1] > levels[0]
    assert result["status"] == "planned_not_executed"


def test_conditional_tail_bound_has_half_weight_at_n_zero() -> None:
    result = conditional_analytic_outer_tail_bound(
        u0=24.0,
        separation_nm=20.0,
        temperature_K=10.0,
        matsubara_indices=(0, 1),
    )
    n0, n1 = result["channel_bounds_J_m2"]
    assert math.isfinite(n0) and n0 > 0.0
    assert n1 == pytest.approx(2.0 * n0)
    assert result["production_usable"] is False


def test_budget_fraction_sensitivity_is_only_a_screen() -> None:
    payload = {
        "cutoff_records": [
            {
                "u_max": 24.0,
                "pairing_results": {
                    "spm": {
                        "matsubara_indices": [0],
                        "combined_comparison_radial_errors_J_m2": [7e-13],
                        "estimated_angular_errors_J_m2": [2e-13],
                        "outer_tolerances_J_m2": [1e-12],
                    }
                },
            }
        ]
    }
    result = budget_fraction_sensitivity(payload)
    by_fraction = {
        row["radial_budget_fraction"]: row["all_channels_passed"]
        for row in result["records"]
    }
    assert by_fraction[0.75] is True
    assert by_fraction[0.85] is False
    assert result["exact_replay_still_required"] is True


def _config(N_candidates: list[int]) -> dict:
    return {
        "outer_tail_config": {
            "joint_config": {
                "radial_config": {
                    "point_config": {
                        "pairings": ["spm"],
                        "N_candidates": N_candidates,
                        "required_consecutive_passes": 2,
                        "logdet_rtol": 0.0015,
                        "logdet_atol": 1e-6,
                    }
                },
                "radial_budget_fraction": 0.8,
                "angular_budget_fraction": 0.2,
            }
        }
    }


def test_policy_length_difference_reports_lengths_and_sequences() -> None:
    result = compare_policy_snapshots(
        (
            ("a", _config([128, 192, 256])),
            ("b", _config([128, 192, 256, 384])),
        )
    )
    row = next(
        item
        for item in result["comparisons"][0]["differences"]
        if item["path"] == "$.microscopic_acceptance.N_candidates.length"
    )
    assert row["reference_value"] == 3
    assert row["compared_value"] == 4
    assert row["reference_sequence"] == [128, 192, 256]
    assert row["compared_sequence"] == [128, 192, 256, 384]
