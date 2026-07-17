from __future__ import annotations

import numpy as np

from lno327.casimir.fixed_outer_q import (
    aggregate_certified_outer_q,
    build_staged_grid_plan,
    build_union_node_manifest,
    compare_ladders,
)
from lno327.constants import KB


def _plan_and_manifest():
    plan = build_staged_grid_plan(
        u_max_values=[1.0, 2.0],
        radial_orders=[1, 2],
        angular_orders=[2, 4],
        angular_offsets=[0.0, 0.5],
    )
    manifest = build_union_node_manifest(
        plan,
        separation_m=20e-9,
        lattice_a_x_m=3.754e-10,
        lattice_a_y_m=3.754e-10,
    )
    return plan, manifest


def _point(pairing: str, label: str, n: int, value: float) -> dict[str, object]:
    shifts = {
        "shift_0:(0.5, 0.5)": {
            "two_plate_logdet": value,
            "hard_physical_passed": True,
        },
        "shift_1:(0.25, 0.75)": {
            "two_plate_logdet": value + 1e-10,
            "hard_physical_passed": True,
        },
    }
    return {
        "pairing": pairing,
        "q_label": label,
        "n": n,
        "sweet_spot": {
            "status": "established",
            "establishment_mode": "strict_consecutive_adjacent",
            "working_N": 8,
            "audit_N": 10,
        },
        "history": [
            {
                "N": 10,
                "shifts": shifts,
                "two_plate_logdet_cross_shift": {
                    "passed": True,
                    "absolute": 1e-10,
                },
            }
        ],
    }


def test_staged_plan_deduplicates_shared_reference_config() -> None:
    plan, manifest = _plan_and_manifest()
    assert plan.reference_spec_id in {spec.spec_id for spec in plan.specs}
    assert plan.ladders["cutoff"][-1] == plan.reference_spec_id
    assert plan.ladders["radial"][-1] == plan.reference_spec_id
    assert plan.ladders["angular"][-1] == plan.reference_spec_id
    assert plan.reference_spec_id in plan.ladders["offset"]
    assert len(plan.specs) < sum(len(values) for values in plan.ladders.values())
    naive_count = sum(grid.node_count for grid in manifest.grids.values())
    assert len(manifest.labels) <= naive_count
    assert np.all(np.linalg.norm(manifest.q_model, axis=1) > 0.0)


def test_certified_primary_shift_is_integrated_with_prime_weight() -> None:
    plan, manifest = _plan_and_manifest()
    payload = {
        "point_results": [
            _point("spm", label, n, -0.2 if n == 0 else -0.1)
            for label in manifest.labels
            for n in (0, 1)
        ]
    }
    results, unresolved = aggregate_certified_outer_q(
        sweet_spot_payload=payload,
        plan=plan,
        manifest=manifest,
        pairings=["spm"],
        matsubara_indices=[0, 1],
        temperature_K=10.0,
    )
    assert unresolved == []
    reference = results[plan.reference_spec_id]
    pairing = reference["pairings"]["spm"]
    grid = manifest.grids[plan.reference_spec_id]
    expected = KB * 10.0 * grid.disk_measure_m_inv2 * (0.5 * -0.2 + -0.1)
    assert pairing["status"] == "integrated"
    assert np.isclose(pairing["partial_free_energy_J_m2"], expected, rtol=2e-15)
    assert pairing["primary_shift_is_canonical_estimator"] is True
    assert pairing["maximum_audit_N"] == 10


def test_order_and_offset_ladders_pass_for_constant_integrand() -> None:
    plan, manifest = _plan_and_manifest()
    payload = {
        "point_results": [
            _point("spm", label, n, -0.2 if n == 0 else -0.1)
            for label in manifest.labels
            for n in (0, 1)
        ]
    }
    results, unresolved = aggregate_certified_outer_q(
        sweet_spot_payload=payload,
        plan=plan,
        manifest=manifest,
        pairings=["spm"],
        matsubara_indices=[0, 1],
        temperature_K=10.0,
    )
    assert unresolved == []
    comparisons = compare_ladders(
        plan=plan,
        config_results=results,
        pairings=["spm"],
        absolute_tolerance_J_m2=1e-18,
        relative_tolerance=1e-12,
    )
    assert comparisons["radial"]["spm"]["final_transition_passed"] is True
    assert comparisons["angular"]["spm"]["final_transition_passed"] is True
    assert comparisons["offset"]["spm"]["all_passed"] is True
    assert comparisons["cutoff"]["spm"]["final_transition_passed"] is False
