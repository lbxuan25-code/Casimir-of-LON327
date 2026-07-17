from __future__ import annotations

import numpy as np
import pytest

from validation.lib.commensurate_orbit_panel_adaptive import (
    clenshaw_curtis_rule,
    integrate_commensurate_orbit_panel_adaptive,
)


@pytest.mark.parametrize("order", [5, 9, 17, 33])
def test_clenshaw_curtis_rules_are_normalized_and_nested(order: int) -> None:
    nodes, weights = clenshaw_curtis_rule(order)
    assert nodes.shape == (order,)
    assert weights.shape == (order,)
    assert np.all(np.diff(nodes) < 0.0)
    assert np.sum(weights) == pytest.approx(2.0, rel=1e-14, abs=1e-14)
    assert np.dot(weights, nodes) == pytest.approx(0.0, abs=1e-14)
    assert np.dot(weights, nodes * nodes) == pytest.approx(
        2.0 / 3.0,
        rel=1e-13,
        abs=1e-13,
    )

    if order > 5:
        previous, _ = clenshaw_curtis_rule((order + 1) // 2)
        np.testing.assert_allclose(nodes[::2], previous, rtol=0.0, atol=2e-15)


def test_constant_complex_vector_primary_and_audit_share_one_panel_state() -> None:
    target = np.asarray([1.25 + 0.5j, -0.75 + 2.0j])

    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        assert points.ndim == 2 and points.shape[1] == 2
        assert np.sum(weights) == pytest.approx(1.0)
        return target

    result = integrate_commensurate_orbit_panel_adaptive(
        evaluator,
        nk=8,
        mx=1,
        my=0,
        max_unique_transverse_evaluations=128,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    assert result.success
    assert result.primary.success
    assert result.audit is not None and result.audit.success
    assert result.primitive_group_agreement_passed
    np.testing.assert_allclose(result.primary.value, target, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.audit.value, target, rtol=1e-12, atol=1e-12)
    assert result.primary.unique_evaluations == result.audit.unique_evaluations
    assert result.transverse_evaluations == 64
    assert result.initial_panel_count == 8
    assert result.pilot_count == 16
    assert not result.refinement_trace
    assert result.point_evaluations == result.chunk_size * result.transverse_evaluations


def test_periodic_localized_function_refines_without_q_specific_dispatch() -> None:
    concentration = 0.95
    center = 2.17
    expected = 1.0 / np.sqrt(1.0 - concentration * concentration)

    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        transverse = points[:, 1]
        localized = 1.0 / (1.0 - concentration * np.cos(transverse - center))
        return np.asarray([np.sum(weights * localized)], dtype=complex)

    result = integrate_commensurate_orbit_panel_adaptive(
        evaluator,
        nk=12,
        mx=1,
        my=0,
        max_unique_transverse_evaluations=256,
        epsabs=2e-7,
        epsrel=2e-6,
    )
    assert result.success
    assert result.value is not None
    assert result.value[0].real == pytest.approx(expected, rel=2e-6, abs=2e-7)
    assert abs(result.value[0].imag) <= 1e-13
    assert result.audit is not None
    assert result.audit.panel_count >= result.primary.panel_count
    assert result.transverse_evaluations <= 256
    assert result.refinement_trace
    assert {entry.stage for entry in result.refinement_trace} <= {"primary", "audit"}
    assert all(entry.required_new_nodes > 0 for entry in result.refinement_trace)
    assert result.full_transverse_period_integrated
    assert not result.symmetry_reduction_applied
    assert not result.q_direction_special_case


def test_budget_stops_before_complete_refinement_and_keeps_finite_snapshot() -> None:
    concentration = 0.99
    center = 2.17

    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        transverse = points[:, 1]
        localized = 1.0 / (1.0 - concentration * np.cos(transverse - center))
        return np.asarray([np.sum(weights * localized)], dtype=complex)

    result = integrate_commensurate_orbit_panel_adaptive(
        evaluator,
        nk=12,
        mx=1,
        my=0,
        max_unique_transverse_evaluations=64,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    assert not result.success
    assert not result.primary.success
    assert result.primary.value is not None
    assert np.isfinite(result.primary.value.real).all()
    assert np.isfinite(result.primary.integral_error_ratio)
    assert result.transverse_evaluations == 64
    assert "panel_boundary_transverse_budget_exceeded" in result.failure_reason
    assert "minimum_required_new=8" in result.failure_reason
    assert result.primary.panel_count == 8
    assert result.primary.refinement_steps == 0


def test_budget_aware_scheduler_uses_last_eight_node_operation() -> None:
    concentration = 0.99
    center = 2.17

    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        transverse = points[:, 1]
        localized = 1.0 / (1.0 - concentration * np.cos(transverse - center))
        return np.asarray([np.sum(weights * localized)], dtype=complex)

    result = integrate_commensurate_orbit_panel_adaptive(
        evaluator,
        nk=12,
        mx=1,
        my=0,
        max_unique_transverse_evaluations=72,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    assert not result.success
    assert result.transverse_evaluations == 72
    assert len(result.refinement_trace) == 1
    entry = result.refinement_trace[0]
    assert entry.operation == "p_refine"
    assert entry.old_order == 9
    assert entry.required_new_nodes == 8
    assert entry.unique_evaluations_after == 72


def test_full_period_cut_does_not_assume_even_or_c4_symmetry() -> None:
    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        transverse = points[:, 1]
        asymmetric = 1.0 + 0.2 * np.sin(transverse) + 0.1 * np.sin(
            2.0 * transverse + 0.37
        )
        return np.asarray([np.sum(weights * asymmetric)], dtype=complex)

    result = integrate_commensurate_orbit_panel_adaptive(
        evaluator,
        nk=12,
        mx=1,
        my=0,
        max_unique_transverse_evaluations=192,
        epsabs=1e-7,
        epsrel=1e-7,
    )
    assert result.success
    assert result.value is not None
    assert result.value[0] == pytest.approx(1.0 + 0.0j, rel=1e-7, abs=1e-7)
    assert result.full_transverse_period_integrated
    assert not result.symmetry_reduction_applied
    assert not result.q_direction_special_case
    pilot_grid = -np.pi + 2.0 * np.pi * np.arange(16) / 16.0
    assert np.min(np.abs(pilot_grid - result.integration_start)) <= 1e-14


def test_zero_weight_monitor_does_not_drive_cut_or_refinement() -> None:
    def run(monitor_frequency: float):
        def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
            transverse = points[:, 1]
            control = np.sum(weights * (2.0 + np.cos(transverse)))
            monitor = np.sum(weights * np.exp(1j * monitor_frequency * transverse))
            return np.asarray([control, monitor], dtype=complex)

        return integrate_commensurate_orbit_panel_adaptive(
            evaluator,
            nk=12,
            mx=1,
            my=0,
            max_unique_transverse_evaluations=256,
            epsabs=1e-9,
            epsrel=1e-9,
            component_group_ids=np.asarray([0, 1]),
            group_names=("control", "monitor"),
            group_control_weights=np.asarray([1.0, 0.0]),
        )

    first = run(97.0)
    second = run(131.0)
    assert first.success and second.success
    assert first.control_group_names == ("control",)
    assert first.monitor_group_names == ("monitor",)
    assert first.value is not None and second.value is not None
    assert first.value[0] == pytest.approx(2.0 + 0.0j, rel=1e-9, abs=1e-9)
    assert second.value[0] == pytest.approx(2.0 + 0.0j, rel=1e-9, abs=1e-9)
    assert first.integration_start == second.integration_start
    assert [
        (entry.panel, entry.old_order, entry.operation, entry.required_new_nodes)
        for entry in first.refinement_trace
    ] == [
        (entry.panel, entry.old_order, entry.operation, entry.required_new_nodes)
        for entry in second.refinement_trace
    ]
