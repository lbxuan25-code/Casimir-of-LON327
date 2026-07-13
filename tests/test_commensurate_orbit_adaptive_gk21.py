from __future__ import annotations

import numpy as np
import pytest

from validation.lib.commensurate_orbit_adaptive_gk21 import (
    CompleteOrbitAggregateWorkspace,
    GK21_ROOT_NODES,
    TransverseEvaluationBudgetExceeded,
    integrate_commensurate_orbit_adaptive_gk21,
)


def test_gk21_root_nodes_are_complete_and_symmetric() -> None:
    assert GK21_ROOT_NODES.shape == (21,)
    assert np.all(np.diff(GK21_ROOT_NODES) > 0.0)
    assert GK21_ROOT_NODES[10] == 0.0
    assert np.allclose(GK21_ROOT_NODES, -GK21_ROOT_NODES[::-1])


def test_complete_orbit_workspace_caches_and_enforces_unique_budget() -> None:
    seen_shapes: list[tuple[int, int]] = []

    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        seen_shapes.append(points.shape)
        assert np.isclose(np.sum(weights), 1.0)
        return np.asarray([np.sum(weights)], dtype=complex)

    workspace = CompleteOrbitAggregateWorkspace(
        evaluator=evaluator,
        nk=8,
        mx=1,
        my=1,
        max_unique_transverse_evaluations=1,
    )
    first = workspace.evaluate_phase(0.125)
    cached = workspace.evaluate_phase(0.125)
    assert np.array_equal(first, cached)
    assert workspace.transverse_evaluations_unique == 1
    assert workspace.cache_hits == 1
    assert workspace.point_evaluations == workspace.points_per_t
    assert seen_shapes == [(workspace.points_per_t, 2)]
    with pytest.raises(TransverseEvaluationBudgetExceeded):
        workspace.evaluate_phase(0.375)


def test_constant_complex_vector_uses_shared_root_cache() -> None:
    target = np.asarray([1.25 + 0.5j, -0.75 + 2.0j])

    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        assert points.ndim == 2 and points.shape[1] == 2
        assert np.isclose(np.sum(weights), 1.0)
        return target

    result = integrate_commensurate_orbit_adaptive_gk21(
        evaluator,
        nk=8,
        mx=1,
        my=0,
        max_unique_transverse_evaluations=64,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    assert result.success
    assert result.primary.success
    assert result.audit is not None and result.audit.success
    assert result.primitive_group_agreement_passed
    assert np.allclose(result.primary.value, target, rtol=1e-12, atol=1e-12)
    assert np.allclose(result.audit.value, target, rtol=1e-12, atol=1e-12)
    assert result.transverse_evaluations == 63
    assert result.cache_hits >= 84
    assert result.point_evaluations == result.chunk_size * 63
    assert result.primary.integral_error_ratio <= 1.0
    assert result.audit.integral_error_ratio <= 1.0


def test_periodic_complex_mode_has_correct_bz_average() -> None:
    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # For m=(1,0), the complete orbit runs along kx and all points in one
        # transverse sample share ky=t.  The BZ average of exp(i t) is zero.
        mode = np.exp(1j * points[:, 1])
        return np.asarray([np.sum(weights * mode), 2.0 - 3.0j])

    result = integrate_commensurate_orbit_adaptive_gk21(
        evaluator,
        nk=12,
        mx=1,
        my=0,
        max_unique_transverse_evaluations=128,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    assert result.success
    assert result.value is not None
    assert abs(result.value[0]) <= 1e-11
    assert result.value[1] == pytest.approx(2.0 - 3.0j, rel=1e-11, abs=1e-11)


def test_budget_failure_is_structured_before_a_complete_primary_result() -> None:
    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.asarray([1.0 + 0.0j])

    result = integrate_commensurate_orbit_adaptive_gk21(
        evaluator,
        nk=8,
        mx=1,
        my=0,
        max_unique_transverse_evaluations=20,
    )
    assert not result.success
    assert result.primary.value is None
    assert result.failure_reason == "transverse_evaluation_budget_exceeded"
    assert result.transverse_evaluations == 20
    assert result.point_evaluations == result.chunk_size * 20


def test_canonical_dwave_groups_keep_ward_rhs_as_monitor() -> None:
    from validation.lib.commensurate_orbit_groups import group_layout

    width = 18 + 2 * 25
    ids, names, weights = group_layout(
        width,
        component_group_ids=None,
        group_names=None,
        group_control_weights=None,
    )
    assert ids.shape == (width,)
    assert names[:3] == ("em_direct", "collective_static", "ward_rhs_monitor")
    assert weights[2] == 0.0
    assert np.all(weights[np.arange(weights.size) != 2] > 0.0)
