from __future__ import annotations

import numpy as np
import pytest

from validation.lib.commensurate_orbit_panel_adaptive_v3 import (
    PanelState,
    _split_children_with_parent_history,
    integrate_commensurate_orbit_panel_adaptive,
)


def test_split_children_retain_parent_error_and_observed_scales() -> None:
    parent = PanelState(
        left=-1.0,
        right=1.0,
        depth=0,
        order=33,
        value=np.asarray([1.0 + 0.0j, 2.0 + 0.0j]),
        group_errors=np.asarray([0.20, 0.40]),
        group_point_scales=np.asarray([5.0, 6.0]),
    )
    left_raw = PanelState(
        left=-1.0,
        right=0.0,
        depth=1,
        order=9,
        value=np.asarray([0.48 + 0.0j, 0.90 + 0.0j]),
        group_errors=np.asarray([1.20, 0.10]),
        group_point_scales=np.asarray([2.0, 7.0]),
    )
    right_raw = PanelState(
        left=0.0,
        right=1.0,
        depth=1,
        order=9,
        value=np.asarray([0.50 + 0.0j, 1.05 + 0.0j]),
        group_errors=np.asarray([0.80, 0.50]),
        group_point_scales=np.asarray([4.0, 3.0]),
    )

    left, right = _split_children_with_parent_history(
        parent,
        left_raw,
        right_raw,
        group_ids=np.asarray([0, 1]),
        norm="max",
    )

    split_discrepancy = np.asarray([0.02, 0.05])
    raw_sum = left_raw.group_errors + right_raw.group_errors
    expected_envelope = np.maximum.reduce(
        (
            parent.group_errors,
            2.0 * split_discrepancy,
            0.25 * raw_sum,
        )
    )
    np.testing.assert_allclose(
        left.group_errors + right.group_errors,
        expected_envelope,
        rtol=0.0,
        atol=1e-15,
    )
    assert np.all(expected_envelope >= parent.group_errors)
    np.testing.assert_allclose(left.group_point_scales, [5.0, 7.0])
    np.testing.assert_allclose(right.group_point_scales, [5.0, 7.0])


def test_v3_constant_integral_preserves_shared_state_and_full_period() -> None:
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
    assert result.quadrature == "nested_clenshaw_curtis_5_9_17_33_split_history"
    assert result.full_transverse_period_integrated
    assert not result.symmetry_reduction_applied
    assert not result.q_direction_special_case
    np.testing.assert_allclose(result.primary.value, target, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.audit.value, target, rtol=1e-12, atol=1e-12)
    assert result.primary.unique_evaluations == result.audit.unique_evaluations
