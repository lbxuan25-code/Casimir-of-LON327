from __future__ import annotations

import numpy as np
import pytest

from validation.lib.commensurate_orbit_gauss_aggregate import (
    integrate_commensurate_orbit_gauss_aggregate,
)
from validation.lib.dwave_commensurate_orbit_gauss import (
    OrbitEvaluationBudgetExceeded,
)


def test_commensurate_orbit_gauss_aggregate_integrates_periodic_vector():
    calls: list[int] = []

    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        calls.append(int(points.shape[0]))
        assert points.shape == (24, 2)
        assert weights.shape == (24,)
        assert np.isclose(np.sum(weights), 1.0)
        kx = points[:, 0]
        ky = points[:, 1]
        values = np.column_stack(
            (
                np.ones(points.shape[0], dtype=complex),
                np.cos(kx),
                np.sin(ky),
                np.cos(kx) ** 2,
                np.exp(1j * (kx + ky)),
            )
        )
        return np.sum(weights[:, None] * values, axis=0)

    result = integrate_commensurate_orbit_gauss_aggregate(
        evaluator,
        nk=24,
        mx=2,
        my=0,
        transverse_order=24,
        shift_s=0.5,
        subgrid_average="auto",
        max_point_evaluations=20_000,
    )

    expected = np.asarray([1.0, 0.0, 0.0, 0.5, 0.0], dtype=complex)
    assert result.success
    assert result.transverse_evaluations == 24
    assert result.panel_count == 1
    assert result.panel_order == 24
    assert result.integration_start == pytest.approx(-np.pi)
    assert result.quadrature == "fixed_gauss_legendre"
    assert result.full_transverse_period_integrated
    assert not result.symmetry_reduction_applied
    assert not result.q_direction_special_case
    assert result.point_evaluations == 24 * 24
    assert calls == [24] * 24
    assert np.allclose(result.value, expected, rtol=1e-12, atol=1e-12)


def test_composite_gauss_integrates_full_asymmetric_period_for_shifted_cuts():
    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        kx = points[:, 0]
        ky = points[:, 1]
        values = np.column_stack(
            (
                np.ones(points.shape[0], dtype=complex),
                np.exp(0.3 * np.cos(kx) + 0.2 * np.sin(ky)),
                np.sin(kx + 2.0 * ky) + 0.4 * np.cos(2.0 * kx - ky),
                np.exp(1j * (2.0 * kx + ky)),
            )
        )
        return np.sum(weights[:, None] * values, axis=0)

    common = dict(
        evaluator=evaluator,
        nk=24,
        mx=3,
        my=2,
        transverse_order=64,
        panel_count=4,
        shift_s=0.5,
        subgrid_average="auto",
        max_point_evaluations=10_000,
    )
    fixed = integrate_commensurate_orbit_gauss_aggregate(
        integration_start=-np.pi,
        **common,
    )
    shifted = integrate_commensurate_orbit_gauss_aggregate(
        integration_start=-np.pi + 0.371,
        **common,
    )

    assert fixed.panel_count == shifted.panel_count == 4
    assert fixed.panel_order == shifted.panel_order == 16
    assert fixed.transverse_evaluations == shifted.transverse_evaluations == 64
    assert fixed.quadrature == shifted.quadrature == "composite_fixed_gauss_legendre"
    assert fixed.point_evaluations == shifted.point_evaluations == 2 * 24 * 64
    assert np.allclose(fixed.value, shifted.value, rtol=2e-12, atol=2e-12)
    assert fixed.value[0] == pytest.approx(1.0, abs=2e-14)


def test_composite_gauss_requires_total_order_divisible_by_panel_count():
    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        raise AssertionError("invalid composite rule must fail before evaluation")

    with pytest.raises(ValueError, match="divisible"):
        integrate_commensurate_orbit_gauss_aggregate(
            evaluator,
            nk=24,
            mx=2,
            my=0,
            transverse_order=50,
            panel_count=4,
            max_point_evaluations=20_000,
        )


def test_commensurate_orbit_gauss_aggregate_rejects_incomplete_order_budget():
    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        raise AssertionError("evaluator must not run when the full order exceeds budget")

    with pytest.raises(OrbitEvaluationBudgetExceeded) as caught:
        integrate_commensurate_orbit_gauss_aggregate(
            evaluator,
            nk=24,
            mx=2,
            my=0,
            transverse_order=24,
            panel_count=4,
            max_point_evaluations=575,
        )

    assert caught.value.maximum == 575
    assert caught.value.attempted == 576
