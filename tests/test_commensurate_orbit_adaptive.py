from __future__ import annotations

import numpy as np

from validation.lib.commensurate_orbit_adaptive import (
    integrate_commensurate_orbit_adaptive_aggregate,
)


def test_commensurate_orbit_adaptive_integrates_periodic_vector():
    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
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

    result = integrate_commensurate_orbit_adaptive_aggregate(
        evaluator,
        nk=24,
        mx=2,
        my=0,
        shift_s=0.5,
        subgrid_average="auto",
        max_point_evaluations=20_000,
        pilot_order=8,
        epsabs=1e-8,
        epsrel=1e-8,
        limit=40,
        quadrature="gk15",
        norm="max",
    )

    expected = np.asarray([1.0, 0.0, 0.0, 0.5, 0.0], dtype=complex)
    assert result.success
    assert result.strategy == "periodic_nested"
    assert result.quadrature == "periodic_trapezoid"
    assert result.point_evaluations == result.final_transverse_order * 24
    assert result.transverse_evaluations == result.final_transverse_order
    assert result.order_history == (8, 16, 32)
    assert result.orbit_shift_steps == 2
    assert result.orbit_origins == (0.5,)
    assert np.allclose(result.value, expected, rtol=2e-7, atol=2e-8)


def test_block_scaling_does_not_promote_tiny_oscillatory_component():
    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        ky = points[:, 1]
        values = np.column_stack(
            (
                np.ones(points.shape[0], dtype=complex),
                1e-12 * np.cos(127.0 * ky),
            )
        )
        return np.sum(weights[:, None] * values, axis=0)

    result = integrate_commensurate_orbit_adaptive_aggregate(
        evaluator,
        nk=24,
        mx=2,
        my=0,
        max_point_evaluations=20_000,
        pilot_order=8,
        epsabs=1e-6,
        epsrel=1e-4,
        component_group_ids=[0, 0],
        group_names=["physical_block"],
        group_control_weights=[1.0],
    )

    assert result.success
    assert result.final_transverse_order == 32
    assert result.point_evaluations == 32 * 24
    assert abs(result.value[0] - 1.0) < 1e-12
    assert abs(result.value[1]) < 1e-11


def test_budget_stops_before_incomplete_nested_level_and_returns_diagnostic():
    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        ky = points[:, 1]
        values = np.column_stack(
            (
                np.ones(points.shape[0], dtype=complex),
                np.cos(13.0 * ky),
            )
        )
        return np.sum(weights[:, None] * values, axis=0)

    result = integrate_commensurate_orbit_adaptive_aggregate(
        evaluator,
        nk=24,
        mx=2,
        my=0,
        max_point_evaluations=24 * 20,
        pilot_order=8,
        epsabs=0.0,
        epsrel=0.0,
        required_consecutive_levels=2,
    )

    assert not result.success
    assert result.status == 2
    assert result.final_transverse_order == 16
    assert result.point_evaluations == 16 * 24
    assert result.order_history == (8, 16)
    assert "last complete level" in result.message
