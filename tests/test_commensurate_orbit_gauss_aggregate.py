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
    assert result.point_evaluations == 24 * 24
    assert calls == [24] * 24
    assert np.allclose(result.value, expected, rtol=1e-12, atol=1e-12)


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
            max_point_evaluations=575,
        )

    assert caught.value.maximum == 575
    assert caught.value.attempted == 576
