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
    assert result.point_evaluations <= 20_000
    assert result.orbit_shift_steps == 2
    assert result.orbit_origins == (0.5,)
    assert np.allclose(result.value, expected, rtol=2e-7, atol=2e-8)
