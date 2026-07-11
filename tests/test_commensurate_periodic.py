from __future__ import annotations

import numpy as np
import pytest

from validation.lib.commensurate_periodic import (
    CommensuratePeriodicGrid,
    integrate_commensurate_periodic_vector,
)


def test_commensurate_grid_constructs_integer_q_and_complete_chunks():
    grid = CommensuratePeriodicGrid(
        nk=8,
        mx=2,
        my=-1,
        shift_x=0.25,
        shift_y=0.75,
        max_points=64,
    )
    assert np.allclose(grid.q_model / grid.step, [2.0, -1.0], rtol=0.0, atol=0.0)
    assert grid.shifted_index(7, 0) == (1, 7)
    chunks = list(grid.iter_point_chunks(13))
    points = np.concatenate(chunks, axis=0)
    assert points.shape == (64, 2)
    assert len(chunks) == 5
    assert np.unique(points, axis=0).shape == points.shape
    assert grid.translation_permutation_exact


def test_commensurate_periodic_vector_integral_uses_equal_weight_average():
    grid = CommensuratePeriodicGrid(nk=12, mx=3, my=2, max_points=144)

    def evaluator(points: np.ndarray) -> np.ndarray:
        kx = points[:, 0]
        ky = points[:, 1]
        return np.column_stack(
            (
                np.ones(points.shape[0]),
                np.exp(1j * kx),
                np.exp(1j * ky),
                np.cos(kx) ** 2 + np.sin(ky) ** 2,
            )
        )

    result = integrate_commensurate_periodic_vector(grid, evaluator, chunk_size=17)
    assert result.point_evaluations == 144
    assert result.chunks == 9
    assert np.allclose(result.value, [1.0, 0.0, 0.0, 1.0], rtol=0.0, atol=2e-15)
    assert "kahan" in result.summation_method


def test_commensurate_periodic_translation_is_an_index_permutation():
    grid = CommensuratePeriodicGrid(nk=11, mx=3, my=2, max_points=121)
    shifted = {
        grid.shifted_index(ix, iy)
        for ix in range(grid.nk)
        for iy in range(grid.nk)
    }
    original = {(ix, iy) for ix in range(grid.nk) for iy in range(grid.nk)}
    assert shifted == original


def test_commensurate_grid_fails_closed_on_budget_and_invalid_q():
    with pytest.raises(RuntimeError, match="exceeded max_points"):
        CommensuratePeriodicGrid(nk=20, mx=1, my=1, max_points=399)
    with pytest.raises(ValueError, match="at least one"):
        CommensuratePeriodicGrid(nk=8, mx=0, my=0, max_points=64)
    with pytest.raises(ValueError, match="principal periodic range"):
        CommensuratePeriodicGrid(nk=8, mx=5, my=0, max_points=64)


def test_commensurate_periodic_audit_runner_imports():
    import validation.run_dwave_static_commensurate_periodic_audit as runner

    assert callable(runner.main)
