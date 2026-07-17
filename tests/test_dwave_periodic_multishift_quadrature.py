from __future__ import annotations

import numpy as np
import pytest

from lno327 import uniform_bz_mesh
from lno327.workflows.dwave_periodic_multishift_quadrature import (
    DWavePeriodicMultishiftOptions,
    build_dwave_periodic_multishift_quadrature,
)


def test_shift_order_one_matches_uniform_midpoint_mesh():
    points, weights, metadata = build_dwave_periodic_multishift_quadrature(
        np.array([0.03, 0.02]),
        DWavePeriodicMultishiftOptions(base_nk=4, shift_order=1),
    )
    expected = uniform_bz_mesh(4)
    assert points.shape == expected.shape
    assert np.allclose(points, expected, rtol=0.0, atol=1e-14)
    assert np.allclose(weights, np.full(16, 1.0 / 16.0), rtol=0.0, atol=1e-15)
    assert metadata["full_periodic_lattice_per_shift"] is True
    assert metadata["local_cell_refinement"] is False


def test_gauss_multishift_is_normalized_and_grouped_into_complete_lattices():
    base_nk = 3
    order = 2
    points, weights, metadata = build_dwave_periodic_multishift_quadrature(
        np.array([0.03, 0.02]),
        DWavePeriodicMultishiftOptions(base_nk=base_nk, shift_order=order),
    )
    assert points.shape == (base_nk**2 * order**2, 2)
    assert weights.shape == (len(points),)
    assert np.all(weights > 0.0)
    assert np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=2e-12)
    assert metadata["num_grid_shifts"] == order**2
    assert metadata["num_points_per_shift"] == base_nk**2
    assert metadata["primitive_merge_before_schur_required"] is True
    block = base_nk**2
    for index in range(order**2):
        block_weights = weights[index * block : (index + 1) * block]
        assert np.allclose(block_weights, block_weights[0], rtol=0.0, atol=1e-16)


def test_multishift_point_budget_is_fail_closed():
    with pytest.raises(RuntimeError, match="max_quadrature_points"):
        build_dwave_periodic_multishift_quadrature(
            np.array([0.03, 0.02]),
            DWavePeriodicMultishiftOptions(
                base_nk=10,
                shift_order=3,
                max_quadrature_points=899,
            ),
        )
