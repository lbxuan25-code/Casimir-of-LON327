from __future__ import annotations

import numpy as np
import pytest

from lno327 import uniform_bz_mesh
from lno327.workflows.dwave_periodic_multishift_quadrature import (
    DWavePeriodicMultishiftOptions,
    build_dwave_periodic_multishift_quadrature,
)
from validation.run_dwave_static_periodic_multishift_scan import _run_task


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


def test_periodic_multishift_runner_smoke_reports_ward_fields():
    row = _run_task(
        {
            "base_nk": 4,
            "shift_order": 1,
            "qx": 0.03,
            "qy": 0.02,
            "temperature_K": 10.0,
            "delta0_eV": 0.1,
            "eta_eV": 1e-8,
            "max_quadrature_points": 100,
            "ward_tolerance": 1e-6,
            "ward_absolute_tolerance": 1e-12,
            "condition_max": 1e12,
            "raw_longitudinal_ceiling": 10.0,
            "longitudinal_tolerance": 1e-7,
            "mixing_tolerance": 10.0,
            "reality_tolerance": 10.0,
            "passivity_tolerance": 10.0,
            "separation_nm": 20.0,
        }
    )
    assert row["base_nk"] == 4
    assert row["shift_order"] == 1
    assert row["num_quadrature_points"] == 16
    assert np.isfinite(row["chi_bar"])
    assert np.isfinite(row["dbar_t"])
    assert "ward_primitive_mixed_ratio_max" in row
    assert "ward_effective_mixed_ratio_max" in row
    assert "projection_eligible" in row
