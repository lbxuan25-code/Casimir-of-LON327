from __future__ import annotations

import numpy as np

from lno327.casimir.material_geometry_outer_qualification import (
    qualify_fixed_outer_geometry_replay,
)
from lno327.casimir.material_geometry_qualification import GeometryEquivalencePolicy
from lno327.casimir.outer_quadrature import build_outer_q_polar_grid


def _grid():
    return build_outer_q_polar_grid(
        separation_m=100e-9,
        lattice_a_x_m=3.87e-10,
        lattice_a_y_m=3.87e-10,
        u_max=4.0,
        radial_order=2,
        angular_order=4,
        angular_offset_fraction=0.5,
    )


def test_identical_logdet_arrays_survive_fixed_outer_reduction_exactly() -> None:
    grid = _grid()
    values = -np.linspace(0.01, 0.16, 2 * grid.node_count).reshape(2, grid.node_count)
    report = qualify_fixed_outer_geometry_replay(
        reference_logdet_by_n_and_node=values,
        candidate_logdet_by_n_and_node=values.copy(),
        matsubara_indices=(0, 1),
        temperature_K=40.0,
        grid=grid,
    )

    assert report.passed is True
    assert report.node_comparison["failed_node_count"] == 0
    assert report.total_comparison["absolute"] == 0.0
    assert report.reference.total_J_m2 == report.candidate.total_J_m2
    assert report.production_casimir_allowed is False


def test_fixed_outer_replay_fails_when_one_point_exceeds_policy() -> None:
    grid = _grid()
    reference = -np.full((1, grid.node_count), 0.1)
    candidate = reference.copy()
    candidate[0, 3] -= 0.01
    report = qualify_fixed_outer_geometry_replay(
        reference_logdet_by_n_and_node=reference,
        candidate_logdet_by_n_and_node=candidate,
        matsubara_indices=(1,),
        temperature_K=40.0,
        grid=grid,
        policy=GeometryEquivalencePolicy(
            absolute_tolerance=1e-15,
            relative_tolerance=1e-8,
        ),
    )

    assert report.passed is False
    assert report.node_comparison["failed_node_count"] == 1
    assert report.total_comparison["passed"] is False
