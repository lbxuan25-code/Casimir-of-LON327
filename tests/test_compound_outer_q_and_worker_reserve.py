from __future__ import annotations

import numpy as np

from lno327.casimir.compound_outer_quadrature import build_compound_outer_q_polar_grid
from lno327.casimir.outer_quadrature import integrate_outer_q
from validation.lib import transverse_point_sweet_spot_engine as sweet_engine
from validation.lib.microscopic_outer_q_preflight import (
    build_staged_grid_plan,
    build_union_node_manifest,
)


def _grid(edges, panel_order=4, angular=8):
    return build_compound_outer_q_polar_grid(
        separation_m=20e-9,
        lattice_a_x_m=3.754e-10,
        lattice_a_y_m=3.754e-10,
        radial_panel_edges=edges,
        radial_panel_order=panel_order,
        angular_order=angular,
        angular_offset_fraction=0.5,
    )


def test_compound_grid_preserves_exact_disk_measure() -> None:
    grid = _grid([0.0, 6.0, 10.0, 14.0])
    actual = integrate_outer_q(np.ones(grid.node_count), grid)
    assert np.isclose(actual, grid.disk_measure_m_inv2, rtol=3e-15, atol=0.0)
    assert grid.metadata["radial_panel_edges"] == [0.0, 6.0, 10.0, 14.0]
    assert grid.metadata["radial_panel_count"] == 3
    assert grid.radial_order == 12


def test_compound_cutoff_increment_is_exact_added_annulus() -> None:
    lower = _grid([0.0, 6.0, 10.0, 14.0])
    upper = _grid([0.0, 6.0, 10.0, 14.0, 18.0])
    lower_value = integrate_outer_q(np.ones(lower.node_count), lower)
    upper_value = integrate_outer_q(np.ones(upper.node_count), upper)
    expected = upper.disk_measure_m_inv2 - lower.disk_measure_m_inv2
    assert np.isclose(upper_value - lower_value, expected, rtol=5e-15, atol=0.0)


def test_cutoff_ladder_reuses_all_earlier_nodes() -> None:
    plan = build_staged_grid_plan(
        u_max_values=[6.0, 10.0, 14.0, 18.0],
        radial_orders=[4, 6],
        angular_orders=[8, 16],
        angular_offsets=[0.0, 0.5],
    )
    manifest = build_union_node_manifest(
        plan,
        separation_m=20e-9,
        lattice_a_x_m=3.754e-10,
        lattice_a_y_m=3.754e-10,
    )
    cutoff_ids = plan.ladders["cutoff"]
    for left_id, right_id in zip(cutoff_ids[:-1], cutoff_ids[1:], strict=True):
        assert set(manifest.labels_by_spec[left_id]) < set(manifest.labels_by_spec[right_id])
    reference = next(spec for spec in plan.specs if spec.spec_id == plan.reference_spec_id)
    assert reference.radial_panel_edges == (0.0, 6.0, 10.0, 14.0, 18.0)
    assert reference.radial_panel_order == 6


def _parse_workers(extra=None):
    values = [
        "--q-point", "q", "0.01", "0.02",
        "--N-candidates", "128", "192", "256",
    ]
    values.extend(extra or [])
    return sweet_engine._parse_args(values)


def test_automatic_worker_budget_reserves_two_affinity_cpus(monkeypatch) -> None:
    monkeypatch.setattr(sweet_engine, "affinity_cpu_count", lambda: 32)
    monkeypatch.delenv("LNO327_CPU_RESERVE", raising=False)
    args = _parse_workers()
    assert args.workers == 30
    assert args.reserved_affinity_cpus == 2


def test_worker_reserve_environment_and_explicit_workers(monkeypatch) -> None:
    monkeypatch.setattr(sweet_engine, "affinity_cpu_count", lambda: 32)
    monkeypatch.setenv("LNO327_CPU_RESERVE", "1")
    assert _parse_workers().workers == 31
    explicit = _parse_workers(["--workers", "12"])
    assert explicit.workers == 12
    assert explicit.worker_budget_source == "explicit_workers"
