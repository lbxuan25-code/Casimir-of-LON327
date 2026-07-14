from __future__ import annotations

import numpy as np

from lno327.workflows.arbitrary_q_vector_adaptive import (
    ArbitraryQVectorAdaptiveOptions,
    build_hierarchical_material_node_cache,
    integrate_arbitrary_q_vector_adaptive,
)
from lno327.workflows.arbitrary_q_vector_adaptive_cached import (
    build_reusable_hierarchical_material_node_cache,
    integrate_arbitrary_q_vector_adaptive_cached,
)
from lno327.workflows.dwave_vector_adaptive_cubature import (
    cubature_cell_gauss_rule,
    initial_cubature_cells,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _state(pairing_name: str):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    reusable = build_reusable_hierarchical_material_node_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=10.0,
        eta_eV=1e-8,
    )
    return model, ansatz, pairing, reusable


def _options() -> ArbitraryQVectorAdaptiveOptions:
    return ArbitraryQVectorAdaptiveOptions(
        coarse_grid=2,
        low_order=2,
        high_order=3,
        relative_tolerance=1e9,
        absolute_tolerance=1e9,
        ward_error_tolerance=1e9,
        max_level=1,
        max_iterations=0,
        refine_fraction=0.5,
        min_refine_cells=1,
        max_cells=64,
        max_evaluation_points=2000,
        cell_batch_size=8,
    )


def test_reusable_counterterm_matches_established_ansatz_without_eigh_rebuild() -> None:
    model, ansatz, pairing, cache = _state("dwave")
    cells = initial_cubature_cells(2)
    rules = [cubature_cell_gauss_rule(cell, 3) for cell in cells]
    points = np.concatenate([item[0] for item in rules], axis=0)
    weights = np.concatenate([item[1] for item in rules], axis=0)
    cache.material_workspace(points, weights, include_counterterm=False)
    midpoint_calls = cache.midpoint_eigh_call_count
    value = cache.counterterm(points, weights)
    expected = ansatz.hs_counterterm(cache.config, points, weights, pairing)
    np.testing.assert_allclose(value, expected, atol=5e-12, rtol=5e-11)
    assert cache.midpoint_eigh_call_count == midpoint_calls
    assert cache.counterterm_shifted_eigh_call_count == 0
    assert cache.counterterm_q0_workspace_build_count == 1
    cached = cache.counterterm(points, weights)
    np.testing.assert_allclose(cached, value, atol=0.0, rtol=0.0)
    assert cache.counterterm_q0_workspace_build_count == 1
    assert cache.metadata()["counterterm_uses_cached_midpoint_eigensystems"] is True


def test_reusable_cached_backend_matches_base_and_reuses_nodes_across_q() -> None:
    model, ansatz, pairing, reusable = _state("spm")
    base_cache = build_hierarchical_material_node_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=10.0,
        eta_eV=1e-8,
    )
    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=np.asarray([0.071, 0.037]),
        adaptive_options=_options(),
    )
    base = integrate_arbitrary_q_vector_adaptive(node_cache=base_cache, **common)
    cached = integrate_arbitrary_q_vector_adaptive_cached(
        node_cache=reusable, **common
    )
    np.testing.assert_allclose(
        cached.packed_primitives,
        base.packed_primitives,
        atol=5e-12,
        rtol=5e-11,
    )
    midpoint_calls = reusable.midpoint_eigh_call_count
    counterterm_builds = reusable.counterterm_q0_workspace_build_count
    integrate_arbitrary_q_vector_adaptive_cached(
        node_cache=reusable,
        **{**common, "q_model": np.asarray([0.043, -0.061])},
    )
    assert reusable.midpoint_eigh_call_count == midpoint_calls
    assert reusable.counterterm_q0_workspace_build_count == counterterm_builds
    assert reusable.node_hits > 0
    assert reusable.counterterm_node_hits > 0
