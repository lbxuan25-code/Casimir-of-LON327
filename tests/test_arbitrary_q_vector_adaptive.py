from __future__ import annotations

from multiprocessing import get_all_start_methods

import numpy as np
import pytest

from lno327.response.primitive_kernel_v2 import evaluate_primitive_batch_from_material
from lno327.workflows.arbitrary_q_parallel import QLabAngleTask
from lno327.workflows.arbitrary_q_vector_adaptive import (
    AdaptiveConvergenceError,
    ArbitraryQVectorAdaptiveOptions,
    ArbitraryQVectorAdaptiveResponseCache,
    build_hierarchical_material_node_cache,
    integrate_arbitrary_q_vector_adaptive,
    integrate_two_plate_angle_batch_vector_adaptive,
)
from lno327.workflows.arbitrary_q_vector_adaptive_cached import (
    build_reusable_hierarchical_material_node_cache,
)
from lno327.workflows.arbitrary_q_vector_adaptive_parallel import (
    ArbitraryQVectorAdaptiveParallelEvaluator,
)
from lno327.workflows.dwave_vector_adaptive_cubature import (
    _tensor_gauss_reference,
    cubature_cell_gauss_rule,
    initial_cubature_cells,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _state(pairing_name: str = "spm", *, reusable: bool = False, **cache_kwargs):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    builder = (
        build_reusable_hierarchical_material_node_cache
        if reusable
        else build_hierarchical_material_node_cache
    )
    cache = builder(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=10.0,
        eta_eV=1e-8,
        **cache_kwargs,
    )
    return model, ansatz, pairing, cache


def _loose_options(*, cell_batch_size: int = 8) -> ArbitraryQVectorAdaptiveOptions:
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
        cell_batch_size=cell_batch_size,
    )


def _strict_nonconverged_options() -> ArbitraryQVectorAdaptiveOptions:
    return ArbitraryQVectorAdaptiveOptions(
        coarse_grid=2,
        low_order=2,
        high_order=3,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        ward_error_tolerance=0.0,
        max_level=0,
        max_iterations=0,
        min_refine_cells=1,
        max_cells=4,
        max_evaluation_points=100,
        cell_batch_size=4,
    )


@pytest.mark.parametrize("pairing_name", ["spm", "dwave"])
def test_initial_high_rule_matches_one_shot_shared_kernel(pairing_name: str) -> None:
    model, ansatz, pairing, cache = _state(pairing_name)
    q = np.asarray([0.071, 0.037])
    xi = np.asarray([0.0, 0.02])
    options = _loose_options(cell_batch_size=2)
    adaptive = integrate_arbitrary_q_vector_adaptive(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=q,
        adaptive_options=options,
        node_cache=cache,
    )
    cells = initial_cubature_cells(options.coarse_grid)
    rules = [cubature_cell_gauss_rule(cell, options.high_order) for cell in cells]
    points = np.concatenate([item[0] for item in rules], axis=0)
    weights = np.concatenate([item[1] for item in rules], axis=0)
    material = cache.material_workspace(points, weights, include_counterterm=True)
    one_shot = evaluate_primitive_batch_from_material(
        material, q, xi, include_counterterm=True
    )
    np.testing.assert_allclose(
        adaptive.packed_primitives, one_shot.packed, atol=5e-12, rtol=5e-11
    )
    assert adaptive.profile.converged
    assert adaptive.profile.counterterm_add_count == 1
    assert adaptive.metadata["primitive_vector_integrated_before_schur"] is True
    assert adaptive.metadata["low_high_rules_share_one_q_workspace_per_cell_batch"] is True
    assert adaptive.operator_ward.passed


def test_low_high_rules_share_one_q_workspace_per_cell_batch() -> None:
    model, ansatz, pairing, cache = _state("spm", reusable=True)
    result = integrate_arbitrary_q_vector_adaptive(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=np.asarray([0.071, 0.037]),
        adaptive_options=_loose_options(cell_batch_size=8),
        node_cache=cache,
    )
    assert result.profile.q_workspace_build_count == 1
    assert result.profile.shifted_eigensystem_build_count == 2
    assert result.profile.total_point_evaluations == 4 * (2**2 + 3**2)


def test_hierarchical_node_cache_reuses_midpoint_eigensystems_across_q() -> None:
    model, ansatz, pairing, cache = _state("spm", reusable=True)
    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        adaptive_options=_loose_options(),
        node_cache=cache,
    )
    first = integrate_arbitrary_q_vector_adaptive(
        q_model=np.asarray([0.071, 0.037]), **common
    )
    second = integrate_arbitrary_q_vector_adaptive(
        q_model=np.asarray([0.043, -0.061]), **common
    )
    assert first.profile.midpoint_eigensystem_build_count > 0
    assert second.profile.midpoint_eigensystem_build_count == 0
    assert second.profile.node_cache_misses == 0
    assert second.profile.node_cache_hits > 0
    assert second.profile.cache_totals_after_call["entries"] == first.profile.cache_totals_after_call["entries"]


def test_refinement_is_deterministic_across_cell_batch_sizes() -> None:
    model, ansatz, pairing, first_cache = _state("dwave", reusable=True)
    _, _, _, second_cache = _state("dwave", reusable=True)
    base = dict(
        coarse_grid=2,
        low_order=2,
        high_order=3,
        relative_tolerance=1e-16,
        absolute_tolerance=1e-18,
        ward_error_tolerance=1e-18,
        max_level=1,
        max_iterations=1,
        refine_fraction=0.5,
        min_refine_cells=1,
        max_cells=64,
        max_evaluation_points=2000,
    )
    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=np.asarray([0.071, 0.037]),
        require_converged=False,
    )
    first = integrate_arbitrary_q_vector_adaptive(
        adaptive_options=ArbitraryQVectorAdaptiveOptions(**base, cell_batch_size=1),
        node_cache=first_cache,
        **common,
    )
    second = integrate_arbitrary_q_vector_adaptive(
        adaptive_options=ArbitraryQVectorAdaptiveOptions(**base, cell_batch_size=8),
        node_cache=second_cache,
        **common,
    )
    np.testing.assert_allclose(
        first.packed_primitives, second.packed_primitives, atol=5e-12, rtol=5e-11
    )
    assert first.profile.iterations == second.profile.iterations == 1
    assert first.profile.accepted_cell_count == second.profile.accepted_cell_count
    assert first.profile.total_point_evaluations == second.profile.total_point_evaluations
    assert first.profile.q_workspace_build_count > second.profile.q_workspace_build_count


def test_nonconverged_strict_call_does_not_pollute_response_cache() -> None:
    model, ansatz, pairing, cache = _state("dwave", reusable=True)
    response_cache = ArbitraryQVectorAdaptiveResponseCache()
    with pytest.raises(AdaptiveConvergenceError):
        integrate_arbitrary_q_vector_adaptive(
            spec=model.spec,
            ansatz=ansatz,
            pairing=pairing,
            xi_eV_values=np.asarray([0.0, 0.02]),
            temperature_K=10.0,
            eta_eV=1e-8,
            q_model=np.asarray([0.071, 0.037]),
            adaptive_options=_strict_nonconverged_options(),
            node_cache=cache,
            response_cache=response_cache,
            require_converged=True,
        )
    assert response_cache.metadata()["entries"] == 0


def test_cached_nonconverged_diagnostic_cannot_bypass_strict_request() -> None:
    model, ansatz, pairing, cache = _state("dwave", reusable=True)
    response_cache = ArbitraryQVectorAdaptiveResponseCache()
    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=np.asarray([0.071, 0.037]),
        adaptive_options=_strict_nonconverged_options(),
        node_cache=cache,
        response_cache=response_cache,
    )
    diagnostic = integrate_arbitrary_q_vector_adaptive(
        require_converged=False, **common
    )
    assert not diagnostic.profile.converged
    assert response_cache.metadata()["entries"] == 1
    with pytest.raises(AdaptiveConvergenceError):
        integrate_arbitrary_q_vector_adaptive(require_converged=True, **common)


def test_two_plate_batch_reuses_exact_q_response() -> None:
    model, ansatz, pairing, cache = _state("spm", reusable=True)
    result = integrate_two_plate_angle_batch_vector_adaptive(
        q_lab=np.asarray([0.071, 0.037]),
        theta_1_rad=0.0,
        theta_2_rad_values=np.asarray([0.0, 0.2]),
        node_cache=cache,
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        adaptive_options=_loose_options(),
    )
    assert result.response_cache_metadata["hits"] == 1
    assert result.plate_2[0] is result.plate_1
    assert not np.array_equal(result.plate_2[1].q_model, result.plate_1.q_model)


def test_tensor_gauss_reference_is_cached_and_rules_are_nonembedded() -> None:
    _tensor_gauss_reference.cache_clear()
    cell = initial_cubature_cells(2)[0]
    cubature_cell_gauss_rule(cell, 2)
    cubature_cell_gauss_rule(cell, 2)
    info = _tensor_gauss_reference.cache_info()
    assert info.misses == 1
    assert info.hits >= 1
    low, _ = _tensor_gauss_reference(2)
    high, _ = _tensor_gauss_reference(3)
    assert not set(map(tuple, low)).issubset(set(map(tuple, high)))


def test_cache_node_budget_fails_closed() -> None:
    model, ansatz, pairing, cache = _state(
        "spm", reusable=True, max_cache_nodes=4
    )
    with pytest.raises(MemoryError):
        integrate_arbitrary_q_vector_adaptive(
            spec=model.spec,
            ansatz=ansatz,
            pairing=pairing,
            xi_eV_values=np.asarray([0.0, 0.02]),
            temperature_K=10.0,
            eta_eV=1e-8,
            q_model=np.asarray([0.071, 0.037]),
            adaptive_options=_loose_options(),
            node_cache=cache,
        )


@pytest.mark.skipif("fork" not in get_all_start_methods(), reason="requires POSIX fork")
def test_vector_adaptive_parallel_matches_serial_and_reports_prewarm(monkeypatch) -> None:
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("OMP_DYNAMIC", "FALSE")
    monkeypatch.setenv("MKL_DYNAMIC", "FALSE")
    model, ansatz, pairing, serial_cache = _state("spm", reusable=True)
    _, _, _, parallel_cache = _state("spm", reusable=True)
    tasks = (
        QLabAngleTask(0, np.asarray([0.071, 0.037]), 0.0, np.asarray([0.0, 0.2])),
        QLabAngleTask(1, np.asarray([0.043, -0.061]), 0.0, np.asarray([0.0, 0.2])),
    )
    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        adaptive_options=_loose_options(),
    )
    with ArbitraryQVectorAdaptiveParallelEvaluator(
        node_cache=serial_cache, process_workers=1, **common
    ) as evaluator:
        serial = evaluator.evaluate(tasks)
    evaluator = ArbitraryQVectorAdaptiveParallelEvaluator(
        node_cache=parallel_cache, process_workers=2, **common
    )
    try:
        parallel = evaluator.evaluate(tasks)
    finally:
        evaluator.close()
    metadata = evaluator.metadata()
    assert metadata["parent_prewarm"]["enabled"] is True
    assert metadata["parent_prewarm"]["unique_nodes_after"] > 0
    assert len(metadata["worker_cache_telemetry"]) == len(tasks)
    assert metadata["worker_cache_final_by_pid"]
    for row in metadata["worker_cache_telemetry"]:
        assert row["cache_delta"]["node_misses"] == 0
    for left, right in zip(serial, parallel, strict=True):
        np.testing.assert_allclose(
            left.result.plate_1.packed_primitives,
            right.result.plate_1.packed_primitives,
            atol=5e-12,
            rtol=5e-11,
        )
        for l_plate, r_plate in zip(left.result.plate_2, right.result.plate_2, strict=True):
            np.testing.assert_allclose(
                l_plate.packed_primitives, r_plate.packed_primitives,
                atol=5e-12, rtol=5e-11,
            )
