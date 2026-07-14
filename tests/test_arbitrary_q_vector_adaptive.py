from __future__ import annotations

from multiprocessing import get_all_start_methods

import numpy as np
import pytest

from lno327.response.primitive_kernel_v2 import evaluate_primitive_batch_from_material
from lno327.workflows.arbitrary_q_parallel import QLabAngleTask
from lno327.workflows.arbitrary_q_vector_adaptive import (
    AdaptiveConvergenceError,
    ArbitraryQVectorAdaptiveOptions,
    build_hierarchical_material_node_cache,
    integrate_arbitrary_q_vector_adaptive,
    integrate_two_plate_angle_batch_vector_adaptive,
)
from lno327.workflows.arbitrary_q_vector_adaptive_parallel import (
    ArbitraryQVectorAdaptiveParallelEvaluator,
)
from lno327.workflows.dwave_vector_adaptive_cubature import (
    cubature_cell_gauss_rule,
    initial_cubature_cells,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _state(pairing_name: str = "spm"):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    cache = build_hierarchical_material_node_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=10.0,
        eta_eV=1e-8,
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
    assert adaptive.metadata["all_frequencies_share_one_adaptive_tree"] is True
    assert adaptive.operator_ward.passed


def test_hierarchical_node_cache_reuses_midpoint_eigensystems_across_q() -> None:
    model, ansatz, pairing, cache = _state("spm")
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
    integrate_arbitrary_q_vector_adaptive(q_model=np.asarray([0.071, 0.037]), **common)
    misses = cache.node_misses
    builds = cache.midpoint_eigh_call_count
    integrate_arbitrary_q_vector_adaptive(q_model=np.asarray([0.043, -0.061]), **common)
    assert cache.node_misses == misses
    assert cache.midpoint_eigh_call_count == builds
    assert cache.node_hits > 0


def test_refinement_is_deterministic_across_cell_batch_sizes() -> None:
    model, ansatz, pairing, first_cache = _state("dwave")
    second_cache = build_hierarchical_material_node_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=10.0,
        eta_eV=1e-8,
    )
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


def test_nonconverged_adaptive_result_fails_closed() -> None:
    model, ansatz, pairing, cache = _state("dwave")
    options = ArbitraryQVectorAdaptiveOptions(
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
    with pytest.raises(AdaptiveConvergenceError):
        integrate_arbitrary_q_vector_adaptive(
            spec=model.spec,
            ansatz=ansatz,
            pairing=pairing,
            xi_eV_values=np.asarray([0.0, 0.02]),
            temperature_K=10.0,
            eta_eV=1e-8,
            q_model=np.asarray([0.071, 0.037]),
            adaptive_options=options,
            node_cache=cache,
        )


def test_two_plate_batch_reuses_exact_q_response() -> None:
    model, ansatz, pairing, cache = _state("spm")
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


@pytest.mark.skipif("fork" not in get_all_start_methods(), reason="requires POSIX fork")
def test_vector_adaptive_parallel_matches_serial(monkeypatch) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("OMP_DYNAMIC", "FALSE")
    monkeypatch.setenv("MKL_DYNAMIC", "FALSE")
    model, ansatz, pairing, serial_cache = _state("spm")
    parallel_cache = build_hierarchical_material_node_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=10.0,
        eta_eV=1e-8,
    )
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
    with ArbitraryQVectorAdaptiveParallelEvaluator(
        node_cache=parallel_cache, process_workers=2, **common
    ) as evaluator:
        parallel = evaluator.evaluate(tasks)
    for left, right in zip(serial, parallel, strict=True):
        np.testing.assert_allclose(
            left.result.plate_1.packed_primitives,
            right.result.plate_1.packed_primitives,
            atol=5e-12,
            rtol=5e-11,
        )
        for l_plate, r_plate in zip(
            left.result.plate_2, right.result.plate_2, strict=True
        ):
            np.testing.assert_allclose(
                l_plate.packed_primitives,
                r_plate.packed_primitives,
                atol=5e-12,
                rtol=5e-11,
            )
