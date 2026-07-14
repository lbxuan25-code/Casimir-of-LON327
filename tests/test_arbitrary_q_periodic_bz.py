from __future__ import annotations

from multiprocessing import get_all_start_methods

import numpy as np
import pytest

from lno327 import KuboConfig
from lno327.response.arbitrary_q_material_cache import build_material_grid_cache
from lno327.response.finite_q_material_workspace_batched import (
    precompute_finite_q_material_workspace_batched,
)
from lno327.response.finite_q_optimized import _vectorized_kubo_factors
from lno327.response.finite_q_q_workspace_batched import (
    precompute_finite_q_q_workspace_batched,
)
from lno327.response.periodic_bz_grid import (
    audit_shift_pair,
    build_periodic_bz_grid,
    exact_float64_key,
)
from lno327.response.primitive_kernel import evaluate_primitive_batch_from_material
from lno327.workflows.arbitrary_q_matsubara import (
    CrystalResponseCache,
    integrate_arbitrary_q_periodic_bz,
    integrate_two_plate_angle_batch,
)
from lno327.workflows.arbitrary_q_parallel import (
    ArbitraryQParallelEvaluator,
    QLabAngleTask,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.dwave_positive_orbit_adaptive import _pack_orbit_primitives
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _wrap(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def _material(pairing_name: str = "spm", n: int = 8):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    grid = build_periodic_bz_grid(n, (0.5, 0.5))
    cache = build_material_grid_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        config=config,
        options=options,
        grid=grid,
    )
    return model, ansatz, pairing, config, options, grid, cache


def test_primary_periodic_grid_has_explicit_inversion_pairs() -> None:
    grid = build_periodic_bz_grid(8, (0.5, 0.5))
    assert grid.internally_inversion_symmetric
    assert grid.ordering.startswith("adjacent_k_minus_k_pairs")
    for index, partner in enumerate(grid.inversion_partner):
        assert grid.inversion_partner[int(partner)] == index
        np.testing.assert_allclose(
            _wrap(grid.points[index] + grid.points[int(partner)]),
            0.0,
            atol=2e-15,
            rtol=0.0,
        )


def test_audit_shift_pair_is_related_by_inversion() -> None:
    first, second, mapping = audit_shift_pair(8)
    for index, partner in enumerate(mapping):
        np.testing.assert_allclose(
            _wrap(first.points[index] + second.points[int(partner)]),
            0.0,
            atol=2e-15,
            rtol=0.0,
        )


def test_exact_float_key_canonicalizes_signed_zero_without_rounding() -> None:
    assert exact_float64_key([0.0, -0.0]) == exact_float64_key([-0.0, 0.0])
    assert exact_float64_key([0.1]) != exact_float64_key([np.nextafter(0.1, 1.0)])
    with pytest.raises(ValueError):
        exact_float64_key([np.nan])


def test_shared_primitive_kernel_matches_historical_pack_layout() -> None:
    model, ansatz, pairing, config, options, grid, _cache = _material("spm")
    material = precompute_finite_q_material_workspace_batched(
        model.spec,
        ansatz,
        grid.points,
        grid.weights,
        config,
        pairing,
        options,
    )
    q = np.asarray([0.071, 0.037])
    xi = np.asarray([0.0, 0.02])
    shared = evaluate_primitive_batch_from_material(material, q, xi)

    workspace = precompute_finite_q_q_workspace_batched(material, q)
    factors = _vectorized_kubo_factors(workspace, xi)
    weighted = 0.5 * workspace.material.k_weights[None, :, None, None] * factors
    blocks = np.einsum(
        "xkmn,kamn,kbmn->xab",
        weighted,
        workspace.left_vertices_band,
        np.conjugate(workspace.right_vertices_band),
        optimize=True,
    )
    historical = _pack_orbit_primitives(workspace=workspace, blocks=blocks)
    np.testing.assert_allclose(shared.packed, historical, atol=2e-13, rtol=2e-13)
    assert shared.operator_ward.passed


def test_streamed_result_is_runtime_chunk_independent_and_counterterm_once() -> None:
    model, ansatz, pairing, _config, _options, _grid, cache = _material("spm")
    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=np.asarray([0.071, 0.037]),
        n=8,
        shift=(0.5, 0.5),
        canonical_reduction_block_size=16,
        material_cache=cache,
    )
    first = integrate_arbitrary_q_periodic_bz(runtime_chunk_size=16, **common)
    second = integrate_arbitrary_q_periodic_bz(runtime_chunk_size=64, **common)
    assert first.operator_ward.passed and second.operator_ward.passed
    assert first.profile.counterterm_add_count == 1
    assert second.profile.counterterm_add_count == 1
    assert first.profile.shifted_eigensystem_build_count == 8
    assert second.profile.shifted_eigensystem_build_count == 8
    for left, right in zip(first.components, second.components, strict=True):
        np.testing.assert_allclose(
            left.gauge_restored,
            right.gauge_restored,
            atol=2e-12,
            rtol=2e-11,
        )
    for left, right in zip(first.rhs, second.rhs, strict=True):
        np.testing.assert_allclose(left.left, right.left, atol=2e-13, rtol=2e-13)


def test_q_lab_angle_batch_reuses_fixed_plate_exact_response() -> None:
    model, ansatz, pairing, _config, _options, _grid, cache = _material("spm")
    response_cache = CrystalResponseCache()
    result = integrate_two_plate_angle_batch(
        q_lab=np.asarray([0.07, 0.03]),
        theta_1_rad=0.0,
        theta_2_rad_values=np.asarray([0.0, 0.17]),
        material_cache=cache,
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        canonical_reduction_block_size=16,
        runtime_chunk_size=32,
        response_cache=response_cache,
    )
    assert result.plate_1 is result.plate_2[0]
    assert result.response_cache_metadata["hits"] >= 1


@pytest.mark.skipif("fork" not in get_all_start_methods(), reason="requires POSIX fork")
def test_q_level_process_pool_preserves_task_order_and_values(monkeypatch) -> None:
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

    model, ansatz, pairing, _config, _options, _grid, cache = _material("spm")
    tasks = (
        QLabAngleTask(3, np.asarray([0.07, 0.03]), 0.0, np.asarray([0.17])),
        QLabAngleTask(1, np.asarray([0.05, 0.02]), 0.0, np.asarray([0.11])),
    )
    common = dict(
        material_cache=cache,
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        canonical_reduction_block_size=16,
        runtime_chunk_size=32,
    )
    with ArbitraryQParallelEvaluator(process_workers=1, **common) as serial:
        expected = serial.evaluate(tasks)
    with ArbitraryQParallelEvaluator(process_workers=2, **common) as parallel:
        observed = parallel.evaluate(tasks)
    assert [item.index for item in observed] == [3, 1]
    for left_task, right_task in zip(expected, observed, strict=True):
        for left, right in zip(
            left_task.result.plate_1.components,
            right_task.result.plate_1.components,
            strict=True,
        ):
            np.testing.assert_allclose(
                left.gauge_restored,
                right.gauge_restored,
                atol=2e-12,
                rtol=2e-11,
            )
