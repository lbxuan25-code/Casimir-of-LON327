from __future__ import annotations

from dataclasses import fields, replace
from multiprocessing import get_all_start_methods

import numpy as np
import pytest

from lno327 import KuboConfig
from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.models.symmetry_bdg_2band.parameters import TwoBandParameters
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec
from lno327.response.arbitrary_q_material_cache import (
    build_material_grid_cache,
    material_cache_fingerprint,
)
from lno327.response.periodic_bz_grid import (
    audit_shift_pair,
    build_periodic_bz_grid,
    exact_float64_key,
)
from lno327.response.primitive_kernel_v2 import (
    evaluate_primitive_batch_from_material,
)
from lno327.workflows.arbitrary_q_matsubara import (
    CrystalResponseCache,
    integrate_arbitrary_q_periodic_bz,
    integrate_two_plate_angle_batch,
    paired_average_arbitrary_q_results,
)
from lno327.workflows.arbitrary_q_parallel import (
    ArbitraryQParallelEvaluator,
    QLabAngleTask,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.commands.matsubara.arbitrary_q_periodic_bz_qualification import (
    _plate_reflection,
)
from validation.lib.dwave_orbit_acceptance import (
    OrbitAcceptancePhysicsConfig,
    evaluate_matsubara_pipeline,
)
from validation.lib.finite_q_validation_models import (
    get_finite_q_validation_model,
)


def _wrap(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def _material(pairing_name: str = "spm", n: int = 8, shift=(0.5, 0.5)):
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
    grid = build_periodic_bz_grid(n, shift)
    cache = build_material_grid_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        config=config,
        options=options,
        grid=grid,
    )
    return model, ansatz, pairing, config, options, grid, cache


def _result_signature(response) -> np.ndarray:
    values = [np.asarray(response.packed_primitives).reshape(-1)]
    for component, rhs in zip(response.components, response.rhs, strict=True):
        for field in (
            "bare_bubble",
            "direct",
            "bare_total",
            "collective_bubble",
            "collective_counterterm",
            "em_collective_left",
            "collective_em_right",
            "gauge_restored",
        ):
            values.append(np.asarray(getattr(component, field)).reshape(-1))
        values.append(np.asarray(rhs.left).reshape(-1))
        values.append(np.asarray(rhs.right).reshape(-1))
    return np.concatenate(values)


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
    assert exact_float64_key([0.1]) != exact_float64_key(
        [np.nextafter(0.1, 1.0)]
    )
    with pytest.raises(ValueError):
        exact_float64_key([np.nan])


def test_streamed_packed_primitives_match_one_shot_counterterm_once() -> None:
    model, ansatz, pairing, _config, _options, _grid, cache = _material("spm")
    q = np.asarray([0.071, 0.037])
    xi = np.asarray([0.0, 0.02])
    one_shot = evaluate_primitive_batch_from_material(
        cache.workspace,
        q,
        xi,
        include_counterterm=True,
    )
    streamed = integrate_arbitrary_q_periodic_bz(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=q,
        n=8,
        shift=(0.5, 0.5),
        canonical_reduction_block_size=16,
        runtime_chunk_size=32,
        material_cache=cache,
    )
    np.testing.assert_allclose(
        streamed.packed_primitives,
        one_shot.packed,
        atol=3e-12,
        rtol=3e-11,
    )
    assert streamed.profile.counterterm_add_count == 1
    assert streamed.operator_ward.passed


def test_streamed_result_is_runtime_chunk_independent() -> None:
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
    np.testing.assert_allclose(
        first.packed_primitives,
        second.packed_primitives,
        atol=2e-13,
        rtol=2e-13,
    )
    assert first.profile.shifted_eigensystem_build_count == 8
    assert second.profile.shifted_eigensystem_build_count == 8


def test_material_cache_fingerprint_changes_for_every_two_band_parameter() -> None:
    model, ansatz, pairing, config, options, grid, _cache = _material("spm")
    base = TwoBandParameters()
    reference = material_cache_fingerprint(
        spec=SymmetryBdG2BandSpec(base),
        ansatz=ansatz,
        pairing=pairing,
        config=config,
        options=options,
        grid=grid,
    )
    for field in fields(TwoBandParameters):
        value = getattr(base, field.name)
        changed = replace(base, **{field.name: float(value) + 0.013})
        observed = material_cache_fingerprint(
            spec=SymmetryBdG2BandSpec(changed),
            ansatz=ansatz,
            pairing=pairing,
            config=config,
            options=options,
            grid=grid,
        )
        assert observed != reference, field.name


def test_material_cache_avoids_grid_rebuild_and_response_cache_key_is_complete(
    monkeypatch,
) -> None:
    model, ansatz, pairing, _config, _options, _grid, cache = _material("spm")
    q = np.asarray([0.071, 0.037])
    xi = np.asarray([0.0, 0.02])
    response_cache = CrystalResponseCache()
    first = integrate_arbitrary_q_periodic_bz(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=q,
        n=8,
        shift=(0.5, 0.5),
        canonical_reduction_block_size=16,
        runtime_chunk_size=16,
        material_cache=cache,
        response_cache=response_cache,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("grid rebuild occurred on material-cache path")

    monkeypatch.setattr(
        "lno327.workflows.arbitrary_q_matsubara.build_periodic_bz_grid",
        forbidden,
    )
    second = integrate_arbitrary_q_periodic_bz(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=q,
        n=8,
        shift=(0.5, 0.5),
        canonical_reduction_block_size=16,
        runtime_chunk_size=64,
        material_cache=cache,
        response_cache=response_cache,
    )
    assert second is first

    base_key = CrystalResponseCache.key(
        cache.fingerprint,
        q,
        xi,
        phase_policy="q_independent",
        canonical_reduction_block_size=16,
        operator_ward_atol=first.operator_ward.atol,
        operator_ward_rtol=first.operator_ward.rtol,
    )
    assert base_key != CrystalResponseCache.key(
        cache.fingerprint,
        q,
        xi,
        phase_policy="q_independent",
        canonical_reduction_block_size=32,
        operator_ward_atol=first.operator_ward.atol,
        operator_ward_rtol=first.operator_ward.rtol,
    )
    assert base_key != CrystalResponseCache.key(
        cache.fingerprint,
        q,
        xi,
        phase_policy="q_independent",
        canonical_reduction_block_size=16,
        operator_ward_atol=2 * first.operator_ward.atol,
        operator_ward_rtol=first.operator_ward.rtol,
    )
    q_ulp = q.copy()
    q_ulp[0] = np.nextafter(q_ulp[0], 1.0)
    assert base_key != CrystalResponseCache.key(
        cache.fingerprint,
        q_ulp,
        xi,
        phase_policy="q_independent",
        canonical_reduction_block_size=16,
        operator_ward_atol=first.operator_ward.atol,
        operator_ward_rtol=first.operator_ward.rtol,
    )


def test_q_domain_rejects_unvalidated_momentum() -> None:
    model, ansatz, pairing, _config, _options, _grid, cache = _material("spm")
    with pytest.raises(ValueError, match="validated microscopic principal domain"):
        integrate_arbitrary_q_periodic_bz(
            spec=model.spec,
            ansatz=ansatz,
            pairing=pairing,
            xi_eV_values=np.asarray([0.0, 0.02]),
            temperature_K=10.0,
            eta_eV=1e-8,
            q_model=np.asarray([np.pi + 1e-6, 0.0]),
            n=8,
            shift=(0.5, 0.5),
            canonical_reduction_block_size=16,
            runtime_chunk_size=16,
            material_cache=cache,
        )


@pytest.mark.parametrize("pairing_name", ["spm", "dwave"])
def test_tiny_arbitrary_q_integrated_physical_pipeline(pairing_name: str) -> None:
    model, ansatz, pairing, _config, _options, _grid, cache = _material(
        pairing_name,
        n=16,
    )
    q = np.asarray([0.071, 0.037])
    result = integrate_arbitrary_q_periodic_bz(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=q,
        n=16,
        shift=(0.5, 0.5),
        canonical_reduction_block_size=32,
        runtime_chunk_size=64,
        material_cache=cache,
    )
    config = OrbitAcceptancePhysicsConfig(
        ward_tolerance=1e-7,
        ward_absolute_tolerance=1e-12,
    )
    assert result.operator_ward.passed
    for xi, component, rhs in zip(
        result.xi_eV_values,
        result.components,
        result.rhs,
        strict=True,
    ):
        physical = evaluate_matsubara_pipeline(
            components=component,
            rhs=rhs,
            q_model=q,
            xi_eV=float(xi),
            config=config,
        )
        assert physical["physical_passed"], physical["error"]


def test_paired_shift_average_occurs_before_nonlinear_postprocessing() -> None:
    first_material = _material("spm", n=8, shift=(0.25, 0.75))
    second_material = _material("spm", n=8, shift=(0.75, 0.25))
    model, ansatz, pairing = first_material[:3]
    q = np.asarray([0.071, 0.037])
    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=q,
        n=8,
        canonical_reduction_block_size=16,
        runtime_chunk_size=32,
    )
    first = integrate_arbitrary_q_periodic_bz(
        shift=(0.25, 0.75),
        material_cache=first_material[-1],
        **common,
    )
    second = integrate_arbitrary_q_periodic_bz(
        shift=(0.75, 0.25),
        material_cache=second_material[-1],
        **common,
    )
    paired = paired_average_arbitrary_q_results(
        first,
        second,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=10.0,
        eta_eV=1e-8,
    )
    np.testing.assert_allclose(
        paired.packed_primitives,
        0.5 * (first.packed_primitives + second.packed_primitives),
        atol=0.0,
        rtol=0.0,
    )
    assert paired.metadata["paired_shift_primitive_average"] is True
    assert paired.metadata["nonlinear_observable_average_forbidden"] is True


def test_two_plate_common_lab_logdet_small_path() -> None:
    model, ansatz, pairing, _config, _options, _grid, cache = _material(
        "spm",
        n=16,
    )
    q_lab = np.asarray([0.071, 0.037])
    theta = np.deg2rad(17.0)
    batch = integrate_two_plate_angle_batch(
        q_lab=q_lab,
        theta_1_rad=0.0,
        theta_2_rad_values=np.asarray([theta]),
        material_cache=cache,
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.02]),
        temperature_K=10.0,
        eta_eV=1e-8,
        canonical_reduction_block_size=32,
        runtime_chunk_size=64,
    )
    config = OrbitAcceptancePhysicsConfig()
    for index, xi in enumerate(batch.plate_1.xi_eV_values):
        r1, p1 = _plate_reflection(
            batch.plate_1.components[index],
            batch.plate_1.rhs[index],
            batch.plate_1.q_model,
            q_lab,
            0.0,
            float(xi),
            config,
        )
        r2, p2 = _plate_reflection(
            batch.plate_2[0].components[index],
            batch.plate_2[0].rhs[index],
            batch.plate_2[0].q_model,
            q_lab,
            theta,
            float(xi),
            config,
        )
        point = passive_sheet_logdet(r1, r2, separation_m=20e-9)
        assert p1 and p2 and np.isfinite(point.logdet)


@pytest.mark.skipif("fork" not in get_all_start_methods(), reason="requires POSIX fork")
def test_q_level_process_pool_preserves_all_plate_values(monkeypatch) -> None:
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
        QLabAngleTask(
            3,
            np.asarray([0.07, 0.03]),
            0.0,
            np.asarray([0.0, 0.17]),
        ),
        QLabAngleTask(
            1,
            np.asarray([0.05, 0.02]),
            0.0,
            np.asarray([0.0, 0.11]),
        ),
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
        left_responses = (
            left_task.result.plate_1,
            *left_task.result.plate_2,
        )
        right_responses = (
            right_task.result.plate_1,
            *right_task.result.plate_2,
        )
        for left, right in zip(left_responses, right_responses, strict=True):
            np.testing.assert_allclose(
                _result_signature(left),
                _result_signature(right),
                atol=3e-12,
                rtol=3e-11,
            )
            assert left.operator_ward.as_dict() == right.operator_ward.as_dict()
        assert right_task.payload_bytes > 0
        assert right_task.worker_rss_bytes > 0
