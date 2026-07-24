from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lno327.casimir.lifshitz_integrand import (
    evaluate_prepared_passive_sheet_pair,
    passive_sheet_logdet,
    prepare_passive_sheet_pair,
)
from lno327.casimir.material_geometry_batch import (
    GeometryBatchCacheIncomplete,
    execute_geometry_batch,
    preflight_geometry_batch,
)
from lno327.casimir.material_geometry_plan import build_geometry_batch_plan
from lno327.casimir.material_response_cache_artifact import (
    CachedCertifiedMaterialResponse,
)
from lno327.casimir.material_response_cache_store import MaterialResponseCacheStore
from lno327.casimir.material_response_engine import MaterialResponseEngineConfig
from lno327.casimir.material_response_snapshot import MaterialResponseSnapshot
from lno327.casimir.material_two_plate import (
    TwoPlateGeometryPolicy,
    assemble_two_plate_logdet,
)
from lno327.electrodynamics.basis import q_lab_to_crystal
from lno327.electrodynamics.conductivity import ConductivityTensor
from lno327.electrodynamics.conventions import (
    PositiveMatsubaraSheetResponse,
    SheetConductivityConversion,
    SheetResponseValidation,
)
from lno327.electrodynamics.static_sheet import (
    STATIC_LOCAL_BASIS,
    StaticSheetResponse,
    StaticSheetValidation,
)


def _config(indices: tuple[int, ...] = (0, 1)) -> MaterialResponseEngineConfig:
    return MaterialResponseEngineConfig(
        pairing_name="spm",
        temperature_K=40.0,
        matsubara_indices=indices,
        n_candidates=(64, 96, 128),
        required_consecutive_passes=1,
        envelope_levels=3,
    )


def _sample_identity(identity) -> dict[str, object]:
    return {
        "schema": "material-response-identity-v1",
        "frequency_sector": identity.frequency_sector,
        "xi_eV_hex": float(identity.xi_eV).hex(),
        "q_crystal_hex": [float(value).hex() for value in identity.q_crystal],
        "material_state_fingerprint": identity.material_state_fingerprint,
        "response_policy_fingerprint": identity.response_policy_fingerprint,
        "primitive_contract_version": identity.primitive_contract_version,
        "phase_hessian_policy": identity.phase_hessian_policy,
        "basis": identity.basis,
    }


def _snapshot(identity) -> MaterialResponseSnapshot:
    q = identity.q_crystal
    if identity.matsubara_index == 0:
        validation = StaticSheetValidation(
            finite=True,
            ward_passed=True,
            relative_imaginary_norm=0.0,
            relative_longitudinal_gauge_residual=0.0,
            relative_density_transverse_mixing=0.0,
            chi_bar=1.0,
            dbar_t=2.0,
            reality_tolerance=1e-8,
            longitudinal_tolerance=1e-6,
            mixing_tolerance=1e-6,
            passivity_tolerance=1e-10,
        )
        response = StaticSheetResponse(
            kernel_lt=np.diag([-1.0, 0.0, -2.0]).astype(complex),
            chi_bar=1.0,
            dbar_t=2.0,
            q_model=q,
            energy_scale_eV=1.0,
            degeneracy=1.0,
            basis=STATIC_LOCAL_BASIS,
            validation=validation,
            metadata={"source": "todo4-test"},
        )
    else:
        tensor = ConductivityTensor(0.4, 0.5, 0.02, 0.02)
        sheet = SheetConductivityConversion(
            tensor=tensor,
            unit_stage="sheet_conductivity",
            unit_label="test-sheet",
            normalization_status="test",
            valid_for_casimir_input=True,
            notes=(),
        )
        tilde = SheetConductivityConversion(
            tensor=tensor,
            unit_stage="reflection_dimensionless_conductivity",
            unit_label="test-tilde",
            normalization_status="test",
            valid_for_casimir_input=True,
            notes=(),
        )
        response = PositiveMatsubaraSheetResponse(
            sigma_model_xy=tensor,
            sigma_sheet_si_xy=sheet,
            sigma_tilde_xy=tilde,
            q_model=q,
            xi_eV=identity.xi_eV,
            degeneracy=1.0,
            basis="crystal_xy",
            metadata={"source": "todo4-test"},
        )
        validation = SheetResponseValidation(
            finite=True,
            relative_imaginary_norm=0.0,
            relative_symmetry_residual=0.0,
            minimum_symmetric_eigenvalue=0.38,
            reality_tolerance=1e-9,
            symmetry_tolerance=1e-9,
            passivity_tolerance=1e-10,
        )
    return MaterialResponseSnapshot(
        frequency_index=0,
        frequency_sector=identity.frequency_sector,
        q_crystal=q,
        xi_eV=identity.xi_eV,
        response=response,
        sheet_validation=validation,
        identity=_sample_identity(identity),
        provenance={"grid": {"n": 96}},
        physical_audit={"hard_physical_passed": True},
    )


def _shift_label(index: int, shift: tuple[float, float]) -> str:
    return f"shift_{index}:{float(shift[0]).hex()}:{float(shift[1]).hex()}"


def _artifact(identity) -> CachedCertifiedMaterialResponse:
    provenance = {
        _shift_label(index, shift): {
            "canonical_reduction_block_size": identity.canonical_reduction_block_size,
            "grid": {
                "N": 96,
                "shift_hex": [float(value).hex() for value in shift],
            },
        }
        for index, shift in enumerate(identity.shifts)
    }
    primary = _shift_label(0, identity.shifts[0])
    return CachedCertifiedMaterialResponse(
        identity=identity,
        snapshot=_snapshot(identity),
        working_N=64,
        audit_N=96,
        primary_shift=primary,
        establishment_mode="strict_consecutive_adjacent",
        certification_evidence={
            "convergence_policy": dict(identity.convergence_policy),
            "required_consecutive_passes": identity.required_consecutive_passes,
            "oscillatory_envelope": {"levels": identity.envelope_levels},
            "observable_error_budget_calibrated": False,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
        audit_provenance_by_shift=provenance,
    )


def _populated_read_only_store(
    tmp_path: Path,
    plan,
) -> MaterialResponseCacheStore:
    writer = MaterialResponseCacheStore(tmp_path, mode="populate")
    for requirement in plan.requirements.values():
        writer.put(_artifact(requirement.identity))
    return MaterialResponseCacheStore(tmp_path, mode="read_only")


def test_prepared_pair_matches_scalar_logdet_for_all_distances() -> None:
    config = _config((1,))
    plan = build_geometry_batch_plan(
        config,
        q_lab_points={"q": np.array([0.015, 0.025])},
        angle_pairs_rad=((0.0, 0.0),),
        separations_m=(50e-9, 100e-9, 200e-9),
    )
    first_requirement = next(iter(plan.requirements.values()))
    snapshot = _snapshot(first_requirement.identity)
    from lno327.casimir.material_geometry import material_response_to_reflection

    reflection, _ = material_response_to_reflection(
        snapshot,
        q_lab=plan.q_lab_by_label["q"],
        theta_rad=0.0,
        policy=plan.policy.reflection_policy,
    )
    prepared = prepare_passive_sheet_pair(reflection, reflection)
    for distance in plan.separations_m:
        scalar = passive_sheet_logdet(
            reflection,
            reflection,
            separation_m=distance,
        )
        batched = evaluate_prepared_passive_sheet_pair(
            prepared,
            separation_m=distance,
        )
        assert batched.logdet == pytest.approx(scalar.logdet, rel=0.0, abs=1e-14)
        np.testing.assert_allclose(
            batched.trace_log_matrix,
            scalar.trace_log_matrix,
            rtol=0.0,
            atol=1e-14,
        )


def test_geometry_plan_uses_exact_rotation_and_deduplicates_requirements() -> None:
    q_lab = np.array([0.015, 0.025])
    theta = 0.31
    plan = build_geometry_batch_plan(
        _config(),
        q_lab_points={"q": q_lab},
        angle_pairs_rad=((0.0, 0.0), (0.0, theta)),
        separations_m=(100e-9,),
    )
    assert len(plan.points) == 4
    assert len(plan.requirements) == 4
    expected = q_lab_to_crystal(q_lab, theta)
    assert any(
        np.array_equal(requirement.q_crystal, expected)
        for requirement in plan.requirements.values()
    )
    for requirement in plan.requirements.values():
        payload = str(requirement.identity.payload)
        for forbidden in ("separation", "theta", "q_lab", "outer", "worker"):
            assert forbidden not in payload
    shifted = build_geometry_batch_plan(
        _config((1,)),
        q_lab_points={"q": q_lab},
        angle_pairs_rad=((0.0, np.nextafter(theta, 1.0)),),
        separations_m=(100e-9,),
    )
    original_keys = {
        key
        for key, requirement in plan.requirements.items()
        if requirement.identity.matsubara_index == 1
    }
    assert not set(shifted.requirements).issubset(original_keys)


@pytest.mark.parametrize("indices", [(0,), (1,), (0, 1)])
def test_read_only_batch_matches_scalar_and_reuses_reflections(
    tmp_path: Path,
    indices: tuple[int, ...],
) -> None:
    plan = build_geometry_batch_plan(
        _config(indices),
        q_lab_points={"q": np.array([0.015, 0.025])},
        angle_pairs_rad=((0.0, 0.0), (0.0, 0.3)),
        separations_m=(50e-9, 100e-9, 200e-9),
    )
    store = _populated_read_only_store(tmp_path, plan)
    result = execute_geometry_batch(plan, cache=store)

    assert result.metadata["response_load_count"] == len(plan.requirements)
    assert result.metadata["microscopic_integration_call_count"] == 0
    assert result.metadata["response_certification_call_count"] == 0
    assert result.metadata["cache_write_count"] == 0
    assert result.metadata["distance_update_count"] == (
        len(plan.points) * len(plan.separations_m)
    )
    assert result.metadata["reflection_build_count"] < (
        2 * len(plan.points) * len(plan.separations_m)
    )

    for spec in plan.points:
        row = result.points[spec.point_id]
        first = result.preflight.snapshots[spec.plate_1_requirement]
        second = result.preflight.snapshots[spec.plate_2_requirement]
        for distance, batch_point in zip(plan.separations_m, row.lifshitz_points):
            scalar = assemble_two_plate_logdet(
                first,
                second,
                q_lab=spec.q_lab,
                theta_1_rad=spec.theta_1_rad,
                theta_2_rad=spec.theta_2_rad,
                policy=TwoPlateGeometryPolicy(
                    separation_m=distance,
                    reflection_policy=plan.policy.reflection_policy,
                    compatibility_tolerance=plan.policy.compatibility_tolerance,
                    eigenvalue_imag_tolerance=plan.policy.eigenvalue_imag_tolerance,
                    eigenvalue_lower_tolerance=plan.policy.eigenvalue_lower_tolerance,
                ),
            )
            assert batch_point.logdet == pytest.approx(
                scalar.logdet,
                rel=0.0,
                abs=1e-14,
            )


def test_preflight_reports_all_misses_and_execute_fails_without_fallback(
    tmp_path: Path,
) -> None:
    plan = build_geometry_batch_plan(
        _config((0, 1)),
        q_lab_points={"q": np.array([0.015, 0.025])},
        angle_pairs_rad=((0.0, 0.3),),
        separations_m=(100e-9,),
    )
    store = MaterialResponseCacheStore(tmp_path, mode="read_only")
    preflight = preflight_geometry_batch(plan, cache=store)
    assert preflight.complete is False
    assert set(preflight.misses) == set(plan.requirements)
    assert preflight.metadata["microscopic_fallback_attempted"] is False
    assert not tmp_path.exists() or list(tmp_path.rglob("*")) == []
    with pytest.raises(GeometryBatchCacheIncomplete):
        execute_geometry_batch(plan, cache=store)


def test_geometry_batch_rejects_non_read_only_cache(tmp_path: Path) -> None:
    plan = build_geometry_batch_plan(
        _config((1,)),
        q_lab_points={"q": np.array([0.015, 0.025])},
        angle_pairs_rad=((0.0, 0.0),),
        separations_m=(100e-9,),
    )
    with pytest.raises(ValueError, match="strict read_only"):
        preflight_geometry_batch(
            plan,
            cache=MaterialResponseCacheStore(tmp_path, mode="populate"),
        )
