from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lno327.casimir import fixed_transverse_point_engine as legacy_engine
from lno327.casimir.material_geometry import material_response_to_reflection
from lno327.casimir.material_geometry_batch import execute_geometry_batch
from lno327.casimir.material_geometry_plan import build_geometry_batch_plan
from lno327.casimir.material_geometry_qualification import (
    qualify_batch_point_against_scalar,
    qualify_matched_legacy_point,
)
from lno327.casimir.material_response_cache_artifact import (
    CachedCertifiedMaterialResponse,
)
from lno327.casimir.material_response_cache_store import MaterialResponseCacheStore
from lno327.casimir.material_response_engine import MaterialResponseEngineConfig
from lno327.casimir.material_response_snapshot import MaterialResponseSnapshot
from lno327.electrodynamics.conductivity import ConductivityTensor
from lno327.electrodynamics.conventions import (
    PositiveMatsubaraSheetResponse,
    SheetConductivityConversion,
    SheetResponseValidation,
)


def _config() -> MaterialResponseEngineConfig:
    return MaterialResponseEngineConfig(
        pairing_name="spm",
        temperature_K=40.0,
        matsubara_indices=(1,),
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
        q_model=identity.q_crystal,
        xi_eV=identity.xi_eV,
        degeneracy=1.0,
        basis="crystal_xy",
        metadata={"source": "todo4-qualification-test"},
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
        q_crystal=identity.q_crystal,
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
    return CachedCertifiedMaterialResponse(
        identity=identity,
        snapshot=_snapshot(identity),
        working_N=64,
        audit_N=96,
        primary_shift=_shift_label(0, identity.shifts[0]),
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


def _batch(tmp_path: Path):
    plan = build_geometry_batch_plan(
        _config(),
        q_lab_points={"q": np.array([0.015, 0.025])},
        angle_pairs_rad=((0.0, 0.3),),
        separations_m=(50e-9, 100e-9),
    )
    writer = MaterialResponseCacheStore(tmp_path, mode="populate")
    for requirement in plan.requirements.values():
        writer.put(_artifact(requirement.identity))
    result = execute_geometry_batch(
        plan,
        cache=MaterialResponseCacheStore(tmp_path, mode="read_only"),
    )
    return result


def test_scalar_qualification_passes_for_every_distance(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    point_id = batch.plan.points[0].point_id
    report = qualify_batch_point_against_scalar(batch, point_id=point_id)

    assert report.passed is True
    assert set(report.comparisons) == {
        float(distance).hex() for distance in batch.plan.separations_m
    }
    assert all(row["trace_log_matrix_close"] for row in report.comparisons.values())
    assert report.production_casimir_allowed is False


def test_legacy_qualification_requires_exact_primary_N_shift_and_reduction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    batch = _batch(tmp_path)
    spec = batch.plan.points[0]
    first_artifact = batch.preflight.artifacts[spec.plate_1_requirement]
    second_artifact = batch.preflight.artifacts[spec.plate_2_requirement]
    first_reflection, _ = material_response_to_reflection(
        first_artifact.snapshot,
        q_lab=spec.q_lab,
        theta_rad=spec.theta_1_rad,
        policy=batch.plan.policy.reflection_policy,
    )
    second_reflection, _ = material_response_to_reflection(
        second_artifact.snapshot,
        q_lab=spec.q_lab,
        theta_rad=spec.theta_2_rad,
        policy=batch.plan.policy.reflection_policy,
    )

    shift_1 = first_artifact.identity.shifts[0]
    shift_2 = second_artifact.identity.shifts[0]
    legacy_plate_1 = SimpleNamespace(
        q_model=np.array(first_artifact.identity.q_crystal),
        metadata={
            "grid": {"N": 64, "shift": list(shift_1)},
            "canonical_reduction_block_size": (
                first_artifact.identity.canonical_reduction_block_size
            ),
        },
    )
    legacy_plate_2 = SimpleNamespace(
        q_model=np.array(second_artifact.identity.q_crystal),
        metadata={
            "grid": {"N": 64, "shift": list(shift_2)},
            "canonical_reduction_block_size": (
                second_artifact.identity.canonical_reduction_block_size
            ),
        },
    )
    legacy_batch = SimpleNamespace(
        q_lab=np.array(spec.q_lab),
        plate_1=legacy_plate_1,
        plate_2=(legacy_plate_2,),
    )
    legacy_args = SimpleNamespace(
        plate_angles_rad=(spec.theta_1_rad, spec.theta_2_rad)
    )

    def fake_plate_state(result, **kwargs):
        if result is legacy_plate_1:
            return first_reflection, {"hard_physical_passed": True}
        if result is legacy_plate_2:
            return second_reflection, {"hard_physical_passed": True}
        raise AssertionError("unexpected legacy plate object")

    monkeypatch.setattr(legacy_engine, "_plate_state", fake_plate_state)
    report = qualify_matched_legacy_point(
        batch,
        point_id=spec.point_id,
        distance_m=50e-9,
        legacy_batch=legacy_batch,
        legacy_frequency_index=0,
        legacy_n=64,
        legacy_xi_eV=first_artifact.identity.xi_eV,
        legacy_args=legacy_args,
    )

    assert report.passed is True
    assert report.comparisons["legacy_working_N_matches"] is True
    assert report.comparisons["legacy_primary_shift_matches"] is True
    assert report.comparisons["legacy_canonical_reduction_matches"] is True
    assert report.comparisons["legacy_exact_q_mapping"] is True

    with pytest.raises(ValueError, match="N/shift/reduction"):
        qualify_matched_legacy_point(
            batch,
            point_id=spec.point_id,
            distance_m=50e-9,
            legacy_batch=legacy_batch,
            legacy_frequency_index=0,
            legacy_n=96,
            legacy_xi_eV=first_artifact.identity.xi_eV,
            legacy_args=legacy_args,
        )
