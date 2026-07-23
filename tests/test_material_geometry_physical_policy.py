"""Fail-closed policy ownership at the material-response/geometry boundary."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lno327.casimir.material_geometry import (
    ReflectionGeometryPolicy,
    material_response_to_reflection,
)
from lno327.casimir.material_response import MaterialResponseSample
from lno327.electrodynamics.conductivity import ConductivityTensor
from lno327.electrodynamics.conventions import (
    PositiveMatsubaraSheetResponse,
    SheetConductivityConversion,
    SheetResponseValidation,
)


def _positive_sample(
    *,
    symmetry_residual: float,
    symmetry_tolerance: float,
    finite: bool = True,
) -> MaterialResponseSample:
    q = np.array([0.015, 0.025])
    xi_eV = 0.05
    tensor = ConductivityTensor(0.3, 0.4, 1.0e-6, 0.0)
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
        xi_eV=xi_eV,
        degeneracy=1.0,
        basis="crystal_xy",
        metadata={"source": "geometry-policy-test"},
    )
    validation = SheetResponseValidation(
        finite=finite,
        relative_imaginary_norm=0.0,
        relative_symmetry_residual=symmetry_residual,
        minimum_symmetric_eigenvalue=0.29,
        reality_tolerance=1e-9,
        symmetry_tolerance=symmetry_tolerance,
        passivity_tolerance=1e-10,
    )
    side = SimpleNamespace(effective_mixed_ratio=0.0)
    return MaterialResponseSample(
        frequency_index=0,
        frequency_sector="positive_matsubara",
        q_crystal=q,
        xi_eV=xi_eV,
        material_cache_fingerprint="geometry-policy-test",
        kernel=SimpleNamespace(q_model=q, xi_eV=xi_eV),
        operator_ward=SimpleNamespace(passed=True),
        effective_ward=SimpleNamespace(
            passed=True,
            left=side,
            right=side,
            schur_condition_number=1.0,
        ),
        strict_static_ward=None,
        response=response,
        sheet_validation=validation,
        metadata={
            "material_state_fingerprint": "geometry-policy-material",
            "response_policy_fingerprint": "geometry-policy-response",
            "primitive_contract_version": "geometry-policy-primitive",
            "post_integral_phase_hessian_policy": "q_independent",
            "basis": "crystal_xy",
        },
    )


def test_geometry_rejects_sample_that_failed_recorded_material_policy() -> None:
    sample = _positive_sample(
        symmetry_residual=1.0e-6,
        symmetry_tolerance=1.0e-9,
        finite=True,
    )
    assert sample.hard_physical_passed is False

    with pytest.raises(ValueError, match="recorded hard physical validation policy"):
        material_response_to_reflection(
            sample,
            q_lab=sample.q_crystal,
            theta_rad=0.0,
        )


def test_geometry_uses_recorded_material_policy_not_adapter_defaults() -> None:
    # The response is intentionally outside the electrodynamics adapter's default
    # 1e-9 symmetry tolerance but inside the material policy recorded on the sample.
    sample = _positive_sample(
        symmetry_residual=1.0e-6,
        symmetry_tolerance=1.0e-5,
        finite=True,
    )
    assert sample.hard_physical_passed is True

    reflection, diagnostics = material_response_to_reflection(
        sample,
        q_lab=sample.q_crystal,
        theta_rad=0.0,
    )

    assert np.isfinite(reflection.matrix_lt).all()
    assert diagnostics["material_validation_gate_required"] is True
    assert diagnostics["material_validation_gate_source"] == "MaterialResponseSample"
    assert diagnostics["adapter_default_policy_revalidation_performed"] is False


def test_diagnostic_geometry_can_explicitly_bypass_material_gate() -> None:
    sample = _positive_sample(
        symmetry_residual=1.0e-6,
        symmetry_tolerance=1.0e-9,
        finite=True,
    )
    reflection, diagnostics = material_response_to_reflection(
        sample,
        q_lab=sample.q_crystal,
        theta_rad=0.0,
        policy=ReflectionGeometryPolicy(require_physical=False),
    )

    assert np.isfinite(reflection.matrix_lt).all()
    assert diagnostics["hard_physical_passed"] is False
    assert diagnostics["material_validation_gate_required"] is False
