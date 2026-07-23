"""Regression and architecture tests for the TODO 2 response/geometry split."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lno327.casimir import material_geometry as geometry
from lno327.casimir import material_response as material
from lno327.casimir import material_response_certification as certification
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


def _operator_ward(*, passed: bool = True) -> SimpleNamespace:
    return SimpleNamespace(passed=passed)


def _effective_ward(*, passed: bool = True) -> SimpleNamespace:
    side = SimpleNamespace(effective_mixed_ratio=0.1)
    return SimpleNamespace(
        passed=passed,
        left=side,
        right=side,
        schur_condition_number=2.0,
    )


def _kernel(q: np.ndarray, xi_eV: float) -> SimpleNamespace:
    return SimpleNamespace(q_model=np.asarray(q, dtype=float), xi_eV=float(xi_eV))


def _static_sample(
    *,
    q: np.ndarray | None = None,
    chi_bar: float = 1.0,
    dbar_t: float = 2.0,
    fingerprint: str = "static-grid",
    hard_passed: bool = True,
) -> material.MaterialResponseSample:
    q_value = np.array([0.02, -0.01]) if q is None else np.asarray(q, dtype=float)
    validation = StaticSheetValidation(
        finite=True,
        ward_passed=hard_passed,
        relative_imaginary_norm=0.0,
        relative_longitudinal_gauge_residual=0.0,
        relative_density_transverse_mixing=0.0,
        chi_bar=chi_bar,
        dbar_t=dbar_t,
        reality_tolerance=1e-8,
        longitudinal_tolerance=1e-6,
        mixing_tolerance=1e-6,
        passivity_tolerance=1e-10,
    )
    sheet = StaticSheetResponse(
        kernel_lt=np.eye(3, dtype=complex),
        chi_bar=chi_bar,
        dbar_t=dbar_t,
        q_model=q_value,
        energy_scale_eV=1.0,
        degeneracy=1.0,
        basis=STATIC_LOCAL_BASIS,
        validation=validation,
        metadata={"source": "test"},
    )
    return material.MaterialResponseSample(
        frequency_index=0,
        frequency_sector="zero_matsubara",
        q_crystal=q_value,
        xi_eV=0.0,
        material_cache_fingerprint=fingerprint,
        kernel=_kernel(q_value, 0.0),
        operator_ward=_operator_ward(passed=hard_passed),
        effective_ward=_effective_ward(passed=hard_passed),
        strict_static_ward=SimpleNamespace(passed=hard_passed),
        response=sheet,
        sheet_validation=validation,
        metadata={},
    )


def _positive_sample(
    *,
    q: np.ndarray | None = None,
    scale: float = 1.0,
    fingerprint: str = "positive-grid",
) -> material.MaterialResponseSample:
    q_value = np.array([0.015, 0.025]) if q is None else np.asarray(q, dtype=float)
    xi_eV = 0.05
    model = ConductivityTensor(scale, 2.0 * scale, 0.1 * scale, 0.1 * scale)
    sheet = SheetConductivityConversion(
        tensor=model,
        unit_stage="sheet_conductivity",
        unit_label="test-sheet",
        normalization_status="test",
        valid_for_casimir_input=True,
        notes=(),
    )
    tilde = SheetConductivityConversion(
        tensor=model,
        unit_stage="reflection_dimensionless_conductivity",
        unit_label="test-tilde",
        normalization_status="test",
        valid_for_casimir_input=True,
        notes=(),
    )
    response = PositiveMatsubaraSheetResponse(
        sigma_model_xy=model,
        sigma_sheet_si_xy=sheet,
        sigma_tilde_xy=tilde,
        q_model=q_value,
        xi_eV=xi_eV,
        degeneracy=1.0,
        basis="crystal_xy",
        metadata={"source": "test"},
    )
    validation = SheetResponseValidation(
        finite=True,
        relative_imaginary_norm=0.0,
        relative_symmetry_residual=0.0,
        minimum_symmetric_eigenvalue=0.5 * scale,
        reality_tolerance=1e-9,
        symmetry_tolerance=1e-9,
        passivity_tolerance=1e-10,
    )
    return material.MaterialResponseSample(
        frequency_index=1,
        frequency_sector="positive_matsubara",
        q_crystal=q_value,
        xi_eV=xi_eV,
        material_cache_fingerprint=fingerprint,
        kernel=_kernel(q_value, xi_eV),
        operator_ward=_operator_ward(),
        effective_ward=_effective_ward(),
        strict_static_ward=None,
        response=response,
        sheet_validation=validation,
        metadata={},
    )


def test_geometry_adapter_does_not_rebuild_material_response(monkeypatch) -> None:
    sample = _static_sample()
    expected = SimpleNamespace(matrix_lt=np.diag([-0.1, -0.2]))
    calls: list[dict[str, object]] = []

    def fake_static(response, **kwargs):
        calls.append({"response": response, **kwargs})
        return expected

    monkeypatch.setattr(geometry, "static_sheet_response_to_reflection", fake_static)
    monkeypatch.setattr(
        geometry,
        "positive_matsubara_sheet_response_to_reflection",
        lambda *args, **kwargs: pytest.fail("positive adapter used for static sample"),
    )

    reflection, diagnostics = geometry.material_response_to_reflection(
        sample,
        q_lab=np.array([0.02, -0.01]),
        theta_rad=0.0,
    )
    assert reflection is expected
    assert calls[0]["response"] is sample.response
    assert diagnostics["reflection_constructed"] is True
    assert diagnostics["material_response_schema"] == material.MATERIAL_RESPONSE_SAMPLE_SCHEMA
    assert diagnostics["q_lab"] == [0.02, -0.01]


def test_positive_response_comparison_uses_crystal_tensor_norm() -> None:
    policy = certification.MaterialResponseConvergencePolicy(
        relative_tolerance=1e-2,
        absolute_tolerance=1e-12,
    )
    comparison = certification.compare_material_responses(
        _positive_sample(scale=1.0),
        _positive_sample(scale=1.005, fingerprint="other-shift"),
        policy=policy,
    )
    assert comparison["comparison_basis"] == "crystal_xy_sigma_tilde_spectral_norm"
    assert comparison["passed"] is True
    assert comparison["matrix"]["passed_by"] == "relative"


def test_static_response_comparison_checks_channels_separately() -> None:
    policy = certification.MaterialResponseConvergencePolicy(
        relative_tolerance=1e-3,
        absolute_tolerance=1e-8,
    )
    comparison = certification.compare_material_responses(
        _static_sample(chi_bar=1.0, dbar_t=1000.0),
        _static_sample(chi_bar=1.01, dbar_t=1000.0, fingerprint="other"),
        policy=policy,
    )
    assert comparison["channels"]["chi_bar"]["passed"] is False
    assert comparison["channels"]["dbar_t"]["passed"] is True
    assert comparison["passed"] is False


def test_level_assessment_is_geometry_independent_and_fail_closed() -> None:
    policy = certification.MaterialResponseConvergencePolicy(
        relative_tolerance=1e-2,
        absolute_tolerance=1e-8,
    )
    previous = {
        "shift_a": _static_sample(chi_bar=1.0, dbar_t=2.0, fingerprint="a0"),
        "shift_b": _static_sample(chi_bar=1.001, dbar_t=2.001, fingerprint="b0"),
    }
    current = {
        "shift_a": _static_sample(chi_bar=1.002, dbar_t=2.002, fingerprint="a1"),
        "shift_b": _static_sample(chi_bar=1.003, dbar_t=2.003, fingerprint="b1"),
    }
    assessment = certification.assess_material_response_level(
        current_by_shift=current,
        previous_by_shift=previous,
        policy=policy,
    )
    assert assessment.hard_physical_closure_across_shifts is True
    assert assessment.cross_shift_all_passed is True
    assert assessment.adjacent_N_all_shifts_passed is True
    assert assessment.accepted_transition is True
    payload = assessment.as_dict()
    assert not {"q_lab", "theta_rad", "separation_nm", "two_plate_logdet"}.intersection(payload)

    failed_current = dict(current)
    failed_current["shift_b"] = _static_sample(
        chi_bar=1.003,
        dbar_t=2.003,
        fingerprint="failed",
        hard_passed=False,
    )
    failed = certification.assess_material_response_level(
        current_by_shift=failed_current,
        previous_by_shift=previous,
        policy=policy,
    )
    assert failed.hard_physical_closure_across_shifts is False
    assert failed.accepted_transition is False


def test_certification_produces_explicit_diagnostic_primary_response() -> None:
    policy = certification.MaterialResponseConvergencePolicy(
        relative_tolerance=1e-2,
        absolute_tolerance=1e-8,
    )
    records: list[certification.MaterialResponseLevelRecord] = []
    previous = None
    for n_grid, offset in ((64, 0.0), (96, 0.001), (128, 0.002)):
        samples = {
            "shift_a": _static_sample(
                chi_bar=1.0 + offset,
                dbar_t=2.0 + offset,
                fingerprint=f"a-{n_grid}",
            ),
            "shift_b": _static_sample(
                chi_bar=1.0005 + offset,
                dbar_t=2.0005 + offset,
                fingerprint=f"b-{n_grid}",
            ),
        }
        assessment = certification.assess_material_response_level(
            current_by_shift=samples,
            previous_by_shift=previous,
            policy=policy,
        )
        records.append(
            certification.MaterialResponseLevelRecord(
                n_grid=n_grid,
                samples_by_shift=samples,
                assessment=assessment,
            )
        )
        previous = samples

    certified = certification.certify_material_response_history(
        records,
        policy=policy,
        required_consecutive_passes=2,
    )
    assert certified is not None
    assert certified.working_N == 96
    assert certified.audit_N == 128
    assert certified.primary_shift == "shift_a"
    assert certified.primary_response is records[-1].samples_by_shift["shift_a"]
    assert certified.status == "response_certified_diagnostic"
    assert certified.production_casimir_allowed is False
    assert certified.evidence["valid_for_casimir_input"] is False


def test_comparison_rejects_q_mismatch_instead_of_rounding() -> None:
    policy = certification.MaterialResponseConvergencePolicy()
    with pytest.raises(ValueError, match="different q_crystal"):
        certification.compare_material_responses(
            _static_sample(q=np.array([0.01, 0.02])),
            _static_sample(q=np.array([0.01, np.nextafter(0.02, 1.0)])),
            policy=policy,
        )
