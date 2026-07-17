from __future__ import annotations

import numpy as np

from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.electrodynamics.static_sheet import (
    static_matsubara_kernel_to_sheet_response,
    static_sheet_response_to_reflection,
)
from lno327.response.effective_kernel import EffectiveEMKernel
from lno327.response.ward_validation import PrimitiveWardRHS, validate_effective_ward_xy
from validation.lib import dwave_orbit_acceptance as acceptance


def _longitudinal_kernel_and_ward():
    q = np.asarray([0.1, 0.0], dtype=float)
    k_eff = np.diag([-1.0, 1e-3, -2.0]).astype(complex)
    kernel = EffectiveEMKernel(
        k_ss=k_eff,
        k_seta=np.zeros((3, 2), dtype=complex),
        k_etas=np.zeros((2, 3), dtype=complex),
        k_etaeta=np.eye(2, dtype=complex),
        k_eff=k_eff,
        q_model=q,
        xi_eV=0.0,
        schur_condition_number=1.0,
        schur_inverse_method="inv",
        metadata={"test": "longitudinal_diagnostic_only"},
    )
    rhs_value = np.asarray([0.0, q[0] * 1e-3, 0.0], dtype=complex)
    rhs = PrimitiveWardRHS(
        left=rhs_value,
        right=rhs_value.copy(),
        q_model=q,
        xi_eV=0.0,
        delta0_eV=0.0,
        metadata={},
    )
    ward = validate_effective_ward_xy(kernel, rhs)
    assert ward.passed is True
    return q, kernel, rhs, ward


def test_static_sheet_longitudinal_excess_is_recorded_but_nonblocking() -> None:
    q, kernel, _, ward = _longitudinal_kernel_and_ward()
    sheet = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward,
        longitudinal_tolerance=1e-6,
    )

    assert sheet.validation.longitudinal_within_tolerance is False
    assert sheet.validation.longitudinal_warning is True
    assert sheet.validation.passed is True
    assert sheet.metadata["longitudinal_hard_gate"] is False

    reflection = static_sheet_response_to_reflection(
        sheet,
        q_lab_model=q,
        theta_rad=0.0,
        require_physical=True,
    )
    point = passive_sheet_logdet(reflection, reflection, separation_m=20e-9)

    assert reflection.metadata["longitudinal_warning"] is True
    assert np.isfinite(point.logdet)


def test_zero_pipeline_uses_ward_as_hard_gate_and_keeps_longitudinal_warning(
    monkeypatch,
) -> None:
    q, kernel, rhs, ward = _longitudinal_kernel_and_ward()
    monkeypatch.setattr(
        acceptance,
        "effective_em_kernel_from_components",
        lambda *args, **kwargs: kernel,
    )
    monkeypatch.setattr(
        acceptance,
        "validate_effective_ward_xy",
        lambda *args, **kwargs: ward,
    )

    state = acceptance.evaluate_zero_matsubara_pipeline(
        components=object(),
        rhs=rhs,
        q_model=q,
        config=acceptance.OrbitAcceptancePhysicsConfig(
            static_longitudinal_tolerance=1e-6,
        ),
    )

    assert state["ward_passed"] is True
    assert state["strict_static_ward_passed"] is False
    assert state["strict_static_hard_gate"] is False
    assert state["static_longitudinal_within_tolerance"] is False
    assert state["static_longitudinal_warning"] is True
    assert "diagnostic warning" in state["warning"]
    assert state["sheet_validation_passed"] is True
    assert state["reflection_constructed"] is True
    assert state["logdet_passed"] is True
    assert state["physical_passed"] is True
    assert state["error"] == ""
