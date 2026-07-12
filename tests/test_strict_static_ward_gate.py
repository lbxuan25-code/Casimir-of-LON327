from __future__ import annotations

import numpy as np
import pytest

from lno327.response.effective_kernel import EffectiveEMKernel
from lno327.response.static_ward_gate import validate_strict_static_ward_closure
from lno327.response.ward_validation import PrimitiveWardRHS, validate_effective_ward_xy


def _kernel_and_ward(kll: float):
    q = np.asarray([0.1, 0.0], dtype=float)
    k_ss = np.diag([0.0, float(kll), 0.0]).astype(complex)
    kernel = EffectiveEMKernel(
        k_ss=k_ss,
        k_seta=np.zeros((3, 2), dtype=complex),
        k_etas=np.zeros((2, 3), dtype=complex),
        k_etaeta=np.eye(2, dtype=complex),
        k_eff=k_ss.copy(),
        q_model=q,
        xi_eV=0.0,
        schur_condition_number=1.0,
        schur_inverse_method="inv",
        metadata={"test": True},
    )
    rhs_value = np.asarray([0.0, q[0] * float(kll), 0.0], dtype=complex)
    rhs = PrimitiveWardRHS(
        left=rhs_value,
        right=rhs_value.copy(),
        q_model=q,
        xi_eV=0.0,
        delta0_eV=0.0,
        metadata={},
    )
    ward = validate_effective_ward_xy(kernel, rhs)
    return kernel, ward


def test_strict_static_gate_passes_zero_longitudinal_kernel():
    kernel, ward = _kernel_and_ward(0.0)
    gate = validate_strict_static_ward_closure(kernel, ward)
    assert ward.passed is True
    assert gate.passed is True
    assert gate.effective_direct_over_q == pytest.approx(0.0)
    assert gate.relative_longitudinal_gauge_residual == pytest.approx(0.0)


def test_strict_static_gate_rejects_nonzero_longitudinal_kernel_even_when_mixed_ward_passes():
    kernel, ward = _kernel_and_ward(1e-3)
    gate = validate_strict_static_ward_closure(kernel, ward)
    assert ward.passed is True
    assert gate.generic_ward_passed is True
    assert gate.effective_direct_over_q == pytest.approx(1e-3)
    assert gate.relative_longitudinal_gauge_residual == pytest.approx(1e-3)
    assert gate.passed is False
    with pytest.raises(ValueError, match="strict exact-static Ward closure failed"):
        gate.require_passed()


def test_strict_static_gate_requires_positive_condition_limit():
    kernel, ward = _kernel_and_ward(0.0)
    with pytest.raises(ValueError, match="condition_max must be positive"):
        validate_strict_static_ward_closure(kernel, ward, condition_max=0.0)
