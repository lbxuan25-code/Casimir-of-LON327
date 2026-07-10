from __future__ import annotations

import numpy as np
import pytest

from lno327.electrodynamics.conventions import (
    model_response_to_reflection_dimensionless,
    positive_matsubara_kernel_to_sheet_response,
    validate_positive_matsubara_sheet_response,
)
from lno327.response.effective_kernel import EffectiveEMKernel


def _kernel(spatial: np.ndarray, *, xi_eV: float = 0.5) -> EffectiveEMKernel:
    matrix = np.zeros((3, 3), dtype=complex)
    matrix[1:, 1:] = np.asarray(spatial, dtype=complex)
    return EffectiveEMKernel(
        k_ss=matrix.copy(),
        k_seta=np.zeros((3, 2), dtype=complex),
        k_etas=np.zeros((2, 3), dtype=complex),
        k_etaeta=np.eye(2, dtype=complex),
        k_eff=matrix,
        q_model=np.array([0.07, -0.03]),
        xi_eV=xi_eV,
        schur_condition_number=1.0,
        schur_inverse_method="solve",
        metadata={"basis": "crystal_A0_xy"},
    )


def test_positive_matsubara_conversion_uses_fixed_minus_sign_and_degeneracy_once():
    kernel = _kernel(np.diag([-1.0, -2.0]))
    response = positive_matsubara_kernel_to_sheet_response(kernel, degeneracy=2.0)

    expected_model = np.diag([4.0, 8.0]).astype(complex)
    expected_tilde = model_response_to_reflection_dimensionless(expected_model).tensor.matrix()
    np.testing.assert_allclose(response.matrix_model, expected_model, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(response.matrix_tilde, expected_tilde, rtol=1e-15, atol=0.0)
    np.testing.assert_array_equal(response.q_model, kernel.q_model)
    assert response.basis == "crystal_xy"
    assert response.metadata["frequency_sector"] == "positive_matsubara"
    assert response.metadata["square_lattice_geometry_factor"] == 1.0


def test_positive_matsubara_conversion_rejects_static_kernel():
    with pytest.raises(ValueError, match="xi_eV > 0"):
        positive_matsubara_kernel_to_sheet_response(
            _kernel(np.diag([-1.0, -2.0]), xi_eV=0.0)
        )


def test_positive_real_symmetric_sheet_passes_single_point_validation():
    response = positive_matsubara_kernel_to_sheet_response(_kernel(np.diag([-1.0, -2.0])))
    report = validate_positive_matsubara_sheet_response(response)
    assert report.passed
    assert report.finite
    assert report.relative_imaginary_norm == 0.0
    assert report.relative_symmetry_residual == 0.0
    assert report.minimum_symmetric_eigenvalue > 0.0
    report.require_passed()


def test_negative_sheet_eigenvalue_fails_passivity_gate():
    response = positive_matsubara_kernel_to_sheet_response(_kernel(np.diag([1.0, -2.0])))
    report = validate_positive_matsubara_sheet_response(response)
    assert not report.passed
    assert report.minimum_symmetric_eigenvalue < 0.0
    with pytest.raises(ValueError, match="failed physical validation"):
        report.require_passed()


def test_complex_nonsymmetric_sheet_fails_reality_and_symmetry_gates():
    response = positive_matsubara_kernel_to_sheet_response(
        _kernel(np.array([[-1.0, 0.3j], [0.0, -2.0]], dtype=complex))
    )
    report = validate_positive_matsubara_sheet_response(response)
    assert not report.passed
    assert report.relative_imaginary_norm > 0.0
    assert report.relative_symmetry_residual > 0.0
