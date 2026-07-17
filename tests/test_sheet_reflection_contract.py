from __future__ import annotations

import numpy as np
import pytest

from lno327.electrodynamics.basis import q_lab_to_crystal
from lno327.electrodynamics.conventions import positive_matsubara_kernel_to_sheet_response
from lno327.electrodynamics.reflection import (
    LAB_LT_TANGENTIAL_E_BASIS,
    positive_matsubara_sheet_response_to_reflection,
)
from lno327.response.effective_kernel import EffectiveEMKernel


def _kernel(spatial: np.ndarray, q_model: np.ndarray, *, xi_eV: float = 0.2) -> EffectiveEMKernel:
    matrix = np.zeros((3, 3), dtype=complex)
    matrix[1:, 1:] = np.asarray(spatial, dtype=complex)
    return EffectiveEMKernel(
        k_ss=matrix.copy(),
        k_seta=np.zeros((3, 2), dtype=complex),
        k_etas=np.zeros((2, 3), dtype=complex),
        k_etaeta=np.eye(2, dtype=complex),
        k_eff=matrix,
        q_model=q_model,
        xi_eV=xi_eV,
        schur_condition_number=1.0,
        schur_inverse_method="solve",
        metadata={"basis": "crystal_A0_xy"},
    )


def _response(spatial: np.ndarray, q_lab: np.ndarray, theta: float, *, degeneracy: float = 1.0):
    q_crystal = q_lab_to_crystal(q_lab, theta)
    kernel = _kernel(spatial, q_crystal)
    return positive_matsubara_kernel_to_sheet_response(kernel, degeneracy=degeneracy)


def test_isotropic_sheet_reflection_is_independent_of_plate_angle():
    q_lab = np.array([0.08, 0.03])
    spatial = -2.0 * np.eye(2)
    first = positive_matsubara_sheet_response_to_reflection(
        _response(spatial, q_lab, 0.0),
        q_lab_model=q_lab,
        theta_rad=0.0,
        lattice_constant_m=3.754e-10,
    )
    second = positive_matsubara_sheet_response_to_reflection(
        _response(spatial, q_lab, 0.73),
        q_lab_model=q_lab,
        theta_rad=0.73,
        lattice_constant_m=3.754e-10,
    )

    np.testing.assert_allclose(first.matrix_lt, second.matrix_lt, rtol=1e-13, atol=1e-13)
    assert first.basis == LAB_LT_TANGENTIAL_E_BASIS
    assert first.sheet_validation.passed


def test_anisotropic_plate_rotation_is_applied_before_lt_projection():
    q_lab = np.array([0.1, 0.0])
    spatial = np.diag([-1.0, -3.0])
    zero = positive_matsubara_sheet_response_to_reflection(
        _response(spatial, q_lab, 0.0),
        q_lab_model=q_lab,
        theta_rad=0.0,
        lattice_constant_m=3.754e-10,
    )
    quarter_turn = positive_matsubara_sheet_response_to_reflection(
        _response(spatial, q_lab, np.pi / 2.0),
        q_lab_model=q_lab,
        theta_rad=np.pi / 2.0,
        lattice_constant_m=3.754e-10,
    )

    np.testing.assert_allclose(
        quarter_turn.sigma_tilde_lt,
        zero.sigma_tilde_lt[::-1, ::-1],
        rtol=1e-13,
        atol=1e-13,
    )


def test_reflection_rejects_response_evaluated_at_wrong_crystal_q():
    q_lab = np.array([0.08, 0.03])
    wrong = positive_matsubara_kernel_to_sheet_response(
        _kernel(-np.eye(2), q_model=q_lab)
    )
    with pytest.raises(ValueError, match="inconsistent with plate orientation"):
        positive_matsubara_sheet_response_to_reflection(
            wrong,
            q_lab_model=q_lab,
            theta_rad=0.4,
            lattice_constant_m=3.754e-10,
        )


def test_strong_passive_sheet_approaches_minus_identity_in_tangential_e_basis():
    q_lab = np.array([0.08, 0.03])
    reflection = positive_matsubara_sheet_response_to_reflection(
        _response(-np.eye(2), q_lab, 0.0, degeneracy=1e12),
        q_lab_model=q_lab,
        theta_rad=0.0,
        lattice_constant_m=3.754e-10,
    )
    np.testing.assert_allclose(reflection.matrix_lt, -np.eye(2), rtol=1e-7, atol=1e-7)


def test_nonpassive_sheet_is_rejected_by_default_but_available_for_diagnostics():
    q_lab = np.array([0.08, 0.03])
    response = _response(np.diag([1.0, -1.0]), q_lab, 0.0)
    with pytest.raises(ValueError, match="failed physical validation"):
        positive_matsubara_sheet_response_to_reflection(
            response,
            q_lab_model=q_lab,
            theta_rad=0.0,
            lattice_constant_m=3.754e-10,
        )

    diagnostic = positive_matsubara_sheet_response_to_reflection(
        response,
        q_lab_model=q_lab,
        theta_rad=0.0,
        lattice_constant_m=3.754e-10,
        require_physical=False,
    )
    assert not diagnostic.sheet_validation.passed
