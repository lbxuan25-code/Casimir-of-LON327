from __future__ import annotations

import numpy as np
import pytest

from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.electrodynamics.conventions import SheetResponseValidation
from lno327.electrodynamics.reflection import LAB_LT_TANGENTIAL_E_BASIS, SheetReflection


def _validation(*, passed: bool = True) -> SheetResponseValidation:
    if passed:
        return SheetResponseValidation(
            finite=True,
            relative_imaginary_norm=0.0,
            relative_symmetry_residual=0.0,
            minimum_symmetric_eigenvalue=0.1,
            reality_tolerance=1e-9,
            symmetry_tolerance=1e-9,
            passivity_tolerance=1e-10,
        )
    return SheetResponseValidation(
        finite=True,
        relative_imaginary_norm=0.0,
        relative_symmetry_residual=0.0,
        minimum_symmetric_eigenvalue=-0.1,
        reality_tolerance=1e-9,
        symmetry_tolerance=1e-9,
        passivity_tolerance=1e-10,
    )


def _reflection(
    matrix: np.ndarray,
    *,
    q: np.ndarray | None = None,
    kappa: float = 2.0,
    validation_passed: bool = True,
) -> SheetReflection:
    q_si = np.array([3.0, 4.0]) if q is None else np.asarray(q, dtype=float)
    return SheetReflection(
        matrix_lt=np.asarray(matrix, dtype=complex),
        sigma_tilde_lt=np.eye(2, dtype=complex),
        q_lab_model=np.array([0.3, 0.4]),
        q_lab_si_m_inv=q_si,
        xi_eV=0.2,
        xi_si_s_inv=7.0,
        kappa_m_inv=kappa,
        theta_rad=0.0,
        basis=LAB_LT_TANGENTIAL_E_BASIS,
        sheet_validation=_validation(passed=validation_passed),
        metadata={},
    )


def test_diagonal_passive_reflections_use_signed_log1p_sum():
    first = _reflection(np.diag([-0.2, -0.5]))
    second = _reflection(np.diag([-0.3, -0.4]))
    point = passive_sheet_logdet(first, second, separation_m=0.1)

    propagation = np.exp(-0.4)
    expected_eigenvalues = propagation * np.array([0.06, 0.2])
    expected = float(np.sum(np.log1p(-expected_eigenvalues)))
    np.testing.assert_allclose(np.sort(point.round_trip_eigenvalues), np.sort(expected_eigenvalues))
    assert point.logdet == pytest.approx(expected)
    assert point.logdet < 0.0
    assert point.metadata["signed_real_logdet"] is True


def test_noncommuting_negative_symmetric_reflections_still_give_real_nonnegative_spectrum():
    theta = 0.43
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    first = _reflection(-np.diag([0.2, 0.5]))
    second_matrix = rotation @ (-np.diag([0.3, 0.6])) @ rotation.T
    second = _reflection(second_matrix)

    point = passive_sheet_logdet(first, second, separation_m=0.2)
    assert np.all(point.product_eigenvalues >= 0.0)
    assert np.all(point.round_trip_eigenvalues < 1.0)
    assert point.logdet <= 0.0


def test_logdet_rejects_incompatible_reflection_kinematics():
    first = _reflection(-0.2 * np.eye(2))
    second = _reflection(-0.3 * np.eye(2), q=np.array([3.0, 4.1]))
    with pytest.raises(ValueError, match="wavevectors do not match"):
        passive_sheet_logdet(first, second, separation_m=0.1)


def test_logdet_rejects_failed_sheet_validation():
    first = _reflection(-0.2 * np.eye(2), validation_passed=False)
    second = _reflection(-0.3 * np.eye(2))
    with pytest.raises(ValueError, match="sheet validations"):
        passive_sheet_logdet(first, second, separation_m=0.1)


def test_logdet_rejects_round_trip_branch_crossing():
    first = _reflection(-2.0 * np.eye(2), kappa=1.0)
    second = _reflection(-2.0 * np.eye(2), kappa=1.0)
    with pytest.raises(ValueError, match="branch point"):
        passive_sheet_logdet(first, second, separation_m=1e-6)
