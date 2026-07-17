from __future__ import annotations

import numpy as np
import pytest

from lno327.electrodynamics.basis import xy_to_lt_rotation
from lno327.electrodynamics.static_gauge_projection import (
    PROJECT_AFTER_VALIDATED_WARD,
    RAW_FAIL_CLOSED,
    static_longitudinal_gauge_projector_lt,
    static_matsubara_kernel_to_sheet_response_with_policy,
)
from lno327.electrodynamics.static_sheet import (
    static_matsubara_kernel_to_sheet_response,
    static_sheet_response_to_reflection,
)
from lno327.response.effective_kernel import EffectiveEMKernel
from lno327.response.ward_validation import (
    PrimitiveWardRHS,
    primitive_ward_vectors_xy,
    validate_effective_ward_xy,
)


def _synthetic_static_contract(
    *,
    kll_noise: float,
    chi_bar: float = 2.0,
    dbar_t: float = 3.0,
) -> tuple[EffectiveEMKernel, PrimitiveWardRHS]:
    q = np.asarray([0.03, 0.02], dtype=float)
    rotation = xy_to_lt_rotation(float(q[0]), float(q[1]))
    transform = np.eye(3, dtype=float)
    transform[1:3, 1:3] = rotation

    kernel_lt = np.zeros((3, 3), dtype=complex)
    kernel_lt[0, 0] = -chi_bar
    kernel_lt[2, 2] = -dbar_t
    kernel_lt[1, 1] = float(kll_noise)
    kernel_lt[0, 1] = 0.1 * float(kll_noise)
    kernel_lt[1, 0] = -0.08 * float(kll_noise)
    kernel_lt[1, 2] = 0.05 * float(kll_noise)
    kernel_lt[2, 1] = -0.04 * float(kll_noise)
    kernel_xy = transform.T @ kernel_lt @ transform

    kernel = EffectiveEMKernel(
        k_ss=kernel_xy,
        k_seta=np.zeros((3, 2), dtype=complex),
        k_etas=np.zeros((2, 3), dtype=complex),
        k_etaeta=np.eye(2, dtype=complex),
        k_eff=kernel_xy,
        q_model=q,
        xi_eV=0.0,
        schur_condition_number=1.0,
        schur_inverse_method="inv",
        metadata={"basis": "crystal_A0_xy"},
    )
    u_left, u_right, _, _ = primitive_ward_vectors_xy(0.0, q, 0.0)
    rhs = PrimitiveWardRHS(
        left=u_left @ kernel_xy,
        right=kernel_xy @ u_right,
        q_model=q,
        xi_eV=0.0,
        delta0_eV=0.0,
        metadata={"formula": "synthetic static identity with L quadrature noise"},
    )
    return kernel, rhs


def _closed_ward(kernel: EffectiveEMKernel, rhs: PrimitiveWardRHS):
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=1e-12,
        absolute_residual_tolerance=1e-14,
    )
    assert ward.passed is True
    return ward


def test_static_projector_removes_entire_longitudinal_row_and_column():
    projector = static_longitudinal_gauge_projector_lt()
    np.testing.assert_array_equal(projector, np.diag([1.0, 0.0, 1.0]))
    assert projector.flags.writeable is False

    matrix = np.arange(9, dtype=float).reshape(3, 3).astype(complex)
    projected = projector @ matrix @ projector
    np.testing.assert_array_equal(projected[1, :], 0.0)
    np.testing.assert_array_equal(projected[:, 1], 0.0)
    np.testing.assert_array_equal(
        projected[np.ix_([0, 2], [0, 2])],
        matrix[np.ix_([0, 2], [0, 2])],
    )


def test_projection_passes_small_raw_leakage_and_preserves_physical_channels():
    kernel, rhs = _synthetic_static_contract(kll_noise=2e-5)
    ward = _closed_ward(kernel, rhs)

    raw = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward,
        longitudinal_tolerance=1e-7,
    )
    assert raw.validation.passed is True
    assert raw.validation.longitudinal_warning is True
    assert raw.validation.relative_longitudinal_gauge_residual > 1e-7
    assert raw.validation.relative_longitudinal_gauge_residual < 1e-5

    projected = static_matsubara_kernel_to_sheet_response_with_policy(
        kernel,
        ward,
        longitudinal_policy=PROJECT_AFTER_VALIDATED_WARD,
        projection_raw_longitudinal_ceiling=1e-5,
        longitudinal_tolerance=1e-7,
    )

    assert projected.validation.passed is True
    assert projected.validation.relative_longitudinal_gauge_residual == 0.0
    np.testing.assert_array_equal(projected.kernel_lt[1, :], 0.0)
    np.testing.assert_array_equal(projected.kernel_lt[:, 1], 0.0)
    np.testing.assert_array_equal(
        projected.kernel_lt[np.ix_([0, 2], [0, 2])],
        raw.kernel_lt[np.ix_([0, 2], [0, 2])],
    )
    assert projected.chi_bar == raw.chi_bar
    assert projected.dbar_t == raw.dbar_t
    assert projected.metadata["gauge_projection_applied"] is True
    assert projected.metadata["physical_A0_T_block_preserved_exactly"] is True
    assert projected.metadata["raw_relative_longitudinal_gauge_residual"] == pytest.approx(
        raw.validation.relative_longitudinal_gauge_residual
    )
    assert projected.metadata["relative_projection_correction_norm"] > 0.0
    assert projected.metadata["relative_projection_correction_norm"] < 1e-5
    assert projected.metadata["raw_kernel_lt"].flags.writeable is False


def test_projection_does_not_change_static_reflection_or_logical_physical_output():
    noisy_kernel, noisy_rhs = _synthetic_static_contract(kll_noise=2e-5)
    noisy_ward = _closed_ward(noisy_kernel, noisy_rhs)
    projected = static_matsubara_kernel_to_sheet_response_with_policy(
        noisy_kernel,
        noisy_ward,
        longitudinal_policy=PROJECT_AFTER_VALIDATED_WARD,
    )

    clean_kernel, clean_rhs = _synthetic_static_contract(kll_noise=0.0)
    clean_ward = _closed_ward(clean_kernel, clean_rhs)
    clean = static_matsubara_kernel_to_sheet_response(clean_kernel, clean_ward)

    projected_reflection = static_sheet_response_to_reflection(
        projected,
        q_lab_model=noisy_kernel.q_model,
        theta_rad=0.0,
    )
    clean_reflection = static_sheet_response_to_reflection(
        clean,
        q_lab_model=clean_kernel.q_model,
        theta_rad=0.0,
    )

    assert projected.chi_bar == pytest.approx(clean.chi_bar, rel=0.0, abs=1e-14)
    assert projected.dbar_t == pytest.approx(clean.dbar_t, rel=0.0, abs=1e-14)
    np.testing.assert_allclose(
        projected_reflection.matrix_lt,
        clean_reflection.matrix_lt,
        rtol=0.0,
        atol=1e-14,
    )
    assert projected_reflection.lambda_l == pytest.approx(
        clean_reflection.lambda_l, rel=0.0, abs=1e-14
    )
    assert projected_reflection.lambda_t == pytest.approx(
        clean_reflection.lambda_t, rel=0.0, abs=1e-14
    )


def test_projection_records_large_raw_longitudinal_leakage_without_blocking():
    kernel, rhs = _synthetic_static_contract(kll_noise=1e-3)
    ward = _closed_ward(kernel, rhs)

    projected = static_matsubara_kernel_to_sheet_response_with_policy(
        kernel,
        ward,
        longitudinal_policy=PROJECT_AFTER_VALIDATED_WARD,
        projection_raw_longitudinal_ceiling=1e-5,
    )

    assert projected.validation.passed is True
    assert projected.metadata["gauge_projection_applied"] is True
    assert projected.metadata["raw_relative_longitudinal_gauge_residual"] > 1e-5
    assert projected.metadata["projected_relative_longitudinal_gauge_residual"] == 0.0


def test_projection_rejects_failed_mixed_ward_even_when_raw_leakage_is_small():
    kernel, rhs = _synthetic_static_contract(kll_noise=2e-5)
    bad_rhs = PrimitiveWardRHS(
        left=rhs.left + np.asarray([1e-3, 0.0, 0.0], dtype=complex),
        right=rhs.right + np.asarray([1e-3, 0.0, 0.0], dtype=complex),
        q_model=rhs.q_model,
        xi_eV=rhs.xi_eV,
        delta0_eV=rhs.delta0_eV,
        metadata={"formula": "deliberately broken Ward RHS"},
    )
    bad_ward = validate_effective_ward_xy(
        kernel,
        bad_rhs,
        residual_tolerance=1e-12,
        absolute_residual_tolerance=0.0,
    )
    assert bad_ward.passed is False

    with pytest.raises(ValueError, match="projection prerequisites failed"):
        static_matsubara_kernel_to_sheet_response_with_policy(
            kernel,
            bad_ward,
            longitudinal_policy=PROJECT_AFTER_VALIDATED_WARD,
        )


def test_raw_policy_records_longitudinal_warning_without_blocking():
    kernel, rhs = _synthetic_static_contract(kll_noise=2e-5)
    ward = _closed_ward(kernel, rhs)

    response = static_matsubara_kernel_to_sheet_response_with_policy(
        kernel,
        ward,
        longitudinal_policy=RAW_FAIL_CLOSED,
        longitudinal_tolerance=1e-7,
    )

    assert response.validation.passed is True
    assert response.validation.longitudinal_warning is True
    assert response.metadata["static_longitudinal_policy"] == RAW_FAIL_CLOSED
    assert response.metadata["gauge_projection_applied"] is False
    assert response.kernel_lt[1, 1] != 0.0


def test_projection_is_forbidden_for_positive_matsubara_frequency():
    kernel, rhs = _synthetic_static_contract(kll_noise=2e-5)
    ward = _closed_ward(kernel, rhs)
    positive = EffectiveEMKernel(
        k_ss=kernel.k_ss,
        k_seta=kernel.k_seta,
        k_etas=kernel.k_etas,
        k_etaeta=kernel.k_etaeta,
        k_eff=kernel.k_eff,
        q_model=kernel.q_model,
        xi_eV=0.01,
        schur_condition_number=kernel.schur_condition_number,
        schur_inverse_method=kernel.schur_inverse_method,
        metadata=kernel.metadata,
    )

    with pytest.raises(ValueError, match="kernel.xi_eV == 0"):
        static_matsubara_kernel_to_sheet_response_with_policy(
            positive,
            ward,
            longitudinal_policy=PROJECT_AFTER_VALIDATED_WARD,
        )
