from __future__ import annotations

import numpy as np
import pytest

from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.constants import C0, E2_OVER_HBAR, EV_TO_J, HBAR, SIGMA0
from lno327.electrodynamics.basis import q_crystal_to_lab, xy_to_lt_rotation
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.electrodynamics.static_sheet import (
    static_matsubara_kernel_to_sheet_response,
    static_sheet_response_to_reflection,
)
from lno327.numerics.grids import uniform_bz_mesh
from lno327.numerics.weights import k_weights
from lno327.response.config import KuboConfig
from lno327.response.effective_kernel import EffectiveEMKernel, effective_em_kernel_from_components
from lno327.response.finite_q import add_bubble, fermi_derivative, kubo_factor
from lno327.response.finite_q_bdg import finite_q_bdg_response_from_model_ansatz
from lno327.response.ward_validation import (
    PrimitiveWardRHS,
    primitive_ward_rhs_from_model_ansatz,
    primitive_ward_vectors_xy,
    validate_effective_ward_xy,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _synthetic_static_contract(
    *,
    q_model: np.ndarray = np.array([0.03, -0.04]),
    chi_bar: float = 2.0,
    dbar_t: float = 3.0,
    energy_scale_eV: float = 1.0,
) -> tuple[EffectiveEMKernel, object]:
    q = np.asarray(q_model, dtype=float)
    projection = xy_to_lt_rotation(float(q[0]), float(q[1]))
    transform = np.eye(3, dtype=float)
    transform[1:3, 1:3] = projection

    kernel_lt = np.zeros((3, 3), dtype=complex)
    kernel_lt[0, 0] = -float(chi_bar) / float(energy_scale_eV)
    kernel_lt[2, 2] = -float(dbar_t) * float(energy_scale_eV)
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
        metadata={"formula": "synthetic exact static contraction"},
    )
    ward = validate_effective_ward_xy(kernel, rhs, residual_tolerance=1e-13)
    assert ward.passed is True
    return kernel, ward


def test_exact_zero_matsubara_kubo_uses_thermodynamic_derivative():
    temperature_eV = 0.01
    expected = fermi_derivative(0.0, 0.0, temperature_eV, 1e-8)
    value = kubo_factor(
        0.0,
        0.0,
        0.5,
        0.5,
        0.0,
        temperature_eV=temperature_eV,
        eta_eV=1e-8,
    )
    assert value == pytest.approx(expected)
    assert np.isreal(value)


def test_add_bubble_selects_static_degenerate_limit_without_q0_flag():
    config = KuboConfig(
        omega_eV=0.0,
        temperature_eV=0.01,
        fermi_level_eV=0.0,
        eta_eV=1e-8,
        output_si=False,
    )
    accumulator = np.zeros((1, 1), dtype=complex)
    state = np.eye(1, dtype=complex)
    vertex = np.eye(1, dtype=complex)
    add_bubble(
        accumulator,
        (vertex,),
        (vertex,),
        np.array([0.0]),
        state,
        np.array([0.5]),
        np.array([0.0]),
        state,
        np.array([0.5]),
        0.0,
        1.0,
        config=config,
        static_limit=False,
    )
    expected = 0.5 * fermi_derivative(0.0, 0.0, 0.01, 1e-8)
    assert accumulator[0, 0] == pytest.approx(expected)
    assert np.isfinite(accumulator[0, 0])


def test_static_sheet_extracts_density_and_transverse_channels():
    kernel, ward = _synthetic_static_contract(chi_bar=2.0, dbar_t=3.0)
    response = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward,
        energy_scale_eV=1.0,
    )

    assert response.validation.passed is True
    assert response.chi_bar == pytest.approx(2.0)
    assert response.dbar_t == pytest.approx(3.0)
    assert response.kernel_lt[0, 0] == pytest.approx(-2.0)
    assert response.kernel_lt[2, 2] == pytest.approx(-3.0)
    assert response.metadata["conductivity_division_forbidden"] is True


def test_static_reflection_uses_fixed_beta_gamma_prefactors_and_rotation():
    q_crystal = np.array([0.03, -0.04])
    theta = 0.37
    q_lab = q_crystal_to_lab(q_crystal, theta)
    kernel, ward = _synthetic_static_contract(q_model=q_crystal, chi_bar=2.0, dbar_t=3.0)
    response = static_matsubara_kernel_to_sheet_response(kernel, ward)
    reflection = static_sheet_response_to_reflection(
        response,
        q_lab_model=q_lab,
        theta_rad=theta,
    )

    lattice = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m
    beta = EV_TO_J * lattice / (HBAR * C0)
    gamma = E2_OVER_HBAR / SIGMA0
    q_norm = np.linalg.norm(q_lab)
    expected_l = (gamma / beta) * 2.0 / q_norm
    expected_t = (gamma * beta) * 3.0 / q_norm

    assert reflection.lambda_l == pytest.approx(expected_l)
    assert reflection.lambda_t == pytest.approx(expected_t)
    assert reflection.matrix_lt[0, 0] == pytest.approx(-expected_l / (2.0 + expected_l))
    assert reflection.matrix_lt[1, 1] == pytest.approx(-expected_t / (2.0 + expected_t))
    assert reflection.matrix_lt[0, 1] == 0.0
    assert reflection.matrix_lt[1, 0] == 0.0
    assert reflection.xi_si_s_inv == 0.0
    assert reflection.kappa_m_inv == pytest.approx(q_norm / lattice)
    assert reflection.metadata["matsubara_prime_weight"] == 0.5


def test_static_strong_screening_limit_and_signed_logdet():
    kernel, ward = _synthetic_static_contract(chi_bar=1e8, dbar_t=1e8)
    response = static_matsubara_kernel_to_sheet_response(kernel, ward)
    reflection = static_sheet_response_to_reflection(
        response,
        q_lab_model=kernel.q_model,
        theta_rad=0.0,
    )

    np.testing.assert_allclose(reflection.matrix_lt, -np.eye(2), rtol=1e-4, atol=1e-4)
    point = passive_sheet_logdet(
        reflection,
        reflection,
        separation_m=20e-9,
    )
    assert point.logdet < 0.0
    assert point.metadata["frequency_sector"] == "zero_matsubara"
    assert point.metadata["zero_matsubara_prime_weight"] == 0.5


def test_static_passivity_failure_is_fail_closed_before_reflection():
    kernel, ward = _synthetic_static_contract(chi_bar=1.0, dbar_t=-0.1)
    response = static_matsubara_kernel_to_sheet_response(kernel, ward)
    assert response.validation.passed is False
    with pytest.raises(ValueError, match="zero-Matsubara sheet response failed hard validation"):
        static_sheet_response_to_reflection(
            response,
            q_lab_model=kernel.q_model,
            theta_rad=0.0,
        )


def test_static_contract_rejects_positive_frequency_kernel():
    kernel, ward = _synthetic_static_contract()
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
        static_matsubara_kernel_to_sheet_response(positive, ward)


def test_real_two_band_arbitrary_q_zero_mode_is_finite_and_ward_closed():
    """Smoke-test exact xi=0; strict static physics gates need converged local runs."""

    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("spm", phase_vertex="bond_endpoint_gauge")
    pairing_params = model.build_pairing_params(0.1)
    q_model = np.array([0.03, 0.02])
    points = uniform_bz_mesh(2)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )

    components = finite_q_bdg_response_from_model_ansatz(
        model.spec,
        ansatz,
        0.0,
        q_model,
        points,
        weights,
        config,
        pairing_params,
    )
    kernel = effective_em_kernel_from_components(
        components,
        q_model=q_model,
        xi_eV=0.0,
    )
    rhs = primitive_ward_rhs_from_model_ansatz(
        model.spec,
        ansatz,
        q_model,
        points,
        weights,
        config,
        pairing_params,
    )
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=1e-6,
    )
    static = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward,
        longitudinal_tolerance=1.0,
        mixing_tolerance=1.0,
        passivity_tolerance=1.0,
    )

    assert np.isfinite(kernel.matrix.real).all()
    assert np.isfinite(kernel.matrix.imag).all()
    assert ward.passed is True
    assert static.validation.ward_passed is True
    assert np.isfinite(static.chi_bar)
    assert np.isfinite(static.dbar_t)
    assert static.metadata["frequency_sector"] == "zero_matsubara"
