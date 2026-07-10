from __future__ import annotations

import numpy as np

from lno327.numerics.grids import uniform_bz_mesh
from lno327.numerics.weights import k_weights
from lno327.response.config import KuboConfig
from lno327.response.effective_kernel import EffectiveEMKernel, effective_em_kernel_from_components
from lno327.response.finite_q_bdg import finite_q_bdg_response_from_model_ansatz
from lno327.response.ward_validation import (
    PrimitiveWardRHS,
    primitive_ward_rhs_from_model_ansatz,
    primitive_ward_vectors_xy,
    project_ward_validation_xy_to_lt,
    validate_effective_ward_xy,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _synthetic_closed_contract() -> tuple[EffectiveEMKernel, PrimitiveWardRHS]:
    q = np.array([0.03, -0.04])
    xi = 0.02
    delta0 = 0.1
    k_ss = np.array(
        [
            [1.2, 0.1j, -0.2],
            [-0.1j, 0.8, 0.05],
            [0.2, -0.03, 0.6],
        ],
        dtype=complex,
    )
    k_seta = np.array(
        [[0.2, 0.05j], [0.1j, 0.3], [-0.15, 0.08j]],
        dtype=complex,
    )
    k_etas = np.array(
        [[0.1, -0.2j, 0.05], [0.04j, 0.25, -0.1]],
        dtype=complex,
    )
    k_etaeta = np.array([[2.0, 0.1j], [-0.1j, 1.5]], dtype=complex)
    inverse = np.linalg.inv(k_etaeta)
    k_eff = k_ss - k_seta @ inverse @ k_etas
    u_left, u_right, w_left, w_right = primitive_ward_vectors_xy(xi, q, delta0)
    rhs_left = u_left @ k_ss + w_left @ k_etas
    rhs_right = k_ss @ u_right + k_seta @ w_right
    kernel = EffectiveEMKernel(
        k_ss=k_ss,
        k_seta=k_seta,
        k_etas=k_etas,
        k_etaeta=k_etaeta,
        k_eff=k_eff,
        q_model=q,
        xi_eV=xi,
        schur_condition_number=float(np.linalg.cond(k_etaeta)),
        schur_inverse_method="inv",
        metadata={"basis": "crystal_A0_xy"},
    )
    rhs = PrimitiveWardRHS(
        left=rhs_left,
        right=rhs_right,
        q_model=q,
        xi_eV=xi,
        delta0_eV=delta0,
        metadata={"formula": "synthetic exact extended identity"},
    )
    return kernel, rhs


def test_arbitrary_q_xy_validator_closes_exact_schur_identity():
    kernel, rhs = _synthetic_closed_contract()
    report = validate_effective_ward_xy(kernel, rhs, residual_tolerance=1e-12)

    assert report.passed is True
    assert report.primitive_closed is True
    assert report.effective_closed is True
    assert report.left.primitive_relative_residual < 1e-13
    assert report.right.primitive_relative_residual < 1e-13
    assert report.left.effective_relative_residual < 1e-13
    assert report.right.effective_relative_residual < 1e-13
    np.testing.assert_allclose(report.u_left, [0.02j, 0.03, -0.04])
    np.testing.assert_allclose(report.u_right, [-0.02j, 0.03, -0.04])


def test_xy_to_lt_is_only_an_orthogonal_residual_projection():
    kernel, rhs = _synthetic_closed_contract()
    perturbed = PrimitiveWardRHS(
        left=rhs.left + np.array([1e-5, -2e-5j, 3e-5]),
        right=rhs.right + np.array([-2e-5j, 1e-5, -1e-5]),
        q_model=rhs.q_model,
        xi_eV=rhs.xi_eV,
        delta0_eV=rhs.delta0_eV,
        metadata={},
    )
    report = validate_effective_ward_xy(kernel, perturbed, residual_tolerance=1e-12)
    diagnostics = project_ward_validation_xy_to_lt(report)

    assert report.passed is False
    np.testing.assert_allclose(diagnostics.xy_norms, diagnostics.lt_norms, rtol=1e-14, atol=1e-14)


def test_rhs_builder_closes_real_two_band_response_for_nonzero_qy():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("spm", phase_vertex="bond_endpoint_gauge")
    pairing_params = model.build_pairing_params(0.1)
    q_model = np.array([0.03, 0.02])
    points = uniform_bz_mesh(2)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.02,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    components = finite_q_bdg_response_from_model_ansatz(
        model.spec,
        ansatz,
        config.omega_eV,
        q_model,
        points,
        weights,
        config,
        pairing_params,
    )
    kernel = effective_em_kernel_from_components(
        components,
        q_model=q_model,
        xi_eV=config.omega_eV,
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
    report = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=1e-7,
    )

    np.testing.assert_array_equal(rhs.q_model, q_model)
    assert rhs.metadata["basis"] == "crystal_A0_xy"
    assert rhs.metadata["formula"] == "R_S = equal_forward - delta_v_mid + qM_mid"
    assert report.metadata["lt_projection_is_diagnostic_only"] is True
    assert report.primitive_closed is True
    assert report.effective_closed is True
    assert report.left.effective_relative_residual < 1e-7
    assert report.right.effective_relative_residual < 1e-7


def test_validator_rejects_mismatched_q_or_frequency():
    kernel, rhs = _synthetic_closed_contract()
    wrong_q = PrimitiveWardRHS(
        left=rhs.left,
        right=rhs.right,
        q_model=np.array([0.02, -0.04]),
        xi_eV=rhs.xi_eV,
        delta0_eV=rhs.delta0_eV,
        metadata={},
    )
    with np.testing.assert_raises_regex(ValueError, "q_model"):
        validate_effective_ward_xy(kernel, wrong_q)

    wrong_xi = PrimitiveWardRHS(
        left=rhs.left,
        right=rhs.right,
        q_model=rhs.q_model,
        xi_eV=0.03,
        delta0_eV=rhs.delta0_eV,
        metadata={},
    )
    with np.testing.assert_raises_regex(ValueError, "xi_eV"):
        validate_effective_ward_xy(kernel, wrong_xi)
