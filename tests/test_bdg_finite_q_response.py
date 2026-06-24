from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation" / "scripts" / "response"))

from bdg_bubble_ward_transfer_common import (  # noqa: E402
    StageSC0bInputs,
    _status_and_dominant,
    audit_contact_remainder_pairing,
    audit_pairing,
    compute_equal_time_remainder,
    convention_summary,
    needed_eta2_contact_if_only_eta2,
    stageSC_0c_status,
)
from bdg_contact_identity_common import (  # noqa: E402
    assess_stageSC_0d,
    bdg_contact_identity_residual,
    normal_contact_identity_residual,
)
from bdg_shifted_grid_assembly_common import (  # noqa: E402
    case_dominant_failure,
    commensurate_grid_spec,
    matrix_fermi_function,
    shifted_trace_direct_terms,
)
from lno327.bdg_finite_q_response import (
    _amplitude_vertex,
    _eta2_phase_vertex,
    _kubo_factor,
    bdg_finite_q_response_imag_axis,
    collective_goldstone_counterterm,
)
from lno327.conductivity import KuboConfig, k_weights, uniform_bz_mesh
from lno327.pairing import PairingAmplitudes
from lno327.ward_response import normal_physical_density_current_response_imag_axis
from lno327.tb_fourier import (
    normal_state_hamiltonian_from_hoppings,
    peierls_hamiltonian_contact_vertex,
    peierls_hamiltonian_vector_vertex,
)


def _inputs(delta0: float = 0.04):
    points = uniform_bz_mesh(3)
    return (
        np.array([0.01, 0.0]),
        points,
        k_weights(points),
        KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False),
        PairingAmplitudes(delta0_eV=delta0),
    )


def test_bdg_finite_q_response_shapes_with_and_without_phase_correction():
    q, points, weights, config, amp = _inputs()
    for include_phase in (False, True):
        result = bdg_finite_q_response_imag_axis(
            "spm",
            config.omega_eV,
            q,
            points,
            weights,
            config,
            amp,
            include_phase_correction=include_phase,
            collective_mode="phase_only",
        )
        assert result.bare_bubble.shape == (3, 3)
        assert result.direct.shape == (3, 3)
        assert result.bare_total.shape == (3, 3)
        assert result.gauge_restored.shape == (3, 3)
        assert result.phase_coupling_left.shape == (3,)
        assert result.phase_coupling_right.shape == (3,)
        assert isinstance(result.phase_phase_bubble, complex)
        assert isinstance(result.phase_phase_direct, complex)
        assert result.phase_phase == result.phase_phase_total
        assert result.minus_schur.shape == (3, 3)
        assert result.plus_schur.shape == (3, 3)
        assert np.all(np.isfinite(result.gauge_restored))
        assert result.metadata["nambu_prefactor"] == 0.5
        assert result.metadata["collective_channels"] == ["global_phase_only"]
        assert result.metadata["phase_phase_total_definition"] == "bubble + direct"


def test_small_phase_phase_has_clear_warning_metadata():
    q, points, weights, config, amp = _inputs(delta0=1e-16)
    with pytest.warns(RuntimeWarning, match="Global phase correction skipped"):
        result = bdg_finite_q_response_imag_axis(
            "spm",
            config.omega_eV,
            q,
            points,
            weights,
            config,
            amp,
            include_phase_correction=True,
            collective_mode="phase_only",
        )
    assert result.metadata["phase_correction_status"] == "singular_phase_phase"
    assert result.metadata["warning"]


def test_schur_complement_defaults_to_minus_sign_and_records_plus_minus_ward():
    q, points, weights, config, amp = _inputs()
    result = bdg_finite_q_response_imag_axis(
        "spm",
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        include_phase_correction=True,
        collective_mode="phase_only",
    )
    expected = result.bare_total - np.outer(result.phase_coupling_left, result.phase_coupling_right) / result.phase_phase
    assert np.allclose(result.gauge_restored, expected)
    assert result.metadata["phase_correction_formula"] == "Pi_GI = Pi_bare - K_mu_theta K_theta_nu / K_theta_theta"
    assert result.metadata["phase_correction_sign_checked"] is True
    assert "ward_residual_minus_schur" in result.metadata
    assert "ward_residual_plus_schur" in result.metadata
    assert result.metadata["selected_gauge_restored"] == "minus_schur"


def test_delta0_zero_uses_true_bdg_by_default_not_normal_backend_shortcut():
    q, points, weights, config, amp = _inputs(delta0=0.0)
    result = bdg_finite_q_response_imag_axis(
        "spm",
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        include_phase_correction=True,
    )
    normal = normal_physical_density_current_response_imag_axis(points, config, q, weights)
    assert "normal_limit_delegated_to" not in result.metadata
    assert result.metadata["normal_backend_reference_used"] is False
    reference = bdg_finite_q_response_imag_axis(
        "spm",
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        include_phase_correction=True,
        use_normal_backend_in_delta0_limit=True,
    )
    assert np.allclose(reference.gauge_restored, normal)
    assert reference.metadata["normal_limit_delegated_to"] == "normal_physical_density_current_response_components_imag_axis"


def test_q0_velocity_approximation_is_not_marked_gauge_closed():
    q, points, weights, config, amp = _inputs()
    result = bdg_finite_q_response_imag_axis(
        "spm",
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        current_vertex="q0_velocity",
        collective_mode="phase_only",
    )
    assert result.metadata["finite_q_current_vertex_status"] == "q0_velocity_vertex_approximation_not_gauge_closed"


def test_phase_vertex_options_and_onsite_s_validation_pairing_run():
    q, points, weights, config, amp = _inputs()
    midpoint = bdg_finite_q_response_imag_axis(
        "onsite_s",
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        phase_vertex="midpoint",
        collective_mode="phase_only",
    )
    symmetric = bdg_finite_q_response_imag_axis(
        "onsite_s",
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        phase_vertex="symmetric_kpm",
        include_phase_phase_direct=False,
        collective_mode="phase_only",
    )
    assert midpoint.metadata["validation_only_pairing"] is True
    assert midpoint.metadata["phase_vertex"] == "midpoint"
    assert symmetric.metadata["phase_vertex"] == "symmetric_kpm"
    assert symmetric.metadata["phase_kernel_status"] == "bubble_only_not_expected_to_gauge_close"


def test_amplitude_phase_collective_shapes_and_goldstone_counterterm():
    q, points, weights, config, amp = _inputs()
    result = bdg_finite_q_response_imag_axis(
        "onsite_s",
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
    )
    assert result.collective_bubble.shape == (2, 2)
    assert result.collective_counterterm.shape == (2, 2)
    assert result.collective_total.shape == (2, 2)
    assert result.em_collective_left.shape == (3, 2)
    assert result.collective_em_right.shape == (2, 3)
    assert result.amplitude_phase_schur.shape == (3, 3)
    assert result.metadata["gauge_restored_selected"] == "amplitude_phase_schur"
    cg = collective_goldstone_counterterm("onsite_s", points, weights, config, amp, "symmetric_kpm")
    assert np.allclose(result.collective_counterterm, cg * np.eye(2))
    assert result.metadata["goldstone_counterterm_Cg"] == cg


def test_amplitude_and_eta2_vertices_have_bdg_shape():
    phi = np.eye(4, dtype=complex)
    assert _amplitude_vertex(phi).shape == (8, 8)
    assert _eta2_phase_vertex(phi).shape == (8, 8)


def test_static_kubo_factor_uses_derivative_limit_for_degenerate_terms():
    value = _kubo_factor(
        0.0,
        0.0,
        0.5,
        0.5,
        0.0,
        static_limit=True,
        temperature_eV=0.01,
        eta_eV=1e-8,
    )
    assert value == pytest.approx(-25.0)


def test_stageSC_0b_helper_uses_stageSC_0_best_convention():
    convention = convention_summary(0.04)
    assert convention["candidate"] == "A"
    assert convention["candidate_ordering"] == "rho_Hp_minus_Hm_rho"
    assert convention["qV_sign"] == -1
    assert convention["C_eta2"] == 2j * 0.04


def test_stageSC_0b_onsite_s_band_pair_identity_is_closed():
    result = audit_pairing(StageSC0bInputs(pairing="onsite_s"))
    assert result["max_band_pair_identity_abs"] < 1e-10


def test_stageSC_0b_records_right_vertex_orientation_difference():
    result = audit_pairing(StageSC0bInputs(pairing="onsite_s"))
    assert "right_vertex_impl_vs_explicit_max_abs" in result
    assert set(result["right_vertex_impl_vs_explicit_by_channel"]) == {"rho", "Vx", "Vy", "eta1", "eta2"}


def test_stageSC_0b_right_vertex_mismatch_prevents_passed():
    status, dominant = _status_and_dominant(
        band_abs=1e-12,
        transfer_abs=1e-12,
        right_abs=1e-6,
        contact_abs=1e-12,
    )
    assert status == "FAILED"
    assert dominant == "right_vertex_orientation"


def test_stageSC_0b_bubble_transfer_mismatch_prevents_passed():
    status, dominant = _status_and_dominant(
        band_abs=1e-12,
        transfer_abs=1e-6,
        right_abs=1e-12,
        contact_abs=1e-12,
    )
    assert status == "FAILED"
    assert dominant == "bubble_transfer"


def test_stageSC_0b_contact_only_remainder_is_not_passed():
    status, dominant = _status_and_dominant(
        band_abs=1e-12,
        transfer_abs=1e-12,
        right_abs=1e-12,
        contact_abs=5e-7,
    )
    assert status == "MONITOR"
    assert dominant == "contact_remainder"


def test_stageSC_0c_equal_time_remainder_records_explicit_reverse_q_orientation():
    result = compute_equal_time_remainder(StageSC0bInputs(pairing="onsite_s"), (0.01, 0.0))
    assert "equal_time_remainder_explicit_reverse_q" in result
    assert "equal_time_remainder_impl_conjugate" in result
    assert "equal_time_remainder_orientation_diff" in result
    assert result["equal_time_remainder_orientation_diff_max_abs"] < 1e-10


def test_stageSC_0c_needed_eta2_contact_formula():
    components = {
        "D_0B_existing": {"available": True, "value": 1.0 + 2.0j},
        "D_xB_existing": {"available": True, "value": -0.5 + 0.25j},
        "D_yB_existing": {"available": False, "value": None},
        "D_eta2B_existing": {"available": False, "value": None},
    }
    e_b = 0.25 - 0.75j
    q = (0.01, 0.02)
    omega = 0.01
    delta0 = 0.04
    expected = -(e_b + 1j * omega * (1.0 + 2.0j) + q[0] * (-0.5 + 0.25j)) / (2j * delta0)
    actual = needed_eta2_contact_if_only_eta2(e_b, components, q, omega, delta0)
    assert actual == pytest.approx(expected)


def test_stageSC_0c_unavailable_direct_blocks_prevent_passed():
    assert stageSC_0c_status(1e-12, 1e-12, unavailable_blocks=True) == "FAILED"


def test_stageSC_0c_large_onsite_contact_residual_fails():
    result = audit_contact_remainder_pairing(StageSC0bInputs(pairing="onsite_s"))
    assert result["max_contact_closure_abs"] >= 1e-5
    assert result["status"] == "FAILED"


def test_stageSC_0c_script_does_not_call_casimir_pipeline():
    script = ROOT / "validation" / "scripts" / "response" / "stageSC_0c_bdg_contact_remainder_decomposition_audit.py"
    text = script.read_text(encoding="utf-8")
    assert "run_material_casimir_figures" not in text
    assert "outputs/material_casimir" not in text


@pytest.mark.parametrize("k_model", [(0.13, 0.27), (0.41, -0.22), (1.11, 0.73)])
@pytest.mark.parametrize("q_model", [(0.01, 0.0), (0.01, 0.01)])
@pytest.mark.parametrize("direction_j", ["x", "y"])
def test_stageSC_0d_normal_contact_identity(k_model, q_model, direction_j):
    residual = normal_contact_identity_residual(k_model, q_model, direction_j)
    assert np.max(np.abs(residual)) < 1e-12


@pytest.mark.parametrize("k_model", [(0.13, 0.27), (0.41, -0.22), (-0.64, 1.37)])
@pytest.mark.parametrize("q_model", [(0.01, 0.0), (0.01, 0.01)])
@pytest.mark.parametrize("direction_j", ["x", "y"])
def test_stageSC_0d_onsite_s_bdg_contact_identity(k_model, q_model, direction_j):
    residual = bdg_contact_identity_residual(k_model, q_model, direction_j)
    assert np.max(np.abs(residual)) < 1e-12


def test_stageSC_0d_script_does_not_call_casimir_pipeline():
    script = (
        ROOT
        / "validation"
        / "scripts"
        / "response"
        / "stageSC_0d_bdg_exact_contact_identity_quadrature_audit.py"
    )
    text = script.read_text(encoding="utf-8")
    assert "run_material_casimir_figures" not in text
    assert "outputs/material_casimir" not in text
    assert '"formal_casimir_ran": False' in text


def test_stageSC_0d_partA_failure_cannot_pass():
    status, dominant, interpretation = assess_stageSC_0d(1e-9, 1e-12, 1e-12)
    assert status == "FAILED"
    assert dominant == "contact_vertex_identity"
    assert interpretation == "contact_vertex_formula_or_bdg_hole_routing_failed"


@pytest.mark.parametrize(
    ("part_b_abs", "fixed_q_abs", "expected"),
    [(1e-4, 1e-4, "shift_invariant_quadrature"), (1e-12, 1e-4, "fixed_q_quadrature")],
)
def test_stageSC_0d_closure_failure_is_not_contact_identity(part_b_abs, fixed_q_abs, expected):
    status, dominant, _ = assess_stageSC_0d(1e-13, part_b_abs, fixed_q_abs)
    assert status == "FAILED"
    assert dominant == expected


def test_stageSC_0e_matrix_fermi_function_is_hermitian_and_reconstructs_eigenbasis():
    hamiltonian = np.array([[0.2, 0.1 - 0.03j], [0.1 + 0.03j, -0.4]], dtype=complex)
    cfg = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, output_si=False)
    actual = matrix_fermi_function(hamiltonian, cfg)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    occupations = 1.0 / (np.exp(np.clip(eigenvalues / cfg.temperature_eV, -700.0, 700.0)) + 1.0)
    expected = eigenvectors @ np.diag(occupations) @ eigenvectors.conjugate().T
    assert np.allclose(actual, actual.conjugate().T)
    assert np.allclose(actual, expected)


def test_stageSC_0e_shifted_trace_and_direct_cancel_for_finite_fourier_toy():
    hopping = 0.7
    terms = [
        ((1, 0), hopping * np.eye(4, dtype=complex)),
        ((-1, 0), hopping * np.eye(4, dtype=complex)),
    ]
    kx, ky = 0.31, 0.0
    qx, qy = 0.2, 0.0
    hamiltonian = normal_state_hamiltonian_from_hoppings(kx, ky, hopping_terms=terms)
    vector_plus = peierls_hamiltonian_vector_vertex(
        kx + 0.5 * qx, ky, -qx, -qy, "x", hopping_terms=terms
    )
    vector_minus = peierls_hamiltonian_vector_vertex(
        kx - 0.5 * qx, ky, -qx, -qy, "x", hopping_terms=terms
    )
    q_contact = qx * peierls_hamiltonian_contact_vertex(
        kx, ky, qx, qy, "x", "x", hopping_terms=terms
    )
    cfg = KuboConfig.from_kelvin(
        omega_eV=0.01,
        temperature_K=300.0,
        fermi_level_eV=1.3,
        output_si=False,
    )
    shifted, direct = shifted_trace_direct_terms(
        hamiltonian, np.eye(4), vector_plus, vector_minus, q_contact, cfg
    )
    assert abs(shifted) > 1e-6
    assert abs(shifted + direct) < 1e-12


def test_stageSC_0e_grid_step_commensurate_half_q_is_one_grid_spacing():
    spec = commensurate_grid_spec(24, "grid_step_commensurate")
    assert spec["q_half_in_grid_steps"] == pytest.approx([1.0, 0.0])
    assert spec["q_half_lands_on_grid"] is True


def test_stageSC_0e_records_distinct_half_and_grid_step_commensurate_grids():
    half = commensurate_grid_spec(24, "half_step_commensurate")
    full = commensurate_grid_spec(24, "grid_step_commensurate")
    assert half["grid_type"] == "half_step_commensurate"
    assert full["grid_type"] == "grid_step_commensurate"
    assert half["q_half_in_grid_steps"] == pytest.approx([0.5, 0.0])
    assert half["q_half_lands_on_grid"] is False
    assert full["q_half_lands_on_grid"] is True


def test_stageSC_0e_direct_failure_has_priority_over_band_shifted_failure():
    dominant = case_dominant_failure(1e-5, 1e-4, 1e-4)
    assert dominant == "direct_expectation_mismatch"


def test_stageSC_0e_script_does_not_call_casimir_pipeline():
    script = (
        ROOT
        / "validation"
        / "scripts"
        / "response"
        / "stageSC_0e_bdg_shifted_grid_response_assembly_audit.py"
    )
    text = script.read_text(encoding="utf-8")
    assert "run_material_casimir_figures" not in text
    assert "outputs/material_casimir" not in text
    assert '"formal_casimir_ran": False' in text
