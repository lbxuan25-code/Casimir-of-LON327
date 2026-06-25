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
from bdg_quadrature_strategy_common import (  # noqa: E402
    composite_uniform_quadrature,
    compute_bdg_components_for_composite_grid,
    recommend_strategy,
    single_composite_schur,
    strategy_origins,
)
from bdg_commensurate_q_common import (  # noqa: E402
    build_dwave_decomposition,
    commensurate_case_status,
    commensurate_q_spec,
)
from stageSC_2d_pairing_bond_collective_vertex_audit import (  # noqa: E402
    operator_ward_rows,
    projection_rows,
    reconstruction_rows,
)
from pairing_bond_goldstone_common import (  # noqa: E402
    goldstone_dimension_rows,
    q0_normalization_rows as goldstone_q0_normalization_rows,
    reconstruction_rows as goldstone_reconstruction_rows,
)
from stageSC_2e_unified_goldstone_tangent_audit import (  # noqa: E402
    build_payload as build_stageSC_2e_payload,
)
from lno327.bdg_finite_q_response import (
    _amplitude_vertex,
    _eta2_phase_vertex,
    _kubo_factor,
    bdg_finite_q_response_imag_axis,
    collective_goldstone_counterterm,
    collective_form_factor,
)
from lno327.conductivity import KuboConfig, k_weights, uniform_bz_mesh
from lno327.pairing import PairingAmplitudes, dwave_pairing_matrix, spm_pairing_matrix
from lno327.pairing_bonds import bond_endpoint_gauge_form_factor, pairing_from_bonds
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


def test_stageSC_0f_composite_quadrature_weights_are_normalized():
    origins = strategy_origins("multi_origin_symmetric", np.array([0.01, 0.01]))
    _, weights = composite_uniform_quadrature(6, origins)
    assert np.sum(weights) == pytest.approx(1.0)


def test_stageSC_0f_multi_origin_composite_point_count():
    origins = strategy_origins("multi_origin_symmetric", np.array([0.01, 0.01]))
    points, _ = composite_uniform_quadrature(5, origins)
    assert len(origins) == 7
    assert points.shape == (5 * 5 * len(origins), 2)


def test_stageSC_0f_schur_is_applied_once_to_composite_kernels():
    bare_a = np.diag([2.0, 3.0, 4.0]).astype(complex)
    bare_b = np.diag([4.0, 5.0, 6.0]).astype(complex)
    left_a = np.ones((3, 2), dtype=complex)
    left_b = 2.0 * np.ones((3, 2), dtype=complex)
    right_a = left_a.T
    right_b = left_b.T
    collective_a = np.diag([2.0, 3.0]).astype(complex)
    collective_b = np.diag([5.0, 7.0]).astype(complex)
    composite, _, _ = single_composite_schur(
        0.5 * (bare_a + bare_b),
        0.5 * (left_a + left_b),
        0.5 * (collective_a + collective_b),
        0.5 * (right_a + right_b),
    )
    schur_a, _, _ = single_composite_schur(bare_a, left_a, collective_a, right_a)
    schur_b, _, _ = single_composite_schur(bare_b, left_b, collective_b, right_b)
    assert not np.allclose(composite, 0.5 * (schur_a + schur_b))


def test_stageSC_0f_grid_step_reference_keeps_shifted_direct_identity():
    n_grid = 4
    q = np.array([4.0 * np.pi / n_grid, 0.0])
    points, weights = composite_uniform_quadrature(n_grid, [(0.0, 0.0)])
    cfg = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    result = compute_bdg_components_for_composite_grid(
        "onsite_s", 0.01, q, points, weights, cfg, chunk_size=32
    )
    assert max(
        result["contact_closure"][channel]["E_shifted_plus_qD_abs"]
        for channel in ("Vx", "Vy")
    ) < 1e-10


def test_stageSC_0f_ordinary_and_multi_origin_are_distinct_strategies():
    q = np.array([0.01, 0.01])
    ordinary = strategy_origins("ordinary_uniform", q)
    multi = strategy_origins("multi_origin_symmetric", q)
    assert len(ordinary) == 1
    assert len(multi) == 7
    assert ordinary != multi


def test_stageSC_0f_script_does_not_call_casimir_pipeline():
    script = (
        ROOT
        / "validation"
        / "scripts"
        / "response"
        / "stageSC_0f_bdg_quadrature_strategy_comparison_audit.py"
    )
    text = script.read_text(encoding="utf-8")
    assert "run_material_casimir_figures" not in text
    assert "outputs/material_casimir" not in text
    assert '"formal_casimir_ran": False' in text


def test_stageSC_0f_no_passing_onsite_strategy_returns_null_recommendation():
    failing_rows = []
    for q in ([0.01, 0.0], [0.01, 0.01]):
        failing_rows.append(
            {
                "pairing": "onsite_s",
                "strategy": "ordinary_uniform",
                "N": 24,
                "num_origins": 1,
                "num_k_points_total": 576,
                "q_model": q,
                "bare_total_ward_max_abs": 1e-4,
                "amplitude_phase_ward_max_abs": 1e-4,
                "Vx": {"E_band_plus_qD_abs": 1e-4},
                "Vy": {"E_band_plus_qD_abs": 1e-4},
            }
        )
    assert recommend_strategy(failing_rows) is None


def test_stageSC_2bC_q_half_lands_on_periodic_grid_step():
    spec = commensurate_q_spec(24, (1, 1))
    assert spec["q_half_in_grid_steps"] == pytest.approx([1.0, 1.0])
    assert spec["q_half_lands_on_grid"] is True


@pytest.mark.parametrize("pairing", ["onsite_s", "spm"])
def test_stageSC_2bC_commensurate_contact_and_ap_ward_close(pairing):
    n_grid = 4
    spec = commensurate_q_spec(n_grid, (1, 0))
    points, weights = composite_uniform_quadrature(n_grid, [(0.0, 0.0)])
    cfg = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    result = compute_bdg_components_for_composite_grid(
        pairing,
        0.01,
        np.asarray(spec["q_model"]),
        points,
        weights,
        cfg,
        phase_vertex="midpoint",
        chunk_size=32,
    )
    contact = max(
        result["contact_closure"][channel]["E_band_plus_qD_abs"]
        for channel in ("Vx", "Vy")
    )
    assert contact < 1e-10
    assert result["amplitude_phase_ward_max_abs"] < 1e-8


def test_stageSC_2bC_bare_ward_is_monitor_only():
    status, dominant = commensurate_case_status(
        "onsite_s",
        contact_closure_abs=1e-14,
        amplitude_phase_ward_abs=1e-14,
        bare_ward_monitor_abs=1.0,
        collective_condition_number=10.0,
    )
    assert status == "PASSED"
    assert dominant == "none"


def test_stageSC_2bC_dwave_failure_builds_form_factor_decomposition():
    cases = []
    for phase_vertex, residual in (("symmetric_kpm", 2e-4), ("midpoint", 4e-4)):
        cases.append(
            {
                "pairing": "dwave",
                "phase_vertex": phase_vertex,
                "N": 24,
                "q_model": [0.5, 0.0],
                "amplitude_phase_ward_max_abs": residual,
                "left_ward_components": {"rho": residual, "x": 0.0, "y": 0.0},
                "right_ward_components": {"rho": residual, "x": 0.0, "y": 0.0},
                "collective_condition_number": 5.0,
                "contact_closure_max_abs": 1e-15,
            }
        )
    decomposition = build_dwave_decomposition(cases)
    assert set(decomposition) >= {"symmetric_kpm", "midpoint", "best_phase_vertex"}
    assert decomposition["best_phase_vertex"] == "symmetric_kpm"
    assert decomposition["residual_ratio_midpoint_over_symmetric"] == pytest.approx(2.0)


def test_stageSC_2bC_script_does_not_call_casimir_pipeline():
    script = (
        ROOT
        / "validation"
        / "scripts"
        / "response"
        / "stageSC_2bC_bdg_amplitude_phase_commensurate_q_audit.py"
    )
    text = script.read_text(encoding="utf-8")
    assert "run_material_casimir_figures" not in text
    assert "outputs/material_casimir" not in text
    assert "bare_total_ward_max_abs is monitor-only" in text


def test_stageSC_2d_pairing_bond_reconstruction_matches_current_pairing_functions():
    amp = PairingAmplitudes(delta0_eV=0.04)
    for kx, ky in [(0.0, 0.0), (0.13, 0.27), (1.11, 0.73)]:
        np.testing.assert_allclose(pairing_from_bonds("onsite_s", kx, ky, amp), amp.delta0_eV * np.eye(4))
        np.testing.assert_allclose(pairing_from_bonds("spm", kx, ky, amp), spm_pairing_matrix(kx, ky, amp))
        np.testing.assert_allclose(pairing_from_bonds("dwave", kx, ky, amp), dwave_pairing_matrix(kx, ky, amp))
    rows = reconstruction_rows(amp)
    assert {row["pairing"]: row["status"] for row in rows} == {
        "onsite_s": "PASSED",
        "spm": "PASSED",
        "dwave": "PASSED",
    }


@pytest.mark.parametrize("pairing", ["onsite_s", "spm", "dwave"])
def test_stageSC_2d_q0_bond_endpoint_phase_vertex_matches_existing_normalization(pairing):
    amp = PairingAmplitudes(delta0_eV=0.04)
    kx, ky = 0.41, -0.22
    exact = bond_endpoint_gauge_form_factor(pairing, kx, ky, 0.0, 0.0, amp)
    existing = collective_form_factor(pairing, kx, ky, 0.0, 0.0, amp, "midpoint")
    np.testing.assert_allclose(exact, existing, atol=1e-14)


def test_stageSC_2d_bond_endpoint_operator_ward_passes_for_reconstructed_pairings():
    rows = operator_ward_rows(PairingAmplitudes(delta0_eV=0.04))
    selected = [row for row in rows if row["phase_vertex"] == "bond_endpoint_gauge"]
    assert {row["pairing"] for row in selected} == {"onsite_s", "spm", "dwave"}
    assert all(row["status"] == "PASSED" for row in selected)


def test_stageSC_2d_collective_basis_projection_residual_is_reported():
    rows = projection_rows(PairingAmplitudes(delta0_eV=0.04))
    assert {row["pairing"] for row in rows} == {"onsite_s", "spm", "dwave"}
    assert all("collective_basis_projection_relative_residual" in row for row in rows)
    assert all(row["num_phase_channels"] >= 1 for row in rows)


def test_stageSC_2d_payload_is_diagnostic_only_and_does_not_call_casimir_pipeline():
    script = ROOT / "validation" / "scripts" / "response" / "stageSC_2d_pairing_bond_collective_vertex_audit.py"
    text = script.read_text(encoding="utf-8")
    assert '"formal_casimir_ran": False' in text
    assert '"production_default_modified": False' in text
    assert "run_material_casimir_figures" not in text
    assert "outputs/material_casimir" not in text


def test_stageSC_2e_goldstone_dimension_is_one_for_total_charge_u1():
    rows = goldstone_dimension_rows()
    assert {row["pairing"] for row in rows} == {"onsite_s", "spm", "dwave"}
    assert all(row["goldstone_dimension"] == 1 for row in rows)
    assert all(row["bond_resolved_internal_modes_are_goldstone"] is False for row in rows)


def test_stageSC_2e_reconstruction_and_q0_goldstone_normalization_pass():
    amp = PairingAmplitudes(delta0_eV=0.04)
    recon = goldstone_reconstruction_rows(amp)
    norm = goldstone_q0_normalization_rows(amp)
    assert all(row["status"] == "PASSED" for row in recon)
    assert all(row["status"] == "PASSED" for row in norm)
    assert all(row["normalization_factor_abs"] == pytest.approx(1.0) for row in norm)


def test_stageSC_2e_payload_reports_dwave_schur_or_internal_diagnosis_without_extra_goldstone():
    payload = build_stageSC_2e_payload(quick=True)
    assert payload["formal_casimir_ran"] is False
    assert payload["production_default_modified"] is False
    assert payload["minimal_collective_basis"]["goldstone_dimension"] == 1
    assert payload["minimal_collective_basis"]["bond_resolved_goldstones_added"] is False
    exact = [
        row
        for row in payload["exact_goldstone_operator_ward"]
        if row["phase_vertex"] == "exact_goldstone_tangent"
    ]
    assert all(row["status"] == "PASSED" for row in exact)
    assert payload["status"] in {
        "PASSED",
        "PARTIAL_PASS_DWAVE_SCHUR_BLOCKED",
        "PARTIAL_PASS_DWAVE_INTERNAL_MODE_NEEDED",
    }
    if payload["status"] != "PASSED":
        assert payload["dwave_internal_mode_diagnosis"]
        assert "not an additional Goldstone" in payload["dwave_internal_mode_diagnosis"][0]["marker"]


def test_stageSC_2e_script_does_not_call_casimir_pipeline():
    script = ROOT / "validation" / "scripts" / "response" / "stageSC_2e_unified_goldstone_tangent_audit.py"
    text = script.read_text(encoding="utf-8")
    assert '"formal_casimir_ran": False' in text
    assert '"production_default_modified": False' in text
    assert "run_material_casimir_figures" not in text
    assert "outputs/material_casimir" not in text
