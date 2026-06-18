from __future__ import annotations

import numpy as np
import pytest

from lno327.bdg_finite_q_response import bdg_finite_q_response_imag_axis
from lno327.conductivity import KuboConfig, k_weights, uniform_bz_mesh
from lno327.pairing import PairingAmplitudes
from lno327.ward_response import normal_physical_density_current_response_imag_axis


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
        )
        assert result.bare_bubble.shape == (3, 3)
        assert result.direct.shape == (3, 3)
        assert result.bare_total.shape == (3, 3)
        assert result.gauge_restored.shape == (3, 3)
        assert result.phase_coupling_left.shape == (3,)
        assert result.phase_coupling_right.shape == (3,)
        assert np.all(np.isfinite(result.gauge_restored))
        assert result.metadata["nambu_prefactor"] == 0.5
        assert result.metadata["collective_channels"] == ["global_phase_only"]


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
    )
    assert result.metadata["finite_q_current_vertex_status"] == "q0_velocity_vertex_approximation_not_gauge_closed"
