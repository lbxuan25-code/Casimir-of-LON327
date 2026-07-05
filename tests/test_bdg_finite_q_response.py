from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lno327.finite_q_engine import bdg_finite_q_response_imag_axis, collective_goldstone_counterterm
from lno327 import bdg_total_kernel_imag_axis
from lno327.response.config import KuboConfig
from lno327.numerics.weights import k_weights
from lno327.numerics.grids import uniform_bz_mesh
from validation.lib.finite_q_diagnostics import run_finite_q_diagnostic
from lno327.finite_q_engine import (
    FiniteQEngineOptions,
    apply_amplitude_phase_schur,
    finite_q_bdg_response_from_ansatz,
)
from lno327.models.lno327_four_orbital.collective import build_pairing_ansatz
from lno327.models.lno327_four_orbital.pairing import pairing_from_bonds
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.finite_q_bdg import finite_q_bdg_response_from_model_ansatz
from lno327.collective.validation import validate_physical_ward_identity
from lno327.response.normal_density_current import normal_physical_density_current_response_imag_axis

ROOT = Path(__file__).resolve().parents[1]


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
        assert result.minus_schur.shape == (3, 3)
        assert result.plus_schur.shape == (3, 3)
        assert np.all(np.isfinite(result.gauge_restored))
        assert result.metadata["nambu_prefactor"] == 0.5
        assert result.metadata["collective_channels"] == ["global_phase_only"]
        assert result.metadata["valid_for_casimir_input"] is False
        assert result.metadata["shared_eigenbasis_q0"] is False


def test_finite_q_wrapper_matches_generic_ansatz_engine():
    q, points, weights, config, amp = _inputs()
    wrapper = bdg_finite_q_response_imag_axis(
        "dwave",
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        phase_vertex="bond_endpoint_gauge",
    )
    engine = finite_q_bdg_response_from_model_ansatz(
        LNO327FourOrbitalSpec(pairing_amplitudes=amp),
        build_pairing_ansatz("dwave", phase_vertex="bond_endpoint_gauge"),
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        FiniteQEngineOptions(),
    )
    for field in (
        "bare_bubble",
        "direct",
        "bare_total",
        "em_collective_left",
        "collective_em_right",
        "collective_bubble",
        "collective_counterterm",
        "collective_total",
        "amplitude_phase_schur",
    ):
        np.testing.assert_allclose(getattr(wrapper, field), getattr(engine, field), rtol=1e-12, atol=1e-12)


def test_generic_finite_q_core_has_no_pairing_name_branching():
    text = (ROOT / "src" / "lno327" / "response" / "finite_q_bdg.py").read_text(encoding="utf-8")
    core = text.split("def finite_q_bdg_response_from_model_ansatz(", maxsplit=1)[1].split(
        "return BdGFiniteQResponseComponents(",
        maxsplit=1,
    )[0]
    forbidden = [
        'pairing == "dwave"',
        "pairing == 'dwave'",
        'pairing == "spm"',
        "pairing == 'spm'",
        'ansatz.name == "dwave"',
        "ansatz.name == 'dwave'",
        'ansatz.name == "spm"',
        "ansatz.name == 'spm'",
    ]
    assert not any(item in core for item in forbidden)


def test_generic_finite_q_engine_does_not_import_legacy_facade():
    text = (ROOT / "src" / "lno327" / "finite_q_engine.py").read_text(encoding="utf-8")
    assert "from .finite_q_engine import" not in text


def test_legacy_finite_q_wrapper_module_was_removed_after_merge():
    assert not (ROOT / "src" / "lno327" / "bdg_finite_q_response.py").exists()


def test_pairing_ansatz_shapes_and_counterterms():
    _, points, weights, config, amp = _inputs()
    for name in ("onsite_s", "spm", "dwave"):
        ansatz = build_pairing_ansatz(name, phase_vertex="bond_endpoint_gauge")
        assert ansatz.mean_pairing(0.2, -0.1, amp).shape == (4, 4)
        vertices = ansatz.collective_vertices(0.2, -0.1, 0.05, 0.0, amp)
        assert len(vertices) == 2
        assert all(vertex.shape == (8, 8) for vertex in vertices)
        assert ansatz.hs_counterterm(config, points, weights, amp).shape == (2, 2)


def test_ward_validation_is_pure_diagnostic():
    response = np.array([[1.0, 0.2j, 0.0], [-0.2j, 0.7, 0.1], [0.0, 0.1, 0.4]], dtype=complex)
    before = response.copy()
    report = validate_physical_ward_identity(response, 0.01, np.array([0.1, 0.0]), tolerance=1e-8)
    np.testing.assert_allclose(response, before)
    assert report.left_residual.shape == (3,)
    assert report.right_residual.shape == (3,)
    assert isinstance(report.passed, bool)


def test_amplitude_phase_schur_helper_matches_matrix_formula():
    bare = np.array([[2.0, 0.1], [0.2, 3.0]], dtype=complex)
    left = np.array([[1.0, 0.5], [0.25, 0.75]], dtype=complex)
    kernel = np.array([[4.0, 0.2], [0.1, 2.0]], dtype=complex)
    right = np.array([[0.5, 0.1], [0.3, 0.8]], dtype=complex)
    result = apply_amplitude_phase_schur(bare, left, kernel, right)
    expected = bare - left @ np.linalg.inv(kernel) @ right
    np.testing.assert_allclose(result.corrected_response, expected)
    assert result.inverse_method == "inv"


def test_raw_finite_q_response_is_not_casimir_ready():
    q, points, weights, config, amp = _inputs()
    result = bdg_finite_q_response_imag_axis(
        "spm",
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        phase_vertex="bond_endpoint_gauge",
    )
    assert result.metadata.get("valid_for_casimir_input") is False
    assert "casimir_gating_status" in result.metadata


def test_q0_finite_q_engine_reuses_shared_bdg_eigenbasis_and_matches_local_raw_kernel():
    _, _, _, config, amp = _inputs()
    points = uniform_bz_mesh(2)
    weights = k_weights(points)
    q0 = np.array([0.0, 0.0])
    for pairing_name in ("spm", "dwave"):
        result = bdg_finite_q_response_imag_axis(
            pairing_name,
            config.omega_eV,
            q0,
            points,
            weights,
            config,
            amp,
            phase_vertex="bond_endpoint_gauge",
            current_vertex="peierls",
            collective_mode="amplitude_phase",
            collective_counterterm="goldstone_gap_equation",
            include_phase_phase_direct=True,
        )
        local = bdg_total_kernel_imag_axis(points, config, pairing_name, amp, weights)
        assert result.metadata["shared_eigenbasis_q0"] is True
        assert result.metadata["shared_eigenbasis_q0_tolerance"] == 1e-14
        np.testing.assert_allclose(result.bare_bubble[1:, 1:], local.paramagnetic, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(result.direct[1:, 1:], -local.total - local.paramagnetic, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(result.bare_total[1:, 1:], -local.total, rtol=1e-6, atol=1e-10)
        assert result.metadata["valid_for_casimir_input"] is False


def test_near_zero_q_uses_shared_basis_but_ordinary_finite_q_does_not():
    _, points, weights, config, amp = _inputs()
    ansatz = build_pairing_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    tiny_q = finite_q_bdg_response_from_ansatz(
        ansatz,
        config.omega_eV,
        np.array([5e-15, 0.0]),
        points,
        weights,
        config,
        amp,
        FiniteQEngineOptions(),
    )
    ordinary_q = finite_q_bdg_response_from_ansatz(
        ansatz,
        config.omega_eV,
        np.array([0.005, 0.0]),
        points,
        weights,
        config,
        amp,
        FiniteQEngineOptions(),
    )
    assert tiny_q.metadata["shared_eigenbasis_q0"] is True
    assert ordinary_q.metadata["shared_eigenbasis_q0"] is False
    assert tiny_q.metadata["valid_for_casimir_input"] is False
    assert ordinary_q.metadata["valid_for_casimir_input"] is False


def test_finite_q_diagnostic_report_defaults_and_gating_for_all_ansatz_names():
    for name in ("onsite_s", "spm", "dwave"):
        report = run_finite_q_diagnostic(name, nk=2)
        assert report.pairing_name == name
        assert report.phase_vertex == "bond_endpoint_gauge"
        assert report.current_vertex == "peierls"
        assert report.collective_mode == "amplitude_phase"
        assert report.collective_counterterm == "goldstone_gap_equation"
        assert report.selected_response_name == "amplitude_phase_schur"
        assert report.valid_for_casimir_input is False
        assert np.isfinite(report.bare_ward_residual_norm)
        assert np.isfinite(report.minus_schur_ward_residual_norm)
        assert np.isfinite(report.amplitude_phase_schur_ward_residual_norm)


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
    assert result.metadata["selected_gauge_restored"] == "minus_schur"
    assert "ward_residual_minus_schur" in result.metadata


def test_delta0_zero_uses_true_bdg_by_default_not_normal_backend_shortcut():
    q, points, weights, config, amp = _inputs(delta0=0.0)
    result = bdg_finite_q_response_imag_axis("spm", config.omega_eV, q, points, weights, config, amp)
    normal = normal_physical_density_current_response_imag_axis(points, config, q, weights)
    assert "normal_limit_delegated_to" not in result.metadata
    reference = bdg_finite_q_response_imag_axis(
        "spm",
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        use_normal_backend_in_delta0_limit=True,
    )
    assert np.allclose(reference.gauge_restored, normal)


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
    assert symmetric.metadata["phase_kernel_status"] == "bubble_only_not_expected_to_gauge_close"


def test_amplitude_phase_collective_shapes_and_goldstone_counterterm():
    q, points, weights, config, amp = _inputs()
    result = bdg_finite_q_response_imag_axis("onsite_s", config.omega_eV, q, points, weights, config, amp)
    assert result.collective_bubble.shape == (2, 2)
    assert result.collective_counterterm.shape == (2, 2)
    assert result.collective_total.shape == (2, 2)
    assert result.amplitude_phase_schur.shape == (3, 3)
    assert result.metadata["gauge_restored_selected"] == "amplitude_phase_schur"
    cg = collective_goldstone_counterterm("onsite_s", points, weights, config, amp, "symmetric_kpm")
    assert np.allclose(result.collective_counterterm, cg * np.eye(2))


def test_pairing_bond_reconstruction_matches_current_pairing_functions():
    amp = PairingAmplitudes(delta0_eV=0.04)
    for kx, ky in ((0.0, 0.0), (0.2, -0.3), (1.1, 0.7)):
        for name in ("onsite_s", "spm", "dwave"):
            ansatz = build_pairing_ansatz(name, phase_vertex="bond_endpoint_gauge")
            np.testing.assert_allclose(pairing_from_bonds(name, kx, ky, amp), ansatz.mean_pairing(kx, ky, amp))


def test_current_finite_q_tests_do_not_import_historical_validation_workflows():
    text = Path(__file__).read_text(encoding="utf-8")
    assert ("validation/scripts" + "/response") not in text
    assert ("stage" + "SC") not in text
    assert ("sys.path" + ".insert") not in text
