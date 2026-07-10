from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from validation.run_static_nk_scan import (
    _collective_channel_diagnostics,
    _kll_decomposition_diagnostics,
    _longitudinal_component_diagnostics,
    _phase_channel_factor_diagnostics,
    _run_one,
    _ward_side_diagnostics,
)


def test_longitudinal_component_decomposition_reproduces_aggregate_norm():
    kernel = np.zeros((3, 3), dtype=complex)
    kernel[0, 1] = 3.0 + 4.0j
    kernel[1, 0] = 2.0
    kernel[1, 1] = 1.0j
    kernel[1, 2] = 6.0
    kernel[2, 1] = 8.0

    diagnostics = _longitudinal_component_diagnostics(kernel, 1.0)
    expected = np.linalg.norm(
        [
            diagnostics["relative_k0l"],
            diagnostics["relative_kl0"],
            diagnostics["relative_kll"],
            diagnostics["relative_klt"],
            diagnostics["relative_ktl"],
        ]
    )
    assert np.isclose(diagnostics["longitudinal_components_relative_norm"], expected)
    assert diagnostics["dominant_longitudinal_component"] == "KTL"


def test_kll_decomposition_reports_fixed_sign_convention_and_cancellations():
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    correction = np.zeros((3, 3), dtype=complex)
    bubble[1, 1] = 5.0
    direct[1, 1] = -3.0
    correction[1, 1] = 1.5

    components = SimpleNamespace(
        bare_bubble=bubble,
        direct=direct,
        bare_total=bubble + direct,
    )
    kernel = SimpleNamespace(
        k_seta=np.asarray([[0.0], [np.sqrt(1.5)], [0.0]], dtype=complex),
        k_etaeta=np.asarray([[1.0]], dtype=complex),
        k_etas=np.asarray([[0.0, np.sqrt(1.5), 0.0]], dtype=complex),
        k_eff=bubble + direct - correction,
        schur_inverse_method="inv",
    )
    diagnostics = _kll_decomposition_diagnostics(
        components,
        kernel,
        np.eye(3),
        1.0,
        1.0,
    )

    assert diagnostics["scaled_kll_bubble_real"] == 5.0
    assert diagnostics["scaled_kll_direct_real"] == -3.0
    assert diagnostics["scaled_kll_bare_total_real"] == 2.0
    assert np.isclose(diagnostics["scaled_kll_collective_correction_real"], 1.5)
    assert np.isclose(diagnostics["scaled_kll_effective_real"], 0.5)
    assert np.isclose(diagnostics["kll_bubble_direct_cancellation_ratio"], 2.0 / 5.0)
    assert np.isclose(diagnostics["kll_schur_cancellation_ratio"], 0.5 / 2.0)
    assert diagnostics["kll_bubble_direct_closure_abs"] < 1e-14
    assert diagnostics["kll_schur_closure_abs"] < 1e-14


def test_collective_channel_decomposition_reconstructs_correction_and_spectrum():
    # Local L couplings are [2, 3] and [5, 7]. With diag(2, 4),
    # eta1-eta1 contributes 5 and eta2-eta2 contributes 21/4.
    kernel = SimpleNamespace(
        k_seta=np.asarray(
            [
                [0.0, 0.0],
                [2.0, 3.0],
                [0.0, 0.0],
            ],
            dtype=complex,
        ),
        k_etaeta=np.diag([2.0, 4.0]).astype(complex),
        k_etas=np.asarray(
            [
                [0.0, 5.0, 0.0],
                [0.0, 7.0, 0.0],
            ],
            dtype=complex,
        ),
        schur_inverse_method="inv",
    )
    diagnostics = _collective_channel_diagnostics(
        kernel,
        np.eye(3),
        1.0,
        10.25,
    )

    assert np.isclose(
        diagnostics["scaled_kll_channel_eta1_amplitude_eta1_amplitude_real"],
        5.0,
    )
    assert diagnostics["scaled_kll_channel_eta1_amplitude_eta2_phase_abs"] == 0.0
    assert diagnostics["scaled_kll_channel_eta2_phase_eta1_amplitude_abs"] == 0.0
    assert np.isclose(
        diagnostics["scaled_kll_channel_eta2_phase_eta2_phase_real"],
        21.0 / 4.0,
    )
    assert np.isclose(diagnostics["scaled_kll_channel_sum_real"], 10.25)
    assert diagnostics["collective_channel_sum_closure_abs"] < 1e-14
    assert np.isclose(diagnostics["collective_singular_value_min"], 2.0)
    assert np.isclose(diagnostics["collective_singular_value_max"], 4.0)
    assert np.isclose(diagnostics["collective_condition_from_svd"], 2.0)
    assert diagnostics["dominant_collective_channel"] == "eta2_phase_eta2_phase"


def test_phase_factor_diagnostics_resolve_couplings_kernel_and_counterterm():
    collective_bubble = np.diag([2.0, -3.0]).astype(complex)
    collective_counterterm = np.diag([0.0, 7.0]).astype(complex)
    collective_total = collective_bubble + collective_counterterm
    components = SimpleNamespace(
        collective_bubble=collective_bubble,
        collective_counterterm=collective_counterterm,
        collective_total=collective_total,
    )
    kernel = SimpleNamespace(
        k_seta=np.asarray(
            [[0.0, 0.0], [0.0, 3.0], [0.0, 0.0]],
            dtype=complex,
        ),
        k_etaeta=collective_total,
        k_etas=np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 7.0, 0.0]],
            dtype=complex,
        ),
        schur_inverse_method="inv",
    )

    diagnostics = _phase_channel_factor_diagnostics(
        components,
        kernel,
        np.eye(3),
        1.0,
        5.25,
    )

    assert np.isclose(diagnostics["raw_phase_left_coupling_abs"], 3.0)
    assert np.isclose(diagnostics["raw_phase_right_coupling_abs"], 7.0)
    assert np.isclose(diagnostics["raw_phase_collective_bubble_real"], -3.0)
    assert np.isclose(diagnostics["raw_phase_collective_counterterm_real"], 7.0)
    assert np.isclose(diagnostics["raw_phase_collective_total_real"], 4.0)
    assert np.isclose(diagnostics["raw_phase_inverse_22_real"], 0.25)
    assert np.isclose(diagnostics["scaled_phase_factorized_correction_real"], 5.25)
    assert np.isclose(diagnostics["phase_bubble_counterterm_cancellation_ratio"], 4.0 / 7.0)
    assert diagnostics["phase_collective_total_closure_abs"] < 1e-14
    assert diagnostics["phase_inverse_scalar_mismatch_relative"] < 1e-14


def test_ward_side_diagnostics_reports_rhs_projection_cancellation():
    side = SimpleNamespace(
        primitive_rhs=np.asarray([3.0, 0.0, 0.0], dtype=complex),
        collective_projection=np.asarray([2.0, 0.0, 0.0], dtype=complex),
        effective_direct=np.asarray([1.0, 0.0, 0.0], dtype=complex),
        effective_predicted=np.asarray([1.0, 0.0, 0.0], dtype=complex),
        effective_residual=np.zeros(3, dtype=complex),
    )
    diagnostics = _ward_side_diagnostics(side, "ward_left")
    assert diagnostics["ward_left_rhs_norm"] == 3.0
    assert diagnostics["ward_left_collective_projection_norm"] == 2.0
    assert np.isclose(
        diagnostics["ward_left_rhs_projection_cancellation_ratio"],
        1.0 / 3.0,
    )
    assert diagnostics["ward_left_direct_prediction_relative_residual"] == 0.0


def test_static_scan_row_contains_resolved_collective_fields():
    row = _run_one(
        {
            "nk": 2,
            "pairing": "spm",
            "qx": 0.03,
            "qy": 0.02,
            "temperature_K": 10.0,
            "delta0_eV": 0.1,
            "eta_eV": 1e-8,
            "ward_tolerance": 1e-7,
        }
    )

    for field in (
        "relative_k0l",
        "relative_kl0",
        "relative_kll",
        "relative_klt",
        "relative_ktl",
        "dominant_longitudinal_component",
        "ward_left_rhs_norm",
        "ward_left_collective_projection_norm",
        "ward_left_rhs_projection_cancellation_ratio",
        "ward_right_rhs_norm",
        "ward_right_collective_projection_norm",
        "ward_right_rhs_projection_cancellation_ratio",
        "scaled_kll_bubble_real",
        "scaled_kll_direct_real",
        "scaled_kll_bare_total_real",
        "scaled_kll_collective_correction_real",
        "scaled_kll_effective_real",
        "kll_bubble_direct_cancellation_ratio",
        "kll_schur_cancellation_ratio",
        "collective_singular_value_min",
        "collective_singular_value_max",
        "collective_condition_from_svd",
        "collective_eigenvalue_small_real",
        "collective_eigenvalue_large_real",
        "raw_k_l_eta1_amplitude_abs",
        "raw_k_l_eta2_phase_abs",
        "raw_k_eta1_amplitude_l_abs",
        "raw_k_eta2_phase_l_abs",
        "scaled_kll_channel_eta1_amplitude_eta1_amplitude_real",
        "scaled_kll_channel_eta1_amplitude_eta2_phase_real",
        "scaled_kll_channel_eta2_phase_eta1_amplitude_real",
        "scaled_kll_channel_eta2_phase_eta2_phase_real",
        "scaled_kll_channel_sum_real",
        "dominant_collective_channel",
        "raw_phase_left_coupling_abs",
        "raw_phase_right_coupling_abs",
        "raw_phase_collective_bubble_real",
        "raw_phase_collective_counterterm_real",
        "raw_phase_collective_total_real",
        "raw_phase_inverse_22_real",
        "scaled_phase_factorized_correction_real",
        "phase_bubble_counterterm_cancellation_ratio",
    ):
        assert field in row

    assert np.isclose(
        row["longitudinal_components_relative_norm"],
        row["relative_longitudinal_gauge_residual"],
    )
    assert np.isclose(
        row["scaled_kll_effective_relative_abs"],
        row["relative_kll"],
    )
    assert np.isclose(
        row["scaled_kll_channel_sum_real"],
        row["scaled_kll_collective_correction_real"],
    )
    assert np.isclose(
        row["scaled_phase_factorized_correction_real"],
        row["scaled_kll_channel_eta2_phase_eta2_phase_real"],
    )
