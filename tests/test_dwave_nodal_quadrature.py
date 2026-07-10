from __future__ import annotations

import numpy as np
import pytest

from lno327.workflows.dwave_nodal_quadrature import (
    DWaveNodalQuadratureOptions,
    build_dwave_nodal_quadrature,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.run_dwave_static_adaptive_scan import _run_task


def _model_inputs(pairing: str = "dwave"):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing, phase_vertex="bond_endpoint_gauge")
    amplitudes = model.build_pairing_params(0.1)
    return model, ansatz, amplitudes


def test_dwave_nodal_quadrature_is_normalized_and_refines_flagged_cells():
    model, ansatz, amplitudes = _model_inputs()
    options = DWaveNodalQuadratureOptions(
        coarse_grid=2,
        adaptive_level=1,
        gauss_order=2,
        sample_order=2,
        quasiparticle_window_eV=1.0,
        normal_window_eV=1.0,
        gap_window_eV=1.0,
        transition_window_eV=1.0,
        transition_shell_eV=2.0,
        max_quadrature_points=1_000,
    )
    points, weights, metadata = build_dwave_nodal_quadrature(
        model.spec,
        ansatz,
        amplitudes,
        np.array([0.03, 0.02]),
        options,
    )

    assert points.ndim == 2 and points.shape[1] == 2
    assert weights.shape == (len(points),)
    assert np.all(weights > 0.0)
    assert np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=2e-12)
    assert np.all(points >= -np.pi)
    assert np.all(points < np.pi)
    assert metadata["integration_strategy"] == "two_band_dwave_nodal_adaptive"
    assert metadata["primitive_merge_before_schur_required"] is True
    assert metadata["parent_child_double_counting"] is False
    assert metadata["num_final_cells"] > metadata["num_base_cells"]
    assert metadata["refinement_history"][0]["num_cells_flagged"] > 0


def test_dwave_nodal_quadrature_rejects_non_dwave_ansatz():
    model, ansatz, amplitudes = _model_inputs("spm")
    with pytest.raises(ValueError, match="ansatz.name == 'dwave'"):
        build_dwave_nodal_quadrature(
            model.spec,
            ansatz,
            amplitudes,
            np.array([0.03, 0.02]),
            DWaveNodalQuadratureOptions(coarse_grid=2, adaptive_level=0, gauss_order=1),
        )


def test_dwave_static_adaptive_runner_smoke_reports_full_contract_fields():
    row = _run_task(
        {
            "coarse_grid": 2,
            "adaptive_level": 0,
            "gauss_order": 1,
            "sample_order": 2,
            "qx": 0.03,
            "qy": 0.02,
            "temperature_K": 10.0,
            "delta0_eV": 0.1,
            "eta_eV": 1e-8,
            "quasiparticle_window_eV": 0.1,
            "normal_window_eV": 0.2,
            "gap_window_eV": 0.1,
            "transition_window_eV": 0.1,
            "transition_shell_eV": 0.3,
            "include_transition_indicator": True,
            "max_quadrature_points": 1_000,
            "ward_tolerance": 1e-6,
            "ward_absolute_tolerance": 1e-12,
            "condition_max": 1e12,
            "raw_longitudinal_ceiling": 10.0,
            "longitudinal_tolerance": 1e-7,
            "mixing_tolerance": 10.0,
            "reality_tolerance": 10.0,
            "passivity_tolerance": 10.0,
            "separation_nm": 20.0,
        }
    )

    assert row["coarse_grid"] == 2
    assert row["adaptive_level"] == 0
    assert row["num_quadrature_points"] == 4
    assert row["ward_passed"] is True
    assert row["ward_condition_ok"] is True
    assert np.isfinite(row["chi_bar"])
    assert np.isfinite(row["dbar_t"])
    assert "projection_eligible" in row
    assert "reflection_constructed" in row
    assert "logdet_passed" in row
