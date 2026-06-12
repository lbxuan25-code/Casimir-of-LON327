from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage4_15_fermi_window_adaptive_quadrature.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_15_adaptive_quadrature", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small_audit():
    module = _load_module()
    return module.run_audit(
        coarse_grid=8,
        max_refinement_levels=[0, 1],
        gauss_order=2,
        q_scales=[1.0],
        temperature_sweep_K=[30.0],
    )


def test_fast_run_outputs_required_top_level_fields(small_audit):
    assert small_audit["stage"] == "Stage 4.15"
    assert "config" in small_audit
    assert "adaptive_results" in small_audit
    assert "uniform_reference_results" in small_audit
    assert "temperature_sweep_results" in small_audit
    assert "diagnostic_status" in small_audit


def test_adaptive_rows_contain_ck_and_quadrature_counts(small_audit):
    row = small_audit["adaptive_results"][0]
    for key in (
        "C",
        "K",
        "C_minus_K",
        "C_minus_K_rel",
        "num_cells_total",
        "num_cells_refined",
        "num_quadrature_points",
    ):
        assert key in row


def test_adaptive_quadrature_uses_same_points_and_weights_for_c_and_k(small_audit):
    assert small_audit["adaptive_results"]
    assert all(row["same_points_weights_for_C_and_K"] is True for row in small_audit["adaptive_results"])
    assert all(row["same_points_weights_for_C_and_K"] is True for row in small_audit["uniform_reference_results"])


def test_boundary_fields_are_all_true(small_audit):
    boundary = small_audit["boundary"]
    for key in (
        "no_main_response_change",
        "no_bubble_sign_change",
        "no_direct_contact_change",
        "no_residual_tuning",
        "no_fitted_contact",
        "no_E_ET_added",
        "no_conductivity_reflection_casimir",
        "does_not_claim_ward_closure",
    ):
        assert boundary[key] is True


def test_diagnostic_status_fields_exist(small_audit):
    status = small_audit["diagnostic_status"]
    assert "adaptive_improvement_status" in status
    assert "refinement_convergence_status" in status
    assert "temperature_sanity_status" in status
    assert "likely_issue" in status
    assert "next_step" in status
