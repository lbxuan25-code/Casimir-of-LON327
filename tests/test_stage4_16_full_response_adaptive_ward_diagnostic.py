from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage4_16_full_response_adaptive_ward_diagnostic.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_16_full_response_adaptive", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small_audit():
    module = _load_module()
    return module.run_audit(coarse_grid=8, max_refinement_levels=[0, 1], gauss_order=2, q_scales=[1.0])


def test_fast_run_outputs_required_top_level_fields(small_audit):
    assert small_audit["stage"] == "Stage 4.16"
    assert "adaptive_response_results" in small_audit
    assert "uniform_reference_results" in small_audit
    assert "diagnostic_status" in small_audit
    assert "boundary" in small_audit


def test_adaptive_result_contains_required_residual_fields(small_audit):
    row = small_audit["adaptive_response_results"][0]
    for key in (
        "left_norm",
        "right_norm",
        "max_norm",
        "left_density_source_abs",
        "left_spatial_source_norm",
        "right_density_observable_abs",
        "right_spatial_observable_norm",
        "num_quadrature_points",
    ):
        assert key in row


def test_bubble_and_direct_use_same_points_and_weights(small_audit):
    assert small_audit["adaptive_response_results"]
    assert all(
        row["same_points_weights_for_bubble_and_direct"] is True
        for row in small_audit["adaptive_response_results"]
    )


def test_boundary_fields_are_all_true(small_audit):
    boundary = small_audit["boundary"]
    for key in (
        "no_main_response_change",
        "no_bubble_sign_change",
        "no_direct_contact_change",
        "no_source_observable_change",
        "no_residual_tuning",
        "no_fitted_contact",
        "no_E_ET_added",
        "no_conductivity_reflection_casimir",
    ):
        assert boundary[key] is True


def test_diagnostic_status_fields_exist(small_audit):
    status = small_audit["diagnostic_status"]
    assert "adaptive_improvement_status" in status
    assert "refinement_convergence_status" in status
    assert "closure_status" in status
    assert "dominant_remaining_channel" in status
    assert "likely_issue" in status
    assert "next_step" in status
