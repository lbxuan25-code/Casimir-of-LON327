from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "response" / "stage4_17_right_ward_source_convention_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage4_17_right_ward_audit", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small_audit():
    module = _load_module()
    return module.run_audit(coarse_grid=8, max_refinement_level=1, gauss_order=2, q_scales=[1.0])


def test_fast_run_outputs_required_top_level_fields(small_audit):
    assert small_audit["stage"] == "Stage 4.17"
    assert "results" in small_audit
    assert "diagnostic_status" in small_audit
    assert "boundary" in small_audit


def test_all_right_ward_candidates_are_computed(small_audit):
    candidates = {item["candidate"] for item in small_audit["results"][0]["right_candidates"]}
    assert candidates == {
        "R_right_plus_omega_plus_q",
        "R_right_plus_omega_minus_q",
        "R_right_minus_omega_plus_q",
        "R_right_minus_omega_minus_q",
    }


def test_best_candidate_and_predicted_candidate_fields_exist(small_audit):
    row = small_audit["results"][0]
    assert "best_right_candidate" in row
    assert "predicted_candidate_norm" in row
    predicted = next(
        item for item in row["right_candidates"] if item["candidate"] == "R_right_plus_omega_minus_q"
    )
    assert predicted["right_norm"] == row["predicted_candidate_norm"]


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
    assert "best_candidate_global" in status
    assert "right_source_sign_status" in status
    assert "closure_status" in status
    assert "dominant_remaining_channel" in status
    assert "likely_issue" in status
    assert "next_step" in status
