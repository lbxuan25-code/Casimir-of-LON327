from __future__ import annotations

from pathlib import Path

import numpy as np

from lno327.dwave_pairing_tangent_diagnostics import run_dwave_pairing_tangent_diagnostics
from lno327.finite_q_ward_scan import run_finite_q_ward_scan
from lno327.goldstone_counterterm_diagnostics import run_goldstone_counterterm_diagnostics
from lno327.q0_bdg_response_alignment import run_q0_bdg_response_alignment

ROOT = Path(__file__).resolve().parents[1]
NEW_DIAGNOSTIC_FILES = (
    ROOT / "src" / "lno327" / "q0_bdg_response_alignment.py",
    ROOT / "src" / "lno327" / "finite_q_ward_scan.py",
    ROOT / "src" / "lno327" / "dwave_pairing_tangent_diagnostics.py",
    ROOT / "src" / "lno327" / "goldstone_counterterm_diagnostics.py",
    ROOT / "docs" / "bdg_finite_q_validation_plan.md",
)


def test_q0_alignment_diagnostics_run_for_all_cases_and_are_finite():
    for pairing_name in ("normal", "onsite_s", "spm", "dwave"):
        report = run_q0_bdg_response_alignment(pairing_name, nk=2)
        assert report.q_model == (0.0, 0.0)
        assert report.mesh_size == 4
        assert report.valid_for_casimir_input is False
        assert report.compared_quantity_names
        assert all(np.isfinite(value) for value in report.matrix_norms.values())
        assert all(np.isfinite(value) for value in report.pairwise_difference_norms.values())
        assert all(np.isfinite(value) for value in report.relative_difference_norms.values())
        assert "valid_for_casimir_input: False" in report.format_text()


def test_finite_q_ward_scan_runs_for_three_pairings_and_is_not_casimir_ready():
    report = run_finite_q_ward_scan(nk=2, q_values=(0.005,), q_directions=((1.0, 0.0),))
    assert report.valid_for_casimir_input is False
    assert {row.pairing_name for row in report.rows} == {"onsite_s", "spm", "dwave"}
    assert {row.response_name for row in report.rows} == {
        "bare_total",
        "minus_schur",
        "amplitude_phase_schur",
    }
    for row in report.rows:
        assert row.valid_for_casimir_input is False
        assert np.isfinite(row.left_ward_residual_norm)
        assert np.isfinite(row.right_ward_residual_norm)
        assert np.isfinite(row.max_ward_residual_norm)
    assert "valid_for_casimir_input: False" in report.format_text()


def test_dwave_reconstruction_and_tangent_diagnostic_reports_structured_errors():
    report = run_dwave_pairing_tangent_diagnostics()
    assert report.valid_for_casimir_input is False
    assert len(report.reconstruction_errors) == len(report.k_points)
    assert len(report.q0_tangent_errors) == len(report.k_points)
    assert np.isfinite(report.max_reconstruction_error)
    assert np.isfinite(report.max_q0_tangent_error)
    assert report.q0_tangent_passed
    assert "valid_for_casimir_input: False" in report.format_text()


def test_goldstone_counterterm_diagnostic_reports_eta2_status_for_all_pairings():
    report = run_goldstone_counterterm_diagnostics(nk=2)
    assert report.valid_for_casimir_input is False
    assert {row.pairing_name for row in report.rows} == {"onsite_s", "spm", "dwave"}
    for row in report.rows:
        assert row.valid_for_casimir_input is False
        assert np.isfinite(row.eta2_kernel_abs)
        assert row.eta2_normalization_status == "eta2 = delta0 * theta"
        assert row.counterterm_only_collective_kernel
    assert "valid_for_casimir_input: False" in report.format_text()


def test_new_diagnostic_names_use_physics_labels():
    forbidden_label = "stage" + "_"
    paths = [*NEW_DIAGNOSTIC_FILES, Path(__file__)]
    for path in paths:
        assert forbidden_label not in path.name
        text = path.read_text(encoding="utf-8")
        assert forbidden_label not in text


def test_new_diagnostic_code_does_not_do_response_fitting_or_repair():
    forbidden = (
        "lst" + "sq",
        "least" + "_squares",
        "poly" + "fit",
        "response" + "_repair",
        "residual" + "_projection",
    )
    for path in NEW_DIAGNOSTIC_FILES[:4]:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(item in text for item in forbidden)
