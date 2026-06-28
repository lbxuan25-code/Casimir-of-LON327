from __future__ import annotations

import importlib.util
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "validation" / "scripts" / "bdg_finite_q"
DIAGNOSTIC_SCRIPT_FILES = (
    SCRIPT_DIR / "q0_bdg_response_alignment.py",
    SCRIPT_DIR / "dwave_raw_bubble_vertex_audit.py",
    SCRIPT_DIR / "q0_local_intraband_decomposition.py",
    SCRIPT_DIR / "finite_q_ward_scan.py",
    SCRIPT_DIR / "dwave_pairing_tangent_diagnostics.py",
    SCRIPT_DIR / "goldstone_counterterm_diagnostics.py",
)
NEW_DIAGNOSTIC_FILES = (
    *DIAGNOSTIC_SCRIPT_FILES,
    ROOT / "validation" / "outputs" / "bdg_finite_q" / "README.md",
    ROOT / "validation" / "outputs" / "bdg_finite_q" / "command.sh",
)
MOVED_CORE_FILES = (
    ROOT / "src" / "lno327" / "q0_bdg_response_alignment.py",
    ROOT / "src" / "lno327" / "finite_q_ward_scan.py",
    ROOT / "src" / "lno327" / "dwave_pairing_tangent_diagnostics.py",
    ROOT / "src" / "lno327" / "goldstone_counterterm_diagnostics.py",
)


def _load_validation_script(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_q0_alignment_diagnostics_run_for_all_cases_and_are_finite():
    module = _load_validation_script("q0_bdg_response_alignment")
    for pairing_name in ("normal", "onsite_s", "spm", "dwave"):
        report = module.run_q0_bdg_response_alignment(pairing_name, nk=2)
        assert report.q_model == (0.0, 0.0)
        assert report.mesh_size == 4
        assert report.valid_for_casimir_input is False
        assert report.compared_quantity_names
        assert all(np.isfinite(value) for value in report.matrix_norms.values())
        assert all(np.isfinite(value) for value in report.pairwise_difference_norms.values())
        assert all(np.isfinite(value) for value in report.relative_difference_norms.values())
        assert "valid_for_casimir_input: False" in report.format_text()


def test_q0_alignment_reports_transformed_comparison_rows():
    module = _load_validation_script("q0_bdg_response_alignment")

    normal_report = module.run_q0_bdg_response_alignment("normal", nk=2)
    normal_pairs = {
        (row.finite_q_quantity, row.transformed_local_quantity)
        for row in normal_report.transformed_comparison_rows
    }
    assert ("finite_q_total_q0", "local_normal_density_current_total_current_block") in normal_pairs
    assert ("finite_q_total_q0", "omega * local_normal_sigma_like") in normal_pairs
    assert ("finite_q_total_q0", "-omega * local_normal_sigma_like") in normal_pairs
    assert normal_report.valid_for_casimir_input is False

    for pairing_name in ("spm", "dwave"):
        report = module.run_q0_bdg_response_alignment(pairing_name, nk=2)
        pairs = {
            (row.finite_q_quantity, row.transformed_local_quantity)
            for row in report.transformed_comparison_rows
        }
        assert ("finite_q_raw_bubble_q0", "local_K_para") in pairs
        assert ("finite_q_raw_bubble_q0", "local_K_para_total") in pairs
        assert ("finite_q_raw_bubble_q0", "local_K_para_interband") in pairs
        assert ("finite_q_raw_bubble_q0", "-local_K_para") in pairs
        assert ("local_K_para_total - finite_q_raw_bubble_q0", "local_K_para_intraband") in pairs
        assert ("finite_q_total_q0", "local_K_total") in pairs
        assert ("finite_q_total_q0", "omega * local_superconducting_response") in pairs
        assert ("finite_q_direct_q0", "local_K_total - local_K_para") in pairs
        assert ("finite_q_direct_q0", "-local_K_total - local_K_para") in pairs
        assert ("finite_q_direct_q0", "local_K_total + local_K_para") in pairs
        assert ("finite_q_minus_schur_q0", "local_K_total") in pairs
        assert ("finite_q_minus_schur_q0", "-local_K_total") in pairs
        assert ("finite_q_minus_schur_q0", "-omega * local_superconducting_response") in pairs
        assert ("finite_q_amplitude_phase_schur_q0", "local_K_total") in pairs
        assert ("finite_q_amplitude_phase_schur_q0", "-local_K_total") in pairs
        assert (
            "finite_q_amplitude_phase_schur_q0",
            "-omega * local_superconducting_response",
        ) in pairs
        assert report.valid_for_casimir_input is False
        assert "transformed comparison table" in report.format_text()


def test_spm_q0_alignment_passes_convention_aware_rule_without_promoting_to_casimir_input():
    module = _load_validation_script("q0_bdg_response_alignment")
    report = module.run_q0_bdg_response_alignment("spm", nk=2)
    assert report.passed is True
    assert report.best_transformed_match["finite_q_raw_bubble_q0"] == "local_K_para"
    passed_pairs = {
        (row.finite_q_quantity, row.transformed_local_quantity)
        for row in report.transformed_comparison_rows
        if row.passes_tolerance
    }
    assert ("finite_q_raw_bubble_q0", "local_K_para") in passed_pairs
    assert ("finite_q_direct_q0", "-local_K_total - local_K_para") in passed_pairs
    assert ("finite_q_total_q0", "-local_K_total") in passed_pairs
    assert ("finite_q_minus_schur_q0", "-local_K_total") in passed_pairs
    assert ("finite_q_amplitude_phase_schur_q0", "-local_K_total") in passed_pairs
    assert report.valid_for_casimir_input is False
    assert any("convention-aware" in note for note in report.pass_fail_notes)


def test_dwave_q0_alignment_uses_intraband_aware_raw_bubble_status():
    module = _load_validation_script("q0_bdg_response_alignment")
    report = module.run_q0_bdg_response_alignment("dwave", nk=3)
    assert report.passed is True
    assert report.valid_for_casimir_input is False
    rows = {
        (row.finite_q_quantity, row.transformed_local_quantity): row
        for row in report.transformed_comparison_rows
    }
    assert rows[("finite_q_raw_bubble_q0", "local_K_para_interband")].passes_tolerance
    assert rows[
        ("local_K_para_total - finite_q_raw_bubble_q0", "local_K_para_intraband")
    ].passes_tolerance
    assert not rows[("finite_q_raw_bubble_q0", "local_K_para_total")].passes_tolerance
    assert not rows[("finite_q_raw_bubble_q0", "local_K_para")].passes_tolerance
    assert any("intraband-aware" in note for note in report.pass_fail_notes)
    assert any("raw-vs-total mismatch 保持可见" in note for note in report.pass_fail_notes)
    assert "valid_for_casimir_input: False" in report.format_text()


def test_dwave_raw_bubble_vertex_audit_reports_roundoff_vertex_match_and_dwave_raw_mismatch():
    module = _load_validation_script("dwave_raw_bubble_vertex_audit")
    report = module.run_dwave_raw_bubble_vertex_audit(nk=3)
    assert report.valid_for_casimir_input is False
    assert report.q_model == (0.0, 0.0)
    assert report.mesh_size == 9
    assert {row.pairing_name for row in report.rows} == {"spm", "dwave"}
    assert report.dwave_specific_mismatch is False
    assert report.raw_vs_total_mismatch_explained_by_intraband is True
    assert report.interpretation == "raw_vs_total_mismatch_explained_by_intraband"
    row_by_pairing = {row.pairing_name: row for row in report.rows}
    assert row_by_pairing["spm"].evidence == "raw_bubble_matches_local_K_para"
    assert row_by_pairing["dwave"].raw_vs_local_rel > 1e-6
    assert row_by_pairing["dwave"].raw_vs_interband_rel < 1e-6
    assert row_by_pairing["dwave"].missing_vs_intraband_rel < 1e-6
    assert row_by_pairing["dwave"].intraband_explanation_supported is True
    assert row_by_pairing["dwave"].vertex_status == "vertex_operator_q0_match"
    assert row_by_pairing["dwave"].evidence == "raw_vs_total_mismatch_explained_by_intraband"
    for row in report.rows:
        assert row.valid_for_casimir_input is False
        assert np.isfinite(row.finite_q_raw_bubble_norm)
        assert np.isfinite(row.local_k_para_norm)
        assert np.isfinite(row.local_k_para_interband_norm)
        assert np.isfinite(row.local_k_para_intraband_norm)
        assert np.isfinite(row.raw_vs_local_abs)
        assert np.isfinite(row.raw_vs_local_rel)
        assert np.isfinite(row.raw_vs_interband_rel)
        assert np.isfinite(row.missing_vs_intraband_rel)
        assert np.isfinite(row.finite_q_vs_local_vertex_max_abs)
        assert np.isfinite(row.finite_q_vs_local_vertex_max_rel)
        assert row.vertex_abs_tolerance == 1e-12
        assert row.vertex_rel_tolerance == 1e-6
        if row.finite_q_vs_local_vertex_max_abs <= row.vertex_abs_tolerance:
            assert row.vertex_status == "vertex_operator_q0_match"
            assert row.evidence != "vertex_operator_level_mismatch"
        assert row.evidence
    text = report.format_text()
    assert "valid_for_casimir_input: False" in text
    assert "d-wave raw-bubble / vertex audit" in text
    assert "raw_vs_total_mismatch_explained_by_intraband" in text
    assert "dwave_specific_raw_bubble_mismatch" not in text
    assert "vertex_operator_level_mismatch" not in text


def test_vertex_roundoff_abs_tolerance_prevents_false_operator_mismatch(monkeypatch):
    module = _load_validation_script("dwave_raw_bubble_vertex_audit")

    def finite_q_vertex(*_args):
        return np.zeros((8, 8), dtype=complex)

    def local_vertex(*_args):
        return 1e-16 * np.eye(8, dtype=complex)

    monkeypatch.setattr(module, "bdg_finite_q_vector_vertex", finite_q_vertex)
    monkeypatch.setattr(module, "bdg_current_vertex", local_vertex)
    max_abs, max_rel, status = module._vertex_difference_max(
        np.array([[0.0, 0.0]]),
        absolute_tolerance=1e-12,
        relative_tolerance=1e-20,
    )
    assert max_abs <= 1e-12
    assert max_rel > 1e-20
    assert status == "vertex_operator_q0_match"


def test_raw_bubble_failure_with_matched_vertex_points_to_bubble_assembly_not_vertex(monkeypatch):
    module = _load_validation_script("dwave_raw_bubble_vertex_audit")

    class FakeResponse:
        bare_bubble = np.zeros((3, 3), dtype=complex)

    class FakeLocal:
        paramagnetic = np.eye(2, dtype=complex)

    monkeypatch.setattr(module, "bdg_finite_q_response_imag_axis", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(module, "bdg_total_kernel_imag_axis", lambda *_args, **_kwargs: FakeLocal())
    monkeypatch.setattr(
        module,
        "_local_k_para_decomposition",
        lambda *_args, **_kwargs: (
            np.eye(2, dtype=complex),
            np.zeros((2, 2), dtype=complex),
            np.eye(2, dtype=complex),
        ),
    )
    monkeypatch.setattr(module, "bdg_finite_q_vector_vertex", lambda *_args: np.zeros((8, 8), dtype=complex))
    monkeypatch.setattr(module, "bdg_current_vertex", lambda *_args: 1e-16 * np.eye(8, dtype=complex))

    config = module.KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    row = module._audit_one_pairing(
        "dwave",
        np.array([[0.0, 0.0]]),
        np.array([1.0]),
        config,
        module.PairingAmplitudes(delta0_eV=0.04),
        raw_relative_tolerance=1e-6,
        vertex_abs_tolerance=1e-12,
        vertex_rel_tolerance=1e-20,
    )
    assert row.raw_vs_local_rel > 1e-6
    assert row.vertex_status == "vertex_operator_q0_match"
    assert row.intraband_explanation_supported is True
    assert row.evidence == "raw_vs_total_mismatch_explained_by_intraband"
    assert row.valid_for_casimir_input is False


def test_q0_local_intraband_decomposition_runs_and_tests_hypothesis_on_tiny_grid():
    module = _load_validation_script("q0_local_intraband_decomposition")
    report = module.run_q0_local_intraband_decomposition(nk=3)
    assert report.valid_for_casimir_input is False
    assert report.q_model == (0.0, 0.0)
    assert report.mesh_size == 9
    assert {row.pairing_name for row in report.rows} == {"spm", "dwave"}
    row_by_pairing = {row.pairing_name: row for row in report.rows}
    for row in report.rows:
        assert row.valid_for_casimir_input is False
        np.testing.assert_allclose(
            row.local_k_para_total,
            row.local_k_para_interband + row.local_k_para_intraband,
            rtol=1e-9,
            atol=1e-10,
        )
        assert row.decomposition_rel <= 1e-9
        assert np.isfinite(row.total_norm)
        assert np.isfinite(row.interband_norm)
        assert np.isfinite(row.intraband_norm)
        assert np.isfinite(row.finite_q_raw_norm)
        assert np.isfinite(row.raw_vs_interband_rel)
        assert np.isfinite(row.missing_vs_intraband_rel)
        assert np.isfinite(row.raw_vs_total_rel)
        assert row.interpretation
    assert row_by_pairing["spm"].hypothesis_supported is True
    assert row_by_pairing["dwave"].raw_vs_total_rel > 1e-6
    assert row_by_pairing["dwave"].hypothesis_supported is True
    assert row_by_pairing["dwave"].raw_vs_interband_rel < row_by_pairing["dwave"].raw_vs_total_rel
    assert (
        row_by_pairing["dwave"].interpretation
        == "q0_raw_bubble_mismatch_consistent_with_missing_local_intraband_contribution"
    )
    text = report.format_text()
    assert "valid_for_casimir_input: False" in text
    assert "Ward closure proof" in text


def test_finite_q_ward_scan_runs_for_three_pairings_and_is_not_casimir_ready():
    module = _load_validation_script("finite_q_ward_scan")
    report = module.run_finite_q_ward_scan(nk=2, q_values=(0.005,), q_directions=((1.0, 0.0),))
    assert report.valid_for_casimir_input is False
    assert {row.pairing_name for row in report.rows} == {"onsite_s", "spm", "dwave"}
    assert {row.response_name for row in report.rows} == {
        "bare_total",
        "minus_schur",
        "amplitude_phase_schur",
    }
    assert report.q0_alignment_prerequisite["spm"] == "convention_aware_pass"
    assert report.q0_alignment_prerequisite["dwave"] == "intraband_aware_pass"
    assert report.q0_alignment_prerequisite["onsite_s"] == "diagnostic_only_not_passed"
    for row in report.rows:
        assert row.valid_for_casimir_input is False
        assert np.isfinite(row.left_ward_residual_norm)
        assert np.isfinite(row.right_ward_residual_norm)
        assert np.isfinite(row.max_ward_residual_norm)
    assert "valid_for_casimir_input: False" in report.format_text()


def test_dwave_reconstruction_and_tangent_diagnostic_reports_structured_errors():
    module = _load_validation_script("dwave_pairing_tangent_diagnostics")
    report = module.run_dwave_pairing_tangent_diagnostics()
    assert report.valid_for_casimir_input is False
    assert len(report.reconstruction_errors) == len(report.k_points)
    assert len(report.q0_tangent_errors) == len(report.k_points)
    assert np.isfinite(report.max_reconstruction_error)
    assert np.isfinite(report.max_q0_tangent_error)
    assert report.q0_tangent_passed
    assert "valid_for_casimir_input: False" in report.format_text()


def test_goldstone_counterterm_diagnostic_reports_eta2_status_for_all_pairings():
    module = _load_validation_script("goldstone_counterterm_diagnostics")
    report = module.run_goldstone_counterterm_diagnostics(nk=2)
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


def test_validation_workflows_are_not_left_in_core_package():
    assert not any(path.exists() for path in MOVED_CORE_FILES)


def test_bdg_finite_q_command_references_existing_current_scripts():
    command_path = ROOT / "validation" / "outputs" / "bdg_finite_q" / "command.sh"
    for line in command_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = shlex.split(stripped)
        if len(parts) >= 2 and parts[0] == "python":
            assert (ROOT / parts[1]).exists()


def test_new_diagnostic_code_does_not_do_response_fitting_or_repair():
    forbidden = (
        "lst" + "sq",
        "least" + "_squares",
        "poly" + "fit",
        "response" + "_repair",
        "residual" + "_projection",
        "fitted" + "_ward",
        "ward" + "_correction",
    )
    for path in DIAGNOSTIC_SCRIPT_FILES:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(item in text for item in forbidden)
