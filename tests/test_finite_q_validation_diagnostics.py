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
    SCRIPT_DIR / "finite_q_ward_scan.py",
    SCRIPT_DIR / "dwave_pairing_tangent_diagnostics.py",
    SCRIPT_DIR / "goldstone_counterterm_diagnostics.py",
    SCRIPT_DIR / "summarize_validation.py",
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
        if pairing_name == "normal":
            report = module.run_q0_bdg_response_alignment(pairing_name, nk=2)
        else:
            report = module.run_q0_bdg_response_alignment(pairing_name, model_name="lno327_four_orbital", nk=2)
        assert report.q_model == (0.0, 0.0)
        assert report.mesh_size == 4
        assert report.valid_for_casimir_input is False
        assert report.status in {
            "diagnostic_only_not_passed",
            "convention_aware_pass",
            "intraband_aware_pass",
        }
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
    assert normal_report.valid_for_casimir_input is False
    assert normal_report.model_name == "symmetry_bdg_2band"

    for pairing_name in ("spm", "dwave"):
        report = module.run_q0_bdg_response_alignment(pairing_name, model_name="lno327_four_orbital", nk=2)
        pairs = {
            (row.finite_q_quantity, row.transformed_local_quantity)
            for row in report.transformed_comparison_rows
        }
        assert ("finite_q_raw_bubble_q0", "local_K_para_total") in pairs
        assert ("finite_q_raw_bubble_q0", "local_K_para_interband") in pairs
        assert ("local_K_para_total - finite_q_raw_bubble_q0", "local_K_para_intraband") in pairs
        assert ("finite_q_direct_q0", "-local_K_total - local_K_para_total") in pairs
        assert ("finite_q_total_q0", "-local_K_total") in pairs
        assert report.valid_for_casimir_input is False
        assert "transformed comparison table" in report.format_text()


def test_spm_q0_alignment_passes_convention_aware_rule_without_promoting_to_casimir_input():
    module = _load_validation_script("q0_bdg_response_alignment")
    report = module.run_q0_bdg_response_alignment("spm", model_name="lno327_four_orbital", nk=2)
    assert report.passed is True
    assert report.status == "convention_aware_pass"
    assert report.best_transformed_match["finite_q_raw_bubble_q0"] in {
        "local_K_para_total",
        "local_K_para_interband",
    }
    passed_pairs = {
        (row.finite_q_quantity, row.transformed_local_quantity)
        for row in report.transformed_comparison_rows
        if row.passes_tolerance
    }
    assert ("finite_q_raw_bubble_q0", "local_K_para_total") in passed_pairs
    assert ("finite_q_direct_q0", "-local_K_total - local_K_para_total") in passed_pairs
    assert ("finite_q_total_q0", "-local_K_total") in passed_pairs
    assert ("finite_q_minus_schur_q0", "-local_K_total") in passed_pairs
    assert ("finite_q_amplitude_phase_schur_q0", "-local_K_total") in passed_pairs
    assert report.valid_for_casimir_input is False
    assert any("convention-aware" in note for note in report.pass_fail_notes)


def test_dwave_q0_alignment_uses_intraband_aware_raw_bubble_status():
    module = _load_validation_script("q0_bdg_response_alignment")
    report = module.run_q0_bdg_response_alignment("dwave", model_name="lno327_four_orbital", nk=3)
    assert report.passed is True
    assert report.status == "intraband_aware_pass"
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
    assert any("intraband-aware" in note for note in report.pass_fail_notes)
    assert any("raw-vs-total" in note for note in report.pass_fail_notes)
    assert "valid_for_casimir_input: False" in report.format_text()


def test_finite_q_ward_scan_runs_for_three_pairings_and_is_not_casimir_ready():
    module = _load_validation_script("finite_q_ward_scan")
    report = module.run_finite_q_ward_scan(
        model_name="lno327_four_orbital",
        nk=2,
        q_values=(0.005,),
        q_directions=((1.0, 0.0),),
    )
    assert report.valid_for_casimir_input is False
    assert {row.pairing_name for row in report.rows} == {"onsite_s", "spm", "dwave"}
    closure_response_names = set(module.WARD_CLOSURE_RESPONSE_NAMES)
    diagnostic_response_names = {"bare_bubble", "direct", "plus_schur"}
    expected_response_names = closure_response_names | diagnostic_response_names
    actual_response_names = {row.response_name for row in report.rows}
    assert closure_response_names == {
        "bare_total",
        "minus_schur",
        "amplitude_phase_schur",
    }
    assert diagnostic_response_names <= actual_response_names
    assert actual_response_names == expected_response_names
    for pairing_name in {"onsite_s", "spm", "dwave"}:
        names_for_pairing = {
            row.response_name for row in report.rows if row.pairing_name == pairing_name
        }
        assert names_for_pairing == expected_response_names
    assert report.q0_alignment_prerequisite["spm"] == "convention_aware_pass"
    assert report.q0_alignment_prerequisite["dwave"] == "intraband_aware_pass"
    assert report.q0_alignment_prerequisite["onsite_s"] == "diagnostic_only_not_passed"
    assert report.q0_precondition_status["dwave"] == "intraband_aware_pass"
    assert report.diagnostic_run_completed is True
    assert report.ward_identity_closed is False
    closure_rows = [
        row for row in report.rows if row.response_name in closure_response_names
    ]
    assert closure_rows
    assert report.ward_identity_closed is bool(
        all(row.max_ward_residual_norm <= 1e-8 for row in closure_rows)
    )
    for row in report.rows:
        assert row.valid_for_casimir_input is False
        assert np.isfinite(row.left_ward_residual_norm)
        assert np.isfinite(row.right_ward_residual_norm)
        assert np.isfinite(row.max_ward_residual_norm)
    assert "valid_for_casimir_input: False" in report.format_text()


def test_default_finite_q_ward_scan_uses_two_band_workspace_model():
    module = _load_validation_script("finite_q_ward_scan")
    report = module.run_finite_q_ward_scan(nk=2, q_values=(0.005,), q_directions=((1.0, 0.0),))

    assert report.model_name == "symmetry_bdg_2band"
    assert report.primary_validation_model is True
    assert report.workspace_evaluation is True
    assert set(report.pairing_names) == {"spm", "dwave"}
    assert report.q0_precondition_status["spm"] == "convention_aware_pass"
    assert report.q0_precondition_status["dwave"] == "convention_aware_pass"
    assert report.valid_for_casimir_input is False


def test_two_band_q0_superconducting_reports_local_bdg_comparator_without_bad_copy():
    module = _load_validation_script("q0_bdg_response_alignment")
    for pairing_name in ("spm", "dwave"):
        report = module.run_q0_bdg_response_alignment(pairing_name, nk=2)
        assert report.model_name == "symmetry_bdg_2band"
        assert report.comparator_family == "local_bdg"
        assert report.q0_comparator_available is True
        assert report.status_reason
        assert report.transformed_comparison_rows
        text = report.format_text()
        assert "onsite_s has no local public comparator" not in text
        pairs = {
            (row.finite_q_quantity, row.transformed_local_quantity)
            for row in report.transformed_comparison_rows
        }
        assert ("finite_q_raw_bubble_q0", "local_BdG_paramagnetic_kernel") in pairs
        assert ("finite_q_direct_q0", "local_BdG_diamagnetic_kernel") in pairs
        assert ("finite_q_total_q0", "-local_BdG_total_kernel") in pairs


def test_current_bdg_finite_q_output_contract_files_are_model_aware():
    status_path = ROOT / "validation" / "outputs" / "bdg_finite_q" / "bdg_finite_q_validation_status.json"
    readme_path = ROOT / "validation" / "outputs" / "bdg_finite_q" / "README.md"
    command_path = ROOT / "validation" / "outputs" / "bdg_finite_q" / "command.sh"

    assert status_path.exists()
    assert readme_path.exists()
    assert command_path.exists()
    status = __import__("json").loads(status_path.read_text(encoding="utf-8"))
    command = command_path.read_text(encoding="utf-8")

    assert status["primary_validation_model"] == "symmetry_bdg_2band"
    assert status["secondary_transfer_model"] == "lno327_four_orbital"
    assert status["valid_for_casimir_input"] is False
    assert status["ward_identity_closed"] is False
    assert status["workspace_evaluation"] is True
    assert "--model symmetry_bdg_2band" in command
    assert "q0_bdg_response_alignment.py" in command
    assert "finite_q_ward_scan.py" in command


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
    text = command_path.read_text(encoding="utf-8")
    assert "dwave_raw_bubble_vertex_audit.py" not in text
    assert "q0_local_intraband_decomposition.py" not in text
    assert "q0_bdg_response_alignment.py" in text
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
