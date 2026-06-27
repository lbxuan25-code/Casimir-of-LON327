from __future__ import annotations

import importlib.util
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "validation" / "scripts" / "bdg_finite_q"
NEW_DIAGNOSTIC_FILES = (
    SCRIPT_DIR / "q0_bdg_response_alignment.py",
    SCRIPT_DIR / "finite_q_ward_scan.py",
    SCRIPT_DIR / "dwave_pairing_tangent_diagnostics.py",
    SCRIPT_DIR / "goldstone_counterterm_diagnostics.py",
    ROOT / "docs" / "bdg_finite_q_validation_plan.md",
    ROOT / "docs" / "finite_q_diagnostic_pipeline.md",
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
    )
    for path in NEW_DIAGNOSTIC_FILES[:4]:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(item in text for item in forbidden)
