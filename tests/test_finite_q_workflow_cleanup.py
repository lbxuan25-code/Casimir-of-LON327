from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_finite_q_material_casimir_candidate_is_not_current_workflow():
    assert not (ROOT / "src" / "lno327" / "material_casimir_figures.py").exists()
    assert not (ROOT / "scripts" / "run_material_casimir_figures.py").exists()
    assert not (ROOT / "scripts" / "plot_material_casimir_figures.py").exists()


def test_old_bdg_finite_q_gate_has_no_active_status_json():
    old_dir = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
    assert not (old_dir / "bdg_finite_q_validation_status.json").exists()
    assert not (old_dir / "command.sh").exists()
    readme = old_dir / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        assert "validation/outputs/bdg_finite_q" in text


def test_historical_superconducting_bdg_workflows_are_not_current_entries():
    response_dir = ROOT / "validation" / "scripts" / "response"
    historical_prefix = "stage" + "SC_"
    assert not list(response_dir.glob(f"{historical_prefix}*.py"))


def test_no_compat_validation_wrappers_remain():
    compat_dir = ROOT / "validation" / "scripts" / "compat"
    if not compat_dir.exists():
        return

    assert not list(compat_dir.glob("*.py"))


def test_current_command_files_do_not_reference_removed_historical_bdg_workflows():
    historical_prefix = "stage" + "SC_"
    for command_path in (ROOT / "validation" / "outputs").glob("**/command.sh"):
        text = command_path.read_text(encoding="utf-8")
        assert historical_prefix not in text


def test_current_tests_do_not_import_historical_response_workflows():
    old_response_path = "validation/scripts" + "/response"
    historical_prefix = "stage" + "SC_"
    sys_path_insert = "sys.path" + ".insert"

    for test_path in (ROOT / "tests").glob("test_*.py"):
        if test_path == Path(__file__):
            continue
        text = test_path.read_text(encoding="utf-8")
        assert old_response_path not in text
        assert historical_prefix not in text
        assert not (sys_path_insert in text and "/response" in text)


def test_current_bdg_finite_q_workflows_do_not_reintroduce_fitting_or_repair():
    workflow_dir = ROOT / "validation" / "scripts" / "bdg_finite_q"
    forbidden_terms = [
        "ls" + "q",
        "least" + "_squares",
        "response" + "_repair",
        "repair" + "_response",
        "residual" + "_projection",
        "project" + "_residual",
        "fitted" + " ward",
        "ward" + "_correction",
    ]

    for script_path in workflow_dir.glob("*.py"):
        text = script_path.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in text


def test_current_bdg_finite_q_status_is_diagnostic_only():
    status_path = ROOT / "validation" / "outputs" / "bdg_finite_q" / "bdg_finite_q_validation_status.json"
    assert status_path.exists()

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["diagnostic_only"] is True
    assert status["valid_for_casimir_input"] is False
    assert status["status"] != "PASSED"


def test_old_bdg_finite_q_output_dir_only_contains_migration_readme():
    old_dir = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
    if not old_dir.exists():
        return

    files = [path.name for path in old_dir.iterdir() if path.is_file()]
    assert files == ["README.md"]


def test_bdg_finite_q_command_targets_existing_current_scripts_only():
    command_path = ROOT / "validation" / "outputs" / "bdg_finite_q" / "command.sh"
    text = command_path.read_text(encoding="utf-8")
    current_prefix = "validation/scripts" + "/bdg_finite_q/"
    old_response_path = "validation/scripts" + "/response/"
    historical_prefix = "stage" + "SC_"

    assert old_response_path not in text
    assert historical_prefix not in text

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("python "):
            continue
        script = line.split()[1]
        assert script.startswith(current_prefix)
        assert (ROOT / script).exists()


def test_readmes_and_commands_do_not_advertise_deleted_current_entries():
    historical_prefix = "stage" + "SC_"
    compat_path = "validation/scripts" + "/compat"
    material_module = "material" + "_casimir_figures"

    docs = [
        ROOT / "validation" / "README.md",
        ROOT / "scripts" / "README.md",
        ROOT / "docs" / "finite_q_diagnostic_pipeline.md",
        ROOT / "docs" / "bdg_finite_q_validation_plan.md",
        ROOT / "validation" / "outputs" / "bdg_finite_q" / "README.md",
        ROOT / "validation" / "outputs" / "bdg_finite_q" / "command.sh",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert historical_prefix not in text
        assert compat_path not in text
        assert material_module not in text
