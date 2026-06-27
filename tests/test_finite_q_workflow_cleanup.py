from __future__ import annotations

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
    assert not compat_dir.exists()


def test_current_command_files_do_not_reference_removed_historical_bdg_workflows():
    historical_prefix = "stage" + "SC_"
    for command_path in (ROOT / "validation" / "outputs").glob("**/command.sh"):
        text = command_path.read_text(encoding="utf-8")
        assert historical_prefix not in text
