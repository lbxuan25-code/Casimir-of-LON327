from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_retired_trees_and_competing_casimir_scripts_are_absent() -> None:
    for relative in (
        "results",
        "sandbox",
        "validation/scripts",
        "scripts/casimir",
        "outputs/normal_state",
        "outputs/pairing",
        "outputs/bdg",
        "outputs/casimir/local_response_distance_scan",
        "outputs/casimir/finite_q_bdg_pipeline",
        "src/lno327/casimir/__main__.py",
        "scripts/full_casimir/workflow.py",
        "scripts/full_casimir/background.sh",
        "scripts/full_casimir/qualification.py",
    ):
        assert not (ROOT / relative).exists(), relative


def test_canonical_route_and_documentation_are_present() -> None:
    for relative in (
        "scripts/full_casimir/__main__.py",
        "scripts/full_casimir/scan.py",
        "scripts/full_casimir/README.md",
        "src/lno327/casimir/production.py",
        "src/lno327/casimir/cli.py",
        "src/lno327/casimir/legacy.py",
        "docs/casimir/README.md",
        "docs/casimir/numerical_contract.md",
        "docs/casimir/operations.md",
        "docs/casimir/legacy_fixed_reference.md",
        "outputs/README.md",
        "outputs/casimir/README.md",
    ):
        assert (ROOT / relative).is_file(), relative


def test_stage_version_and_handoff_documents_are_absent() -> None:
    for relative in (
        "docs/casimir_production_chain_v1.md",
        "docs/casimir_adaptive_radial_v1.md",
        "docs/casimir_adaptive_angular_v1.md",
        "docs/casimir_adaptive_joint_v1.md",
        "docs/casimir_adaptive_outer_tail_v1.md",
        "docs/casimir_adaptive_matsubara_v1.md",
        "docs/full_outer_integration_handoff.md",
        "docs/outer_q_integration_contract.md",
    ):
        assert not (ROOT / relative).exists(), relative


def test_active_validation_surface_is_grouped() -> None:
    root = ROOT / "validation"
    assert (root / "__main__.py").is_file()
    assert (root / "commands/ward").is_dir()
    assert (root / "commands/static").is_dir()
    assert (root / "commands/matsubara").is_dir()
    assert (root / "lib/finite_q_validation_models.py").is_file()


def test_validation_root_has_no_bare_runner_or_analyzer_modules() -> None:
    assert {path.name for path in (ROOT / "validation").glob("*.py")} == {
        "__init__.py",
        "__main__.py",
    }


def test_generated_outputs_are_fully_ignored() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/outputs/**" in text
    assert "!/outputs/README.md" in text
    assert "!/outputs/casimir/README.md" in text
    assert "validation/outputs/**" in text
    for token in (
        "!outputs/**/summary",
        "!outputs/**/status",
        "!outputs/**/run_config",
        "!outputs/**/command",
    ):
        assert token not in text
