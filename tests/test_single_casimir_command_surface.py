from __future__ import annotations

from pathlib import Path
import tomllib

import lno327.casimir as casimir
from scripts.full_casimir import __main__ as unified


ROOT = Path(__file__).resolve().parents[1]


def test_unified_dispatcher_is_the_only_operational_command_surface() -> None:
    assert set(unified._COMMANDS) == {
        "plan",
        "run",
        "resources",
        "diagnose",
        "audit",
        "shift-audit",
        "torque",
        "plot",
        "data",
        "layout",
    }
    forbidden = {
        "legacy-workflow",
        "qualification",
        "qualification-holdout",
        "qualification-verify",
        "pilots",
        "scan",
        "all",
    }
    assert forbidden.isdisjoint(unified._COMMANDS)


def test_competing_calculation_scripts_are_removed() -> None:
    removed = (
        "scripts/full_casimir/workflow.py",
        "scripts/full_casimir/background.sh",
        "scripts/full_casimir/background_runner.sh",
        "scripts/full_casimir/qualification.py",
        "scripts/full_casimir/qualification_prepare.py",
        "scripts/full_casimir/qualification_holdout.py",
        "scripts/full_casimir/qualification_holdout_group.py",
        "scripts/full_casimir/qualification_verify.py",
        "scripts/full_casimir/cache_migration.py",
        "scripts/full_casimir/cache_extension.py",
        "src/lno327/casimir/qualification.py",
        "src/lno327/casimir/__main__.py",
    )
    assert all(not (ROOT / relative).exists() for relative in removed)


def test_package_install_does_not_create_a_competing_console_command() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "scripts" not in payload["project"]


def test_package_metadata_points_to_the_unified_operational_route() -> None:
    metadata = casimir.casimir_layer_metadata()
    assert metadata["canonical_operational_entrypoint"] == (
        "python -m scripts.full_casimir"
    )
    assert metadata["canonical_plan_command"].endswith(" plan")
    assert metadata["canonical_run_command"].endswith(" run")
    assert metadata["package_command_present"] is False
    assert metadata["installed_console_command_present"] is False
    assert metadata["legacy_calculation_scripts_present"] is False


def test_only_plan_and_run_define_or_execute_formal_work() -> None:
    assert unified._COMMANDS["plan"] == ("scripts.full_casimir.scan", ("plan",))
    assert unified._COMMANDS["run"] == ("scripts.full_casimir.scan", ("run",))
    for command, target in unified._COMMANDS.items():
        if command in {"plan", "run"}:
            continue
        assert target != unified._COMMANDS["run"]
