from __future__ import annotations

import ast
from pathlib import Path


CAMPAIGN = Path(
    "src/lno327/casimir/material_geometry_qualification_campaign.py"
)
COMPATIBILITY = Path(
    "src/lno327/casimir/material_geometry_qualification_compatibility.py"
)
EXECUTION = Path(
    "src/lno327/casimir/material_geometry_qualification_execution.py"
)
IO = Path("src/lno327/casimir/material_geometry_qualification_io.py")
CLI = Path(
    "validation/commands/casimir/todo4_representative_qualification.py"
)
CORE = (
    Path("src/lno327/casimir/material_geometry_plan.py"),
    Path("src/lno327/casimir/material_geometry_batch.py"),
    Path("src/lno327/casimir/material_geometry.py"),
    Path("src/lno327/casimir/lifshitz_integrand.py"),
)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return tuple(modules)


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def test_qualification_preparation_modules_exist_and_cli_is_validation_only() -> None:
    for path in (CAMPAIGN, COMPATIBILITY, EXECUTION, IO, CLI):
        assert path.is_file()
    assert str(CLI).startswith("validation/commands/")
    assert "production_casimir_allowed" in CLI.read_text(encoding="utf-8") or (
        "production_casimir_allowed"
        in EXECUTION.read_text(encoding="utf-8")
    )


def test_campaign_planner_has_no_cache_io_or_microscopic_orchestration() -> None:
    forbidden = (
        "lno327.casimir.material_response_cache_store",
        "lno327.casimir.material_response_cached_engine",
        "lno327.casimir.material_geometry_legacy_replay",
        "lno327.workflows",
    )
    violations = [
        module
        for module in _imports(CAMPAIGN)
        if any(_matches(module, prefix) for prefix in forbidden)
    ]
    assert violations == []


def test_artifact_io_has_no_physics_dependencies() -> None:
    assert all(not module.startswith("lno327") for module in _imports(IO))


def test_core_geometry_cannot_import_qualification_campaign_execution() -> None:
    forbidden = (
        "lno327.casimir.material_geometry_qualification_campaign",
        "lno327.casimir.material_geometry_qualification_compatibility",
        "lno327.casimir.material_geometry_qualification_execution",
        "lno327.casimir.material_geometry_qualification_io",
    )
    for path in CORE:
        violations = [
            module
            for module in _imports(path)
            if any(_matches(module, prefix) for prefix in forbidden)
        ]
        assert violations == []


def test_microscopic_population_and_legacy_replay_exist_only_in_execution_layer() -> None:
    execution = EXECUTION.read_text(encoding="utf-8")
    campaign = CAMPAIGN.read_text(encoding="utf-8")
    compatibility = COMPATIBILITY.read_text(encoding="utf-8")

    assert "evaluate_material_response_ladder_cached" in execution
    assert "run_matched_legacy_geometry_replay" in execution
    assert "evaluate_material_response_ladder_cached" not in campaign
    assert "run_matched_legacy_geometry_replay" not in campaign
    assert "evaluate_material_response_ladder_cached" not in compatibility
    assert "run_matched_legacy_geometry_replay" not in compatibility
