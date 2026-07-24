"""Repository-level dependency guards for TODO 4."""
from __future__ import annotations

import ast
from pathlib import Path

CORE_PLAN = Path("src/lno327/casimir/material_geometry_plan.py")
CORE_BATCH = Path("src/lno327/casimir/material_geometry_batch.py")
QUALIFICATION = Path("src/lno327/casimir/material_geometry_qualification.py")


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


def test_todo4_modules_exist_and_core_batch_has_no_microscopic_or_outer_imports() -> None:
    assert CORE_PLAN.is_file()
    assert CORE_BATCH.is_file()
    assert QUALIFICATION.is_file()
    forbidden = (
        "lno327.casimir.material_response_engine",
        "lno327.casimir.material_response_cached_engine",
        "lno327.casimir.fixed_transverse_point_engine",
        "lno327.casimir.fixed_transverse_point_certification",
        "lno327.casimir.outer",
        "lno327.casimir.fixed_outer_q",
        "lno327.workflows",
    )
    violations = [
        module
        for module in _imports(CORE_BATCH)
        if any(_matches(module, prefix) for prefix in forbidden)
    ]
    assert violations == []


def test_geometry_plan_has_no_cache_io_or_observable_dependencies() -> None:
    forbidden = (
        "lno327.casimir.material_response_cache_store",
        "lno327.casimir.material_response_cache_codec",
        "lno327.casimir.lifshitz_integrand",
        "lno327.casimir.material_two_plate",
        "lno327.casimir.outer",
        "lno327.casimir.fixed_outer_q",
    )
    violations = [
        module
        for module in _imports(CORE_PLAN)
        if any(_matches(module, prefix) for prefix in forbidden)
    ]
    assert violations == []


def test_legacy_engine_is_quarantined_to_qualification_boundary() -> None:
    core_text = CORE_PLAN.read_text(encoding="utf-8") + CORE_BATCH.read_text(
        encoding="utf-8"
    )
    qualification_text = QUALIFICATION.read_text(encoding="utf-8")
    assert "fixed_transverse_point_engine" not in core_text
    assert "fixed_transverse_point_engine" in qualification_text


def test_core_batch_is_strict_read_only_and_has_no_fallback_call_surface() -> None:
    text = CORE_BATCH.read_text(encoding="utf-8")
    assert 'cache.mode != "read_only"' in text
    assert "evaluate_material_response_ladder" not in text
    assert "integrate_arbitrary_q_periodic_bz" not in text
    assert "cache.put(" not in text
    assert '"microscopic_integration_call_count": 0' in text
    assert '"response_certification_call_count": 0' in text
