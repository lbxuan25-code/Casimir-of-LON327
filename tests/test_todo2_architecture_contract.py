"""Repository-level dependency guards for the TODO 2 architecture."""
from __future__ import annotations

import ast
from pathlib import Path


MATERIAL_MODULES = (
    Path("src/lno327/casimir/material_response.py"),
    Path("src/lno327/casimir/material_response_certification.py"),
    Path("src/lno327/casimir/material_response_engine.py"),
)
GEOMETRY_MODULES = (
    Path("src/lno327/casimir/material_geometry.py"),
    Path("src/lno327/casimir/material_two_plate.py"),
)
LEGACY_MODULES = (
    "lno327.casimir.fixed_transverse_point_engine",
    "lno327.casimir.fixed_transverse_point_certification",
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


def test_new_todo2_route_never_imports_legacy_transverse_certifier() -> None:
    violations = [
        f"{path}:{module}"
        for path in (*MATERIAL_MODULES, *GEOMETRY_MODULES)
        for module in _imports(path)
        if any(_matches(module, prefix) for prefix in LEGACY_MODULES)
    ]
    assert violations == []


def test_material_layer_has_no_geometry_or_observable_dependencies() -> None:
    forbidden = (
        "lno327.casimir.material_geometry",
        "lno327.casimir.material_two_plate",
        "lno327.casimir.lifshitz",
        "lno327.casimir.outer",
        "lno327.electrodynamics.reflection",
    )
    violations = [
        f"{path}:{module}"
        for path in MATERIAL_MODULES
        for module in _imports(path)
        if any(_matches(module, prefix) for prefix in forbidden)
    ]
    assert violations == []


def test_geometry_layer_has_no_microscopic_or_certification_fallback() -> None:
    forbidden = (
        "lno327.workflows",
        "lno327.response.arbitrary_q",
        "lno327.casimir.material_response_engine",
        "lno327.casimir.material_response_certification",
    )
    violations = [
        f"{path}:{module}"
        for path in GEOMETRY_MODULES
        for module in _imports(path)
        if any(_matches(module, prefix) for prefix in forbidden)
    ]
    assert violations == []
