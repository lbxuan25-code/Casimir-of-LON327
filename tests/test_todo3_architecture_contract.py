"""Repository-level dependency and identity guards for TODO 3."""
from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from lno327.casimir.material_response_cache_identity import MaterialResponseCacheIdentity

PERSISTENCE_MODULES = (
    Path("src/lno327/casimir/material_response_snapshot.py"),
    Path("src/lno327/casimir/material_response_cache_errors.py"),
    Path("src/lno327/casimir/material_response_cache_identity.py"),
    Path("src/lno327/casimir/material_response_cache_artifact.py"),
    Path("src/lno327/casimir/material_response_cache_codec.py"),
    Path("src/lno327/casimir/material_response_cache_store.py"),
)
CACHE_ORCHESTRATION = Path("src/lno327/casimir/material_response_cached_engine.py")
GEOMETRY_MODULES = (
    Path("src/lno327/casimir/material_geometry.py"),
    Path("src/lno327/casimir/material_two_plate.py"),
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


def test_persistence_layer_has_no_geometry_or_outer_dependencies() -> None:
    assert all(path.is_file() for path in PERSISTENCE_MODULES)
    forbidden = (
        "lno327.casimir.material_geometry",
        "lno327.casimir.material_two_plate",
        "lno327.casimir.lifshitz",
        "lno327.casimir.outer",
        "lno327.electrodynamics.reflection",
        "lno327.workflows",
    )
    violations = [
        f"{path}:{module}"
        for path in PERSISTENCE_MODULES
        for module in _imports(path)
        if any(_matches(module, prefix) for prefix in forbidden)
    ]
    assert violations == []


def test_geometry_layer_cannot_import_cache_writer_or_cached_engine() -> None:
    forbidden = (
        "lno327.casimir.material_response_cache_artifact",
        "lno327.casimir.material_response_cache_codec",
        "lno327.casimir.material_response_cache_store",
        "lno327.casimir.material_response_cached_engine",
        "lno327.casimir.material_response_engine",
        "lno327.workflows",
    )
    violations = [
        f"{path}:{module}"
        for path in GEOMETRY_MODULES
        for module in _imports(path)
        if any(_matches(module, prefix) for prefix in forbidden)
    ]
    assert violations == []


def test_cache_orchestration_does_not_import_geometry() -> None:
    forbidden = (
        "lno327.casimir.material_geometry",
        "lno327.casimir.material_two_plate",
        "lno327.casimir.lifshitz",
        "lno327.casimir.outer",
    )
    violations = [
        module
        for module in _imports(CACHE_ORCHESTRATION)
        if any(_matches(module, prefix) for prefix in forbidden)
    ]
    assert violations == []


def test_cache_identity_type_cannot_accept_geometry_or_runtime_fields() -> None:
    names = {field.name for field in fields(MaterialResponseCacheIdentity)}
    forbidden = {
        "separation_m",
        "distance_m",
        "theta_rad",
        "theta_1_rad",
        "theta_2_rad",
        "q_lab",
        "outer_order",
        "quadrature_node",
        "worker_count",
        "runtime_chunk_size",
        "cache_root",
    }
    assert names.isdisjoint(forbidden)


def test_old_in_memory_crystal_cache_is_not_persistent_store_base() -> None:
    store_path = Path("src/lno327/casimir/material_response_cache_store.py")
    codec_path = Path("src/lno327/casimir/material_response_cache_codec.py")
    combined = store_path.read_text(encoding="utf-8") + codec_path.read_text(encoding="utf-8")
    assert "CrystalResponseCache" not in combined
    assert all("pickle" not in _imports(path) for path in PERSISTENCE_MODULES)
    assert "allow_pickle=False" in codec_path.read_text(encoding="utf-8")


def test_cache_store_is_not_a_monolithic_codec_module() -> None:
    imports = _imports(Path("src/lno327/casimir/material_response_cache_store.py"))
    assert "lno327.casimir.material_response_cache_codec" in imports
    assert "lno327.electrodynamics.conventions" not in imports
    assert "lno327.electrodynamics.static_sheet" not in imports
