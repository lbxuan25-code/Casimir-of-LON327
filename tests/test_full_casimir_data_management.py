from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from scripts.full_casimir.data_management import (
    ARCHIVE_PLAN_SCHEMA,
    CATALOG_SCHEMA,
    REGISTRY_SCHEMA,
    build_archive_plan,
    build_data_catalog,
    execute_archive_plan,
    write_archive_plan,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(
    root: Path,
    name: str,
    *,
    complete: bool = True,
    established: int = 2,
    unresolved: int = 0,
) -> Path:
    run = root / "runs" / name
    config = {
        "schema": "config-v1",
        "outer_tail_config": {
            "joint_config": {
                "radial_config": {
                    "point_config": {
                        "pairings": ["spm"],
                        "temperature_K": 10.0,
                        "separation_m": 20e-9,
                        "plate_angles_deg": [0.0, 0.0],
                        "N_candidates": [128, 192, 256],
                        "logdet_rtol": 0.0015,
                        "logdet_atol": 1e-6,
                        "required_consecutive_passes": 2,
                    }
                },
                "radial_budget_fraction": 0.8,
                "angular_budget_fraction": 0.2,
            },
            "cutoff_u_values": [6.0, 10.0],
        },
        "matsubara_cutoff_values": [1, 3],
    }
    _write_json(run / "config.json", config)
    _write_json(run / "manifest.json", {"schema": "manifest-v1", "git_commit": "abc"})
    if complete:
        _write_json(
            run / "result.json",
            {
                "schema": "result-v1",
                "status": "unresolved",
                "production_casimir_allowed": False,
                "all_microscopic_nodes_certified": unresolved == 0,
                "outer_tail_estimated": False,
                "matsubara_tail_estimated": False,
                "termination_reason": "test",
            },
        )
        _write_json(run / "summary.json", {"schema": "summary-v1", "status": "unresolved"})
    entries = []
    for index in range(established + unresolved):
        entries.append(
            {
                "pairing": "spm",
                "n": index % 2,
                "point_result": {
                    "sweet_spot": {
                        "status": "established" if index < established else "not_established"
                    }
                },
            }
        )
    _write_json(
        run / "cache" / "certified_points.json",
        {"schema": "cache-v1", "entries": entries},
    )
    return run


def test_catalog_separates_scientific_and_lifecycle_state(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "casimir"
    _make_run(root, "active", established=3)
    _make_run(root, "incomplete", complete=False)
    registry = root / "catalog" / "registry.json"
    _write_json(
        registry,
        {
            "schema": REGISTRY_SCHEMA,
            "runs": {
                "active": {
                    "lifecycle_state": "active",
                    "retention_action": "keep_hot",
                    "note": "current evidence",
                }
            },
        },
    )
    catalog = build_data_catalog(root, registry_path=registry)
    assert catalog["schema"] == CATALOG_SCHEMA
    by_name = {row["run_name"]: row for row in catalog["runs"]}
    assert by_name["active"]["scientific_state"] == "unresolved"
    assert by_name["active"]["lifecycle_state"] == "active"
    assert by_name["active"]["retention_action"] == "keep_hot"
    assert by_name["active"]["physics_identity"]["plate_angles_deg"] == [0.0, 0.0]
    assert by_name["incomplete"]["scientific_state"] == "incomplete"
    assert by_name["incomplete"]["lifecycle_state"] == "unclassified"


def test_archive_plan_requires_explicit_registry_action(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "casimir"
    source = _make_run(root, "legacy")
    registry = root / "catalog" / "registry.json"
    _write_json(
        registry,
        {
            "schema": REGISTRY_SCHEMA,
            "runs": {
                "legacy": {
                    "lifecycle_state": "legacy_exploratory",
                    "retention_action": "archive",
                    "note": "old scan",
                }
            },
        },
    )
    catalog = build_data_catalog(root, registry_path=registry)
    plan = build_archive_plan(catalog, archive_root=root / "archive")
    assert plan["schema"] == ARCHIVE_PLAN_SCHEMA
    assert plan["item_count"] == 1
    assert plan["items"][0]["run_name"] == source.name
    assert plan["items"][0]["source_removal_authorized"] is False


def test_archive_execution_verifies_and_never_removes_source(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "casimir"
    source = _make_run(root, "legacy")
    registry = root / "catalog" / "registry.json"
    _write_json(
        registry,
        {
            "schema": REGISTRY_SCHEMA,
            "runs": {
                "legacy": {
                    "lifecycle_state": "superseded",
                    "retention_action": "archive",
                    "note": "",
                }
            },
        },
    )
    catalog = build_data_catalog(root, registry_path=registry)
    plan = build_archive_plan(catalog, archive_root=root / "archive")
    plan_path = write_archive_plan(plan, root / "catalog" / "archive_plan.json")

    with pytest.raises(ValueError, match="confirmation"):
        execute_archive_plan(plan_path, confirm_plan_sha256="wrong")

    result = execute_archive_plan(
        plan_path,
        confirm_plan_sha256=plan["plan_sha256"],
    )
    assert result["source_removal_performed"] is False
    assert source.is_dir()
    archive_path = Path(result["results"][0]["archive_path"])
    assert archive_path.is_file()
    assert Path(result["results"][0]["archive_manifest_path"]).is_file()
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("config.json") for name in names)


def test_archive_plan_detects_source_changes(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "casimir"
    source = _make_run(root, "legacy")
    registry = root / "catalog" / "registry.json"
    _write_json(
        registry,
        {
            "schema": REGISTRY_SCHEMA,
            "runs": {
                "legacy": {
                    "lifecycle_state": "superseded",
                    "retention_action": "archive",
                    "note": "",
                }
            },
        },
    )
    catalog = build_data_catalog(root, registry_path=registry)
    plan = build_archive_plan(catalog, archive_root=root / "archive")
    plan_path = write_archive_plan(plan, root / "catalog" / "archive_plan.json")
    (source / "late.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        execute_archive_plan(
            plan_path,
            confirm_plan_sha256=plan["plan_sha256"],
        )
