from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.full_casimir.data_management import (
    build_archive_plan,
    build_data_catalog,
    execute_archive_plan,
    write_archive_execution,
    write_archive_plan,
)
from scripts.full_casimir.data_retention import (
    PRUNE_CONFIRMATION,
    build_prune_plan,
    execute_prune_plan,
    pack_json_report,
    verify_archive_plan,
    write_archive_verification,
    write_prune_plan,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(root: Path, name: str) -> Path:
    run = root / "runs" / name
    _write_json(
        run / "config.json",
        {
            "schema": "config-v1",
            "pairings": ["spm"],
            "plate_angles_deg": [0.0, 0.0],
            "temperature_K": 10.0,
            "separation_nm": 20.0,
            "logdet_rtol": 0.0015,
            "N_candidates": [128, 192],
        },
    )
    _write_json(run / "manifest.json", {"schema": "manifest-v1", "git_commit": "abc"})
    _write_json(
        run / "result.json",
        {
            "schema": "result-v1",
            "status": "unresolved",
            "termination_reason": "test",
            "production_casimir_allowed": False,
        },
    )
    _write_json(run / "summary.json", {"schema": "summary-v1", "status": "unresolved"})
    _write_json(
        run / "cache" / "certified_points.json",
        {
            "schema": "cache-v1",
            "entries": [
                {
                    "point_result": {
                        "sweet_spot": {"status": "established"},
                    }
                }
            ],
        },
    )
    (run / "notes.txt").write_text("retention evidence\n", encoding="utf-8")
    return run


def _registry(path: Path, name: str, lifecycle: str = "superseded") -> Path:
    _write_json(
        path,
        {
            "schema": "casimir-data-registry-v1",
            "runs": {
                name: {
                    "lifecycle_state": lifecycle,
                    "retention_action": "archive",
                    "note": "test",
                }
            },
        },
    )
    return path


def _archive_fixture(tmp_path: Path, *, lifecycle: str = "superseded") -> dict:
    root = tmp_path / "casimir"
    name = "spm_test_run"
    source = _make_run(root, name)
    registry = _registry(root / "catalog" / "registry.json", name, lifecycle)
    catalog = build_data_catalog(root, registry_path=registry)
    plan = build_archive_plan(catalog, archive_root=root / "archive")
    plan_path = write_archive_plan(plan, root / "catalog" / "archive_plan.json")
    execution = execute_archive_plan(
        plan_path,
        confirm_plan_sha256=plan["plan_sha256"],
    )
    write_archive_execution(execution, root / "catalog" / "archive_execution.json")
    return {
        "root": root,
        "name": name,
        "source": source,
        "registry": registry,
        "catalog": catalog,
        "archive_plan": plan,
        "archive_plan_path": plan_path,
    }


def test_archive_verification_restores_every_file(tmp_path: Path) -> None:
    fixture = _archive_fixture(tmp_path)
    report = verify_archive_plan(fixture["archive_plan_path"])
    assert report["all_archives_restored_and_verified"] is True
    assert report["result_count"] == 1
    row = report["results"][0]
    assert row["status"] == "restored_and_verified"
    assert row["restored_file_count"] == fixture["archive_plan"]["items"][0][
        "source_file_count"
    ]
    assert fixture["source"].is_dir()


def test_prune_plan_rejects_frozen_evidence(tmp_path: Path) -> None:
    fixture = _archive_fixture(tmp_path, lifecycle="frozen_evidence")
    verification = verify_archive_plan(fixture["archive_plan_path"])
    catalog = build_data_catalog(fixture["root"], registry_path=fixture["registry"])
    with pytest.raises(ValueError, match="protected lifecycle"):
        build_prune_plan(
            catalog,
            verification,
            selected_runs=(fixture["name"],),
        )


def test_prune_requires_exact_phrase_and_preserves_archive(tmp_path: Path) -> None:
    fixture = _archive_fixture(tmp_path)
    verification = verify_archive_plan(fixture["archive_plan_path"])
    verification_path = fixture["root"] / "catalog" / "archive_verification.json"
    write_archive_verification(verification, verification_path)
    catalog = build_data_catalog(fixture["root"], registry_path=fixture["registry"])
    plan = build_prune_plan(
        catalog,
        verification,
        selected_runs=(fixture["name"],),
    )
    plan_path = write_prune_plan(plan, fixture["root"] / "catalog" / "prune_plan.json")

    with pytest.raises(ValueError, match="deletion confirmation"):
        execute_prune_plan(
            plan_path,
            confirm_plan_sha256=plan["plan_sha256"],
            confirm_delete="wrong",
        )
    assert fixture["source"].is_dir()

    report = execute_prune_plan(
        plan_path,
        confirm_plan_sha256=plan["plan_sha256"],
        confirm_delete=PRUNE_CONFIRMATION,
    )
    assert report["removed_run_count"] == 1
    assert report["released_bytes"] > 0
    assert not fixture["source"].exists()
    archive = Path(fixture["archive_plan"]["items"][0]["archive_path"])
    manifest = Path(fixture["archive_plan"]["items"][0]["archive_manifest_path"])
    assert archive.is_file()
    assert manifest.is_file()


def test_prune_plan_detects_source_change_after_verification(tmp_path: Path) -> None:
    fixture = _archive_fixture(tmp_path)
    verification = verify_archive_plan(fixture["archive_plan_path"])
    (fixture["source"] / "notes.txt").write_text("changed\n", encoding="utf-8")
    catalog = build_data_catalog(fixture["root"], registry_path=fixture["registry"])
    with pytest.raises(ValueError, match="source changed"):
        build_prune_plan(
            catalog,
            verification,
            selected_runs=(fixture["name"],),
        )


def test_report_pack_externalizes_large_lists_and_verifies_reconstruction(
    tmp_path: Path,
) -> None:
    source = tmp_path / "convergence_audit.json"
    payload = {
        "schema": "full-casimir-convergence-audit-v2",
        "decision": {"status": "production_change_not_authorized"},
        "candidate_details": [
            {"index": index, "history": [index, index + 1, index + 2]}
            for index in range(500)
        ],
        "small": [1, 2, 3],
    }
    _write_json(source, payload)
    compact = tmp_path / "convergence_audit.compact.json"
    pack_root = tmp_path / "convergence_audit.pack"
    manifest = tmp_path / "convergence_audit.pack_manifest.json"
    report = pack_json_report(
        source,
        compact_path=compact,
        pack_root=pack_root,
        manifest_path=manifest,
        threshold_bytes=1000,
    )
    assert report["reconstruction_verified"] is True
    assert report["source_report_preserved"] is True
    assert report["sidecar_count"] >= 1
    assert compact.stat().st_size < source.stat().st_size
    assert manifest.is_file()
    compact_payload = json.loads(compact.read_text(encoding="utf-8"))
    assert compact_payload["decision"]["status"] == "production_change_not_authorized"
    assert compact_payload["candidate_details"]["schema"] == (
        "casimir-data-json-sidecar-reference-v1"
    )
