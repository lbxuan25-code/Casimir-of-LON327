from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.full_casimir.output_layout import (
    build_output_layout_audit,
    value_digest,
    write_output_layout_audit,
)
from scripts.full_casimir.output_layout_migration import (
    LAYOUT_FINALIZE_CONFIRMATION,
    build_layout_finalize_plan,
    build_layout_migration_plan,
    execute_layout_finalize_plan,
    stage_layout_migration,
    write_layout_finalize_plan,
    write_layout_migration_plan,
    write_layout_stage_execution,
)


def _write_tar(path: Path, member_name: str, payload: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as handle:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))


def _write_legacy_n896_diagnostics(root: Path) -> Path:
    diagnostics = root / "diagnostics"
    nested = diagnostics / "N896_unresolved_diagnostics"
    cases = nested / "cases"
    case = cases / "spm_T10K_d20nm_theta_p000deg_N896_grid2"
    case.mkdir(parents=True)
    (diagnostics / "N896_cache_primary_and_replay.json").write_text(
        json.dumps({"status": "legacy"}), encoding="utf-8"
    )
    (nested / "overview.json").write_text("{}", encoding="utf-8")
    (nested / "system_info.json").write_text("{}", encoding="utf-8")
    for name, payload in {
        "config.json": {},
        "manifest.json": {"schema": "full-casimir-run-manifest"},
        "result.json": {"schema": "adaptive-matsubara-casimir-result-v1"},
        "summary.json": {"schema": "full-casimir-run-summary"},
    }.items():
        (case / name).write_text(json.dumps(payload), encoding="utf-8")
    return diagnostics


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    root = repo / "outputs" / "casimir"
    for name in ("runs", "archive", "catalog", "reports", "workflow_logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("layout\n", encoding="utf-8")

    logs = root / "N896_scan_logs"
    logs.mkdir()
    (logs / "scan.log").write_text("legacy scan\n", encoding="utf-8")
    (logs / "status.txt").write_text("done\n", encoding="utf-8")

    runtime_budget = root / "0deg_runtime_budget_pilot_logs"
    runtime_budget.mkdir()
    (runtime_budget / "pilot.log").write_text("legacy pilot\n", encoding="utf-8")

    _write_tar(
        root / "0deg_pilot_v2_diagnostics.tar.gz",
        "outputs/casimir/diagnostics/summary.json",
        b"{}",
    )
    _write_tar(
        root / "dwave_0deg_pilot_cache.tar.gz",
        "outputs/casimir/runs/dwave/cache/certified_points.json",
        b"{}",
    )
    _write_legacy_n896_diagnostics(root)
    return repo, root


def _audit_plan_and_stage(repo: Path, root: Path) -> tuple[dict, Path, dict, Path]:
    audit = build_output_layout_audit(root, repo_root=repo)
    audit_path = root / "catalog" / "output_layout_audit.json"
    write_output_layout_audit(
        audit,
        json_path=audit_path,
        tsv_path=root / "catalog" / "output_layout_audit.tsv",
    )
    plan = build_layout_migration_plan(audit_path)
    plan_path = root / "catalog" / "output_layout_migration_plan.json"
    write_layout_migration_plan(plan, plan_path)
    stage = stage_layout_migration(
        plan_path,
        confirm_plan_sha256=plan["plan_sha256"],
    )
    stage_path = root / "catalog" / "output_layout_stage_execution.json"
    write_layout_stage_execution(stage, stage_path)
    return plan, plan_path, stage, stage_path


def test_path_aware_audit_classifies_frozen_n896_diagnostics(tmp_path: Path) -> None:
    repo, root = _fixture_repo(tmp_path)
    generic = repo / "scripts" / "generic.py"
    generic.parent.mkdir(parents=True)
    generic.write_text(
        "def diagnostics(value):\n    return value\n",
        encoding="utf-8",
    )

    audit = build_output_layout_audit(root, repo_root=repo)
    entries = {entry["name"]: entry for entry in audit["entries"]}

    assert audit["migration_blockers"] == []
    assert audit["legacy_entry_count"] == 5
    assert entries["diagnostics"]["classification"] == "known_legacy"
    assert entries["diagnostics"]["kind"] == "legacy_n896_diagnostics_directory"
    assert entries["diagnostics"]["runtime_reference_count"] == 0
    assert entries["diagnostics"]["proposed_destination"].endswith(
        "archive/legacy/diagnostics/N896_unresolved_diagnostics.tar.gz"
    )


def test_layout_migration_stages_verifies_and_finalizes_all_legacy_entries(
    tmp_path: Path,
) -> None:
    repo, root = _fixture_repo(tmp_path)
    plan, plan_path, stage, stage_path = _audit_plan_and_stage(repo, root)
    assert plan["item_count"] == 5
    assert stage["result_count"] == 5
    assert stage["all_sources_preserved"] is True
    for row in stage["results"]:
        assert Path(row["source_path"]).exists()
        assert Path(row["destination_path"]).is_file()
        assert Path(row["manifest_path"]).is_file()
        assert row["verification_mode"] in {"temporary_restore", "exact_byte_copy"}

    finalize = build_layout_finalize_plan(plan_path, stage_path)
    finalize_path = root / "catalog" / "output_layout_finalize_plan.json"
    write_layout_finalize_plan(finalize, finalize_path)

    with pytest.raises(ValueError, match="confirmation must equal"):
        execute_layout_finalize_plan(
            finalize_path,
            confirm_plan_sha256=finalize["plan_sha256"],
            confirm_delete="DELETE",
        )

    execution = execute_layout_finalize_plan(
        finalize_path,
        confirm_plan_sha256=finalize["plan_sha256"],
        confirm_delete=LAYOUT_FINALIZE_CONFIRMATION,
    )
    assert execution["removed_entry_count"] == 5
    assert execution["all_sources_removed"] is True
    assert execution["all_destinations_preserved"] is True

    refreshed = build_output_layout_audit(root, repo_root=repo)
    assert refreshed["layout_normalized"] is True
    assert refreshed["legacy_entry_count"] == 0
    assert refreshed["migration_blockers"] == []


def test_layout_stage_rejects_source_change_after_planning(tmp_path: Path) -> None:
    repo, root = _fixture_repo(tmp_path)
    audit = build_output_layout_audit(root, repo_root=repo)
    audit_path = root / "catalog" / "output_layout_audit.json"
    write_output_layout_audit(
        audit,
        json_path=audit_path,
        tsv_path=root / "catalog" / "output_layout_audit.tsv",
    )
    plan = build_layout_migration_plan(audit_path)
    plan_path = root / "catalog" / "output_layout_migration_plan.json"
    write_layout_migration_plan(plan, plan_path)

    (root / "N896_scan_logs" / "scan.log").write_text(
        "changed after planning\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source changed since planning"):
        stage_layout_migration(
            plan_path,
            confirm_plan_sha256=plan["plan_sha256"],
        )
    assert (root / "N896_scan_logs").is_dir()
    assert not (root / "archive" / "legacy" / "logs" / "N896_scan_logs.tar.gz").exists()


def test_finalize_rejects_replaced_stage_execution_after_planning(tmp_path: Path) -> None:
    repo, root = _fixture_repo(tmp_path)
    _, plan_path, _, stage_path = _audit_plan_and_stage(repo, root)
    finalize = build_layout_finalize_plan(plan_path, stage_path)
    finalize_path = root / "catalog" / "output_layout_finalize_plan.json"
    write_layout_finalize_plan(finalize, finalize_path)

    changed = json.loads(stage_path.read_text(encoding="utf-8"))
    changed["results"][0]["destination_bytes"] += 1
    changed.pop("stage_sha256")
    changed["stage_sha256"] = value_digest(changed)
    stage_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="changed after finalize planning"):
        execute_layout_finalize_plan(
            finalize_path,
            confirm_plan_sha256=finalize["plan_sha256"],
            confirm_delete=LAYOUT_FINALIZE_CONFIRMATION,
        )
    for item in finalize["items"]:
        assert Path(item["source_path"]).exists()


def test_finalize_plan_rejects_self_consistent_manifest_replacement(tmp_path: Path) -> None:
    repo, root = _fixture_repo(tmp_path)
    _, plan_path, stage, stage_path = _audit_plan_and_stage(repo, root)
    first = stage["results"][0]
    manifest_path = Path(first["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_path"] = str(root / "replacement")
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = value_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest relation mismatch"):
        build_layout_finalize_plan(plan_path, stage_path)
    for row in stage["results"]:
        assert Path(row["source_path"]).exists()
