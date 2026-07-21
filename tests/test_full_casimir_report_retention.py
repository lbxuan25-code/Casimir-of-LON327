from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.full_casimir.data_retention import pack_json_report
from scripts.full_casimir.report_retention import (
    REPORT_PRUNE_CONFIRMATION,
    build_report_prune_plan,
    execute_report_prune_plan,
    write_report_prune_plan,
)


def _packed_report(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "convergence_audit.json"
    source.write_text(
        json.dumps(
            {
                "schema": "audit-v1",
                "decision": {"status": "not_authorized"},
                "trace": [
                    {"index": index, "value": float(index) / 7.0}
                    for index in range(250)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    compact = tmp_path / "convergence_audit.compact.json"
    pack_root = tmp_path / "convergence_audit.pack"
    manifest = tmp_path / "convergence_audit.pack_manifest.json"
    pack_json_report(
        source,
        compact_path=compact,
        pack_root=pack_root,
        manifest_path=manifest,
        threshold_bytes=100,
    )
    return source, compact, pack_root, manifest


def test_report_prune_requires_verified_pack_and_exact_confirmation(tmp_path: Path) -> None:
    source, compact, pack_root, manifest = _packed_report(tmp_path)
    plan = build_report_prune_plan(manifest)
    plan_path = tmp_path / "report_prune_plan.json"
    write_report_prune_plan(plan, plan_path)

    assert source.is_file()
    assert compact.is_file()
    assert any(pack_root.iterdir())
    assert plan["source_bytes"] == source.stat().st_size
    assert plan["reconstruction_verified"] is True

    with pytest.raises(ValueError, match="confirmation SHA-256"):
        execute_report_prune_plan(
            plan_path,
            confirm_plan_sha256="wrong",
            confirm_delete=REPORT_PRUNE_CONFIRMATION,
        )
    with pytest.raises(ValueError, match="confirmation must equal"):
        execute_report_prune_plan(
            plan_path,
            confirm_plan_sha256=plan["plan_sha256"],
            confirm_delete="DELETE",
        )

    report = execute_report_prune_plan(
        plan_path,
        confirm_plan_sha256=plan["plan_sha256"],
        confirm_delete=REPORT_PRUNE_CONFIRMATION,
    )
    assert report["source_removed"] is True
    assert report["released_bytes"] == plan["source_bytes"]
    assert not source.exists()
    assert compact.is_file()
    assert any(pack_root.iterdir())


def test_report_prune_rejects_source_change_after_planning(tmp_path: Path) -> None:
    source, _, _, manifest = _packed_report(tmp_path)
    plan = build_report_prune_plan(manifest)
    plan_path = tmp_path / "report_prune_plan.json"
    write_report_prune_plan(plan, plan_path)

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["decision"]["status"] = "changed"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source report size changed|source report SHA-256 changed"):
        execute_report_prune_plan(
            plan_path,
            confirm_plan_sha256=plan["plan_sha256"],
            confirm_delete=REPORT_PRUNE_CONFIRMATION,
        )
    assert source.is_file()


def test_report_prune_rejects_sidecar_change(tmp_path: Path) -> None:
    source, _, pack_root, manifest = _packed_report(tmp_path)
    plan = build_report_prune_plan(manifest)
    plan_path = tmp_path / "report_prune_plan.json"
    write_report_prune_plan(plan, plan_path)

    sidecar = next(pack_root.iterdir())
    sidecar.write_bytes(sidecar.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="sidecar size mismatch|sidecar SHA-256 mismatch"):
        execute_report_prune_plan(
            plan_path,
            confirm_plan_sha256=plan["plan_sha256"],
            confirm_delete=REPORT_PRUNE_CONFIRMATION,
        )
    assert source.is_file()
