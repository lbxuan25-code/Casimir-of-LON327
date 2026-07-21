from __future__ import annotations

import json
from pathlib import Path

from scripts.full_casimir.output_layout_review import build_reviewed_output_layout_audit


def _write_case(case: Path) -> None:
    case.mkdir(parents=True)
    for name, payload in {
        "config.json": {},
        "manifest.json": {"schema": "full-casimir-run-manifest"},
        "result.json": {"schema": "adaptive-matsubara-casimir-result-v1"},
        "summary.json": {"schema": "full-casimir-run-summary"},
    }.items():
        (case / name).write_text(json.dumps(payload), encoding="utf-8")


def test_reviewed_audit_accepts_extra_top_level_n896_evidence_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "outputs" / "casimir"
    for name in ("runs", "archive", "catalog", "reports", "workflow_logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("layout\n", encoding="utf-8")

    diagnostics = root / "diagnostics"
    nested = diagnostics / "N896_unresolved_diagnostics"
    cases = nested / "cases"
    _write_case(cases / "spm_T10K_d20nm_theta_p000deg_N896_grid2")
    (diagnostics / "N896_cache_primary_and_replay.json").write_text("{}", encoding="utf-8")
    (diagnostics / "N896_notes.txt").write_text("legacy evidence\n", encoding="utf-8")
    (nested / "overview.json").write_text("{}", encoding="utf-8")
    (nested / "system_info.json").write_text("{}", encoding="utf-8")

    audit = build_reviewed_output_layout_audit(root, repo_root=repo)
    entry = next(item for item in audit["entries"] if item["name"] == "diagnostics")

    assert entry["classification"] == "known_legacy"
    assert entry["kind"] == "legacy_n896_diagnostics_directory"
    assert entry["runtime_reference_count"] == 0
    assert audit["legacy_entry_count"] == 1
    assert audit["review_required_count"] == 0
    assert audit["migration_blockers"] == []


def test_reviewed_audit_rejects_mixed_diagnostics_directories(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "outputs" / "casimir"
    (root / "runs").mkdir(parents=True)

    diagnostics = root / "diagnostics"
    nested = diagnostics / "N896_unresolved_diagnostics"
    cases = nested / "cases"
    _write_case(cases / "spm_T10K_d20nm_theta_p000deg_N896_grid2")
    (diagnostics / "N896_cache_primary_and_replay.json").write_text("{}", encoding="utf-8")
    (nested / "overview.json").write_text("{}", encoding="utf-8")
    (nested / "system_info.json").write_text("{}", encoding="utf-8")
    (diagnostics / "current_diagnostics").mkdir()

    audit = build_reviewed_output_layout_audit(root, repo_root=repo)
    entry = next(item for item in audit["entries"] if item["name"] == "diagnostics")

    assert entry["classification"] == "review_required"
    assert audit["migration_blockers"] == ["manual review required: diagnostics"]
