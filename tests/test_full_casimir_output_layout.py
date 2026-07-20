from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from scripts.full_casimir.output_layout import (
    LAYOUT_AUDIT_SCHEMA,
    build_output_layout_audit,
    write_output_layout_audit,
)


def _write_tar(path: Path, member_name: str, payload: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as handle:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))


def test_layout_audit_classifies_canonical_legacy_and_review_entries(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    root = repo / "outputs" / "casimir"
    for name in ("runs", "archive", "catalog", "reports", "workflow_logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("layout\n", encoding="utf-8")

    legacy_logs = root / "N896_scan_logs"
    legacy_logs.mkdir()
    (legacy_logs / "scan.log").write_text("old scan\n", encoding="utf-8")

    diagnostics = root / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "summary.json").write_text(
        json.dumps({"schema": "legacy-diagnostics-v1", "status": "complete"}),
        encoding="utf-8",
    )

    snapshot = root / "dwave_0deg_pilot_cache.tar.gz"
    _write_tar(snapshot, "dwave/cache/certified_points.json", b"{}")

    source = repo / "scripts" / "reader.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        'path = "outputs/casimir/N896_scan_logs"\n',
        encoding="utf-8",
    )

    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    audit = build_output_layout_audit(root, repo_root=repo)
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    assert before == after
    assert audit["schema"] == LAYOUT_AUDIT_SCHEMA
    assert audit["layout_normalized"] is False
    assert audit["legacy_entry_count"] == 2
    assert audit["review_required_count"] == 1
    assert audit["unexpected_entry_count"] == 0
    assert audit["safety"]["read_only"] is True

    entries = {entry["name"]: entry for entry in audit["entries"]}
    assert entries["runs"]["classification"] == "canonical"
    assert entries["N896_scan_logs"]["classification"] == "known_legacy"
    assert entries["N896_scan_logs"]["reference_count"] == 1
    assert entries["N896_scan_logs"]["tree_digest"]
    assert entries["diagnostics"]["classification"] == "review_required"
    assert entries["diagnostics"]["json_files"] == [
        {
            "relative_path": "summary.json",
            "parse_error": None,
            "schema": "legacy-diagnostics-v1",
            "top_level_type": "dict",
        }
    ]
    assert entries["dwave_0deg_pilot_cache.tar.gz"]["tar_summary"][
        "unsafe_member_count"
    ] == 0
    assert entries["dwave_0deg_pilot_cache.tar.gz"]["tar_summary"][
        "top_level_roots"
    ] == ["dwave"]
    assert any("still referenced" in item for item in audit["migration_blockers"])


def test_layout_audit_flags_unexpected_and_unsafe_archives(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "outputs" / "casimir"
    root.mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "mystery.bin").write_bytes(b"mystery")
    unsafe = root / "0deg_pilot_v2_diagnostics.tar.gz"
    _write_tar(unsafe, "../escape.txt")

    audit = build_output_layout_audit(root, repo_root=repo)
    entries = {entry["name"]: entry for entry in audit["entries"]}

    assert audit["unexpected_entry_count"] == 1
    assert entries["mystery.bin"]["classification"] == "unexpected"
    assert entries["0deg_pilot_v2_diagnostics.tar.gz"]["tar_summary"][
        "unsafe_member_count"
    ] == 1
    assert any("unsafe archive members" in item for item in audit["migration_blockers"])


def test_layout_audit_writer_emits_json_and_tsv(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "outputs" / "casimir"
    (root / "runs").mkdir(parents=True)
    audit = build_output_layout_audit(root, repo_root=repo)

    json_path, tsv_path = write_output_layout_audit(
        audit,
        json_path=root / "catalog" / "output_layout_audit.json",
        tsv_path=root / "catalog" / "output_layout_audit.tsv",
    )

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["audit_sha256"] == audit["audit_sha256"]
    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("classification\tkind\tbytes")
    assert any("canonical_directory" in line and line.endswith("runs\t") for line in lines[1:])
