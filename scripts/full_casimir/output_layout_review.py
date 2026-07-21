from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .output_layout import (
    LAYOUT_AUDIT_SCHEMA,
    LEGACY_N896_DIAGNOSTICS_DESTINATION,
    build_output_layout_audit,
    value_digest,
)


_REQUIRED_CASE_FILES = {"config.json", "manifest.json", "result.json", "summary.json"}


def _is_relaxed_legacy_n896_diagnostics(path: Path) -> bool:
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        return False

    primary = root / "N896_cache_primary_and_replay.json"
    nested = root / "N896_unresolved_diagnostics"
    cases = nested / "cases"
    if not primary.is_file() or not cases.is_dir():
        return False
    if not (nested / "overview.json").is_file() or not (nested / "system_info.json").is_file():
        return False

    try:
        top_level = list(root.iterdir())
    except OSError:
        return False

    top_directories = [item for item in top_level if item.is_dir() and not item.is_symlink()]
    if top_directories != [nested]:
        return False
    if any(item.is_symlink() or (not item.is_file() and not item.is_dir()) for item in top_level):
        return False

    case_directories = sorted(item for item in cases.iterdir() if item.is_dir() and not item.is_symlink())
    if not case_directories:
        return False
    if any(item.is_symlink() or (not item.is_file() and not item.is_dir()) for item in cases.iterdir()):
        return False

    for case in case_directories:
        if not case.name.endswith("_N896_grid2"):
            return False
        file_names = {item.name for item in case.iterdir() if item.is_file() and not item.is_symlink()}
        if not _REQUIRED_CASE_FILES.issubset(file_names):
            return False

    return True


def _rebuild_summary(payload: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(str(entry["classification"]) for entry in payload["entries"])
    blockers: list[str] = []
    for entry in payload["entries"]:
        classification = entry["classification"]
        if classification == "unexpected":
            blockers.append(f"unexpected root entry: {entry['name']}")
        if classification == "review_required":
            blockers.append(f"manual review required: {entry['name']}")
        if classification == "known_legacy" and entry.get("runtime_reference_count"):
            blockers.append(f"legacy entry still has runtime path references: {entry['name']}")

        type_counts = entry.get("entry_type_counts", {})
        if type_counts.get("symlink") or type_counts.get("other"):
            blockers.append(f"unsupported filesystem entry type: {entry['name']}")
        tar_summary = entry.get("tar_summary")
        if isinstance(tar_summary, Mapping):
            if tar_summary.get("parse_error"):
                blockers.append(f"unreadable archive: {entry['name']}")
            elif tar_summary.get("unsafe_member_count"):
                blockers.append(f"unsafe archive members: {entry['name']}")

    payload["classification_counts"] = dict(sorted(counts.items()))
    payload["legacy_entry_count"] = int(counts.get("known_legacy", 0))
    payload["review_required_count"] = int(counts.get("review_required", 0))
    payload["unexpected_entry_count"] = int(counts.get("unexpected", 0))
    payload["layout_normalized"] = not any(
        entry["classification"] in {"known_legacy", "review_required", "unexpected"}
        for entry in payload["entries"]
    )
    payload["migration_blockers"] = blockers
    payload["safety"]["diagnostics_requires_manual_review"] = bool(
        counts.get("review_required", 0)
    )
    payload["safety"]["legacy_n896_diagnostics_signature"] = (
        "required N896 primary JSON; unique top-level diagnostics directory; "
        "overview/system_info; nonempty _N896_grid2 cases with config/manifest/result/summary"
    )
    payload.pop("audit_sha256", None)
    payload["audit_sha256"] = value_digest(payload)
    return payload


def build_reviewed_output_layout_audit(
    casimir_root: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    payload = build_output_layout_audit(casimir_root, repo_root=repo_root)
    if payload.get("schema") != LAYOUT_AUDIT_SCHEMA:
        raise ValueError(f"unexpected layout audit schema: {payload.get('schema')}")

    for entry in payload["entries"]:
        if entry.get("name") != "diagnostics" or entry.get("classification") != "review_required":
            continue
        if not _is_relaxed_legacy_n896_diagnostics(Path(str(entry["path"]))):
            continue
        entry["classification"] = "known_legacy"
        entry["kind"] = "legacy_n896_diagnostics_directory"
        entry["proposed_destination"] = LEGACY_N896_DIAGNOSTICS_DESTINATION

    return _rebuild_summary(payload)


__all__ = ["build_reviewed_output_layout_audit"]
