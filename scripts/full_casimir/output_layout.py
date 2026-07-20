from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

LAYOUT_AUDIT_SCHEMA = "casimir-output-layout-audit-v1"

CANONICAL_DIRECTORIES = {
    "archive",
    "catalog",
    "postprocessed",
    "reports",
    "runs",
    "workflow_logs",
}
CANONICAL_FILES = {"README.md"}
REVIEW_DIRECTORIES = {"diagnostics"}
KNOWN_LEGACY_ENTRIES: Mapping[str, Mapping[str, str]] = {
    "0deg_runtime_budget_pilot_logs": {
        "kind": "legacy_log_directory",
        "proposed_destination": "archive/legacy/logs/0deg_runtime_budget_pilot_logs.tar.gz",
    },
    "N896_scan_logs": {
        "kind": "legacy_log_directory",
        "proposed_destination": "archive/legacy/logs/N896_scan_logs.tar.gz",
    },
    "0deg_pilot_v2_diagnostics.tar.gz": {
        "kind": "legacy_snapshot_archive",
        "proposed_destination": "archive/legacy/snapshots/0deg_pilot_v2_diagnostics.tar.gz",
    },
    "dwave_0deg_pilot_cache.tar.gz": {
        "kind": "legacy_snapshot_archive",
        "proposed_destination": "archive/legacy/snapshots/dwave_0deg_pilot_cache.tar.gz",
    },
}

_TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_SKIP_REPOSITORY_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "outputs",
    "venv",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _tree_rows(path: Path, *, hash_files: bool) -> list[dict[str, Any]]:
    root = Path(path).resolve()
    rows: list[dict[str, Any]] = []
    if root.is_symlink():
        info = root.lstat()
        return [
            {
                "relative_path": ".",
                "type": "symlink",
                "bytes": int(info.st_size),
                "mode": stat.S_IMODE(info.st_mode),
                "sha256": None,
            }
        ]
    if root.is_file():
        info = root.stat()
        return [
            {
                "relative_path": root.name,
                "type": "file",
                "bytes": int(info.st_size),
                "mode": stat.S_IMODE(info.st_mode),
                "sha256": _sha256(root) if hash_files else None,
            }
        ]
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root).as_posix()
        info = candidate.lstat()
        if candidate.is_symlink():
            item_type = "symlink"
            digest = None
        elif candidate.is_dir():
            item_type = "directory"
            digest = None
        elif candidate.is_file():
            item_type = "file"
            digest = _sha256(candidate) if hash_files else None
        else:
            item_type = "other"
            digest = None
        rows.append(
            {
                "relative_path": relative,
                "type": item_type,
                "bytes": int(info.st_size) if item_type != "directory" else 0,
                "mode": stat.S_IMODE(info.st_mode),
                "sha256": digest,
            }
        )
    return rows


def _json_summary(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() != ".json" or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"parse_error": f"{type(exc).__name__}: {exc}", "schema": None}
    return {
        "parse_error": None,
        "schema": value.get("schema") if isinstance(value, Mapping) else None,
        "top_level_type": type(value).__name__,
    }


def _tar_summary(path: Path) -> dict[str, Any] | None:
    if not path.name.endswith((".tar.gz", ".tgz", ".tar")):
        return None
    try:
        with tarfile.open(path, "r:*") as handle:
            members = handle.getmembers()
    except (OSError, tarfile.TarError) as exc:
        return {
            "parse_error": f"{type(exc).__name__}: {exc}",
            "member_count": 0,
            "unsafe_member_count": None,
        }

    type_counts: Counter[str] = Counter()
    top_level_roots: set[str] = set()
    unsafe: list[str] = []
    member_bytes = 0
    for member in members:
        pure = PurePosixPath(member.name)
        if pure.parts:
            top_level_roots.add(pure.parts[0])
        if pure.is_absolute() or ".." in pure.parts:
            unsafe.append(member.name)
        if member.issym() or member.islnk() or member.isdev():
            unsafe.append(member.name)
        if member.isdir():
            member_type = "directory"
        elif member.isfile():
            member_type = "file"
            member_bytes += int(member.size)
        elif member.issym():
            member_type = "symlink"
        elif member.islnk():
            member_type = "hardlink"
        elif member.isdev():
            member_type = "device"
        else:
            member_type = "other"
        type_counts[member_type] += 1
    return {
        "parse_error": None,
        "member_count": len(members),
        "member_file_bytes": member_bytes,
        "member_type_counts": dict(sorted(type_counts.items())),
        "top_level_roots": sorted(top_level_roots),
        "unsafe_member_count": len(set(unsafe)),
        "unsafe_members": sorted(set(unsafe))[:20],
    }


def _repository_text_files(repo_root: Path) -> Iterable[Path]:
    root = Path(repo_root).resolve()
    for directory, names, filenames in os.walk(root):
        names[:] = sorted(
            name for name in names if name not in _SKIP_REPOSITORY_DIRECTORIES
        )
        base = Path(directory)
        for filename in sorted(filenames):
            path = base / filename
            if path.suffix.lower() in _TEXT_SUFFIXES:
                yield path


def _reference_role(relative_path: str) -> str:
    if relative_path == "scripts/full_casimir/output_layout.py":
        return "contract_definition"
    if relative_path.startswith("tests/"):
        return "test"
    if relative_path.startswith("docs/") or relative_path.endswith(".md"):
        return "documentation"
    return "runtime"


def _reference_scan(
    repo_root: Path,
    entry_names: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    names = tuple(sorted(set(str(name) for name in entry_names)))
    findings: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for path in _repository_text_files(repo_root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        role = _reference_role(relative)
        for line_number, line in enumerate(lines, start=1):
            for name in names:
                canonical_reference = f"outputs/casimir/{name}"
                if canonical_reference not in line and name not in line:
                    continue
                findings[name].append(
                    {
                        "path": relative,
                        "line": line_number,
                        "role": role,
                        "text": line.strip()[:500],
                    }
                )
    return findings


def _entry_classification(name: str, path: Path) -> tuple[str, str, str | None]:
    if name in CANONICAL_DIRECTORIES and path.is_dir():
        return "canonical", "canonical_directory", None
    if name in CANONICAL_FILES and path.is_file():
        return "canonical", "canonical_file", None
    if name in REVIEW_DIRECTORIES and path.is_dir():
        return "review_required", "ambiguous_diagnostics_directory", (
            "Inspect contents and references before deciding whether to retain, "
            "split, or archive this root-level directory."
        )
    legacy = KNOWN_LEGACY_ENTRIES.get(name)
    if legacy is not None:
        return "known_legacy", str(legacy["kind"]), str(legacy["proposed_destination"])
    return "unexpected", "unclassified_root_entry", None


def build_output_layout_audit(
    casimir_root: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    root = Path(casimir_root).resolve()
    repository = Path(repo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Casimir output root is missing: {root}")
    root.relative_to(repository)

    top_level = sorted(root.iterdir(), key=lambda item: item.name)
    reference_names = [
        item.name
        for item in top_level
        if item.name not in CANONICAL_DIRECTORIES | CANONICAL_FILES
    ]
    references = _reference_scan(repository, reference_names)

    entries: list[dict[str, Any]] = []
    for path in top_level:
        classification, kind, proposed_destination = _entry_classification(path.name, path)
        detailed = classification in {"known_legacy", "review_required", "unexpected"}
        rows = _tree_rows(path, hash_files=detailed)
        file_rows = [row for row in rows if row["type"] == "file"]
        type_counts = Counter(str(row["type"]) for row in rows)
        entry_references = references.get(path.name, [])
        runtime_references = [
            row for row in entry_references if row.get("role") == "runtime"
        ]
        entry = {
            "name": path.name,
            "path": str(path),
            "classification": classification,
            "kind": kind,
            "proposed_destination": proposed_destination,
            "bytes": sum(int(row["bytes"]) for row in file_rows),
            "file_count": len(file_rows),
            "entry_type_counts": dict(sorted(type_counts.items())),
            "tree_digest": _digest(rows) if detailed else None,
            "sha256": _sha256(path) if path.is_file() else None,
            "references": entry_references,
            "reference_count": len(entry_references),
            "runtime_reference_count": len(runtime_references),
            "reference_role_counts": dict(
                sorted(Counter(str(row.get("role")) for row in entry_references).items())
            ),
            "tar_summary": _tar_summary(path) if path.is_file() else None,
            "json_files": [],
        }
        if detailed and path.is_dir():
            json_files = []
            for candidate in sorted(path.rglob("*.json")):
                summary = _json_summary(candidate)
                if summary is not None:
                    json_files.append(
                        {
                            "relative_path": candidate.relative_to(path).as_posix(),
                            **summary,
                        }
                    )
            entry["json_files"] = json_files
        entries.append(entry)

    counts = Counter(str(entry["classification"]) for entry in entries)
    blockers = []
    for entry in entries:
        if entry["classification"] == "unexpected":
            blockers.append(f"unexpected root entry: {entry['name']}")
        if entry["classification"] == "review_required":
            blockers.append(f"manual review required: {entry['name']}")
        if (
            entry["classification"] == "known_legacy"
            and entry["runtime_reference_count"]
        ):
            blockers.append(
                f"legacy entry still has runtime references: {entry['name']}"
            )
        tar_summary = entry.get("tar_summary")
        if isinstance(tar_summary, Mapping) and tar_summary.get("unsafe_member_count"):
            blockers.append(f"unsafe archive members: {entry['name']}")

    payload = {
        "schema": LAYOUT_AUDIT_SCHEMA,
        "repo_root": str(repository),
        "casimir_root": str(root),
        "canonical_directories": sorted(CANONICAL_DIRECTORIES),
        "canonical_files": sorted(CANONICAL_FILES),
        "review_directories": sorted(REVIEW_DIRECTORIES),
        "entry_count": len(entries),
        "classification_counts": dict(sorted(counts.items())),
        "legacy_entry_count": int(counts.get("known_legacy", 0)),
        "review_required_count": int(counts.get("review_required", 0)),
        "unexpected_entry_count": int(counts.get("unexpected", 0)),
        "layout_normalized": not any(
            entry["classification"] in {"known_legacy", "review_required", "unexpected"}
            for entry in entries
        ),
        "migration_blockers": blockers,
        "entries": entries,
        "safety": {
            "read_only": True,
            "source_modified": False,
            "legacy_entries_are_not_moved": True,
            "diagnostics_requires_manual_review": True,
            "only_runtime_references_block_migration": True,
        },
    }
    payload["audit_sha256"] = _digest(payload)
    return payload


def write_output_layout_audit(
    audit: Mapping[str, Any],
    *,
    json_path: Path,
    tsv_path: Path,
) -> tuple[Path, Path]:
    json_target = _atomic_json_write(Path(json_path), dict(audit))
    tsv_target = Path(tsv_path).resolve()
    tsv_target.parent.mkdir(parents=True, exist_ok=True)
    temporary = tsv_target.with_name(f".{tsv_target.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "classification",
                "kind",
                "bytes",
                "file_count",
                "reference_count",
                "runtime_reference_count",
                "name",
                "proposed_destination",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for entry in audit.get("entries", []):
            writer.writerow(
                {
                    "classification": entry.get("classification"),
                    "kind": entry.get("kind"),
                    "bytes": entry.get("bytes"),
                    "file_count": entry.get("file_count"),
                    "reference_count": entry.get("reference_count"),
                    "runtime_reference_count": entry.get("runtime_reference_count"),
                    "name": entry.get("name"),
                    "proposed_destination": entry.get("proposed_destination") or "",
                }
            )
    os.replace(temporary, tsv_target)
    return json_target, tsv_target


__all__ = [
    "CANONICAL_DIRECTORIES",
    "CANONICAL_FILES",
    "KNOWN_LEGACY_ENTRIES",
    "LAYOUT_AUDIT_SCHEMA",
    "REVIEW_DIRECTORIES",
    "build_output_layout_audit",
    "write_output_layout_audit",
]
