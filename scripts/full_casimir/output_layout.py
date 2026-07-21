from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import stat
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

LAYOUT_AUDIT_SCHEMA = "casimir-output-layout-audit-v2"

CANONICAL_DIRECTORIES = {
    "archive",
    "catalog",
    "postprocessed",
    "reports",
    "runs",
    "workflow_logs",
}
CANONICAL_FILES = {"README.md"}
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
LEGACY_N896_DIAGNOSTICS_DESTINATION = (
    "archive/legacy/diagnostics/N896_unresolved_diagnostics.tar.gz"
)

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
_SELF_REPORT_NAMES = {"output_layout_audit.json", "output_layout_audit.tsv"}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def value_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def tree_rows(path: Path, *, hash_files: bool) -> list[dict[str, Any]]:
    root = Path(path).absolute()
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
                "sha256": sha256_file(root) if hash_files else None,
            }
        ]
    if not root.is_dir():
        info = root.lstat()
        return [
            {
                "relative_path": ".",
                "type": "other",
                "bytes": int(info.st_size),
                "mode": stat.S_IMODE(info.st_mode),
                "sha256": None,
            }
        ]
    for candidate in sorted(root.rglob("*")):
        if (
            root.name == "catalog"
            and candidate.parent == root
            and candidate.name in _SELF_REPORT_NAMES
        ):
            continue
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
            digest = sha256_file(candidate) if hash_files else None
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


def output_entry_snapshot(path: Path) -> dict[str, Any]:
    source = Path(path).absolute()
    rows = tree_rows(source, hash_files=True)
    files = [row for row in rows if row["type"] == "file"]
    types = Counter(str(row["type"]) for row in rows)
    return {
        "path": str(source),
        "kind": (
            "symlink"
            if source.is_symlink()
            else "directory"
            if source.is_dir()
            else "file"
            if source.is_file()
            else "other"
        ),
        "bytes": sum(int(row["bytes"]) for row in files),
        "file_count": len(files),
        "entry_type_counts": dict(sorted(types.items())),
        "tree_digest": value_digest(rows),
        "sha256": sha256_file(source) if source.is_file() and not source.is_symlink() else None,
        "tree_rows": rows,
    }


def _json_summary(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() != ".json" or not path.is_file() or path.is_symlink():
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
    if path.is_symlink() or not path.name.endswith((".tar.gz", ".tgz", ".tar")):
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


def _reference_patterns(name: str) -> tuple[re.Pattern[str], ...]:
    escaped = re.escape(name)
    canonical = re.compile(rf"outputs[/\\]casimir[/\\]{escaped}(?:[/\\]|[\"']|$)")
    if name != "diagnostics":
        return (canonical, re.compile(escaped))
    constructed = re.compile(
        rf"(?:casimir_root|output_root|DEFAULT_OUTPUT_ROOT\.parent)\s*/\s*[\"']{escaped}[\"']"
    )
    split_literal = re.compile(
        rf"[\"']outputs[\"']\s*/\s*[\"']casimir[\"']\s*/\s*[\"']{escaped}[\"']"
    )
    return canonical, constructed, split_literal


def _reference_scan(
    repo_root: Path,
    entry_names: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    names = tuple(sorted(set(str(name) for name in entry_names)))
    patterns = {name: _reference_patterns(name) for name in names}
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
                if not any(pattern.search(line) for pattern in patterns[name]):
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


def _is_legacy_n896_diagnostics(path: Path) -> bool:
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        return False
    allowed_top_level = {
        "N896_cache_primary_and_replay.json",
        "N896_unresolved_diagnostics",
    }
    try:
        top_level = {child.name for child in root.iterdir()}
    except OSError:
        return False
    if top_level != allowed_top_level:
        return False
    primary = root / "N896_cache_primary_and_replay.json"
    nested = root / "N896_unresolved_diagnostics"
    cases = nested / "cases"
    if not primary.is_file() or not cases.is_dir():
        return False
    case_directories = sorted(child for child in cases.iterdir() if child.is_dir())
    if not case_directories:
        return False
    required = {"config.json", "manifest.json", "result.json", "summary.json"}
    for case in case_directories:
        if not case.name.endswith("_N896_grid2"):
            return False
        if not required.issubset({child.name for child in case.iterdir() if child.is_file()}):
            return False
    rows = tree_rows(root, hash_files=False)
    types = Counter(str(row["type"]) for row in rows)
    return not types.get("symlink") and not types.get("other")


def _entry_classification(name: str, path: Path) -> tuple[str, str, str | None]:
    if name in CANONICAL_DIRECTORIES and path.is_dir() and not path.is_symlink():
        return "canonical", "canonical_directory", None
    if name in CANONICAL_FILES and path.is_file() and not path.is_symlink():
        return "canonical", "canonical_file", None
    if name == "diagnostics" and _is_legacy_n896_diagnostics(path):
        return (
            "known_legacy",
            "legacy_n896_diagnostics_directory",
            LEGACY_N896_DIAGNOSTICS_DESTINATION,
        )
    if name == "diagnostics" and path.is_dir() and not path.is_symlink():
        return "review_required", "ambiguous_diagnostics_directory", (
            "Contents do not match the frozen N896 legacy signature; inspect before migration."
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
        snapshot = output_entry_snapshot(path) if detailed else None
        rows = snapshot["tree_rows"] if snapshot is not None else tree_rows(path, hash_files=False)
        file_rows = [row for row in rows if row["type"] == "file"]
        type_counts = Counter(str(row["type"]) for row in rows)
        entry_references = references.get(path.name, [])
        runtime_references = [
            row for row in entry_references if row.get("role") == "runtime"
        ]
        regular_file = path.is_file() and not path.is_symlink()
        entry = {
            "name": path.name,
            "path": str(path),
            "classification": classification,
            "kind": kind,
            "proposed_destination": proposed_destination,
            "bytes": sum(int(row["bytes"]) for row in file_rows),
            "file_count": len(file_rows),
            "entry_type_counts": dict(sorted(type_counts.items())),
            "tree_digest": snapshot["tree_digest"] if snapshot is not None else None,
            "sha256": sha256_file(path) if regular_file else None,
            "references": entry_references,
            "reference_count": len(entry_references),
            "runtime_reference_count": len(runtime_references),
            "reference_role_counts": dict(
                sorted(Counter(str(row.get("role")) for row in entry_references).items())
            ),
            "tar_summary": _tar_summary(path) if regular_file else None,
            "json_files": [],
        }
        if detailed and path.is_dir() and not path.is_symlink():
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
    blockers: list[str] = []
    for entry in entries:
        if entry["classification"] == "unexpected":
            blockers.append(f"unexpected root entry: {entry['name']}")
        if entry["classification"] == "review_required":
            blockers.append(f"manual review required: {entry['name']}")
        if entry["classification"] == "known_legacy" and entry["runtime_reference_count"]:
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

    payload = {
        "schema": LAYOUT_AUDIT_SCHEMA,
        "repo_root": str(repository),
        "casimir_root": str(root),
        "canonical_directories": sorted(CANONICAL_DIRECTORIES),
        "canonical_files": sorted(CANONICAL_FILES),
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
            "unknown_diagnostics_requires_manual_review": True,
            "n896_diagnostics_signature_is_machine_checked": True,
            "only_runtime_path_references_block_migration": True,
            "symlinks_and_special_files_block_migration": True,
            "self_generated_reports_excluded_from_catalog_stats": True,
        },
    }
    payload["audit_sha256"] = value_digest(payload)
    return payload


def write_output_layout_audit(
    audit: Mapping[str, Any],
    *,
    json_path: Path,
    tsv_path: Path,
) -> tuple[Path, Path]:
    json_target = atomic_json_write(Path(json_path), dict(audit))
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
    "LEGACY_N896_DIAGNOSTICS_DESTINATION",
    "atomic_json_write",
    "build_output_layout_audit",
    "canonical_bytes",
    "output_entry_snapshot",
    "sha256_file",
    "tree_rows",
    "value_digest",
    "write_output_layout_audit",
]
