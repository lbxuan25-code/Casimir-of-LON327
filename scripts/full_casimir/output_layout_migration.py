from __future__ import annotations

import gzip
import json
import os
import shutil
import stat
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from .output_layout import (
    LAYOUT_AUDIT_SCHEMA,
    atomic_json_write,
    output_entry_snapshot,
    sha256_file,
    tree_rows,
    value_digest,
)

LAYOUT_MIGRATION_PLAN_SCHEMA = "casimir-output-layout-migration-plan-v1"
LAYOUT_STAGE_EXECUTION_SCHEMA = "casimir-output-layout-stage-execution-v1"
LAYOUT_FINALIZE_PLAN_SCHEMA = "casimir-output-layout-finalize-plan-v1"
LAYOUT_FINALIZE_EXECUTION_SCHEMA = "casimir-output-layout-finalize-execution-v1"
LAYOUT_FINALIZE_CONFIRMATION = "REMOVE_STAGED_LEGACY_ROOT_ENTRIES"


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _without_digest(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result.pop(field, None)
    return result


def _require_self_digest(payload: Mapping[str, Any], field: str, label: str) -> None:
    expected = str(payload.get(field, ""))
    if not expected or value_digest(_without_digest(payload, field)) != expected:
        raise ValueError(f"{label} digest mismatch")


def _require_snapshot(entry: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(entry["source_path"])).absolute()
    if not source.exists() and not source.is_symlink():
        raise FileNotFoundError(f"layout source is missing: {source}")
    current = output_entry_snapshot(source)
    for field in ("kind", "bytes", "file_count", "entry_type_counts", "tree_digest", "sha256"):
        if current.get(field) != entry.get(f"source_{field}"):
            raise ValueError(f"layout source changed since planning: {source} ({field})")
    types = current["entry_type_counts"]
    if types.get("symlink") or types.get("other"):
        raise ValueError(f"unsupported layout source entry type: {source}")
    return current


def build_layout_migration_plan(audit_path: Path) -> dict[str, Any]:
    path = Path(audit_path).resolve()
    audit = _read(path)
    if not isinstance(audit, Mapping) or audit.get("schema") != LAYOUT_AUDIT_SCHEMA:
        raise ValueError(f"layout audit must use schema {LAYOUT_AUDIT_SCHEMA}")
    _require_self_digest(audit, "audit_sha256", "layout audit")
    blockers = list(audit.get("migration_blockers", []))
    if blockers:
        raise ValueError(f"layout audit has migration blockers: {blockers}")

    root = Path(str(audit["casimir_root"])).resolve()
    legacy_root = (root / "archive" / "legacy").resolve()
    items: list[dict[str, Any]] = []
    for entry in audit.get("entries", []):
        if entry.get("classification") != "known_legacy":
            continue
        source = Path(str(entry["path"])).absolute()
        snapshot = output_entry_snapshot(source)
        if snapshot["tree_digest"] != entry.get("tree_digest"):
            raise ValueError(f"legacy source changed after audit: {source}")
        destination = (root / str(entry["proposed_destination"])).resolve()
        destination.relative_to(legacy_root)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"layout migration destination already exists: {destination}")
        action = "archive_directory" if snapshot["kind"] == "directory" else "copy_file"
        if action == "copy_file" and snapshot["kind"] != "file":
            raise ValueError(f"unsupported legacy source kind: {source}")
        items.append(
            {
                "name": str(entry["name"]),
                "legacy_kind": str(entry["kind"]),
                "action": action,
                "source_path": str(source),
                "source_kind": snapshot["kind"],
                "source_bytes": snapshot["bytes"],
                "source_file_count": snapshot["file_count"],
                "source_entry_type_counts": snapshot["entry_type_counts"],
                "source_tree_digest": snapshot["tree_digest"],
                "source_sha256": snapshot["sha256"],
                "destination_path": str(destination),
                "manifest_path": str(destination.with_name(f"{destination.name}.manifest.json")),
            }
        )
    if not items:
        raise ValueError("layout audit contains no legacy entries to migrate")
    items.sort(key=lambda row: row["name"])
    payload = {
        "schema": LAYOUT_MIGRATION_PLAN_SCHEMA,
        "audit_path": str(path),
        "audit_sha256": audit["audit_sha256"],
        "casimir_root": str(root),
        "item_count": len(items),
        "source_total_bytes": sum(int(row["source_bytes"]) for row in items),
        "items": items,
        "safety": {
            "stage_creates_verified_destinations_only": True,
            "stage_does_not_remove_sources": True,
            "finalize_requires_separate_plan": True,
        },
    }
    payload["plan_sha256"] = value_digest(payload)
    return payload


def write_layout_migration_plan(plan: Mapping[str, Any], path: Path) -> Path:
    return atomic_json_write(Path(path), dict(plan))


def _normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = stat.S_IMODE(info.mode)
    return info


def _create_directory_archive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                candidates = [source, *sorted(source.rglob("*"))]
                for candidate in candidates:
                    if candidate.is_symlink():
                        raise ValueError(f"cannot archive symlink: {candidate}")
                    relative = candidate.relative_to(source.parent).as_posix()
                    info = _normalized_tar_info(archive.gettarinfo(str(candidate), arcname=relative))
                    if candidate.is_dir():
                        archive.addfile(info)
                    elif candidate.is_file():
                        with candidate.open("rb") as handle:
                            archive.addfile(info, handle)
                    else:
                        raise ValueError(f"cannot archive special filesystem entry: {candidate}")


def _safe_restore_directory_archive(
    archive_path: Path,
    *,
    expected_root_name: str,
    expected_rows: Sequence[Mapping[str, Any]],
) -> None:
    with TemporaryDirectory(prefix="casimir-layout-restore-") as temporary:
        target = Path(temporary)
        with tarfile.open(archive_path, "r:*") as archive:
            members = archive.getmembers()
            for member in members:
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                    raise ValueError(f"unsafe layout archive member: {member.name}")
                if pure.parts[0] != expected_root_name:
                    raise ValueError(f"unexpected layout archive root: {member.name}")
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"unsafe layout archive member type: {member.name}")
                destination = target.joinpath(*pure.parts)
                destination.resolve().relative_to(target.resolve())
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    os.chmod(destination, stat.S_IMODE(member.mode))
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"cannot read layout archive member: {member.name}")
                    with destination.open("wb") as output:
                        shutil.copyfileobj(extracted, output)
                    os.chmod(destination, stat.S_IMODE(member.mode))
                else:
                    raise ValueError(f"unsupported layout archive member: {member.name}")
        restored_root = target / expected_root_name
        restored_rows = tree_rows(restored_root, hash_files=True)
        if restored_rows != [dict(row) for row in expected_rows]:
            raise RuntimeError(f"restored layout archive tree mismatch: {archive_path}")


def _stage_manifest(item: Mapping[str, Any], destination: Path, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "casimir-output-layout-staged-artifact-v1",
        "name": item["name"],
        "legacy_kind": item["legacy_kind"],
        "action": item["action"],
        "source_path": item["source_path"],
        "source_kind": item["source_kind"],
        "source_bytes": item["source_bytes"],
        "source_file_count": item["source_file_count"],
        "source_tree_digest": item["source_tree_digest"],
        "source_sha256": item["source_sha256"],
        "source_tree_rows": snapshot["tree_rows"],
        "destination_path": str(destination),
        "destination_bytes": destination.stat().st_size,
        "destination_sha256": sha256_file(destination),
        "restore_verified": True,
        "source_preserved": Path(str(item["source_path"])).exists(),
    }


def stage_layout_migration(
    plan_path: Path,
    *,
    confirm_plan_sha256: str,
) -> dict[str, Any]:
    path = Path(plan_path).resolve()
    plan = _read(path)
    if not isinstance(plan, Mapping) or plan.get("schema") != LAYOUT_MIGRATION_PLAN_SCHEMA:
        raise ValueError(f"layout migration plan must use schema {LAYOUT_MIGRATION_PLAN_SCHEMA}")
    _require_self_digest(plan, "plan_sha256", "layout migration plan")
    if confirm_plan_sha256 != plan.get("plan_sha256"):
        raise ValueError("layout migration plan confirmation SHA-256 does not match")

    prepared: list[tuple[Path, Path, dict[str, Any], Path]] = []
    try:
        for item in plan["items"]:
            snapshot = _require_snapshot(item)
            destination = Path(str(item["destination_path"])).resolve()
            manifest_path = Path(str(item["manifest_path"])).resolve()
            if destination.exists() or manifest_path.exists():
                raise FileExistsError(f"staged layout destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.stage.tmp")
            if temporary.exists():
                temporary.unlink()
            source = Path(str(item["source_path"])).absolute()
            if item["action"] == "archive_directory":
                _create_directory_archive(source, temporary)
                _safe_restore_directory_archive(
                    temporary,
                    expected_root_name=source.name,
                    expected_rows=snapshot["tree_rows"],
                )
            elif item["action"] == "copy_file":
                shutil.copyfile(source, temporary)
                if sha256_file(temporary) != item.get("source_sha256"):
                    raise RuntimeError(f"staged file copy SHA-256 mismatch: {source}")
            else:
                raise ValueError(f"unsupported layout migration action: {item['action']}")
            manifest = _stage_manifest(item, temporary, snapshot)
            prepared.append((temporary, destination, manifest, manifest_path))

        results: list[dict[str, Any]] = []
        for temporary, destination, manifest, manifest_path in prepared:
            os.replace(temporary, destination)
            manifest["destination_path"] = str(destination)
            manifest["destination_bytes"] = destination.stat().st_size
            manifest["destination_sha256"] = sha256_file(destination)
            manifest["manifest_sha256"] = value_digest(manifest)
            atomic_json_write(manifest_path, manifest)
            results.append(
                {
                    "name": manifest["name"],
                    "status": "staged_and_verified",
                    "source_path": manifest["source_path"],
                    "source_still_present": Path(manifest["source_path"]).exists(),
                    "destination_path": str(destination),
                    "destination_bytes": manifest["destination_bytes"],
                    "destination_sha256": manifest["destination_sha256"],
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest["manifest_sha256"],
                }
            )
    except Exception:
        for temporary, _, _, _ in prepared:
            if temporary.exists():
                temporary.unlink()
        raise

    report = {
        "schema": LAYOUT_STAGE_EXECUTION_SCHEMA,
        "plan_path": str(path),
        "plan_sha256": plan["plan_sha256"],
        "result_count": len(results),
        "results": results,
        "source_removal_performed": False,
        "all_sources_preserved": all(row["source_still_present"] for row in results),
    }
    report["stage_sha256"] = value_digest(report)
    return report


def write_layout_stage_execution(report: Mapping[str, Any], path: Path) -> Path:
    return atomic_json_write(Path(path), dict(report))


def _verify_staged_result(item: Mapping[str, Any], staged: Mapping[str, Any]) -> None:
    snapshot = _require_snapshot(item)
    destination = Path(str(item["destination_path"])).resolve()
    manifest_path = Path(str(item["manifest_path"])).resolve()
    if not destination.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"staged layout artifact is missing: {destination}")
    manifest = _read(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"invalid staged layout manifest: {manifest_path}")
    _require_self_digest(manifest, "manifest_sha256", "staged layout manifest")
    if sha256_file(destination) != manifest.get("destination_sha256"):
        raise ValueError(f"staged layout destination SHA-256 mismatch: {destination}")
    if staged.get("destination_sha256") != manifest.get("destination_sha256"):
        raise ValueError(f"layout stage report does not match manifest: {destination}")
    if item["action"] == "archive_directory":
        _safe_restore_directory_archive(
            destination,
            expected_root_name=Path(str(item["source_path"])).name,
            expected_rows=snapshot["tree_rows"],
        )
    elif item["action"] == "copy_file":
        if manifest.get("destination_sha256") != item.get("source_sha256"):
            raise ValueError(f"staged file copy differs from source: {destination}")


def build_layout_finalize_plan(
    migration_plan_path: Path,
    stage_execution_path: Path,
) -> dict[str, Any]:
    migration_path = Path(migration_plan_path).resolve()
    stage_path = Path(stage_execution_path).resolve()
    migration = _read(migration_path)
    stage = _read(stage_path)
    if migration.get("schema") != LAYOUT_MIGRATION_PLAN_SCHEMA:
        raise ValueError("invalid layout migration plan schema")
    if stage.get("schema") != LAYOUT_STAGE_EXECUTION_SCHEMA:
        raise ValueError("invalid layout stage execution schema")
    _require_self_digest(migration, "plan_sha256", "layout migration plan")
    _require_self_digest(stage, "stage_sha256", "layout stage execution")
    if stage.get("plan_sha256") != migration.get("plan_sha256"):
        raise ValueError("layout stage execution belongs to a different migration plan")
    staged_by_name = {str(row["name"]): row for row in stage.get("results", [])}
    if set(staged_by_name) != {str(row["name"]) for row in migration["items"]}:
        raise ValueError("layout stage result set does not match migration plan")

    items: list[dict[str, Any]] = []
    for item in migration["items"]:
        staged = staged_by_name[str(item["name"])]
        _verify_staged_result(item, staged)
        items.append(dict(item))
    payload = {
        "schema": LAYOUT_FINALIZE_PLAN_SCHEMA,
        "migration_plan_path": str(migration_path),
        "migration_plan_sha256": migration["plan_sha256"],
        "stage_execution_path": str(stage_path),
        "stage_sha256": stage["stage_sha256"],
        "item_count": len(items),
        "source_total_bytes": sum(int(row["source_bytes"]) for row in items),
        "items": items,
        "required_confirmation_phrase": LAYOUT_FINALIZE_CONFIRMATION,
    }
    payload["plan_sha256"] = value_digest(payload)
    return payload


def write_layout_finalize_plan(plan: Mapping[str, Any], path: Path) -> Path:
    return atomic_json_write(Path(path), dict(plan))


def execute_layout_finalize_plan(
    plan_path: Path,
    *,
    confirm_plan_sha256: str,
    confirm_delete: str,
) -> dict[str, Any]:
    path = Path(plan_path).resolve()
    plan = _read(path)
    if not isinstance(plan, Mapping) or plan.get("schema") != LAYOUT_FINALIZE_PLAN_SCHEMA:
        raise ValueError(f"layout finalize plan must use schema {LAYOUT_FINALIZE_PLAN_SCHEMA}")
    _require_self_digest(plan, "plan_sha256", "layout finalize plan")
    if confirm_plan_sha256 != plan.get("plan_sha256"):
        raise ValueError("layout finalize plan confirmation SHA-256 does not match")
    if confirm_delete != LAYOUT_FINALIZE_CONFIRMATION:
        raise ValueError(f"layout deletion confirmation must equal {LAYOUT_FINALIZE_CONFIRMATION}")

    migration = _read(Path(str(plan["migration_plan_path"])))
    stage = _read(Path(str(plan["stage_execution_path"])))
    _require_self_digest(migration, "plan_sha256", "layout migration plan")
    _require_self_digest(stage, "stage_sha256", "layout stage execution")
    staged_by_name = {str(row["name"]): row for row in stage["results"]}
    root = Path(str(migration["casimir_root"])).resolve()

    for item in plan["items"]:
        source = Path(str(item["source_path"])).absolute()
        if source.parent.resolve() != root:
            raise ValueError(f"layout finalize source is not a direct root entry: {source}")
        if source.name != item["name"]:
            raise ValueError(f"layout finalize source name mismatch: {source}")
        _verify_staged_result(item, staged_by_name[str(item["name"])])

    results: list[dict[str, Any]] = []
    for item in plan["items"]:
        source = Path(str(item["source_path"])).absolute()
        if source.is_dir() and not source.is_symlink():
            shutil.rmtree(source)
        elif source.is_file() and not source.is_symlink():
            source.unlink()
        else:
            raise ValueError(f"layout finalize source has unsupported type: {source}")
        results.append(
            {
                "name": item["name"],
                "status": "staged_source_removed",
                "source_path": str(source),
                "source_still_present": source.exists() or source.is_symlink(),
                "destination_path": item["destination_path"],
                "destination_preserved": Path(str(item["destination_path"])).is_file(),
                "manifest_preserved": Path(str(item["manifest_path"])).is_file(),
            }
        )
    report = {
        "schema": LAYOUT_FINALIZE_EXECUTION_SCHEMA,
        "plan_path": str(path),
        "plan_sha256": plan["plan_sha256"],
        "removed_entry_count": len(results),
        "released_bytes": plan["source_total_bytes"],
        "results": results,
        "all_sources_removed": all(not row["source_still_present"] for row in results),
        "all_destinations_preserved": all(
            row["destination_preserved"] and row["manifest_preserved"] for row in results
        ),
    }
    report["execution_sha256"] = value_digest(report)
    return report


def write_layout_finalize_execution(report: Mapping[str, Any], path: Path) -> Path:
    return atomic_json_write(Path(path), dict(report))


__all__ = [
    "LAYOUT_FINALIZE_CONFIRMATION",
    "LAYOUT_FINALIZE_EXECUTION_SCHEMA",
    "LAYOUT_FINALIZE_PLAN_SCHEMA",
    "LAYOUT_MIGRATION_PLAN_SCHEMA",
    "LAYOUT_STAGE_EXECUTION_SCHEMA",
    "build_layout_finalize_plan",
    "build_layout_migration_plan",
    "execute_layout_finalize_plan",
    "stage_layout_migration",
    "write_layout_finalize_execution",
    "write_layout_finalize_plan",
    "write_layout_migration_plan",
    "write_layout_stage_execution",
]
