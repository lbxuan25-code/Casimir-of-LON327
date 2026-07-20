from __future__ import annotations

import gzip
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .data_management import (
    ARCHIVE_SCHEMA,
    CATALOG_SCHEMA,
    _digest,
    _read,
    _sha,
    _tree,
    _write,
)

VERIFY_SCHEMA = "casimir-data-archive-verification-v1"
PRUNE_PLAN_SCHEMA = "casimir-data-prune-plan-v1"
PRUNE_EXECUTION_SCHEMA = "casimir-data-prune-execution-v1"
REPORT_PACK_SCHEMA = "casimir-data-report-pack-v1"
SIDECAR_REFERENCE_SCHEMA = "casimir-data-json-sidecar-reference-v1"
PRUNE_CONFIRMATION = "DELETE_VERIFIED_ARCHIVED_RUN_SOURCES"

_PROTECTED_LIFECYCLES = {"active", "frozen_evidence"}


def _selected_items(
    items: Sequence[Mapping[str, Any]],
    selected_runs: Sequence[str],
) -> list[Mapping[str, Any]]:
    selected = {str(value) for value in selected_runs}
    known = {str(item.get("run_name")) for item in items}
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown selected runs: {sorted(unknown)}")
    return [
        item
        for item in items
        if not selected or str(item.get("run_name")) in selected
    ]


def _safe_members(
    handle: tarfile.TarFile,
    *,
    run_name: str,
) -> list[tarfile.TarInfo]:
    output: list[tarfile.TarInfo] = []
    for member in handle.getmembers():
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe archive member: {member.name}")
        if not pure.parts or pure.parts[0] != run_name:
            raise ValueError(f"archive member lies outside run root: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"unsupported archive member type: {member.name}")
        if not (member.isdir() or member.isfile()):
            raise ValueError(f"unsupported archive member: {member.name}")
        output.append(member)
    return output


def _restore_archive(
    archive: Path,
    *,
    run_name: str,
    destination: Path,
) -> Path:
    with tarfile.open(archive, "r:gz") as handle:
        members = _safe_members(handle, run_name=run_name)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.resolve().relative_to(destination.resolve())
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(member.mode & 0o777)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            target.chmod(member.mode & 0o777)
    restored = destination / run_name
    if not restored.is_dir():
        raise RuntimeError(f"restored run root is missing: {run_name}")
    return restored


def verify_archive_plan(
    plan_path: Path,
    *,
    selected_runs: Sequence[str] = (),
) -> dict[str, Any]:
    plan = _read(Path(plan_path))
    if not isinstance(plan, Mapping) or plan.get("schema") != "casimir-data-archive-plan-v1":
        raise ValueError("invalid archive plan schema")
    items = _selected_items(plan.get("items", []), selected_runs)
    results = []
    with tempfile.TemporaryDirectory(prefix="lno327_archive_restore_") as temporary:
        root = Path(temporary)
        for index, item in enumerate(items):
            run_name = str(item["run_name"])
            archive = Path(str(item["archive_path"])).resolve()
            manifest_path = Path(str(item["archive_manifest_path"])).resolve()
            if not archive.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"archive or manifest is missing for {run_name}")
            manifest = _read(manifest_path)
            if not isinstance(manifest, Mapping) or manifest.get("schema") != ARCHIVE_SCHEMA:
                raise ValueError(f"invalid archive manifest for {run_name}")
            archive_sha = _sha(archive)
            if archive_sha != manifest.get("archive_sha256"):
                raise ValueError(f"archive SHA-256 mismatch for {run_name}")
            if manifest.get("source_tree_digest") != item.get("source_tree_digest"):
                raise ValueError(f"archive/source plan mismatch for {run_name}")

            destination = root / f"restore_{index:04d}"
            destination.mkdir(parents=True)
            restored = _restore_archive(
                archive,
                run_name=run_name,
                destination=destination,
            )
            restored_files = _tree(restored)
            expected_files = list(manifest.get("source_files", []))
            if restored_files != expected_files:
                raise RuntimeError(f"restored tree differs from manifest: {run_name}")
            restored_digest = _digest(restored_files)
            if restored_digest != item.get("source_tree_digest"):
                raise RuntimeError(f"restored tree digest mismatch: {run_name}")
            results.append(
                {
                    "run_name": run_name,
                    "status": "restored_and_verified",
                    "archive_path": str(archive),
                    "archive_bytes": archive.stat().st_size,
                    "archive_sha256": archive_sha,
                    "archive_manifest_path": str(manifest_path),
                    "source_tree_digest": restored_digest,
                    "restored_file_count": len(restored_files),
                }
            )
    payload = {
        "schema": VERIFY_SCHEMA,
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": plan["plan_sha256"],
        "selected_runs": sorted(str(value) for value in selected_runs),
        "result_count": len(results),
        "results": results,
        "all_archives_restored_and_verified": True,
    }
    payload["verification_sha256"] = _digest(payload)
    return payload


def write_archive_verification(report: Mapping[str, Any], path: Path) -> Path:
    return _write(Path(path), dict(report))


def build_prune_plan(
    catalog: Mapping[str, Any],
    verification: Mapping[str, Any],
    *,
    selected_runs: Sequence[str],
) -> dict[str, Any]:
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"catalog must use schema {CATALOG_SCHEMA}")
    if verification.get("schema") != VERIFY_SCHEMA:
        raise ValueError(f"verification must use schema {VERIFY_SCHEMA}")
    if verification.get("all_archives_restored_and_verified") is not True:
        raise ValueError("archive verification is incomplete")
    selected = {str(value) for value in selected_runs}
    if not selected:
        raise ValueError("prune planning requires explicit --run selections")

    rows = {str(row["run_name"]): row for row in catalog.get("runs", [])}
    verified = {str(row["run_name"]): row for row in verification.get("results", [])}
    unknown = selected - rows.keys()
    if unknown:
        raise ValueError(f"unknown selected runs: {sorted(unknown)}")
    unverified = selected - verified.keys()
    if unverified:
        raise ValueError(f"selected runs lack restored archive verification: {sorted(unverified)}")

    root = Path(str(catalog["casimir_root"])).resolve()
    items = []
    for run_name in sorted(selected):
        row = rows[run_name]
        lifecycle = str(row.get("lifecycle_state"))
        if lifecycle in _PROTECTED_LIFECYCLES:
            raise ValueError(
                f"protected lifecycle cannot be pruned: {run_name} ({lifecycle})"
            )
        if row.get("retention_action") != "archive":
            raise ValueError(f"run is not registered for archive retention: {run_name}")
        source = (root / str(row["relative_path"])).resolve()
        source.relative_to(root / "runs")
        if not source.is_dir():
            raise FileNotFoundError(f"source run is missing: {run_name}")
        files = _tree(source)
        verification_row = verified[run_name]
        source_digest = _digest(files)
        if source_digest != verification_row.get("source_tree_digest"):
            raise ValueError(f"source changed after archive verification: {run_name}")
        archive = Path(str(verification_row["archive_path"])).resolve()
        if _sha(archive) != verification_row.get("archive_sha256"):
            raise ValueError(f"archive changed after verification: {run_name}")
        items.append(
            {
                "run_name": run_name,
                "lifecycle_state": lifecycle,
                "source_path": str(source),
                "source_tree_digest": source_digest,
                "source_file_count": len(files),
                "source_total_bytes": sum(item["bytes"] for item in files),
                "archive_path": str(archive),
                "archive_sha256": verification_row["archive_sha256"],
                "archive_manifest_path": verification_row["archive_manifest_path"],
                "restored_archive_verified": True,
            }
        )
    payload = {
        "schema": PRUNE_PLAN_SCHEMA,
        "casimir_root": str(root),
        "catalog_sha256": catalog["catalog_sha256"],
        "verification_sha256": verification["verification_sha256"],
        "item_count": len(items),
        "source_total_bytes": sum(item["source_total_bytes"] for item in items),
        "items": items,
        "safety": {
            "protected_lifecycles": sorted(_PROTECTED_LIFECYCLES),
            "explicit_run_selection_required": True,
            "exact_plan_hash_required": True,
            "exact_confirmation_phrase": PRUNE_CONFIRMATION,
            "archive_restore_verification_required": True,
        },
    }
    payload["plan_sha256"] = _digest(payload)
    return payload


def write_prune_plan(plan: Mapping[str, Any], path: Path) -> Path:
    return _write(Path(path), dict(plan))


def execute_prune_plan(
    plan_path: Path,
    *,
    confirm_plan_sha256: str,
    confirm_delete: str,
) -> dict[str, Any]:
    plan = _read(Path(plan_path))
    if not isinstance(plan, Mapping) or plan.get("schema") != PRUNE_PLAN_SCHEMA:
        raise ValueError(f"prune plan must use schema {PRUNE_PLAN_SCHEMA}")
    if confirm_plan_sha256 != plan.get("plan_sha256"):
        raise ValueError("prune plan confirmation SHA-256 does not match")
    if confirm_delete != PRUNE_CONFIRMATION:
        raise ValueError(f"deletion confirmation must equal {PRUNE_CONFIRMATION}")

    root = Path(str(plan["casimir_root"])).resolve()
    prepared = []
    for item in plan.get("items", []):
        source = Path(str(item["source_path"])).resolve()
        source.relative_to(root / "runs")
        if not source.is_dir():
            raise FileNotFoundError(f"source run is missing: {item['run_name']}")
        files = _tree(source)
        if _digest(files) != item.get("source_tree_digest"):
            raise ValueError(f"source tree changed since prune planning: {item['run_name']}")
        archive = Path(str(item["archive_path"])).resolve()
        manifest_path = Path(str(item["archive_manifest_path"])).resolve()
        if not archive.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"verified archive is missing: {item['run_name']}")
        manifest = _read(manifest_path)
        if manifest.get("schema") != ARCHIVE_SCHEMA:
            raise ValueError(f"invalid archive manifest: {item['run_name']}")
        if _sha(archive) != item.get("archive_sha256"):
            raise ValueError(f"archive changed since prune planning: {item['run_name']}")
        if manifest.get("source_tree_digest") != item.get("source_tree_digest"):
            raise ValueError(f"manifest/source mismatch: {item['run_name']}")
        prepared.append((item, source))

    results = []
    for item, source in prepared:
        shutil.rmtree(source)
        results.append(
            {
                "run_name": item["run_name"],
                "status": "verified_source_removed",
                "source_path": str(source),
                "source_still_present": source.exists(),
                "released_bytes": item["source_total_bytes"],
                "archive_path": item["archive_path"],
                "archive_sha256": item["archive_sha256"],
            }
        )
    return {
        "schema": PRUNE_EXECUTION_SCHEMA,
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": plan["plan_sha256"],
        "removed_run_count": len(results),
        "released_bytes": sum(row["released_bytes"] for row in results),
        "results": results,
        "verified_archives_preserved": True,
    }


def write_prune_execution(report: Mapping[str, Any], path: Path) -> Path:
    return _write(Path(path), dict(report))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pointer_token(pointer: str) -> str:
    tail = pointer.rsplit("/", 1)[-1] if pointer else "root"
    clean = "".join(character if character.isalnum() else "_" for character in tail)
    return (clean or "list")[:48]


def _pack_value(
    value: Any,
    *,
    pointer: str,
    pack_root: Path,
    threshold_bytes: int,
    sidecars: list[dict[str, Any]],
) -> Any:
    if isinstance(value, list):
        raw = _canonical_bytes(value)
        if pointer and len(raw) >= threshold_bytes:
            digest = _digest(value)
            name = f"{len(sidecars):04d}_{_pointer_token(pointer)}_{digest[:12]}.json.gz"
            path = pack_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as output:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=output,
                    compresslevel=6,
                    mtime=0,
                ) as compressed:
                    compressed.write(raw)
            row = {
                "json_pointer": pointer,
                "sidecar_path": str(path),
                "item_count": len(value),
                "uncompressed_bytes": len(raw),
                "compressed_bytes": path.stat().st_size,
                "value_sha256": digest,
                "sidecar_sha256": _sha(path),
            }
            sidecars.append(row)
            return {
                "schema": SIDECAR_REFERENCE_SCHEMA,
                **row,
            }
        return [
            _pack_value(
                item,
                pointer=f"{pointer}/{index}",
                pack_root=pack_root,
                threshold_bytes=threshold_bytes,
                sidecars=sidecars,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _pack_value(
                child,
                pointer=f"{pointer}/{str(key).replace('~', '~0').replace('/', '~1')}",
                pack_root=pack_root,
                threshold_bytes=threshold_bytes,
                sidecars=sidecars,
            )
            for key, child in value.items()
        }
    return value


def _unpack_value(value: Any) -> Any:
    if isinstance(value, Mapping) and value.get("schema") == SIDECAR_REFERENCE_SCHEMA:
        path = Path(str(value["sidecar_path"]))
        if _sha(path) != value.get("sidecar_sha256"):
            raise ValueError(f"report sidecar SHA-256 mismatch: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            restored = json.load(handle)
        if _digest(restored) != value.get("value_sha256"):
            raise ValueError(f"report sidecar value mismatch: {path}")
        return restored
    if isinstance(value, Mapping):
        return {str(key): _unpack_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_unpack_value(item) for item in value]
    return value


def pack_json_report(
    report_path: Path,
    *,
    compact_path: Path,
    pack_root: Path,
    manifest_path: Path,
    threshold_bytes: int = 1_000_000,
) -> dict[str, Any]:
    if threshold_bytes <= 0:
        raise ValueError("threshold_bytes must be positive")
    source = Path(report_path).resolve()
    original = _read(source)
    sidecars: list[dict[str, Any]] = []
    compact = _pack_value(
        original,
        pointer="",
        pack_root=Path(pack_root).resolve(),
        threshold_bytes=int(threshold_bytes),
        sidecars=sidecars,
    )
    _write(Path(compact_path), compact)
    restored = _unpack_value(compact)
    if _digest(restored) != _digest(original):
        raise RuntimeError("packed report failed reconstruction verification")
    payload = {
        "schema": REPORT_PACK_SCHEMA,
        "source_report": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": _sha(source),
        "source_value_sha256": _digest(original),
        "compact_report": str(Path(compact_path).resolve()),
        "compact_bytes": Path(compact_path).stat().st_size,
        "compact_sha256": _sha(Path(compact_path)),
        "pack_root": str(Path(pack_root).resolve()),
        "threshold_bytes": int(threshold_bytes),
        "sidecar_count": len(sidecars),
        "sidecar_total_bytes": sum(row["compressed_bytes"] for row in sidecars),
        "sidecars": sidecars,
        "reconstruction_verified": True,
        "source_report_preserved": source.is_file(),
    }
    payload["pack_sha256"] = _digest(payload)
    _write(Path(manifest_path), payload)
    return payload


__all__ = [
    "PRUNE_CONFIRMATION",
    "PRUNE_EXECUTION_SCHEMA",
    "PRUNE_PLAN_SCHEMA",
    "REPORT_PACK_SCHEMA",
    "VERIFY_SCHEMA",
    "build_prune_plan",
    "execute_prune_plan",
    "pack_json_report",
    "verify_archive_plan",
    "write_archive_verification",
    "write_prune_execution",
    "write_prune_plan",
]
