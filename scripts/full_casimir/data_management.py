from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

CATALOG_SCHEMA = "casimir-data-catalog-v1"
REGISTRY_SCHEMA = "casimir-data-registry-v1"
PLAN_SCHEMA = "casimir-data-archive-plan-v1"
ARCHIVE_PLAN_SCHEMA = PLAN_SCHEMA
ARCHIVE_SCHEMA = "casimir-data-archive-manifest-v1"
EXECUTION_SCHEMA = "casimir-data-archive-execution-v1"

_REQUIRED = ("config.json", "manifest.json", "result.json", "summary.json")
_LIFECYCLE = {
    "unclassified", "active", "frozen_evidence", "superseded",
    "legacy_exploratory", "abandoned", "scratch", "archived",
}
_RETENTION = {"review", "keep_hot", "keep_cold", "archive"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _write(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _digest(payload: Any) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find(obj: Any, names: Sequence[str]) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            value = obj.get(name)
            if value is not None and not isinstance(value, (Mapping, list, tuple)):
                return value
        for value in obj.values():
            found = _find(value, names)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find(value, names)
            if found is not None:
                return found
    return None


def _sequence(obj: Any, name: str) -> list[Any]:
    if isinstance(obj, Mapping):
        value = obj.get(name)
        if isinstance(value, (list, tuple)):
            return list(value)
        for child in obj.values():
            found = _sequence(child, name)
            if found:
                return found
    elif isinstance(obj, (list, tuple)):
        for child in obj:
            found = _sequence(child, name)
            if found:
                return found
    return []


def _optional_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        return _read(path), None
    except ValueError as exc:
        return None, str(exc)


def _registry(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    payload = _read(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"registry must use schema {REGISTRY_SCHEMA}")
    raw = payload.get("runs", {})
    if not isinstance(raw, Mapping):
        raise ValueError("registry runs must be a mapping")
    output: dict[str, dict[str, str]] = {}
    for name, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"registry entry {name!r} must be a mapping")
        lifecycle = str(value.get("lifecycle_state", "unclassified"))
        retention = str(value.get("retention_action", "review"))
        if lifecycle not in _LIFECYCLE:
            raise ValueError(f"unknown lifecycle_state for {name}: {lifecycle}")
        if retention not in _RETENTION:
            raise ValueError(f"unknown retention_action for {name}: {retention}")
        output[str(name)] = {
            "lifecycle_state": lifecycle,
            "retention_action": retention,
            "note": str(value.get("note", "")),
        }
    return output


def _scientific(
    missing: Sequence[str], errors: Mapping[str, str], result: Mapping[str, Any]
) -> str:
    if missing:
        return "incomplete"
    if errors:
        return "corrupt"
    flags = (
        _find(result, ("production_casimir_allowed",)),
        _find(result, ("all_microscopic_nodes_certified",)),
        _find(result, ("outer_tail_estimated",)),
        _find(result, ("matsubara_tail_estimated",)),
    )
    if all(value is True for value in flags):
        return "certified"
    status = str(result.get("status", "unknown"))
    return "unresolved" if status == "unresolved" else "diagnostic_complete" if status in {
        "adaptive_tail_bounded", "adaptive_finite_partial"
    } else "unknown"


def _run_record(run: Path, root: Path, registry: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    files = [path for path in run.rglob("*") if path.is_file()]
    payloads: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    schemas: dict[str, Any] = {}
    errors: dict[str, str] = {}
    missing: list[str] = []
    for name in _REQUIRED:
        path = run / name
        if not path.is_file():
            missing.append(name)
            continue
        hashes[name] = _sha(path)
        payload, error = _optional_json(path)
        if error:
            errors[name] = error
        else:
            payloads[name] = payload
            schemas[name] = payload.get("schema") if isinstance(payload, Mapping) else None

    config = payloads.get("config.json")
    manifest = payloads.get("manifest.json")
    result = payloads.get("result.json")
    summary = payloads.get("summary.json")
    config = config if isinstance(config, Mapping) else {}
    manifest = manifest if isinstance(manifest, Mapping) else {}
    result = result if isinstance(result, Mapping) else {}
    summary = summary if isinstance(summary, Mapping) else {}

    cache_path = run / "cache" / "certified_points.json"
    cache, cache_error = _optional_json(cache_path)
    if cache_error:
        errors["cache/certified_points.json"] = cache_error
    entries = cache.get("entries", []) if isinstance(cache, Mapping) else []
    entries = entries if isinstance(entries, list) else []
    status_counts: Counter[str] = Counter()
    for entry in entries:
        point = entry.get("point_result") if isinstance(entry, Mapping) else None
        sweet = point.get("sweet_spot") if isinstance(point, Mapping) else None
        status_counts[
            str(sweet.get("status", "<missing>"))
            if isinstance(sweet, Mapping)
            else "<missing>"
        ] += 1

    separation_nm = _find(config, ("separation_nm", "distance_nm"))
    if separation_nm is None:
        try:
            separation_nm = float(_find(config, ("separation_m", "distance_m"))) * 1e9
        except (TypeError, ValueError):
            separation_nm = None

    lifecycle = registry.get(run.name, {
        "lifecycle_state": "unclassified",
        "retention_action": "review",
        "note": "",
    })
    newest = max((path.stat().st_mtime for path in files), default=run.stat().st_mtime)
    return {
        "run_name": run.name,
        "relative_path": run.relative_to(root).as_posix(),
        "total_bytes": sum(path.stat().st_size for path in files),
        "file_count": len(files),
        "newest_modified_utc": datetime.fromtimestamp(newest, timezone.utc).isoformat(),
        "scientific_state": _scientific(missing, errors, result),
        "lifecycle_state": lifecycle["lifecycle_state"],
        "retention_action": lifecycle["retention_action"],
        "registry_note": lifecycle["note"],
        "missing_required_artifacts": missing,
        "parse_errors": errors,
        "artifact_hashes": hashes,
        "artifact_schemas": schemas,
        "physics_identity": {
            "pairings": _sequence(config, "pairings"),
            "temperature_K": _find(config, ("temperature_K",)),
            "separation_nm": separation_nm,
            "plate_angles_deg": _sequence(config, "plate_angles_deg"),
        },
        "numerical_policy": {
            "logdet_rtol": _find(config, ("logdet_rtol",)),
            "logdet_atol": _find(config, ("logdet_atol",)),
            "required_consecutive_passes": _find(config, ("required_consecutive_passes",)),
            "N_candidates": _sequence(config, "N_candidates"),
            "radial_budget_fraction": _find(config, ("radial_budget_fraction",)),
            "angular_budget_fraction": _find(config, ("angular_budget_fraction",)),
        },
        "result_state": {
            "status": result.get("status"),
            "summary_status": summary.get("status"),
            "termination_reason": result.get("termination_reason"),
            "production_casimir_allowed": _find(result, ("production_casimir_allowed",)),
            "all_microscopic_nodes_certified": _find(
                result, ("all_microscopic_nodes_certified",)
            ),
            "outer_tail_estimated": _find(result, ("outer_tail_estimated",)),
            "matsubara_tail_estimated": _find(result, ("matsubara_tail_estimated",)),
        },
        "cache": {
            "present": cache_path.is_file(),
            "bytes": cache_path.stat().st_size if cache_path.is_file() else 0,
            "sha256": _sha(cache_path) if cache_path.is_file() else None,
            "schema": cache.get("schema") if isinstance(cache, Mapping) else None,
            "entry_count": len(entries),
            "status_counts": dict(status_counts),
        },
        "manifest_commit": _find(
            manifest, ("git_commit", "commit_sha", "git_sha", "code_commit")
        ),
    }


def build_data_catalog(casimir_root: Path, registry_path: Path | None = None) -> dict[str, Any]:
    root = Path(casimir_root).resolve()
    registry = _registry(registry_path)
    runs_root = root / "runs"
    runs = [
        _run_record(path, root, registry)
        for path in sorted(runs_root.iterdir())
        if path.is_dir()
    ] if runs_root.is_dir() else []
    global_artifacts = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in {"runs", "catalog", "archive"}:
            continue
        top = relative.parts[0] if relative.parts else ""
        kind = "derived_report" if top == "reports" else "derived_diagnostics" if top == "diagnostics" else "runtime_log" if "log" in top.lower() else "snapshot_archive" if path.name.endswith((".tar.gz", ".tgz", ".tar.zst")) else "layout_document" if path.name == "README.md" else "unclassified"
        global_artifacts.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
            "artifact_class": kind,
        })
    payload = {
        "schema": CATALOG_SCHEMA,
        "created_at_utc": _now(),
        "casimir_root": str(root),
        "registry_path": str(registry_path.resolve()) if registry_path else None,
        "run_count": len(runs),
        "total_run_bytes": sum(row["total_bytes"] for row in runs),
        "global_artifact_count": len(global_artifacts),
        "total_global_artifact_bytes": sum(row["bytes"] for row in global_artifacts),
        "scientific_state_counts": dict(Counter(row["scientific_state"] for row in runs)),
        "lifecycle_state_counts": dict(Counter(row["lifecycle_state"] for row in runs)),
        "retention_action_counts": dict(Counter(row["retention_action"] for row in runs)),
        "runs": runs,
        "global_artifacts": global_artifacts,
    }
    payload["catalog_sha256"] = _digest(payload)
    return payload


def write_data_catalog(catalog: Mapping[str, Any], catalog_root: Path) -> tuple[Path, Path]:
    root = Path(catalog_root)
    json_path = _write(root / "run_catalog.json", dict(catalog))
    tsv_path = root / "run_catalog.tsv"
    fields = (
        "run_name", "scientific_state", "lifecycle_state", "retention_action",
        "total_bytes", "pairings", "plate_angles_deg", "result_status",
        "termination_reason", "cache_entries", "cache_established",
        "cache_not_established", "manifest_commit", "registry_note",
    )
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fields)
        for row in catalog.get("runs", []):
            counts = row["cache"]["status_counts"]
            writer.writerow([
                row["run_name"], row["scientific_state"], row["lifecycle_state"],
                row["retention_action"], row["total_bytes"],
                json.dumps(row["physics_identity"]["pairings"]),
                json.dumps(row["physics_identity"]["plate_angles_deg"]),
                row["result_state"]["status"], row["result_state"]["termination_reason"],
                row["cache"]["entry_count"], counts.get("established", 0),
                sum(value for key, value in counts.items() if key != "established"),
                row["manifest_commit"], row["registry_note"],
            ])
    return json_path, tsv_path


def write_registry_template(catalog: Mapping[str, Any], path: Path) -> Path:
    return _write(Path(path), {
        "schema": REGISTRY_SCHEMA,
        "created_at_utc": _now(),
        "instructions": (
            "Set lifecycle_state and retention_action explicitly. Archive creates a "
            "verified compressed copy and never removes the source."
        ),
        "runs": {
            row["run_name"]: {
                "lifecycle_state": row["lifecycle_state"],
                "retention_action": row["retention_action"],
                "note": row.get("registry_note", ""),
            }
            for row in catalog.get("runs", [])
        },
    })


def _tree(root: Path) -> list[dict[str, Any]]:
    return [{
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "mode": path.stat().st_mode & 0o777,
    } for path in sorted(value for value in root.rglob("*") if value.is_file())]


def build_archive_plan(catalog: Mapping[str, Any], archive_root: Path) -> dict[str, Any]:
    root = Path(str(catalog["casimir_root"])).resolve()
    archive = Path(archive_root).resolve()
    items = []
    for row in catalog.get("runs", []):
        if row.get("retention_action") != "archive":
            continue
        source = (root / row["relative_path"]).resolve()
        source.relative_to(root)
        if not source.is_dir():
            raise ValueError(f"archive source is missing: {source}")
        files = _tree(source)
        destination = archive / "runs" / f"{source.name}.tar.gz"
        items.append({
            "run_name": source.name,
            "source_path": str(source),
            "source_tree_digest": _digest(files),
            "source_file_count": len(files),
            "source_total_bytes": sum(item["bytes"] for item in files),
            "archive_path": str(destination),
            "archive_manifest_path": str(destination) + ".manifest.json",
            "source_removal_authorized": False,
        })
    payload = {
        "schema": PLAN_SCHEMA,
        "created_at_utc": _now(),
        "casimir_root": str(root),
        "archive_root": str(archive),
        "item_count": len(items),
        "items": items,
        "safety": {
            "source_removal_supported": False,
            "source_directories_remain_untouched": True,
        },
    }
    payload["plan_sha256"] = _digest(payload)
    return payload


def write_archive_plan(plan: Mapping[str, Any], path: Path) -> Path:
    return _write(Path(path), dict(plan))


def _archive(item: Mapping[str, Any], root: Path) -> dict[str, Any]:
    source = Path(str(item["source_path"])).resolve()
    source.relative_to(root)
    files = _tree(source)
    if _digest(files) != item["source_tree_digest"]:
        raise ValueError(f"source tree changed since planning: {source.name}")
    destination = Path(str(item["archive_path"])).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tarfile.open(tmp, "w:gz", compresslevel=6) as handle:
        handle.add(source, arcname=source.name, recursive=True)
    tmp.replace(destination)
    with tarfile.open(destination, "r:gz") as handle:
        members = sorted(member.name for member in handle.getmembers() if member.isfile())
    expected = sorted(f"{source.name}/{row['path']}" for row in files)
    if members != expected:
        raise RuntimeError(f"archive verification failed: {source.name}")
    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "created_at_utc": _now(),
        "run_name": source.name,
        "source_tree_digest": item["source_tree_digest"],
        "source_files": files,
        "archive_path": str(destination),
        "archive_bytes": destination.stat().st_size,
        "archive_sha256": _sha(destination),
        "source_removed": False,
    }
    manifest_path = Path(str(item["archive_manifest_path"]))
    _write(manifest_path, manifest)
    return {
        "run_name": source.name,
        "status": "archived_and_verified",
        "archive_path": str(destination),
        "archive_manifest_path": str(manifest_path),
        "archive_bytes": manifest["archive_bytes"],
        "archive_sha256": manifest["archive_sha256"],
        "source_path": str(source),
        "source_still_present": source.is_dir(),
    }


def execute_archive_plan(
    plan_path: Path,
    confirm_plan_sha256: str,
    selected_runs: Sequence[str] = (),
) -> dict[str, Any]:
    plan = _read(Path(plan_path))
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"archive plan must use schema {PLAN_SCHEMA}")
    if confirm_plan_sha256 != plan.get("plan_sha256"):
        raise ValueError("archive plan confirmation SHA-256 does not match")
    selected = set(selected_runs)
    known = {item["run_name"] for item in plan.get("items", [])}
    if selected - known:
        raise ValueError(f"unknown selected runs: {sorted(selected - known)}")
    root = Path(str(plan["casimir_root"])).resolve()
    results = [
        _archive(item, root)
        for item in plan.get("items", [])
        if not selected or item["run_name"] in selected
    ]
    return {
        "schema": EXECUTION_SCHEMA,
        "created_at_utc": _now(),
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": plan["plan_sha256"],
        "selected_runs": sorted(selected),
        "results": results,
        "source_removal_performed": False,
    }


def write_archive_execution(report: Mapping[str, Any], path: Path) -> Path:
    return _write(Path(path), dict(report))


__all__ = [
    "CATALOG_SCHEMA", "REGISTRY_SCHEMA", "PLAN_SCHEMA", "ARCHIVE_PLAN_SCHEMA",
    "build_archive_plan", "build_data_catalog", "execute_archive_plan",
    "write_archive_execution", "write_archive_plan", "write_data_catalog",
    "write_registry_template",
]
