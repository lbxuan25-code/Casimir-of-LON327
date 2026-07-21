from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .data_management import ARCHIVE_SCHEMA, REGISTRY_SCHEMA, _digest, _read, _sha


def _registry_rows(path: str | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    registry_path = Path(path)
    if not registry_path.is_file():
        return {}
    payload = _read(registry_path)
    if not isinstance(payload, Mapping) or payload.get("schema") != REGISTRY_SCHEMA:
        return {}
    rows = payload.get("runs", {})
    return rows if isinstance(rows, Mapping) else {}


def augment_catalog_with_archives(catalog: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(catalog["casimir_root"])).resolve()
    registry = _registry_rows(catalog.get("registry_path"))
    archive_root = root / "archive" / "runs"
    records = []
    if archive_root.is_dir():
        for manifest_path in sorted(archive_root.glob("*.tar.gz.manifest.json")):
            try:
                manifest = _read(manifest_path)
            except ValueError as exc:
                records.append(
                    {
                        "run_name": manifest_path.name.removesuffix(
                            ".tar.gz.manifest.json"
                        ),
                        "manifest_path": str(manifest_path),
                        "parse_error": str(exc),
                        "archive_present": False,
                    }
                )
                continue
            if not isinstance(manifest, Mapping) or manifest.get("schema") != ARCHIVE_SCHEMA:
                records.append(
                    {
                        "run_name": manifest_path.name.removesuffix(
                            ".tar.gz.manifest.json"
                        ),
                        "manifest_path": str(manifest_path),
                        "parse_error": "invalid archive manifest schema",
                        "archive_present": False,
                    }
                )
                continue
            run_name = str(manifest.get("run_name"))
            archive = Path(str(manifest.get("archive_path"))).resolve()
            entry = registry.get(run_name, {})
            entry = entry if isinstance(entry, Mapping) else {}
            records.append(
                {
                    "run_name": run_name,
                    "archive_path": str(archive),
                    "archive_present": archive.is_file(),
                    "archive_bytes": archive.stat().st_size if archive.is_file() else None,
                    "archive_sha256": _sha(archive) if archive.is_file() else None,
                    "manifest_path": str(manifest_path.resolve()),
                    "manifest_sha256": _sha(manifest_path),
                    "source_tree_digest": manifest.get("source_tree_digest"),
                    "source_file_count": len(manifest.get("source_files", [])),
                    "source_present": (root / "runs" / run_name).is_dir(),
                    "lifecycle_state": str(
                        entry.get("lifecycle_state", "unclassified")
                    ),
                    "retention_action": str(
                        entry.get("retention_action", "review")
                    ),
                    "registry_note": str(entry.get("note", "")),
                    "parse_error": None,
                }
            )
    payload = dict(catalog)
    payload["archived_run_count"] = len(records)
    payload["total_archive_bytes"] = sum(
        int(row.get("archive_bytes") or 0) for row in records
    )
    payload["archive_source_presence_counts"] = dict(
        Counter(
            "source_present" if row.get("source_present") else "archive_only"
            for row in records
            if row.get("parse_error") is None
        )
    )
    payload["archived_runs"] = records
    payload.pop("catalog_sha256", None)
    payload["catalog_sha256"] = _digest(payload)
    return payload


__all__ = ["augment_catalog_with_archives"]
