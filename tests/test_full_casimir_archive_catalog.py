from __future__ import annotations

import json
from pathlib import Path

from scripts.full_casimir.archive_catalog import augment_catalog_with_archives
from scripts.full_casimir.data_management import _sha


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cold_archive_remains_visible_after_source_removal(tmp_path: Path) -> None:
    root = tmp_path / "casimir"
    run_name = "spm_legacy"
    source = root / "runs" / run_name
    source.mkdir(parents=True)
    archive = root / "archive" / "runs" / f"{run_name}.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"archive")
    manifest = archive.with_name(archive.name + ".manifest.json")
    _write(
        manifest,
        {
            "schema": "casimir-data-archive-manifest-v1",
            "run_name": run_name,
            "archive_path": str(archive.resolve()),
            "archive_sha256": _sha(archive),
            "source_tree_digest": "tree",
            "source_files": [{"path": "result.json"}],
        },
    )
    registry = root / "catalog" / "registry.json"
    _write(
        registry,
        {
            "schema": "casimir-data-registry-v1",
            "runs": {
                run_name: {
                    "lifecycle_state": "legacy_exploratory",
                    "retention_action": "archive",
                    "note": "cold",
                }
            },
        },
    )
    base = {
        "schema": "casimir-data-catalog-v1",
        "casimir_root": str(root.resolve()),
        "registry_path": str(registry.resolve()),
        "run_count": 1,
        "runs": [],
        "catalog_sha256": "old",
    }
    hot = augment_catalog_with_archives(base)
    assert hot["archived_run_count"] == 1
    assert hot["archived_runs"][0]["source_present"] is True
    assert hot["archive_source_presence_counts"] == {"source_present": 1}

    source.rmdir()
    cold = augment_catalog_with_archives(base)
    assert cold["archived_runs"][0]["source_present"] is False
    assert cold["archive_source_presence_counts"] == {"archive_only": 1}
