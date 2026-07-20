from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .data_management import _digest, _read, _sha, _write
from .data_retention import REPORT_PACK_SCHEMA, _unpack_value

REPORT_PRUNE_PLAN_SCHEMA = "casimir-data-report-prune-plan-v1"
REPORT_PRUNE_EXECUTION_SCHEMA = "casimir-data-report-prune-execution-v1"
REPORT_PRUNE_CONFIRMATION = "DELETE_RECONSTRUCTABLE_PACKED_REPORT_SOURCE"


def _verified_pack(manifest_path: Path) -> tuple[dict[str, Any], Any]:
    path = Path(manifest_path).resolve()
    manifest = _read(path)
    if not isinstance(manifest, Mapping) or manifest.get("schema") != REPORT_PACK_SCHEMA:
        raise ValueError(f"report pack manifest must use schema {REPORT_PACK_SCHEMA}")

    payload = dict(manifest)
    expected_pack_sha = str(payload.pop("pack_sha256", ""))
    if not expected_pack_sha or _digest(payload) != expected_pack_sha:
        raise ValueError("report pack manifest digest mismatch")
    if manifest.get("reconstruction_verified") is not True:
        raise ValueError("report pack was not marked reconstruction-verified")

    compact = Path(str(manifest["compact_report"])).resolve()
    if not compact.is_file():
        raise FileNotFoundError(f"compact report is missing: {compact}")
    if compact.stat().st_size != int(manifest["compact_bytes"]):
        raise ValueError("compact report size mismatch")
    if _sha(compact) != manifest.get("compact_sha256"):
        raise ValueError("compact report SHA-256 mismatch")

    sidecars = manifest.get("sidecars", [])
    if not isinstance(sidecars, list):
        raise ValueError("report pack sidecars must be a list")
    if len(sidecars) != int(manifest.get("sidecar_count", -1)):
        raise ValueError("report pack sidecar count mismatch")

    compressed_total = 0
    for row in sidecars:
        if not isinstance(row, Mapping):
            raise ValueError("report pack sidecar entry must be a mapping")
        sidecar = Path(str(row["sidecar_path"])).resolve()
        if not sidecar.is_file():
            raise FileNotFoundError(f"report sidecar is missing: {sidecar}")
        if sidecar.stat().st_size != int(row["compressed_bytes"]):
            raise ValueError(f"report sidecar size mismatch: {sidecar}")
        if _sha(sidecar) != row.get("sidecar_sha256"):
            raise ValueError(f"report sidecar SHA-256 mismatch: {sidecar}")
        compressed_total += sidecar.stat().st_size
    if compressed_total != int(manifest.get("sidecar_total_bytes", -1)):
        raise ValueError("report pack total sidecar byte count mismatch")

    compact_value = _read(compact)
    restored = _unpack_value(compact_value)
    if _digest(restored) != manifest.get("source_value_sha256"):
        raise RuntimeError("packed report no longer reconstructs the source value")
    return dict(manifest), restored


def build_report_prune_plan(manifest_path: Path) -> dict[str, Any]:
    manifest, _ = _verified_pack(manifest_path)
    source = Path(str(manifest["source_report"])).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source report is missing: {source}")
    if source.stat().st_size != int(manifest["source_bytes"]):
        raise ValueError("source report size changed after packing")
    if _sha(source) != manifest.get("source_sha256"):
        raise ValueError("source report SHA-256 changed after packing")
    source_value = _read(source)
    if _digest(source_value) != manifest.get("source_value_sha256"):
        raise ValueError("source report value changed after packing")

    payload = {
        "schema": REPORT_PRUNE_PLAN_SCHEMA,
        "manifest_path": str(Path(manifest_path).resolve()),
        "pack_sha256": manifest["pack_sha256"],
        "source_report": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": manifest["source_sha256"],
        "source_value_sha256": manifest["source_value_sha256"],
        "compact_report": manifest["compact_report"],
        "compact_sha256": manifest["compact_sha256"],
        "sidecar_count": manifest["sidecar_count"],
        "sidecar_total_bytes": manifest["sidecar_total_bytes"],
        "reconstruction_verified": True,
        "required_confirmation_phrase": REPORT_PRUNE_CONFIRMATION,
    }
    payload["plan_sha256"] = _digest(payload)
    return payload


def write_report_prune_plan(plan: Mapping[str, Any], path: Path) -> Path:
    return _write(Path(path), dict(plan))


def execute_report_prune_plan(
    plan_path: Path,
    *,
    confirm_plan_sha256: str,
    confirm_delete: str,
) -> dict[str, Any]:
    path = Path(plan_path).resolve()
    plan = _read(path)
    if not isinstance(plan, Mapping) or plan.get("schema") != REPORT_PRUNE_PLAN_SCHEMA:
        raise ValueError(f"report prune plan must use schema {REPORT_PRUNE_PLAN_SCHEMA}")
    if confirm_plan_sha256 != plan.get("plan_sha256"):
        raise ValueError("report prune plan confirmation SHA-256 does not match")
    if confirm_delete != REPORT_PRUNE_CONFIRMATION:
        raise ValueError(
            f"report deletion confirmation must equal {REPORT_PRUNE_CONFIRMATION}"
        )

    manifest, _ = _verified_pack(Path(str(plan["manifest_path"])))
    if manifest.get("pack_sha256") != plan.get("pack_sha256"):
        raise ValueError("report pack changed since prune planning")
    if manifest.get("compact_sha256") != plan.get("compact_sha256"):
        raise ValueError("compact report changed since prune planning")

    source = Path(str(plan["source_report"])).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source report is missing: {source}")
    if source.stat().st_size != int(plan["source_bytes"]):
        raise ValueError("source report size changed since prune planning")
    if _sha(source) != plan.get("source_sha256"):
        raise ValueError("source report SHA-256 changed since prune planning")
    if _digest(_read(source)) != plan.get("source_value_sha256"):
        raise ValueError("source report value changed since prune planning")

    source.unlink()
    return {
        "schema": REPORT_PRUNE_EXECUTION_SCHEMA,
        "plan_path": str(path),
        "plan_sha256": plan["plan_sha256"],
        "source_report": str(source),
        "source_removed": not source.exists(),
        "released_bytes": plan["source_bytes"],
        "compact_report": plan["compact_report"],
        "compact_report_preserved": Path(str(plan["compact_report"])).is_file(),
        "sidecar_count": plan["sidecar_count"],
        "sidecar_total_bytes": plan["sidecar_total_bytes"],
        "pack_reconstruction_verified_before_removal": True,
    }


def write_report_prune_execution(report: Mapping[str, Any], path: Path) -> Path:
    return _write(Path(path), dict(report))


__all__ = [
    "REPORT_PRUNE_CONFIRMATION",
    "REPORT_PRUNE_EXECUTION_SCHEMA",
    "REPORT_PRUNE_PLAN_SCHEMA",
    "build_report_prune_plan",
    "execute_report_prune_plan",
    "write_report_prune_execution",
    "write_report_prune_plan",
]
