"""Matched evidence preflight for TODO 4 legacy qualification."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from lno327.casimir.material_geometry_batch import preflight_geometry_batch
from lno327.casimir.material_geometry_qualification_campaign import (
    Todo4QualificationCampaign,
)
from lno327.casimir.material_geometry_qualification_io import (
    atomic_write_json,
    source_commit,
)
from lno327.casimir.material_response_cache_store import MaterialResponseCacheStore


def legacy_compatibility_payload(
    campaign: Todo4QualificationCampaign,
    *,
    cache_root: Path,
) -> dict[str, Any]:
    cache = MaterialResponseCacheStore(cache_root, mode="read_only")
    artifacts: dict[str, object] = {}
    missing: set[str] = set()
    for entry in campaign.entries:
        preflight = preflight_geometry_batch(entry.geometry_plan, cache=cache)
        artifacts.update(preflight.artifacts)
        missing.update(preflight.misses)

    evidence = [
        {
            "identity_sha256": key,
            "working_N": artifact.working_N,
            "audit_N": artifact.audit_N,
            "primary_shift": artifact.primary_shift,
            "establishment_mode": artifact.establishment_mode,
        }
        for key, artifact in sorted(artifacts.items())
    ]
    records: list[dict[str, Any]] = []
    incompatible = 0
    pending = 0
    for entry in campaign.entries:
        for point in entry.geometry_plan.points:
            first = artifacts.get(point.plate_1_requirement)
            second = artifacts.get(point.plate_2_requirement)
            if first is None or second is None:
                status = "pending_cache_miss"
                compatible = False
                pending += 1
                first_N = None
                second_N = None
                first_shift = None
                second_shift = None
            else:
                first_N = int(first.working_N)
                second_N = int(second.working_N)
                first_shift = str(first.primary_shift)
                second_shift = str(second.primary_shift)
                compatible = bool(
                    first_N == second_N and first_shift == second_shift
                )
                status = "compatible" if compatible else "incompatible"
                if not compatible:
                    incompatible += 1
            records.append(
                {
                    "plan_id": entry.plan_id,
                    "point_id": point.point_id,
                    "plate_1_requirement": point.plate_1_requirement,
                    "plate_2_requirement": point.plate_2_requirement,
                    "plate_1_working_N": first_N,
                    "plate_2_working_N": second_N,
                    "plate_1_primary_shift": first_shift,
                    "plate_2_primary_shift": second_shift,
                    "status": status,
                    "compatible": compatible,
                }
            )

    cache_complete = not missing
    ready = bool(cache_complete and incompatible == 0 and pending == 0)
    return {
        "schema": "todo4-legacy-evidence-compatibility-v1",
        "campaign_id": campaign.campaign_id,
        "manifest_sha256": campaign.manifest_sha256,
        "source_commit": source_commit(),
        "cache_root": str(cache_root),
        "artifact_evidence": evidence,
        "records": records,
        "summary": {
            "unique_artifact_count": len(artifacts),
            "missing_identity_count": len(missing),
            "legacy_pair_count": len(records),
            "incompatible_pair_count": incompatible,
            "pending_pair_count": pending,
            "qualification_ready": ready,
        },
        "contract": {
            "same_working_N_required": True,
            "same_primary_shift_required": True,
            "common_N_or_shift_search_performed": False,
            "microscopic_fallback_attempted": False,
            "diagnostic_only": True,
            "production_casimir_allowed": False,
        },
    }


def write_legacy_compatibility(
    campaign: Todo4QualificationCampaign,
    *,
    output_dir: Path,
    cache_root: Path,
    require_ready: bool,
) -> dict[str, Any]:
    payload = legacy_compatibility_payload(campaign, cache_root=cache_root)
    atomic_write_json(
        Path(output_dir) / "legacy_compatibility.json",
        payload,
    )
    if require_ready and not payload["summary"]["qualification_ready"]:
        raise RuntimeError(
            "legacy qualification evidence is incompatible: both plates must "
            "share working N and exact primary shift before replay"
        )
    return payload


__all__ = [
    "legacy_compatibility_payload",
    "write_legacy_compatibility",
]
