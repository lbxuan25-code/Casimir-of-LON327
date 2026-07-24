"""Execution stages for the frozen TODO 4 diagnostic qualification campaign."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lno327.casimir.material_geometry_batch import (
    execute_geometry_batch,
    preflight_geometry_batch,
)
from lno327.casimir.material_geometry_legacy_replay import (
    run_matched_legacy_geometry_replay,
)
from lno327.casimir.material_geometry_outer_qualification import (
    qualify_fixed_outer_geometry_replay,
)
from lno327.casimir.material_geometry_qualification import (
    qualify_batch_point_against_scalar,
)
from lno327.casimir.material_geometry_qualification_campaign import (
    Todo4QualificationCampaign,
    Todo4QualificationPlanEntry,
    build_todo4_qualification_campaign,
    fixed_outer_equivalence_policy,
    geometry_equivalence_policy,
    load_todo4_qualification_manifest,
    qualification_plan_payload,
)
from lno327.casimir.material_geometry_qualification_io import (
    atomic_write_json,
    atomic_write_npz,
    jsonable,
    load_json,
    point_token,
    slug,
    source_commit,
    tracked_tree_clean,
    write_frozen_json,
)
from lno327.casimir.material_response_cache_store import MaterialResponseCacheStore
from lno327.casimir.material_response_cached_engine import (
    evaluate_material_response_ladder_cached,
)


def load_campaign(path: Path) -> Todo4QualificationCampaign:
    return build_todo4_qualification_campaign(
        load_todo4_qualification_manifest(path)
    )


def plan_path(output_dir: Path) -> Path:
    return Path(output_dir) / "qualification_plan.json"


def current_plan(campaign: Todo4QualificationCampaign) -> dict[str, Any]:
    return qualification_plan_payload(campaign, source_commit=source_commit())


def require_frozen_plan(
    campaign: Todo4QualificationCampaign,
    output_dir: Path,
) -> dict[str, Any]:
    path = plan_path(output_dir)
    if not path.is_file():
        raise RuntimeError(f"frozen plan is missing; run plan first: {path}")
    frozen = load_json(path)
    current = current_plan(campaign)
    if frozen.get("plan_sha256") != current.get("plan_sha256"):
        raise RuntimeError(
            "frozen plan differs from current manifest/source; do not continue"
        )
    if frozen.get("source_commit") != source_commit():
        raise RuntimeError("frozen plan source commit differs from current HEAD")
    return frozen


def freeze_plan(
    campaign: Todo4QualificationCampaign,
    *,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not tracked_tree_clean():
        raise RuntimeError("tracked source tree is dirty; commit changes before freezing")
    payload = current_plan(campaign)
    write_frozen_json(plan_path(output_dir), payload)
    write_frozen_json(
        Path(output_dir) / "qualification_plan.sha256",
        {"plan_sha256": payload["plan_sha256"]},
    )
    atomic_write_json(
        Path(output_dir) / "plan_run.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "output_dir": str(output_dir),
            "source_commit": payload["source_commit"],
            "plan_sha256": payload["plan_sha256"],
            "summary": payload["summary"],
            "diagnostic_only": True,
            "production_casimir_allowed": False,
        },
    )
    return payload


def preflight_payload(
    campaign: Todo4QualificationCampaign,
    *,
    cache_root: Path,
) -> dict[str, Any]:
    cache = MaterialResponseCacheStore(cache_root, mode="read_only")
    status_by_identity: dict[str, str] = {}
    plan_records: list[dict[str, Any]] = []
    for entry in campaign.entries:
        preflight = preflight_geometry_batch(entry.geometry_plan, cache=cache)
        for key in preflight.hits:
            status_by_identity[key] = "hit"
        for key in preflight.misses:
            previous = status_by_identity.get(key)
            if previous == "hit":
                raise RuntimeError("one identity was both a cache hit and miss")
            status_by_identity[key] = "miss"
        plan_records.append(
            {
                "plan_id": entry.plan_id,
                "plan_sha256": entry.geometry_plan.sha256,
                "hit_count": len(preflight.hits),
                "miss_count": len(preflight.misses),
                "hits": list(preflight.hits),
                "misses": list(preflight.misses),
                "metadata": jsonable(preflight.metadata),
            }
        )
    unique = campaign.unique_requirements
    missing = [
        {
            "identity_sha256": key,
            "identity": jsonable(unique[key].identity.payload),
            "q_crystal_hex": [
                float(unique[key].q_crystal[0]).hex(),
                float(unique[key].q_crystal[1]).hex(),
            ],
        }
        for key in sorted(unique)
        if status_by_identity.get(key) == "miss"
    ]
    hits = [
        key
        for key in sorted(unique)
        if status_by_identity.get(key) == "hit"
    ]
    return {
        "schema": "todo4-qualification-cache-preflight-v1",
        "campaign_id": campaign.campaign_id,
        "manifest_sha256": campaign.manifest_sha256,
        "source_commit": source_commit(),
        "cache_root": str(cache_root),
        "cache_mode": "read_only",
        "plan_records": plan_records,
        "summary": {
            "unique_response_identity_count": len(unique),
            "cache_hit_count": len(hits),
            "cache_miss_count": len(missing),
            "complete": not missing,
        },
        "hits": hits,
        "missing": missing,
        "contract": {
            "microscopic_fallback_attempted": False,
            "cache_write_attempted": False,
            "exact_identity_only": True,
            "diagnostic_only": True,
            "production_casimir_allowed": False,
        },
    }


def write_preflight(
    campaign: Todo4QualificationCampaign,
    *,
    output_dir: Path,
    cache_root: Path,
    require_complete: bool,
) -> dict[str, Any]:
    frozen = require_frozen_plan(campaign, output_dir)
    payload = preflight_payload(campaign, cache_root=cache_root)
    payload["plan_sha256"] = frozen["plan_sha256"]
    name = (
        "cache_preflight_after.json"
        if require_complete
        else "cache_preflight_before.json"
    )
    atomic_write_json(Path(output_dir) / name, payload)
    atomic_write_json(
        Path(output_dir) / "cache_miss_manifest.json",
        {
            "schema": "todo4-cache-miss-manifest-v1",
            "campaign_id": campaign.campaign_id,
            "plan_sha256": frozen["plan_sha256"],
            "source_commit": source_commit(),
            "cache_root": str(cache_root),
            "missing": payload["missing"],
            "cache_miss_count": payload["summary"]["cache_miss_count"],
            "diagnostic_only": True,
            "production_casimir_allowed": False,
        },
    )
    if require_complete and not payload["summary"]["complete"]:
        raise RuntimeError("TODO 4 cache preflight is incomplete")
    return payload


def validate_shard(index: int, count: int) -> None:
    if count <= 0:
        raise ValueError("shard_count must be positive")
    if index < 0 or index >= count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")


def _config_by_pairing(
    campaign: Todo4QualificationCampaign,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for entry in campaign.entries:
        result.setdefault(
            entry.pairing_name,
            entry.geometry_plan.response_config,
        )
    return result


def populate_shard(
    campaign: Todo4QualificationCampaign,
    *,
    output_dir: Path,
    cache_root: Path,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    frozen = require_frozen_plan(campaign, output_dir)
    validate_shard(shard_index, shard_count)
    groups = campaign.populate_groups
    selected = [
        group
        for position, group in enumerate(groups)
        if position % shard_count == shard_index
    ]
    configs = _config_by_pairing(campaign)
    cache = MaterialResponseCacheStore(cache_root, mode="populate")
    records: list[dict[str, Any]] = []
    failed = False
    for pairing, q_hex, q in selected:
        try:
            result = evaluate_material_response_ladder_cached(
                configs[pairing],
                q_crystal=q,
                cache=cache,
            )
            frequencies = {
                str(index): {
                    "source": frequency.source,
                    "established": frequency.established,
                    "cache_identity_sha256": frequency.cache_identity.sha256,
                    "xi_eV_hex": float(frequency.xi_eV).hex(),
                }
                for index, frequency in result.frequencies.items()
            }
            established = bool(result.all_requested_established)
            failed = failed or not established
            records.append(
                {
                    "pairing_name": pairing,
                    "q_crystal_hex": list(q_hex),
                    "status": "established" if established else "unresolved",
                    "frequencies": frequencies,
                    "metadata": jsonable(result.metadata),
                }
            )
        except Exception as exc:
            failed = True
            records.append(
                {
                    "pairing_name": pairing,
                    "q_crystal_hex": list(q_hex),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    payload = {
        "schema": "todo4-qualification-populate-shard-v1",
        "campaign_id": campaign.campaign_id,
        "plan_sha256": frozen["plan_sha256"],
        "source_commit": source_commit(),
        "cache_root": str(cache_root),
        "cache_mode": "populate",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "total_group_count": len(groups),
        "selected_group_count": len(selected),
        "records": records,
        "passed": not failed,
        "diagnostic_only": True,
        "production_casimir_allowed": False,
    }
    path = (
        Path(output_dir)
        / "populate"
        / f"shard_{shard_index:03d}_of_{shard_count:03d}.json"
    )
    atomic_write_json(path, payload)
    return payload


def geometry_file(output_dir: Path, entry: Todo4QualificationPlanEntry) -> Path:
    return Path(output_dir) / "geometry" / f"{slug(entry.plan_id)}.json"


def geometry_npz_file(
    output_dir: Path,
    entry: Todo4QualificationPlanEntry,
) -> Path:
    return Path(output_dir) / "geometry" / f"{slug(entry.plan_id)}.npz"


def legacy_file(
    output_dir: Path,
    entry: Todo4QualificationPlanEntry,
    point_id: str,
) -> Path:
    return (
        Path(output_dir)
        / "legacy"
        / slug(entry.plan_id)
        / f"{point_token(point_id)}.json"
    )


def _geometry_payload(
    entry: Todo4QualificationPlanEntry,
    result: object,
    *,
    policy: object,
) -> tuple[dict[str, Any], dict[str, np.ndarray], bool]:
    point_records: list[dict[str, Any]] = []
    scalar_passed = True
    logdet_rows: list[list[float]] = []
    point_ids: list[str] = []
    q_labels: list[str] = []
    matsubara_indices: list[int] = []
    for point in entry.geometry_plan.points:
        point_result = result.points[point.point_id]
        report = qualify_batch_point_against_scalar(
            result,
            point_id=point.point_id,
            policy=policy,
        )
        scalar_passed = scalar_passed and bool(report.passed)
        values = [float(item.logdet) for item in point_result.lifshitz_points]
        logdet_rows.append(values)
        point_ids.append(point.point_id)
        q_labels.append(point.q_label)
        matsubara_indices.append(point.matsubara_index)
        point_records.append(
            {
                "point_id": point.point_id,
                "q_label": point.q_label,
                "q_lab": point.q_lab.tolist(),
                "matsubara_index": point.matsubara_index,
                "theta_1_rad": point.theta_1_rad,
                "theta_2_rad": point.theta_2_rad,
                "distances_m": list(point_result.distances_m),
                "logdet": values,
                "scalar_vs_batch": jsonable(report),
            }
        )
    payload = {
        "schema": "todo4-qualification-geometry-result-v1",
        "plan_id": entry.plan_id,
        "kind": entry.kind,
        "pairing_name": entry.pairing_name,
        "plan_sha256": entry.geometry_plan.sha256,
        "legacy_distance_m": entry.legacy_distance_m,
        "metadata": jsonable(result.metadata),
        "points": point_records,
        "scalar_vs_batch_passed": scalar_passed,
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
    }
    arrays = {
        "point_id": np.asarray(point_ids),
        "q_label": np.asarray(q_labels),
        "matsubara_index": np.asarray(matsubara_indices, dtype=int),
        "distances_m": np.asarray(entry.geometry_plan.separations_m, dtype=float),
        "logdet": np.asarray(logdet_rows, dtype=float),
    }
    return payload, arrays, scalar_passed


def execute_campaign_geometry(
    campaign: Todo4QualificationCampaign,
    *,
    output_dir: Path,
    cache_root: Path,
) -> dict[str, Any]:
    frozen = require_frozen_plan(campaign, output_dir)
    preflight = preflight_payload(campaign, cache_root=cache_root)
    if not preflight["summary"]["complete"]:
        atomic_write_json(
            Path(output_dir) / "cache_preflight_after.json",
            preflight,
        )
        raise RuntimeError("strict read-only geometry cannot run with cache misses")
    cache = MaterialResponseCacheStore(cache_root, mode="read_only")
    policy = geometry_equivalence_policy(campaign.manifest)
    records: list[dict[str, Any]] = []
    passed = True
    for entry in campaign.entries:
        result = execute_geometry_batch(entry.geometry_plan, cache=cache)
        payload, arrays, scalar_passed = _geometry_payload(
            entry,
            result,
            policy=policy,
        )
        payload["campaign_plan_sha256"] = frozen["plan_sha256"]
        atomic_write_json(geometry_file(output_dir, entry), payload)
        atomic_write_npz(geometry_npz_file(output_dir, entry), **arrays)
        passed = passed and scalar_passed
        records.append(
            {
                "plan_id": entry.plan_id,
                "geometry_output": str(geometry_file(output_dir, entry)),
                "npz_output": str(geometry_npz_file(output_dir, entry)),
                "scalar_vs_batch_passed": scalar_passed,
                "metadata": jsonable(result.metadata),
            }
        )
    summary = {
        "schema": "todo4-qualification-geometry-summary-v1",
        "campaign_id": campaign.campaign_id,
        "campaign_plan_sha256": frozen["plan_sha256"],
        "source_commit": source_commit(),
        "records": records,
        "passed": passed,
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
    }
    atomic_write_json(Path(output_dir) / "geometry_summary.json", summary)
    return summary


def legacy_items(
    campaign: Todo4QualificationCampaign,
) -> tuple[tuple[Todo4QualificationPlanEntry, str], ...]:
    items: list[tuple[Todo4QualificationPlanEntry, str]] = []
    for entry in campaign.entries:
        for point in entry.geometry_plan.points:
            items.append((entry, point.point_id))
    return tuple(sorted(items, key=lambda item: (item[0].plan_id, item[1])))


def execute_legacy_shard(
    campaign: Todo4QualificationCampaign,
    *,
    output_dir: Path,
    cache_root: Path,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    frozen = require_frozen_plan(campaign, output_dir)
    validate_shard(shard_index, shard_count)
    preflight = preflight_payload(campaign, cache_root=cache_root)
    if not preflight["summary"]["complete"]:
        raise RuntimeError("legacy qualification requires a complete response cache")
    items = legacy_items(campaign)
    selected = [
        item
        for position, item in enumerate(items)
        if position % shard_count == shard_index
    ]
    by_plan: dict[str, list[str]] = {}
    entry_by_id = {entry.plan_id: entry for entry in campaign.entries}
    for entry, point_id in selected:
        by_plan.setdefault(entry.plan_id, []).append(point_id)
    cache = MaterialResponseCacheStore(cache_root, mode="read_only")
    policy = geometry_equivalence_policy(campaign.manifest)
    records: list[dict[str, Any]] = []
    failed = False
    for plan_id, point_ids in by_plan.items():
        entry = entry_by_id[plan_id]
        batch = execute_geometry_batch(entry.geometry_plan, cache=cache)
        point_map = {point.point_id: point for point in entry.geometry_plan.points}
        for point_id in point_ids:
            point = point_map[point_id]
            path = legacy_file(output_dir, entry, point_id)
            try:
                replay = run_matched_legacy_geometry_replay(
                    batch,
                    point_id=point_id,
                    distance_m=entry.legacy_distance_m,
                    policy=policy,
                )
                comparison = replay.report.comparisons["logdet"]
                payload = {
                    "schema": "todo4-qualification-legacy-point-v1",
                    "campaign_id": campaign.campaign_id,
                    "campaign_plan_sha256": frozen["plan_sha256"],
                    "plan_id": entry.plan_id,
                    "kind": entry.kind,
                    "pairing_name": entry.pairing_name,
                    "point_id": point_id,
                    "q_label": point.q_label,
                    "matsubara_index": point.matsubara_index,
                    "distance_m": entry.legacy_distance_m,
                    "reference_logdet": float(comparison["reference"]),
                    "candidate_logdet": float(comparison["candidate"]),
                    "passed": bool(replay.report.passed),
                    "replay": jsonable(replay),
                    "diagnostic_only": True,
                    "production_casimir_allowed": False,
                }
                failed = failed or not payload["passed"]
            except Exception as exc:
                failed = True
                payload = {
                    "schema": "todo4-qualification-legacy-point-v1",
                    "campaign_id": campaign.campaign_id,
                    "campaign_plan_sha256": frozen["plan_sha256"],
                    "plan_id": entry.plan_id,
                    "kind": entry.kind,
                    "pairing_name": entry.pairing_name,
                    "point_id": point_id,
                    "q_label": point.q_label,
                    "matsubara_index": point.matsubara_index,
                    "distance_m": entry.legacy_distance_m,
                    "passed": False,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "diagnostic_only": True,
                    "production_casimir_allowed": False,
                }
            atomic_write_json(path, payload)
            records.append(
                {
                    "plan_id": entry.plan_id,
                    "point_id": point_id,
                    "output": str(path),
                    "passed": bool(payload["passed"]),
                }
            )
    summary = {
        "schema": "todo4-qualification-legacy-shard-v1",
        "campaign_id": campaign.campaign_id,
        "campaign_plan_sha256": frozen["plan_sha256"],
        "source_commit": source_commit(),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "total_point_count": len(items),
        "selected_point_count": len(selected),
        "records": records,
        "passed": not failed,
        "diagnostic_only": True,
        "production_casimir_allowed": False,
    }
    path = (
        Path(output_dir)
        / "legacy"
        / f"shard_{shard_index:03d}_of_{shard_count:03d}.json"
    )
    atomic_write_json(path, summary)
    return summary


def verify_campaign(
    campaign: Todo4QualificationCampaign,
    *,
    output_dir: Path,
    cache_root: Path,
) -> dict[str, Any]:
    frozen = require_frozen_plan(campaign, output_dir)
    preflight = preflight_payload(campaign, cache_root=cache_root)
    atomic_write_json(
        Path(output_dir) / "cache_preflight_after.json",
        preflight,
    )
    geometry_records: list[dict[str, Any]] = []
    legacy_records: list[dict[str, Any]] = []
    all_passed = bool(preflight["summary"]["complete"])
    for entry in campaign.entries:
        geometry = load_json(geometry_file(output_dir, entry))
        geometry_passed = bool(geometry.get("scalar_vs_batch_passed"))
        all_passed = all_passed and geometry_passed
        geometry_records.append(
            {
                "plan_id": entry.plan_id,
                "passed": geometry_passed,
                "output": str(geometry_file(output_dir, entry)),
            }
        )
        for point in entry.geometry_plan.points:
            legacy = load_json(legacy_file(output_dir, entry, point.point_id))
            passed = bool(legacy.get("passed"))
            all_passed = all_passed and passed
            legacy_records.append(
                {
                    "plan_id": entry.plan_id,
                    "point_id": point.point_id,
                    "passed": passed,
                    "output": str(
                        legacy_file(output_dir, entry, point.point_id)
                    ),
                }
            )

    outer_records: list[dict[str, Any]] = []
    outer_policy = fixed_outer_equivalence_policy(campaign.manifest)
    for entry in campaign.entries:
        if entry.kind != "fixed_outer":
            continue
        if entry.outer_grid is None:
            raise RuntimeError("fixed outer entry has no grid")
        indices = tuple(entry.geometry_plan.response_config.matsubara_indices)
        labels = tuple(entry.geometry_plan.q_lab_by_label)
        reference = np.empty((len(indices), len(labels)), dtype=float)
        candidate = np.empty_like(reference)
        points = {
            (point.matsubara_index, point.q_label): point
            for point in entry.geometry_plan.points
        }
        for n_index, matsubara_index in enumerate(indices):
            for q_index, label in enumerate(labels):
                point = points[(matsubara_index, label)]
                legacy = load_json(
                    legacy_file(output_dir, entry, point.point_id)
                )
                reference[n_index, q_index] = float(legacy["reference_logdet"])
                candidate[n_index, q_index] = float(legacy["candidate_logdet"])
        report = qualify_fixed_outer_geometry_replay(
            reference_logdet_by_n_and_node=reference,
            candidate_logdet_by_n_and_node=candidate,
            matsubara_indices=indices,
            temperature_K=entry.geometry_plan.response_config.temperature_K,
            grid=entry.outer_grid,
            policy=outer_policy,
        )
        path = Path(output_dir) / "fixed_outer" / f"{slug(entry.plan_id)}.json"
        atomic_write_json(
            path,
            {
                "schema": "todo4-qualification-fixed-outer-result-v1",
                "campaign_id": campaign.campaign_id,
                "campaign_plan_sha256": frozen["plan_sha256"],
                "plan_id": entry.plan_id,
                "report": jsonable(report),
                "passed": bool(report.passed),
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "production_casimir_allowed": False,
            },
        )
        all_passed = all_passed and bool(report.passed)
        outer_records.append(
            {
                "plan_id": entry.plan_id,
                "passed": bool(report.passed),
                "output": str(path),
            }
        )

    payload = {
        "schema": "todo4-representative-qualification-verification-v1",
        "campaign_id": campaign.campaign_id,
        "campaign_plan_sha256": frozen["plan_sha256"],
        "source_commit": source_commit(),
        "cache_preflight_complete": bool(preflight["summary"]["complete"]),
        "geometry": geometry_records,
        "legacy": legacy_records,
        "fixed_outer": outer_records,
        "passed": all_passed,
        "status": {
            "todo4_representative_qualification_passed": all_passed,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
            "observable_error_budget_calibrated": False,
        },
        "diagnostic_only": True,
    }
    atomic_write_json(Path(output_dir) / "verification.json", payload)
    return payload


__all__ = [
    "current_plan",
    "execute_campaign_geometry",
    "execute_legacy_shard",
    "freeze_plan",
    "geometry_file",
    "geometry_npz_file",
    "legacy_file",
    "legacy_items",
    "load_campaign",
    "plan_path",
    "populate_shard",
    "preflight_payload",
    "require_frozen_plan",
    "validate_shard",
    "verify_campaign",
    "write_preflight",
]
