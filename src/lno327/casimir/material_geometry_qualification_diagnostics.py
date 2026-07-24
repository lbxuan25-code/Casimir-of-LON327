"""Read-only unresolved-response diagnostics for the TODO 4 qualification campaign."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from lno327.casimir.material_geometry_qualification_campaign import (
    Todo4QualificationCampaign,
)
from lno327.casimir.material_geometry_qualification_execution import (
    require_frozen_plan,
    validate_shard,
)
from lno327.casimir.material_geometry_qualification_io import (
    atomic_write_json,
    source_commit,
)
from lno327.casimir.material_response_cache_errors import MaterialResponseCacheMiss
from lno327.casimir.material_response_cache_store import MaterialResponseCacheStore
from lno327.casimir.material_response_cached_engine import (
    build_material_response_cache_identity,
    build_material_response_identity_context,
)
from lno327.casimir.material_response_certification_diagnostics import (
    summarize_material_frequency_result,
)
from lno327.casimir.material_response_engine import (
    MaterialResponseEngineConfig,
    evaluate_material_response_ladder,
)

TODO4_UNRESOLVED_DIAGNOSTIC_SHARD_SCHEMA = (
    "todo4-unresolved-response-diagnostic-shard-v1"
)


def _config_by_pairing(
    campaign: Todo4QualificationCampaign,
) -> dict[str, MaterialResponseEngineConfig]:
    result: dict[str, MaterialResponseEngineConfig] = {}
    for entry in campaign.entries:
        config = entry.geometry_plan.response_config
        previous = result.setdefault(entry.pairing_name, config)
        if previous.as_dict() != config.as_dict():
            raise ValueError("one pairing has inconsistent response configurations")
    return result


def _missing_indices(
    config: MaterialResponseEngineConfig,
    *,
    q_crystal: np.ndarray,
    cache: MaterialResponseCacheStore,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    context = build_material_response_identity_context(config)
    missing: list[int] = []
    identities: list[str] = []
    for index in config.matsubara_indices:
        identity = build_material_response_cache_identity(
            config,
            q_crystal=q_crystal,
            matsubara_index=index,
            context=context,
        )
        identities.append(identity.sha256)
        try:
            cache.get(identity)
        except MaterialResponseCacheMiss:
            missing.append(index)
    return tuple(missing), tuple(identities)


def diagnose_unresolved_shard(
    campaign: Todo4QualificationCampaign,
    *,
    output_dir: Path,
    cache_root: Path,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    """Re-evaluate only exact cache misses and persist compact failure evidence.

    This stage never writes the certified-response cache.  An unresolved response is
    an expected diagnostic result and does not make the command fail; only execution
    or serialization errors do.
    """

    frozen = require_frozen_plan(campaign, output_dir)
    validate_shard(shard_index, shard_count)
    configs = _config_by_pairing(campaign)
    cache = MaterialResponseCacheStore(cache_root, mode="read_only")

    missing_groups: list[
        tuple[str, tuple[str, str], np.ndarray, tuple[int, ...], tuple[str, ...]]
    ] = []
    for pairing, q_hex, q in campaign.populate_groups:
        missing, identities = _missing_indices(
            configs[pairing],
            q_crystal=q,
            cache=cache,
        )
        if missing:
            missing_groups.append((pairing, q_hex, q, missing, identities))

    selected = [
        group
        for position, group in enumerate(missing_groups)
        if position % shard_count == shard_index
    ]
    records: list[dict[str, Any]] = []
    errors = 0
    unresolved = 0
    established = 0
    for pairing, q_hex, q, missing, identities in selected:
        config = configs[pairing]
        try:
            diagnostic_config = replace(config, matsubara_indices=missing)
            result = evaluate_material_response_ladder(
                diagnostic_config,
                q_crystal=q,
            )
            frequencies = {}
            for index in missing:
                summary = summarize_material_frequency_result(
                    result.frequencies[index],
                    policy=diagnostic_config.convergence_policy,
                    required_consecutive_passes=(
                        diagnostic_config.required_consecutive_passes
                    ),
                    envelope_levels=diagnostic_config.envelope_levels,
                )
                frequencies[str(index)] = summary
                if summary["status"] == "established":
                    established += 1
                else:
                    unresolved += 1
            records.append(
                {
                    "pairing_name": pairing,
                    "q_crystal_hex": list(q_hex),
                    "requested_cache_identity_sha256": list(identities),
                    "missing_matsubara_indices": list(missing),
                    "evaluated_n_candidates": list(result.evaluated_n_candidates),
                    "frequencies": frequencies,
                    "status": "diagnostic_completed",
                }
            )
        except Exception as exc:
            errors += 1
            records.append(
                {
                    "pairing_name": pairing,
                    "q_crystal_hex": list(q_hex),
                    "requested_cache_identity_sha256": list(identities),
                    "missing_matsubara_indices": list(missing),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    payload = {
        "schema": TODO4_UNRESOLVED_DIAGNOSTIC_SHARD_SCHEMA,
        "campaign_id": campaign.campaign_id,
        "plan_sha256": frozen["plan_sha256"],
        "source_commit": source_commit(),
        "cache_root": str(cache_root),
        "cache_mode": "read_only",
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "total_missing_group_count": len(missing_groups),
        "selected_missing_group_count": len(selected),
        "unresolved_frequency_count": unresolved,
        "established_on_diagnostic_replay_count": established,
        "error_count": errors,
        "records": records,
        "diagnostic_completed": errors == 0,
        "cache_write_attempted": False,
        "certified_artifact_created": False,
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
    }
    path = (
        Path(output_dir)
        / "unresolved_diagnostics"
        / f"shard_{shard_index:03d}_of_{shard_count:03d}.json"
    )
    atomic_write_json(path, payload)
    return payload


__all__ = [
    "TODO4_UNRESOLVED_DIAGNOSTIC_SHARD_SCHEMA",
    "diagnose_unresolved_shard",
]
