"""Read-only unresolved-response diagnostics for the TODO 4 qualification campaign."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

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
    "todo4-unresolved-response-diagnostic-shard-v2"
)


def normalize_diagnostic_n_candidates(
    base_n_candidates: Sequence[int],
    override: Sequence[int] | None,
) -> tuple[int, ...]:
    """Validate a controlled diagnostic ladder with one exact overlap anchor.

    The base qualification identity is never changed by this helper.  An override is
    diagnostic-only and must begin at the final base N so the old and extended
    histories share one exact level.  At least two strictly larger levels are
    required so adjacent-N and three-level envelope evidence remain available.
    """

    base = tuple(int(value) for value in base_n_candidates)
    if not base:
        raise ValueError("base_n_candidates must be nonempty")
    if override is None:
        return base

    values = tuple(int(value) for value in override)
    if len(values) < 3:
        raise ValueError("diagnostic N override requires at least three levels")
    if any(value <= 0 or value % 2 != 0 for value in values):
        raise ValueError("diagnostic N candidates must be positive even integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("diagnostic N candidates must be strictly increasing and unique")
    if values[0] != base[-1]:
        raise ValueError(
            "diagnostic N override must start at the final base N overlap anchor"
        )
    if values[1] <= base[-1]:
        raise ValueError("diagnostic N override must extend beyond the base ladder")
    return values


def diagnostic_ladder_tag(n_candidates: Sequence[int]) -> str:
    values = tuple(int(value) for value in n_candidates)
    if not values:
        raise ValueError("n_candidates must be nonempty")
    return "N" + "-".join(str(value) for value in values)


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


def _diagnostic_identity_sha256(
    config: MaterialResponseEngineConfig,
    *,
    q_crystal: np.ndarray,
    indices: Sequence[int],
) -> dict[str, str]:
    context = build_material_response_identity_context(config)
    return {
        str(index): build_material_response_cache_identity(
            config,
            q_crystal=q_crystal,
            matsubara_index=int(index),
            context=context,
        ).sha256
        for index in indices
    }


def diagnose_unresolved_shard(
    campaign: Todo4QualificationCampaign,
    *,
    output_dir: Path,
    cache_root: Path,
    shard_index: int,
    shard_count: int,
    n_candidates_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Re-evaluate only exact base-cache misses and persist compact failure evidence.

    This stage never writes the certified-response cache.  An unresolved response is
    an expected diagnostic result and does not make the command fail; only execution
    or serialization errors do.  A controlled N override changes only the in-memory
    diagnostic ladder and is recorded separately from every base cache identity.
    """

    frozen = require_frozen_plan(campaign, output_dir)
    validate_shard(shard_index, shard_count)
    configs = _config_by_pairing(campaign)
    cache = MaterialResponseCacheStore(cache_root, mode="read_only")

    base_ladders = {config.n_candidates for config in configs.values()}
    if len(base_ladders) != 1:
        raise ValueError("diagnostic campaign requires one shared base N ladder")
    base_n_candidates = next(iter(base_ladders))
    diagnostic_n_candidates = normalize_diagnostic_n_candidates(
        base_n_candidates,
        n_candidates_override,
    )
    override_active = diagnostic_n_candidates != tuple(base_n_candidates)

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
    for pairing, q_hex, q, missing, base_identities in selected:
        config = configs[pairing]
        try:
            diagnostic_config = replace(
                config,
                matsubara_indices=missing,
                n_candidates=diagnostic_n_candidates,
            )
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
                    "base_requested_cache_identity_sha256": list(base_identities),
                    "diagnostic_identity_sha256": _diagnostic_identity_sha256(
                        diagnostic_config,
                        q_crystal=q,
                        indices=missing,
                    ),
                    "missing_matsubara_indices": list(missing),
                    "base_n_candidates": list(base_n_candidates),
                    "diagnostic_n_candidates": list(diagnostic_n_candidates),
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
                    "base_requested_cache_identity_sha256": list(base_identities),
                    "missing_matsubara_indices": list(missing),
                    "base_n_candidates": list(base_n_candidates),
                    "diagnostic_n_candidates": list(diagnostic_n_candidates),
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
        "base_n_candidates": list(base_n_candidates),
        "diagnostic_n_candidates": list(diagnostic_n_candidates),
        "diagnostic_ladder_tag": diagnostic_ladder_tag(diagnostic_n_candidates),
        "diagnostic_n_override_active": override_active,
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
        "base_cache_identity_changed": False,
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
    }
    diagnostic_root = Path(output_dir) / "unresolved_diagnostics"
    if override_active:
        diagnostic_root = diagnostic_root / diagnostic_ladder_tag(
            diagnostic_n_candidates
        )
    path = diagnostic_root / f"shard_{shard_index:03d}_of_{shard_count:03d}.json"
    atomic_write_json(path, payload)
    return payload


__all__ = [
    "TODO4_UNRESOLVED_DIAGNOSTIC_SHARD_SCHEMA",
    "diagnose_unresolved_shard",
    "diagnostic_ladder_tag",
    "normalize_diagnostic_n_candidates",
]
