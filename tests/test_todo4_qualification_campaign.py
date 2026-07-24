from __future__ import annotations

import json
from pathlib import Path

from lno327.casimir.material_geometry_qualification_campaign import (
    build_todo4_qualification_campaign,
    load_todo4_qualification_manifest,
    qualification_plan_payload,
)
from lno327.casimir.material_geometry_qualification_execution import (
    preflight_payload,
    validate_shard,
)
from lno327.casimir.material_geometry_qualification_io import (
    write_frozen_json,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation/configs/casimir/todo4_representative_v1.json"


def _campaign():
    return build_todo4_qualification_campaign(
        load_todo4_qualification_manifest(MANIFEST)
    )


def test_representative_manifest_is_sparse_but_covers_required_route_classes() -> None:
    campaign = _campaign()

    assert campaign.campaign_id == "todo4-representative-v1"
    assert len(campaign.entries) == 5
    assert sum(entry.kind == "direct" for entry in campaign.entries) == 4
    assert sum(entry.kind == "fixed_outer" for entry in campaign.entries) == 1
    assert {entry.pairing_name for entry in campaign.entries} == {"spm", "dwave"}
    assert len(campaign.unique_requirements) == 20
    assert len(campaign.populate_groups) == 10
    assert sum(len(entry.geometry_plan.points) for entry in campaign.entries) == 16

    direct = [entry for entry in campaign.entries if entry.kind == "direct"]
    assert all(entry.geometry_plan.response_config.matsubara_indices == (0, 1) for entry in direct)
    assert all(len(entry.geometry_plan.separations_m) == 3 for entry in direct)
    assert {
        tuple(entry.geometry_plan.angle_pairs_rad[0])
        for entry in direct
    } == {(0.0, 0.0), (0.0, 0.31)}


def test_frozen_plan_payload_is_deterministic_and_explicitly_nonproduction() -> None:
    campaign = _campaign()
    first = qualification_plan_payload(campaign, source_commit="abc123")
    second = qualification_plan_payload(campaign, source_commit="abc123")

    assert first == second
    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["summary"] == {
        "plan_count": 5,
        "direct_plan_count": 4,
        "fixed_outer_plan_count": 1,
        "geometry_point_count": 16,
        "unique_response_identity_count": 20,
        "populate_group_count": 10,
        "expected_reflection_build_count": 20,
        "expected_prepared_pair_count": 16,
        "expected_distance_update_count": 32,
    }
    assert first["contract"]["geometry_cache_mode"] == "read_only"
    assert first["contract"]["populate_separate_from_geometry"] is True
    assert first["contract"]["production_casimir_allowed"] is False


def test_empty_read_only_preflight_reports_all_exact_misses_without_creating_root(
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    cache_root = tmp_path / "absent-cache"

    payload = preflight_payload(campaign, cache_root=cache_root)

    assert payload["summary"] == {
        "unique_response_identity_count": 20,
        "cache_hit_count": 0,
        "cache_miss_count": 20,
        "complete": False,
    }
    assert len(payload["missing"]) == 20
    assert cache_root.exists() is False
    assert payload["contract"]["microscopic_fallback_attempted"] is False
    assert payload["contract"]["cache_write_attempted"] is False


def test_populate_groups_partition_without_overlap_or_loss() -> None:
    groups = _campaign().populate_groups
    shard_count = 4
    partitions = [
        {
            (pairing, q_hex)
            for position, (pairing, q_hex, _q) in enumerate(groups)
            if position % shard_count == shard_index
        }
        for shard_index in range(shard_count)
    ]

    assert sum(len(partition) for partition in partitions) == len(groups)
    assert set().union(*partitions) == {
        (pairing, q_hex) for pairing, q_hex, _q in groups
    }
    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1 :]:
            assert left.isdisjoint(right)
    for shard_index in range(shard_count):
        validate_shard(shard_index, shard_count)


def test_frozen_json_refuses_to_replace_different_scientific_plan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plan.json"
    write_frozen_json(path, {"schema": "test", "value": 1})
    write_frozen_json(path, {"schema": "test", "value": 1})

    try:
        write_frozen_json(path, {"schema": "test", "value": 2})
    except RuntimeError as exc:
        assert "refusing to replace" in str(exc)
    else:
        raise AssertionError("different frozen plan was overwritten")

    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 1
