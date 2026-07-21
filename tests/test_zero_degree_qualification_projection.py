from __future__ import annotations

import json
from pathlib import Path

from lno327.casimir.certified_point_provider import (
    certified_point_policy_fingerprint,
    certified_point_policy_payload,
)
from lno327.casimir.production import build_full_casimir_config
from scripts.full_casimir.cache_migration import CACHE_SCHEMA
from scripts.full_casimir.config import RuntimeResources, case_name
from scripts.full_casimir.data_management import _sha
from scripts.full_casimir.qualification import (
    LOGDET_ATOL,
    LOGDET_RTOL,
    N_CANDIDATES,
    PROFILE,
    REQUIRED_CONSECUTIVE_PASSES,
    SOURCE_PROFILE,
    _projection_one,
)


def _plate() -> dict:
    return {
        "sheet_validation_passed": True,
        "reflection_constructed": True,
        "reflection_norm": 0.4,
    }


def _state(value: float) -> dict:
    return {
        "two_plate_logdet": value,
        "hard_physical_passed": True,
        "plate_1": _plate(),
        "plate_2": _plate(),
    }


def _point() -> dict:
    return {
        "pairing": "spm",
        "q_label": "q0",
        "n": 0,
        "sweet_spot": {
            "status": "not_established",
            "working_N": None,
            "audit_N": None,
        },
        "history": [
            {"N": 128, "shifts": {"a": _state(-1.0), "b": _state(-1.0005)}},
            {"N": 192, "shifts": {"a": _state(-1.001), "b": _state(-1.0014)}},
            {"N": 256, "shifts": {"a": _state(-1.0015), "b": _state(-1.0018)}},
            {"N": 384, "shifts": {"a": _state(-1.0017), "b": _state(-1.0019)}},
        ],
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_projection_reuses_history_and_preserves_source(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    source_run = output_root / case_name("spm", 0, profile=SOURCE_PROFILE)
    source_cache = source_run / "cache" / "certified_points.json"
    source_full = build_full_casimir_config(
        point_cache_path=source_cache,
        pairings=("spm",),
        plate_angles_deg=(0.0, 0.0),
        N_candidates=N_CANDIDATES[:8],
        logdet_rtol=1.5e-3,
        logdet_atol=LOGDET_ATOL,
        required_consecutive_passes=REQUIRED_CONSECUTIVE_PASSES,
        radial_budget_fraction=0.8,
    )
    source_point = source_full.outer_tail_config.joint_config.radial_config.point_config
    entry = {
        "pairing": "spm",
        "n": 0,
        "qx_hex": float(0.1).hex(),
        "qy_hex": float(0.2).hex(),
        "point_result": _point(),
    }
    _write_json(source_run / "config.json", source_full.as_dict())
    for name in ("manifest.json", "result.json", "summary.json"):
        _write_json(source_run / name, {"schema": name})
    _write_json(
        source_cache,
        {
            "schema": CACHE_SCHEMA,
            "policy_fingerprint": certified_point_policy_fingerprint(
                source_point, frequency_extendable=True
            ),
            "frequency_extendable": True,
            "active_matsubara_indices": [0, 1],
            "point_policy": certified_point_policy_payload(
                source_point, frequency_extendable=True
            ),
            "entries": [entry],
        },
    )
    source_sha = _sha(source_cache)

    target_run = output_root / case_name("spm", 0, profile=PROFILE)
    target_cache = target_run / "cache" / "certified_points.json"
    target_full = build_full_casimir_config(
        point_cache_path=target_cache,
        pairings=("spm",),
        plate_angles_deg=(0.0, 0.0),
        N_candidates=N_CANDIDATES,
        logdet_rtol=LOGDET_RTOL,
        logdet_atol=LOGDET_ATOL,
        required_consecutive_passes=REQUIRED_CONSECUTIVE_PASSES,
        radial_budget_fraction=0.8,
        workers=2,
        parallel_mode="q",
        memory_budget_gb=4.0,
        max_context_workers=1,
    )
    report = _projection_one(
        pairing="spm",
        output_root=output_root,
        source_profile=SOURCE_PROFILE,
        target_profile=PROFILE,
        target_config=target_full,
    )

    assert report["retained_entry_count"] == 1
    assert report["omitted_entry_count"] == 0
    assert report["changed_decision_count"] == 1
    assert _sha(source_cache) == source_sha
    target = json.loads(target_cache.read_text(encoding="utf-8"))
    projected = target["entries"][0]["point_result"]
    assert projected["sweet_spot"]["status"] == "established"
    assert len(projected["history"]) == 4
    assert projected["qualification_projection"]["source_history_preserved"] is True
    assert target["policy_fingerprint"] == certified_point_policy_fingerprint(
        target_full.outer_tail_config.joint_config.radial_config.point_config,
        frequency_extendable=True,
    )

    repeated = _projection_one(
        pairing="spm",
        output_root=output_root,
        source_profile=SOURCE_PROFILE,
        target_profile=PROFILE,
        target_config=target_full,
    )
    assert repeated["projection_sha256"] == report["projection_sha256"]
