from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from scripts.full_casimir.data_management import _digest, _read, _write
from scripts.full_casimir.qualification import HOLDOUT_PLAN_SCHEMA
from scripts.full_casimir.qualification_holdout import (
    build_policy,
    execute,
    load_checkpoint,
    write_checkpoint,
)
from scripts.full_casimir.qualification_holdout_group import build_groups


def _item(index: int) -> dict:
    return {
        "identity": ["spm" if index % 2 == 0 else "dwave", index % 2, float(index).hex(), float(index + 1).hex()],
        "pairing": "spm" if index % 2 == 0 else "dwave",
        "n": index % 2,
        "qx_hex": float(index).hex(),
        "qy_hex": float(index + 1).hex(),
        "q_model": [float(index), float(index + 1)],
        "reasons": ["test"],
        "candidate_audit_N": 256 + 64 * index,
        "candidate_values_by_shift": {"a": -1.0, "b": -1.0},
        "predicted_local_uncertainty": 1e-3,
        "safety_factor": 2.0,
        "acceptance_threshold": 2e-3,
        "holdout_N": [320 + 64 * index, 384 + 64 * index],
    }


def _plan(count: int) -> dict:
    payload = {
        "schema": HOLDOUT_PLAN_SCHEMA,
        "profile": "0deg_qualification_v5",
        "source_profile": "0deg_pilot_v4",
        "source_artifact_sha256": {"spm": {}, "dwave": {}},
        "target_cache_sha256": {"spm": "x", "dwave": "y"},
        "items": [_item(index) for index in range(count)],
    }
    payload["plan_sha256"] = _digest(payload)
    return payload


def _policy(groups: int = 3) -> dict:
    return build_policy(
        max_concurrent_groups=groups,
        workers_per_group=2,
        parallel_mode_per_group="context",
        memory_budget_gb_per_group=2.0,
        max_context_workers_per_group=2,
        total_worker_budget=groups * 2,
        total_memory_budget_gb=groups * 2.0,
    )


def _record(group: dict) -> dict:
    return {
        "group_id": group["group_id"],
        "group": group["key"],
        "point_count": len(group["items"]),
        "wall_seconds": 0.01,
        "execution_levels": [],
        "stdout_tail": "",
        "stderr_tail": "",
        "results": [
            {
                "identity": list(item["identity"]),
                "reasons": list(item["reasons"]),
                "predicted_local_uncertainty": item["predicted_local_uncertainty"],
                "safety_factor": item["safety_factor"],
                "levels": [],
                "passed": True,
            }
            for item in group["items"]
        ],
        "all_points_passed": True,
    }


def test_policy_rejects_resource_oversubscription() -> None:
    with pytest.raises(ValueError, match="worker product"):
        build_policy(
            max_concurrent_groups=6,
            workers_per_group=4,
            parallel_mode_per_group="context",
            memory_budget_gb_per_group=2.0,
            max_context_workers_per_group=3,
            total_worker_budget=18,
            total_memory_budget_gb=20.0,
        )


def test_execute_runs_groups_concurrently_and_writes_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(6)
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "holdout.json"
    checkpoint = tmp_path / "checkpoint.json"
    _write(plan_path, plan)

    lock = threading.Lock()
    active = 0
    maximum = 0

    def fake_run(group, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return _record(group)

    monkeypatch.setattr(
        "scripts.full_casimir.qualification_holdout._verify_bound_inputs",
        lambda plan, output_root: None,
    )
    monkeypatch.setattr(
        "scripts.full_casimir.qualification_holdout.run_group",
        fake_run,
    )

    report = execute(
        plan_path=plan_path,
        confirm_plan_sha256=plan["plan_sha256"],
        output_root=tmp_path,
        output_path=output,
        checkpoint_path=checkpoint,
        policy=_policy(3),
    )

    assert maximum >= 2
    assert report["all_points_passed"] is True
    assert report["checkpoint_group_count"] == 6
    saved = _read(checkpoint)
    assert saved["completed_group_count"] == 6
    assert output.is_file()


def test_execute_resumes_without_repeating_completed_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(4)
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "holdout.json"
    checkpoint = tmp_path / "checkpoint.json"
    _write(plan_path, plan)
    groups = build_groups(plan)
    policy = _policy(2)
    first = _record(groups[0])
    write_checkpoint(
        checkpoint,
        plan_sha256=plan["plan_sha256"],
        policy=policy,
        total_groups=len(groups),
        completed={groups[0]["group_id"]: first},
    )

    called = []

    def fake_run(group, **kwargs):
        called.append(group["group_id"])
        return _record(group)

    monkeypatch.setattr(
        "scripts.full_casimir.qualification_holdout._verify_bound_inputs",
        lambda plan, output_root: None,
    )
    monkeypatch.setattr(
        "scripts.full_casimir.qualification_holdout.run_group",
        fake_run,
    )

    report = execute(
        plan_path=plan_path,
        confirm_plan_sha256=plan["plan_sha256"],
        output_root=tmp_path,
        output_path=output,
        checkpoint_path=checkpoint,
        policy=policy,
    )

    assert groups[0]["group_id"] not in called
    assert len(called) == len(groups) - 1
    assert report["checkpoint_group_count"] == len(groups)
    loaded = load_checkpoint(
        checkpoint,
        plan_sha256=plan["plan_sha256"],
        policy=policy,
    )
    assert len(loaded) == len(groups)
