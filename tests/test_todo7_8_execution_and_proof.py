from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
import subprocess

import pytest

from scripts.full_casimir.config import RuntimeResources
from scripts.full_casimir.energy import ProductionRunOptions
from scripts.full_casimir.execution_control import (
    CampaignRunLock,
    build_recovery_report,
)
from scripts.full_casimir.identity import (
    atomic_json,
    build_case_identity,
    case_sidecars,
    finalize_plan,
)
from scripts.full_casimir import run_command
from scripts.full_casimir.reproducibility import (
    build_source_proof,
    verify_reproducibility_bundle,
    write_reproducibility_bundle,
)


def test_live_campaign_owner_blocks_duplicate_execution(tmp_path: Path) -> None:
    first = CampaignRunLock(
        campaign_root=tmp_path,
        campaign_id="campaign-test",
        plan_sha256="a" * 64,
        mode="resume",
        stale_after_seconds=60,
        heartbeat_interval_seconds=60,
    ).acquire()
    try:
        with pytest.raises(RuntimeError, match="live local owner"):
            CampaignRunLock(
                campaign_root=tmp_path,
                campaign_id="campaign-test",
                plan_sha256="a" * 64,
                mode="resume",
                stale_after_seconds=60,
                heartbeat_interval_seconds=60,
            ).acquire()
    finally:
        first.release()
    assert not first.owner_path.exists()


def test_explicit_resume_can_archive_and_take_over_stale_remote_lock(
    tmp_path: Path,
) -> None:
    replacement = CampaignRunLock(
        campaign_root=tmp_path,
        campaign_id="campaign-stale",
        plan_sha256="b" * 64,
        mode="resume",
        stale_after_seconds=1,
        heartbeat_interval_seconds=60,
        take_over_stale=True,
    )
    replacement.lock_root.mkdir(parents=True)
    old_heartbeat = replacement.lock_root / "old.heartbeat.json"
    old_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    atomic_json(
        replacement.owner_path,
        {
            "schema": "full-casimir-campaign-lock-v1",
            "campaign_id": "campaign-stale",
            "plan_sha256": "b" * 64,
            "token": "old-token",
            "hostname": "remote-host.invalid",
            "pid": 12345,
            "mode": "resume",
            "started_at_utc": old_time,
            "heartbeat_path": old_heartbeat.name,
        },
    )
    atomic_json(
        old_heartbeat,
        {
            "schema": "full-casimir-campaign-lock-heartbeat-v1",
            "token": "old-token",
            "heartbeat_at_utc": old_time,
        },
    )

    replacement.acquire()
    try:
        owner = json.loads(replacement.owner_path.read_text(encoding="utf-8"))
        assert owner["token"] == replacement.token
        history = list((replacement.lock_root / "history").glob("*.json"))
        assert len(history) == 1
        archived = json.loads(history[0].read_text(encoding="utf-8"))
        assert archived["previous_owner"]["token"] == "old-token"
    finally:
        replacement.release()


def _recovery_plan() -> dict:
    case_identity = build_case_identity(
        campaign_id="campaign-recovery",
        pairing="spm",
        temperature_K=10.0,
        separation_nm=20.0,
        plate_angles_deg=(0.0, 0.0),
    )
    return {
        "campaign_id": "campaign-recovery",
        "campaign_sha256": "c" * 64,
        "scientific_policy_sha256": "d" * 64,
        "plan_sha256": "e" * 64,
        "code_identity": {"git_commit": "f" * 40},
        "cases": [{"case": "spm_case", "case_identity": case_identity}],
    }


def test_recovery_report_recognizes_last_atomic_cache_checkpoint(
    tmp_path: Path,
) -> None:
    plan = _recovery_plan()
    run = tmp_path / "runs" / "spm_case"
    identity, cache_identity = case_sidecars(
        case_identity=plan["cases"][0]["case_identity"],
        campaign_sha256=plan["campaign_sha256"],
        scientific_policy_sha256=plan["scientific_policy_sha256"],
        git_commit=plan["code_identity"]["git_commit"],
    )
    atomic_json(run / "identity.json", identity)
    atomic_json(run / "cache" / "identity.json", cache_identity)
    atomic_json(
        run / "manifest.json",
        {"status": "running", "attempt_count": 2},
    )
    atomic_json(run / "cache" / "certified_points.json", {"entries": []})
    (run / "cache" / "discarded.tmp").write_text("partial", encoding="utf-8")

    report = build_recovery_report(campaign_dir=tmp_path, plan=plan)

    assert report["resumable_case_count"] == 1
    row = report["cases"][0]
    assert row["checkpoint_usable"] is True
    assert row["manifest_status"] == "running"
    assert row["orphan_temporary_files"] == ["cache/discarded.tmp"]


def test_guarded_run_retries_only_within_explicit_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = {
        "campaign_id": "campaign-retry",
        "plan_sha256": "1" * 64,
        "cases": [],
    }
    resources = RuntimeResources((0, 1), (0,), (1,))
    options = ProductionRunOptions(campaign_root=tmp_path)
    modes: list[str] = []

    def fake_runner(*, plan, mode, resources, options):
        modes.append(mode)
        campaign = options.campaign_root / plan["campaign_id"]
        (campaign / "reports").mkdir(parents=True, exist_ok=True)
        return 1 if len(modes) == 1 else 0

    proof_calls: list[int] = []
    monkeypatch.setattr(
        run_command,
        "write_reproducibility_bundle",
        lambda **kwargs: proof_calls.append(int(kwargs["final_exit_code"])),
    )

    status = run_command.execute_guarded_plan(
        plan=plan,
        mode="fresh",
        resources=resources,
        options=options,
        max_engineering_retries=1,
        retry_delay_seconds=0,
        stale_lock_seconds=60,
        lock_heartbeat_seconds=60,
        take_over_stale_lock=False,
        execution_record={},
        runner=fake_runner,
    )

    assert status == 0
    assert modes == ["fresh", "resume"]
    assert proof_calls == [1, 0]
    assert not (tmp_path / ".locks" / "campaign-retry.lock.json").exists()


def _init_git_repository(path: Path) -> str:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test User"],
        check=True,
    )
    (path / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "source.py"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "source"],
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def test_reproducibility_bundle_links_source_plan_and_artifact_digests(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    commit = _init_git_repository(repository)
    source = build_source_proof(repository_root=repository)
    assert source["git_commit"] == commit

    campaign = tmp_path / "campaign-proof"
    (campaign / "plans").mkdir(parents=True)
    (campaign / "runs" / "case").mkdir(parents=True)
    (campaign / "reports").mkdir(parents=True)
    plan = finalize_plan(
        {
            "schema": "full-casimir-production-plan-v1",
            "campaign_id": "campaign-proof",
            "campaign_sha256": "2" * 64,
            "scientific_policy_sha256": "3" * 64,
            "code_identity": {
                "git_commit": commit,
                "tracked_worktree_clean": True,
            },
            "scientific_policy": {"schema": "policy"},
            "case_count": 0,
            "pairings": [],
            "angles_deg": [],
            "distances_nm": [],
            "temperature_K": 10.0,
            "cases": [],
        }
    )
    atomic_json(campaign / "plans" / f"{plan['plan_sha256']}.json", plan)
    atomic_json(campaign / "campaign.json", {"campaign_id": "campaign-proof"})
    atomic_json(campaign / "policy.json", {"schema": "policy"})
    atomic_json(campaign / "runs" / "case" / "result.json", {"value": 1})
    resources = RuntimeResources((0,), (0,), ())

    write_reproducibility_bundle(
        campaign_dir=campaign,
        plan=plan,
        resources=resources,
        execution_options={"parallel_mode": "serial"},
        final_exit_code=0,
        attempt_count=1,
        repository_root=repository,
    )
    verified = verify_reproducibility_bundle(
        campaign_dir=campaign,
        repository_root=repository,
    )
    assert verified["passed"] is True

    atomic_json(campaign / "runs" / "case" / "result.json", {"value": 2})
    damaged = verify_reproducibility_bundle(
        campaign_dir=campaign,
        repository_root=repository,
    )
    assert damaged["passed"] is False
    assert any("artifact digest mismatch" in value for value in damaged["failures"])
