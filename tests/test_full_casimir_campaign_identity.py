from __future__ import annotations

from pathlib import Path

import pytest

from lno327.casimir.run_identity import (
    scientific_config_payload,
    scientific_config_sha256,
)
from scripts.full_casimir.identity import (
    PLAN_SCHEMA,
    POLICY_SCHEMA,
    build_campaign_identity,
    finalize_plan,
    prepare_campaign,
)


def _policy(*, logdet_rtol: float = 0.002) -> dict[str, object]:
    return {
        "schema": POLICY_SCHEMA,
        "model": {"delta0_eV": 0.1, "eta_eV": 1e-8, "degeneracy": 1.0},
        "microscopic": {
            "N_candidates": [128, 192, 256],
            "required_consecutive_passes": 2,
            "logdet_rtol": logdet_rtol,
            "logdet_atol": 1e-6,
        },
    }


def _plan(*, logdet_rtol: float = 0.002, cases=None) -> dict[str, object]:
    policy = _policy(logdet_rtol=logdet_rtol)
    code = {"git_commit": "b" * 40, "tracked_worktree_clean": True}
    campaign = build_campaign_identity(scientific_policy=policy, code_identity=code)
    return finalize_plan(
        {
            "schema": PLAN_SCHEMA,
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign["campaign_sha256"],
            "scientific_policy_sha256": campaign["scientific_policy_sha256"],
            "code_identity": code,
            "scientific_policy": policy,
            "case_count": len(cases or []),
            "cases": list(cases or []),
        }
    )


def _prepare(
    *,
    campaign_root: Path,
    plan: dict[str, object],
    mode: str,
) -> Path:
    return prepare_campaign(
        campaign_root=campaign_root,
        plan=plan,
        mode=mode,
        current_code_identity=plan["code_identity"],
    )


def test_execution_only_config_fields_do_not_change_scientific_identity() -> None:
    first = {
        "point_cache_path": "/a/cache.json",
        "certifier_q_batch_size": 64,
        "outer": {
            "point_config": {
                "temperature_K": 10.0,
                "workers": 8,
                "parallel_mode": "q",
                "memory_budget_gb": 8.0,
                "max_context_workers": 1,
                "N_candidates": [128, 192, 256],
            }
        },
    }
    second = {
        "point_cache_path": "/b/cache.json",
        "certifier_q_batch_size": 512,
        "outer": {
            "point_config": {
                "temperature_K": 10.0,
                "workers": 24,
                "parallel_mode": "context",
                "memory_budget_gb": 48.0,
                "max_context_workers": 6,
                "N_candidates": [128, 192, 256],
            }
        },
    }
    assert scientific_config_payload(first) == scientific_config_payload(second)
    assert scientific_config_sha256(first) == scientific_config_sha256(second)


def test_scientific_change_does_change_resume_identity() -> None:
    first = {"point": {"N_candidates": [128, 192, 256], "workers": 8}}
    second = {"point": {"N_candidates": [128, 192, 384], "workers": 8}}
    assert scientific_config_sha256(first) != scientific_config_sha256(second)


def test_fresh_and_resume_campaign_state_machine(tmp_path: Path) -> None:
    plan = _plan()
    root = tmp_path / "production"
    campaign = _prepare(campaign_root=root, plan=plan, mode="fresh")
    assert campaign.is_dir()
    assert (campaign / "campaign.json").is_file()
    assert (campaign / "policy.json").is_file()
    assert (campaign / "plans" / f"{plan['plan_sha256']}.json").is_file()

    with pytest.raises(FileExistsError):
        _prepare(campaign_root=root, plan=plan, mode="fresh")

    assert _prepare(campaign_root=root, plan=plan, mode="resume") == campaign


def test_resume_requires_existing_campaign(tmp_path: Path) -> None:
    plan = _plan()
    with pytest.raises(FileNotFoundError):
        _prepare(
            campaign_root=tmp_path / "production",
            plan=plan,
            mode="resume",
        )


def test_campaign_execution_requires_the_frozen_git_commit(tmp_path: Path) -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="current Git commit does not match"):
        prepare_campaign(
            campaign_root=tmp_path / "production",
            plan=plan,
            mode="fresh",
            current_code_identity={
                "git_commit": "c" * 40,
                "tracked_worktree_clean": True,
            },
        )


def test_campaign_execution_requires_a_clean_worktree(tmp_path: Path) -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="clean tracked worktree"):
        prepare_campaign(
            campaign_root=tmp_path / "production",
            plan=plan,
            mode="fresh",
            current_code_identity={
                "git_commit": "b" * 40,
                "tracked_worktree_clean": False,
            },
        )


def test_same_campaign_can_register_an_additional_case_plan(tmp_path: Path) -> None:
    first = _plan(cases=[{"case": "a"}])
    root = tmp_path / "production"
    campaign = _prepare(campaign_root=root, plan=first, mode="fresh")
    second = _plan(cases=[{"case": "a"}, {"case": "b"}])
    assert second["campaign_id"] == first["campaign_id"]
    _prepare(campaign_root=root, plan=second, mode="resume")
    assert (campaign / "plans" / f"{second['plan_sha256']}.json").is_file()


def test_policy_change_creates_a_new_campaign_identity() -> None:
    assert _plan(logdet_rtol=0.0015)["campaign_id"] != _plan(
        logdet_rtol=0.002
    )["campaign_id"]
