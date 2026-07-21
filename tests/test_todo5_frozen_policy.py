from __future__ import annotations

import pytest

from lno327.casimir.fixed_chain import DEFAULT_SHIFTS
from lno327.casimir.production import build_full_casimir_config
from lno327.response.arbitrary_q_formal_policy import (
    EXECUTION_STRATEGY,
    THREAD_POLICY_ID,
)
from scripts.full_casimir import scan
from scripts.full_casimir.config import (
    DEFAULT_CANONICAL_BLOCK,
    DEFAULT_CERTIFIER_Q_BATCH_SIZE,
    DEFAULT_LOGDET_ATOL,
    DEFAULT_LOGDET_RTOL,
    DEFAULT_MATSUBARA_CUTOFFS,
    DEFAULT_N_CANDIDATES,
    DEFAULT_OUTER_CUTOFFS_U,
    DEFAULT_RUNTIME_CHUNK,
    DEFAULT_TRANSVERSE_SHIFTS,
    EXECUTION_POLICY_ID,
    LOCAL_EXECUTION_PROFILE,
    MICROSCOPIC_POLICY_ID,
    SERVER_EXECUTION_PROFILE,
    execution_profile,
    microscopic_policy_payload,
    qualification_candidate_payload,
    select_runtime_resources,
)
from scripts.full_casimir.energy import ProductionRunOptions


def _code_identity() -> dict[str, object]:
    return {
        "git_commit": "a" * 40,
        "tracked_worktree_clean": True,
    }


def test_transverse_policy_is_frozen_and_pairing_blind() -> None:
    expected_N = (
        128,
        192,
        256,
        384,
        512,
        640,
        768,
        896,
        1024,
        1152,
        1280,
    )
    assert DEFAULT_N_CANDIDATES == expected_N
    assert DEFAULT_LOGDET_RTOL == pytest.approx(2.0e-3)
    assert DEFAULT_LOGDET_ATOL == pytest.approx(1.0e-6)
    assert DEFAULT_TRANSVERSE_SHIFTS == DEFAULT_SHIFTS

    payload = microscopic_policy_payload()
    assert payload["policy_id"] == MICROSCOPIC_POLICY_ID
    assert payload["status"] == "frozen"
    assert payload["pairing_blind"] is True
    assert tuple(payload["N_candidates"]) == expected_N
    assert tuple(tuple(row) for row in payload["shifts"]) == DEFAULT_SHIFTS
    assert payload["required_consecutive_passes"] == 2


def test_top_level_plan_serializes_frozen_transverse_and_candidate_outer_policy() -> None:
    args = scan._parser().parse_args(
        [
            "plan",
            "--pairings",
            "spm",
            "dwave",
            "--distances-nm",
            "20",
            "--angles-deg",
            "0",
        ]
    )
    plan = scan.build_scan_plan(args, code_identity=_code_identity())
    policy = plan["scientific_policy"]
    microscopic = policy["microscopic"]
    outer = policy["outer_integration"]
    matsubara = policy["matsubara"]

    assert tuple(microscopic["N_candidates"]) == DEFAULT_N_CANDIDATES
    assert microscopic["logdet_rtol"] == pytest.approx(DEFAULT_LOGDET_RTOL)
    assert microscopic["logdet_atol"] == pytest.approx(DEFAULT_LOGDET_ATOL)
    assert tuple(outer["cutoff_u_values"]) == DEFAULT_OUTER_CUTOFFS_U
    assert tuple(matsubara["cutoff_values"]) == DEFAULT_MATSUBARA_CUTOFFS

    candidates = qualification_candidate_payload()
    assert candidates["outer"]["status"].startswith("candidate_pending")
    assert candidates["matsubara"]["status"].startswith("candidate_pending")


def test_full_builder_uses_same_transverse_policy_and_engine_blocks() -> None:
    config = build_full_casimir_config(pairings=("spm", "dwave"))
    point = config.outer_tail_config.joint_config.radial_config.point_config

    assert point.N_candidates == DEFAULT_N_CANDIDATES
    assert point.shifts == DEFAULT_TRANSVERSE_SHIFTS
    assert point.logdet_rtol == pytest.approx(DEFAULT_LOGDET_RTOL)
    assert point.logdet_atol == pytest.approx(DEFAULT_LOGDET_ATOL)
    assert point.required_consecutive_passes == 2
    assert point.canonical_block == DEFAULT_CANONICAL_BLOCK
    assert point.runtime_chunk == DEFAULT_RUNTIME_CHUNK
    assert config.outer_tail_config.cutoff_u_values == DEFAULT_OUTER_CUTOFFS_U
    assert config.matsubara_cutoff_values == DEFAULT_MATSUBARA_CUTOFFS


def test_execution_profiles_freeze_local_and_server_resource_rules() -> None:
    assert EXECUTION_POLICY_ID == "full-casimir-execution-policy-v1"
    assert execution_profile(LOCAL_EXECUTION_PROFILE.name) is LOCAL_EXECUTION_PROFILE
    assert execution_profile(SERVER_EXECUTION_PROFILE.name) is SERVER_EXECUTION_PROFILE

    assert LOCAL_EXECUTION_PROFILE.parallel_mode == "q"
    assert LOCAL_EXECUTION_PROFILE.certifier_q_batch_size == 512
    assert LOCAL_EXECUTION_PROFILE.canonical_block == 4096
    assert LOCAL_EXECUTION_PROFILE.runtime_chunk == 16384
    assert LOCAL_EXECUTION_PROFILE.max_context_workers == 1

    assert SERVER_EXECUTION_PROFILE.parallel_mode == "q"
    assert SERVER_EXECUTION_PROFILE.worker_cap == 0
    resources = select_runtime_resources(
        available_cpus=tuple(range(64)),
        reserve_logical_cpus=SERVER_EXECUTION_PROFILE.reserve_logical_cpus,
        worker_cap=SERVER_EXECUTION_PROFILE.worker_cap,
    )
    assert resources.workers == 62


def test_formal_run_defaults_match_qualified_q_parallel_contract() -> None:
    args = scan._parser().parse_args(
        [
            "run",
            "--plan",
            "plan.json",
            "--confirm-plan-sha256",
            "x",
            "--fresh",
        ]
    )
    assert args.parallel_mode == LOCAL_EXECUTION_PROFILE.parallel_mode
    assert args.certifier_q_batch_size == DEFAULT_CERTIFIER_Q_BATCH_SIZE
    assert args.memory_budget_gb == LOCAL_EXECUTION_PROFILE.memory_budget_gb
    assert args.max_context_workers == LOCAL_EXECUTION_PROFILE.max_context_workers

    options = ProductionRunOptions()
    assert options.parallel_mode == "q"
    assert options.certifier_q_batch_size == 512
    assert options.memory_budget_gb == 16.0
    assert options.max_context_workers == 1

    assert EXECUTION_STRATEGY == (
        "persistent_fork_q_lab_angle_batch_tasks_ordered_parent_collection"
    )
    assert THREAD_POLICY_ID == "single_thread_blas_omp_v1"
