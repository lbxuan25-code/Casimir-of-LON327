from __future__ import annotations

import json

import pytest

from lno327.response.arbitrary_q_formal_policy import (
    EXECUTION_STRATEGY,
    FORMAL_POLICY_ID,
    THREAD_POLICY_ID,
    config_fingerprint,
    validate_numerical_formal_config,
    validate_performance_formal_config,
)
from validation.commands.matsubara import arbitrary_q_periodic_bz_qualification as qualification
from validation.commands.matsubara.arbitrary_q_periodic_bz_qualification_gate import (
    _authorize_output,
    _load_manifest,
)


def _performance_config() -> dict[str, object]:
    return {
        "pairings": ["spm", "dwave"],
        "N": 128,
        "q_tasks": 8,
        "workers": 8,
        "matsubara_indices": [0, 1, 2, 4, 8],
        "canonical_block_size": 4096,
        "runtime_chunk_sizes": [4096, 16384],
        "minimum_speedup": 4.0,
        "minimum_cpu_wall_ratio": 4.0,
        "maximum_pool_overhead_fraction": 0.05,
        "execution_strategy": EXECUTION_STRATEGY,
        "thread_policy_id": THREAD_POLICY_ID,
    }


def _numerical_config() -> dict[str, object]:
    return {
        "pairings": ["spm", "dwave"],
        "N_values": [256, 384, 512],
        "reference_nk": 1256,
        "reference_order": 384,
        "matsubara_indices": [0, 1, 8],
        "primitive_tolerance": 1e-3,
        "reflection_tolerance": 3e-4,
        "logdet_tolerance": 3e-4,
        "diagonal_observable_tolerance": 1e-3,
        "canonical_block_size": 4096,
        "runtime_chunk_size": 16384,
        "workers": 8,
        "execution_strategy": EXECUTION_STRATEGY,
        "thread_policy_id": THREAD_POLICY_ID,
    }


def _manifest(head: str, *, config: dict[str, object] | None = None) -> dict[str, object]:
    selected = _performance_config() if config is None else dict(config)
    return {
        "schema": "arbitrary-q-performance-preflight-v2",
        "git_head": head,
        "created_at_utc": "2026-07-14T00:00:00+00:00",
        "formal_policy_id": FORMAL_POLICY_ID,
        "formal_policy_passed": True,
        "config_fingerprint": config_fingerprint(selected),
        "exact_command": "python -m validation matsubara arbitrary-q-performance-preflight",
        "hardware": {"hardware_fingerprint": "hardware-test"},
        "actual_threadpool_passed": True,
        "config": selected,
        "arbitrary_q_performance_contract": "formal_preflight_passed",
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
        "passed": True,
    }


def test_frozen_performance_policy_rejects_loosened_values() -> None:
    config = _performance_config()
    config.update(
        {
            "pairings": ["spm"],
            "N": 2,
            "q_tasks": 2,
            "workers": 2,
            "matsubara_indices": [0, 1],
            "minimum_speedup": 0.0,
            "minimum_cpu_wall_ratio": 0.0,
            "maximum_pool_overhead_fraction": 1.0,
        }
    )
    result = validate_performance_formal_config(config)
    assert not result.passed
    assert result.violations


def test_frozen_numerical_policy_rejects_loosened_values() -> None:
    config = _numerical_config()
    config.update(
        {
            "pairings": ["spm"],
            "N_values": [2, 4, 6],
            "reference_nk": 32,
            "reference_order": 16,
            "matsubara_indices": [0, 1],
            "primitive_tolerance": 1.0,
            "reflection_tolerance": 1.0,
            "logdet_tolerance": 1.0,
            "diagonal_observable_tolerance": 1.0,
            "workers": 2,
        }
    )
    result = validate_numerical_formal_config(config)
    assert not result.passed
    assert result.violations


def test_qualification_gate_accepts_only_same_head_formal_manifest(tmp_path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_manifest("abc123")), encoding="utf-8")
    record = _load_manifest(
        path,
        git_head="abc123",
        qualification_config=_numerical_config(),
    )
    assert record["passed"] is True
    assert record["git_head"] == "abc123"
    assert record["contract"] == "formal_preflight_passed"
    assert record["formal_policy_id"] == FORMAL_POLICY_ID


def test_qualification_gate_rejects_stale_or_forged_manifest(tmp_path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_manifest("old-head")), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match"):
        _load_manifest(
            path,
            git_head="new-head",
            qualification_config=_numerical_config(),
        )

    forged = _manifest("new-head")
    forged["config_fingerprint"] = "forged"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(SystemExit, match="fingerprint"):
        _load_manifest(
            path,
            git_head="new-head",
            qualification_config=_numerical_config(),
        )


def test_qualification_gate_rejects_loose_manifest_and_incompatible_execution(tmp_path) -> None:
    path = tmp_path / "preflight.json"
    loose = _performance_config()
    loose["minimum_speedup"] = 0.0
    path.write_text(json.dumps(_manifest("abc123", config=loose)), encoding="utf-8")
    with pytest.raises(SystemExit, match="frozen policy"):
        _load_manifest(
            path,
            git_head="abc123",
            qualification_config=_numerical_config(),
        )

    path.write_text(json.dumps(_manifest("abc123")), encoding="utf-8")
    incompatible = _numerical_config()
    incompatible["workers"] = 4
    with pytest.raises(SystemExit, match="incompatible"):
        _load_manifest(
            path,
            git_head="abc123",
            qualification_config=incompatible,
        )


def test_qualification_gate_rejects_unknown_current_head(tmp_path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_manifest("abc123")), encoding="utf-8")
    with pytest.raises(SystemExit, match="resolvable git HEAD"):
        _load_manifest(
            path,
            git_head="unknown",
            qualification_config=_numerical_config(),
        )


def test_direct_core_result_cannot_authorize_without_public_gate(tmp_path) -> None:
    config = _numerical_config()
    path = tmp_path / "qualification.json"
    diagnostic = {
        "schema": "arbitrary-q-periodic-bz-diagnostic-result-v2",
        "passed": True,
        "formal_policy_id": FORMAL_POLICY_ID,
        "formal_policy_passed": True,
        "config_fingerprint": config_fingerprint(config),
        "arbitrary_q_microscopic_contract": "diagnostic_result_passed",
        "authorization_source": "none_direct_core_cannot_authorize",
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }
    path.write_text(json.dumps(diagnostic), encoding="utf-8")
    before = json.loads(path.read_text(encoding="utf-8"))
    assert before["arbitrary_q_microscopic_contract"] == "diagnostic_result_passed"
    assert before["authorization_source"] == "none_direct_core_cannot_authorize"

    _authorize_output(
        path,
        performance_record={"passed": True},
        numerical_config=config,
    )
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["arbitrary_q_microscopic_contract"] == (
        "qualified_for_diagnostic_outer_integration"
    )
    assert after["formal_authorization_passed"] is True


def test_default_public_numerical_configuration_satisfies_formal_policy() -> None:
    parsed = qualification._args([])
    formal = qualification._formal_config(parsed)
    assert validate_numerical_formal_config(formal).passed
