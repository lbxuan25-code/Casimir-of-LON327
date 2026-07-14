from __future__ import annotations

import json

import pytest

from lno327.response.arbitrary_q_formal_policy import (
    EXECUTION_STRATEGY,
    FORMAL_POLICY_ID,
    MODEL_WORKLOAD_ID,
    OUTER_Q_BATCH_WORKLOAD_ID,
    PERFORMANCE_WORKLOAD_ID,
    QUALIFICATION_AUDIT_WORKLOAD_ID,
    QUALIFICATION_MATRIX_ID,
    QUALIFICATION_PRIMARY_WORKLOAD_ID,
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
from validation.lib.source_tree_provenance import provenance_compatibility


def _provenance(head: str = "abc123") -> dict[str, object]:
    return {
        "git_head": head,
        "git_tree_sha": f"tree-{head}",
        "worktree_clean": True,
        "source_tree_fingerprint": f"source-{head}",
        "tracked_index_fingerprint": f"index-{head}",
    }


def _performance_config() -> dict[str, object]:
    return {
        "performance_workload_id": PERFORMANCE_WORKLOAD_ID,
        "model_workload_id": MODEL_WORKLOAD_ID,
        "workload_classes": [
            OUTER_Q_BATCH_WORKLOAD_ID,
            QUALIFICATION_PRIMARY_WORKLOAD_ID,
            QUALIFICATION_AUDIT_WORKLOAD_ID,
        ],
        "pairings": ["spm", "dwave"],
        "N": 128,
        "q_tasks": 8,
        "workers": 8,
        "qualification_primary_tasks": 4,
        "qualification_primary_workers": 4,
        "qualification_audit_tasks": 1,
        "qualification_audit_workers": 1,
        "matsubara_indices": [0, 1, 2, 4, 8],
        "canonical_block_size": 4096,
        "runtime_chunk_sizes": [4096, 16384],
        "minimum_speedup": 4.0,
        "minimum_cpu_wall_ratio": 4.0,
        "maximum_pool_overhead_fraction": 0.05,
        "comparison_atol": 2e-12,
        "comparison_rtol": 2e-11,
        "temperature_K": 10.0,
        "delta0_eV": 0.1,
        "eta_eV": 1e-8,
        "execution_strategy": EXECUTION_STRATEGY,
        "thread_policy_id": THREAD_POLICY_ID,
    }


def _numerical_config() -> dict[str, object]:
    return {
        "qualification_matrix_id": QUALIFICATION_MATRIX_ID,
        "model_workload_id": MODEL_WORKLOAD_ID,
        "pairings": ["spm", "dwave"],
        "N_values": [256, 384, 512],
        "reference_nk": 1256,
        "reference_order": 384,
        "reference_panel_count": 16,
        "reference_workers": 8,
        "reference_task_size": 4,
        "matsubara_indices": [0, 1, 8],
        "primitive_tolerance": 1e-3,
        "primitive_atol": 1e-12,
        "reflection_tolerance": 3e-4,
        "reflection_atol": 1e-12,
        "logdet_tolerance": 3e-4,
        "logdet_atol": 1e-14,
        "diagonal_observable_tolerance": 1e-3,
        "diagonal_observable_atol": 1e-12,
        "ward_tolerance": 1e-7,
        "ward_absolute_tolerance": 1e-12,
        "temperature_K": 10.0,
        "delta0_eV": 0.1,
        "eta_eV": 1e-8,
        "separation_nm": 20.0,
        "canonical_block_size": 4096,
        "runtime_chunk_size": 16384,
        "primary_workers": 4,
        "audit_workers": 1,
        "execution_strategy": EXECUTION_STRATEGY,
        "thread_policy_id": THREAD_POLICY_ID,
    }


def _workload(identifier: str, *, workers: int) -> dict[str, object]:
    return {
        "workload_id": identifier,
        "task_count": 8 if identifier == OUTER_Q_BATCH_WORKLOAD_ID else (4 if workers > 1 else 1),
        "workers": workers,
        "speedup": 4.5 if workers > 1 else 1.0,
        "pool_overhead_fraction": 0.01 if workers > 1 else 0.0,
        "parallel_metadata": {
            "process_workers": workers,
            "pool_startup_seconds": 0.01 if workers > 1 else 0.0,
            "pool_shutdown_seconds": 0.01 if workers > 1 else 0.0,
        },
        "passed": True,
    }


def _manifest(
    head: str = "abc123",
    *,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    selected = _performance_config() if config is None else dict(config)
    workloads = [
        _workload(OUTER_Q_BATCH_WORKLOAD_ID, workers=8),
        _workload(QUALIFICATION_PRIMARY_WORKLOAD_ID, workers=4),
        _workload(QUALIFICATION_AUDIT_WORKLOAD_ID, workers=1),
    ]
    return {
        "schema": "arbitrary-q-performance-preflight-v3",
        **_provenance(head),
        "created_at_utc": "2026-07-14T00:00:00+00:00",
        "formal_policy_id": FORMAL_POLICY_ID,
        "formal_policy_passed": True,
        "config_fingerprint": config_fingerprint(selected),
        "exact_command": "python -m validation matsubara arbitrary-q-performance-preflight",
        "hardware": {"hardware_fingerprint": "hardware-test"},
        "actual_threadpool_passed": True,
        "config": selected,
        "pairings": [
            {"pairing": "spm", "records": [{"workloads": workloads, "passed": True}]},
            {"pairing": "dwave", "records": [{"workloads": workloads, "passed": True}]},
        ],
        "metric_passed": True,
        "diagnostic_nonformal_requested": False,
        "arbitrary_q_performance_contract": "formal_preflight_passed",
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
        "passed": True,
    }


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("comparison_atol", 1e30),
        ("comparison_rtol", 1e30),
        ("temperature_K", 11.0),
        ("delta0_eV", 0.2),
        ("eta_eV", 1e-4),
        ("performance_workload_id", "forged"),
        ("qualification_primary_workers", 8),
        ("qualification_audit_workers", 8),
    ],
)
def test_performance_policy_rejects_every_remaining_bypass(field: str, bad: object) -> None:
    config = _performance_config()
    config[field] = bad
    result = validate_performance_formal_config(config)
    assert not result.passed
    assert any(field in violation for violation in result.violations)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("primitive_atol", 1e20),
        ("reflection_atol", 1e20),
        ("logdet_atol", 1e20),
        ("diagonal_observable_atol", 1e20),
        ("ward_tolerance", 1.0),
        ("ward_absolute_tolerance", 1.0),
        ("temperature_K", 12.0),
        ("delta0_eV", 0.2),
        ("eta_eV", 1e-4),
        ("separation_nm", 2000.0),
        ("reference_panel_count", 8),
        ("qualification_matrix_id", "forged"),
    ],
)
def test_numerical_policy_rejects_every_remaining_bypass(field: str, bad: object) -> None:
    config = _numerical_config()
    config[field] = bad
    result = validate_numerical_formal_config(config)
    assert not result.passed
    assert any(field in violation for violation in result.violations)


def test_gate_accepts_only_clean_identical_source_and_all_workloads(tmp_path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    record = _load_manifest(
        path,
        current_provenance=_provenance(),
        qualification_config=_numerical_config(),
    )
    assert record["passed"] is True
    assert record["git_tree_sha"] == "tree-abc123"
    assert set(record["workload_evidence"]) == {
        OUTER_Q_BATCH_WORKLOAD_ID,
        QUALIFICATION_PRIMARY_WORKLOAD_ID,
        QUALIFICATION_AUDIT_WORKLOAD_ID,
    }


def test_gate_rejects_dirty_stale_or_forged_source(tmp_path) -> None:
    path = tmp_path / "preflight.json"
    dirty = _manifest()
    dirty["worktree_clean"] = False
    path.write_text(json.dumps(dirty), encoding="utf-8")
    with pytest.raises(SystemExit, match="source tree"):
        _load_manifest(
            path,
            current_provenance=_provenance(),
            qualification_config=_numerical_config(),
        )

    path.write_text(json.dumps(_manifest("old")), encoding="utf-8")
    with pytest.raises(SystemExit, match="source tree"):
        _load_manifest(
            path,
            current_provenance=_provenance("new"),
            qualification_config=_numerical_config(),
        )

    forged = _manifest()
    forged["config_fingerprint"] = "forged"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(SystemExit, match="fingerprint"):
        _load_manifest(
            path,
            current_provenance=_provenance(),
            qualification_config=_numerical_config(),
        )


def test_gate_rejects_missing_workload_or_unmeasured_shutdown(tmp_path) -> None:
    path = tmp_path / "preflight.json"
    missing = _manifest()
    for pairing in missing["pairings"]:  # type: ignore[index]
        pairing["records"][0]["workloads"] = [  # type: ignore[index]
            row
            for row in pairing["records"][0]["workloads"]  # type: ignore[index]
            if row["workload_id"] != QUALIFICATION_AUDIT_WORKLOAD_ID
        ]
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(SystemExit, match="lacks required workload"):
        _load_manifest(
            path,
            current_provenance=_provenance(),
            qualification_config=_numerical_config(),
        )

    shutdown = _manifest()
    workload = shutdown["pairings"][0]["records"][0]["workloads"][0]  # type: ignore[index]
    workload["parallel_metadata"]["pool_shutdown_seconds"] = 0.0  # type: ignore[index]
    path.write_text(json.dumps(shutdown), encoding="utf-8")
    with pytest.raises(SystemExit, match="shutdown"):
        _load_manifest(
            path,
            current_provenance=_provenance(),
            qualification_config=_numerical_config(),
        )


def test_provenance_compatibility_detects_tree_and_cleanliness() -> None:
    assert not provenance_compatibility(_provenance(), _provenance())
    changed = _provenance()
    changed["git_tree_sha"] = "other"
    assert any("git_tree_sha" in item for item in provenance_compatibility(_provenance(), changed))
    dirty = _provenance()
    dirty["worktree_clean"] = False
    assert any("clean" in item for item in provenance_compatibility(_provenance(), dirty))


def test_direct_core_result_requires_public_clean_gate(tmp_path) -> None:
    config = _numerical_config()
    provenance = _provenance()
    path = tmp_path / "qualification.json"
    diagnostic = {
        "schema": "arbitrary-q-periodic-bz-diagnostic-result-v3",
        **provenance,
        "passed": True,
        "formal_evidence_eligible": True,
        "actual_threadpool_passed": True,
        "diagnostic_nonformal_requested": False,
        "formal_policy_id": FORMAL_POLICY_ID,
        "formal_policy_passed": True,
        "config_fingerprint": config_fingerprint(config),
        "arbitrary_q_microscopic_contract": "diagnostic_result_passed",
        "authorization_source": "none_direct_core_execution_never_authorizes_outer_integration",
        "numerical_q_coverage": {
            "principal_supported_domain_is_not_claimed_as_qualified": True,
            "qualified_outer_q_envelope_established": False,
        },
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }
    path.write_text(json.dumps(diagnostic), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["arbitrary_q_microscopic_contract"] == "diagnostic_result_passed"

    _authorize_output(
        path,
        performance_record={"passed": True, **provenance},
        numerical_config=config,
        current_provenance=provenance,
    )
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["arbitrary_q_microscopic_contract"] == "qualified_for_diagnostic_outer_integration"
    assert after["formal_authorization_passed"] is True
    assert after["valid_for_casimir_input"] is False


def test_default_public_numerical_configuration_satisfies_formal_policy() -> None:
    parsed = qualification._args([])
    formal = qualification._formal_config(parsed)
    assert validate_numerical_formal_config(formal).passed
