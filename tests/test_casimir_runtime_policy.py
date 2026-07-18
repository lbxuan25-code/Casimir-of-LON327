from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lno327.casimir.adaptive_joint_q import _usable_radial_estimate
from lno327.casimir.adaptive_matsubara_tail import (
    _outer_run_usable,
    _scaled_outer_tail_config,
)
from lno327.casimir.adaptive_outer_tail import _joint_run_usable
from lno327.casimir.certified_point_provider import CertifiedOuterQProvider
from lno327.casimir.fixed_chain import (
    FixedCasimirConfig,
    FixedCasimirExecutionError,
    _CertificationRun,
)
from lno327.casimir.production import build_full_casimir_config


def _point(pairing: str, label: str, n: int, value: float) -> dict:
    return {
        "pairing": pairing,
        "q_label": label,
        "n": n,
        "sweet_spot": {
            "status": "established",
            "working_N": 192,
            "audit_N": 256,
            "establishment_mode": "strict_consecutive_adjacent",
        },
        "history": [
            {
                "N": 256,
                "two_plate_logdet_cross_shift": {"passed": True},
                "shifts": {
                    "primary": {
                        "two_plate_logdet": value,
                        "hard_physical_passed": True,
                    }
                },
            }
        ],
    }


def _successful_run(current, manifest, output):
    payload = {
        "schema": "transverse-point-sweet-spot-v4",
        "run_complete": True,
        "all_requested_sweet_spots_established": True,
        "point_results": [
            _point(pairing, label, n, float(index + n + 1))
            for index, label in enumerate(manifest.labels)
            for pairing in current.pairings
            for n in current.matsubara_indices
        ],
    }
    return _CertificationRun(payload, "stdout", "stderr", ("python",))


def test_provider_splits_large_q_requests_into_bounded_batches(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def runner(current, manifest, output):
        calls.append(len(manifest.labels))
        return _successful_run(current, manifest, output)

    q = np.asarray([[float(i), float(i + 1)] for i in range(5)])
    provider = CertifiedOuterQProvider(
        FixedCasimirConfig(matsubara_indices=(0,)),
        cache_path=tmp_path / "points.json",
        runner=runner,
        certifier_q_batch_size=2,
    )
    batch = provider.evaluate(q)

    assert batch.all_established
    assert calls == [2, 2, 1]
    assert batch.certification_batches == 3
    statistics = provider.performance_statistics()
    assert statistics["certification_batches"] == 3
    assert statistics["certification_failed_batches"] == 0
    assert statistics["certification_attempts"] == 3
    assert statistics["certifier_q_batch_size"] == 2
    assert [
        record["status"] for record in statistics["certifier_batch_records"]
    ] == ["succeeded", "succeeded", "succeeded"]


def test_provider_persists_completed_chunks_and_failure_context(
    tmp_path: Path,
) -> None:
    calls = 0

    def runner(current, manifest, output):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FixedCasimirExecutionError("synthetic certifier failure")
        return _successful_run(current, manifest, output)

    cache = tmp_path / "points.json"
    q = np.asarray([[float(i), float(i + 1)] for i in range(5)])
    provider = CertifiedOuterQProvider(
        FixedCasimirConfig(matsubara_indices=(0,)),
        cache_path=cache,
        runner=runner,
        certifier_q_batch_size=2,
    )

    with pytest.raises(
        FixedCasimirExecutionError,
        match=r"chunk=2/3.*synthetic certifier failure",
    ):
        provider.evaluate(q)

    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 2
    telemetry = json.loads(
        cache.with_suffix(".telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["certification_batches"] == 1
    assert telemetry["certification_failed_batches"] == 1
    assert telemetry["certification_attempts"] == 2
    assert telemetry["certifier_batch_records"][-1]["status"] == "failed"
    assert (
        telemetry["certifier_batch_records"][-1]["exception_message"]
        == "synthetic certifier failure"
    )

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("persisted successful chunk must be reusable")

    restored = CertifiedOuterQProvider(
        FixedCasimirConfig(matsubara_indices=(0,)),
        cache_path=cache,
        runner=forbidden_runner,
        certifier_q_batch_size=2,
    )
    replay = restored.evaluate(q[:2])
    assert replay.all_established
    assert replay.new_q_count == 0


def test_controller_layers_preserve_provider_failure_reason() -> None:
    radial = SimpleNamespace(
        all_microscopic_nodes_certified=False,
        termination_reason="point_provider_failure",
        unresolved_points=({"reason": "certifier q batch failed: boom"},),
        status="unresolved",
        radial_converged=False,
    )
    usable, reason = _usable_radial_estimate(radial)
    assert not usable
    assert reason == (
        "point_provider_failure: certifier q batch failed: boom"
    )

    joint = SimpleNamespace(
        all_microscopic_nodes_certified=False,
        termination_reason=f"radial_run_unresolved: {reason}",
    )
    assert _joint_run_usable(joint) == (
        False,
        f"radial_run_unresolved: {reason}",
    )

    outer = SimpleNamespace(
        all_microscopic_nodes_certified=False,
        termination_reason=f"finite_domain_run_unresolved: {reason}",
    )
    assert _outer_run_usable(outer) == (
        False,
        f"finite_domain_run_unresolved: {reason}",
    )


def test_runtime_error_budget_policy_uses_current_term_count() -> None:
    spm = build_full_casimir_config(
        pairings=("spm",),
        total_free_energy_rtol=5e-3,
        total_free_energy_atol_J_m2=1e-12,
    )
    dwave = build_full_casimir_config(
        pairings=("dwave",),
        total_free_energy_rtol=5e-3,
        total_free_energy_atol_J_m2=1e-12,
    )

    assert spm.matsubara_cutoff_values == (1, 3, 7, 11, 15, 23, 31)
    assert spm.finite_matsubara_budget_fraction == pytest.approx(0.7)
    assert spm.matsubara_tail_budget_fraction == pytest.approx(0.3)

    outer = spm.outer_tail_config
    assert outer.finite_domain_budget_fraction == pytest.approx(0.7)
    assert outer.tail_budget_fraction == pytest.approx(0.3)
    assert outer.joint_budget_fraction_within_finite == pytest.approx(0.8)
    assert outer.offset_budget_fraction_within_finite == pytest.approx(0.2)

    spm_joint = outer.joint_config
    dwave_joint = dwave.outer_tail_config.joint_config
    assert spm_joint.radial_budget_fraction == pytest.approx(0.85)
    assert spm_joint.angular_budget_fraction == pytest.approx(0.15)
    assert dwave_joint.radial_budget_fraction == pytest.approx(0.75)
    assert dwave_joint.angular_budget_fraction == pytest.approx(0.25)

    cutoff_1 = _scaled_outer_tail_config(spm, 1)
    cutoff_31 = _scaled_outer_tail_config(spm, 31)
    assert cutoff_1.total_outer_rtol == pytest.approx(5e-3 * 0.7 / 2)
    assert cutoff_31.total_outer_rtol == pytest.approx(5e-3 * 0.7 / 32)
    assert cutoff_1.total_outer_atol_J_m2 == pytest.approx(1e-12 * 0.7 / 2)
