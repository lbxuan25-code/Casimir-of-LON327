from __future__ import annotations

import numpy as np
import pytest

from lno327.workflows import cpu_parallel
from lno327.workflows.cpu_parallel import (
    choose_cpu_parallel_plan,
    estimate_context_bytes,
    numpy_array_bytes,
)


def test_auto_prefers_q_axis_when_it_fills_worker_budget(monkeypatch) -> None:
    monkeypatch.setattr(cpu_parallel, "affinity_cpu_count", lambda: 8)
    plan = choose_cpu_parallel_plan(
        mode="auto",
        requested_workers=8,
        context_count=6,
        max_q_tasks_per_context=12,
        estimated_context_bytes=1_000_000,
        memory_budget_gb=1.0,
        q_parallel_supported=True,
    )
    assert plan.total_worker_budget == 8
    assert plan.strategy == "q"
    assert plan.q_workers == 8
    assert plan.context_workers == 1
    assert plan.nested_process_pools is False


def test_auto_uses_context_axis_for_sparse_q_scan_when_memory_allows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cpu_parallel, "affinity_cpu_count", lambda: 8)
    plan = choose_cpu_parallel_plan(
        mode="auto",
        requested_workers=8,
        context_count=6,
        max_q_tasks_per_context=2,
        estimated_context_bytes=100_000_000,
        memory_budget_gb=2.0,
        q_parallel_supported=True,
    )
    assert plan.strategy == "context"
    assert plan.context_workers == 6
    assert plan.q_workers == 1
    assert plan.estimated_process_utilization == 6


def test_context_axis_is_memory_capped_and_auto_falls_back_to_q(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cpu_parallel, "affinity_cpu_count", lambda: 8)
    plan = choose_cpu_parallel_plan(
        mode="auto",
        requested_workers=8,
        context_count=6,
        max_q_tasks_per_context=3,
        estimated_context_bytes=700_000_000,
        memory_budget_gb=1.0,
        q_parallel_supported=True,
    )
    assert plan.memory_limited_context_workers == 1
    assert plan.strategy == "q"
    assert plan.q_workers == 3


def test_forced_q_mode_rejects_platform_without_fork_support() -> None:
    with pytest.raises(ValueError, match="not supported"):
        choose_cpu_parallel_plan(
            mode="q",
            requested_workers=4,
            context_count=3,
            max_q_tasks_per_context=4,
            estimated_context_bytes=1000,
            q_parallel_supported=False,
        )


def test_context_estimate_uses_observed_bytes_and_safety_factor() -> None:
    estimate = estimate_context_bytes(
        point_count=100,
        observed_bytes_per_point=10_000.0,
        safety_factor=2.0,
        fallback_bytes_per_point=1_000.0,
        fixed_overhead_bytes=0,
    )
    assert estimate == 2_000_000


def test_numpy_array_bytes_counts_shared_storage_once() -> None:
    base = np.zeros(128, dtype=np.complex128)
    payload = {"base": base, "view": base[::2], "duplicate": base}
    assert numpy_array_bytes(payload) == base.nbytes
