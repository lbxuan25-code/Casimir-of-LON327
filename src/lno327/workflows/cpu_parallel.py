"""Resource-aware single-layer CPU parallel planning.

The numerical workflows in this repository keep BLAS/OpenMP single-threaded and use
exactly one process-parallel layer.  A caller may parallelize either

* q/angle tasks that share one readonly material cache; or
* independent material contexts such as pairing/shift combinations.

The planner chooses between those axes from CPU affinity, available memory, task
multiplicity and the estimated live bytes of one material context.  It never
recommends nested process pools.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Literal

import numpy as np

ParallelMode = Literal["auto", "serial", "q", "context"]
ParallelStrategy = Literal["serial", "q", "context"]

_GIB = 1024**3
_DEFAULT_MEMORY_FRACTION = 0.70


def affinity_cpu_count() -> int:
    """Return CPUs available to this process, honoring scheduler affinity."""

    getter = getattr(os, "sched_getaffinity", None)
    if callable(getter):
        try:
            count = len(getter(0))
        except OSError:
            count = 0
        if count > 0:
            return int(count)
    return max(int(os.cpu_count() or 1), 1)


def available_memory_bytes() -> int:
    """Return currently available memory with dependency-free fallbacks."""

    try:
        import psutil  # type: ignore

        value = int(psutil.virtual_memory().available)
        if value > 0:
            return value
    except (ImportError, OSError, AttributeError, ValueError):
        pass

    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        value = pages * page_size
        if value > 0:
            return value
    except (AttributeError, OSError, ValueError):
        pass
    return 0


def resolve_worker_budget(requested_workers: int) -> int:
    """Resolve ``0`` as automatic CPU-affinity count."""

    requested = int(requested_workers)
    if requested < 0:
        raise ValueError("requested_workers must be non-negative")
    available = affinity_cpu_count()
    return available if requested == 0 else min(requested, available)


def resolve_memory_budget_bytes(memory_budget_gb: float) -> int:
    """Resolve ``0`` as a conservative fraction of available memory."""

    requested = float(memory_budget_gb)
    if not np.isfinite(requested) or requested < 0.0:
        raise ValueError("memory_budget_gb must be finite and non-negative")
    if requested > 0.0:
        return max(int(requested * _GIB), 1)
    available = available_memory_bytes()
    if available <= 0:
        return 0
    return max(int(_DEFAULT_MEMORY_FRACTION * available), 1)


def estimate_context_bytes(
    *,
    point_count: int,
    observed_bytes_per_point: float | None,
    safety_factor: float,
    fallback_bytes_per_point: float = 16_384.0,
    fixed_overhead_bytes: int = 256 * 1024**2,
) -> int:
    """Estimate live bytes for one material context.

    ``observed_bytes_per_point`` should come from exact ndarray accounting of a
    completed material cache.  The fallback is intentionally conservative and is
    used only before a run has produced such telemetry.
    """

    points = int(point_count)
    if points <= 0:
        raise ValueError("point_count must be positive")
    factor = float(safety_factor)
    if not np.isfinite(factor) or factor < 1.0:
        raise ValueError("safety_factor must be finite and at least one")
    fallback = float(fallback_bytes_per_point)
    if not np.isfinite(fallback) or fallback <= 0.0:
        raise ValueError("fallback_bytes_per_point must be finite and positive")
    observed = None if observed_bytes_per_point is None else float(observed_bytes_per_point)
    if observed is not None and (not np.isfinite(observed) or observed <= 0.0):
        raise ValueError("observed_bytes_per_point must be finite and positive")
    per_point = fallback if observed is None else max(observed, fallback / 4.0)
    estimate = float(fixed_overhead_bytes) + factor * per_point * points
    return max(int(np.ceil(estimate)), 1)


def numpy_array_bytes(value: Any) -> int:
    """Count unique NumPy array storage reachable from an object graph."""

    seen_objects: set[int] = set()
    seen_buffers: set[int] = set()

    def visit(item: Any) -> int:
        object_id = id(item)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)

        if isinstance(item, np.ndarray):
            root = item
            while isinstance(root.base, np.ndarray):
                root = root.base
            buffer_id = id(root)
            if buffer_id in seen_buffers:
                return 0
            seen_buffers.add(buffer_id)
            return int(root.nbytes)
        if isinstance(item, dict):
            return sum(visit(key) + visit(element) for key, element in item.items())
        if isinstance(item, (tuple, list, set, frozenset)):
            return sum(visit(element) for element in item)
        if hasattr(item, "__dict__"):
            return visit(vars(item))
        return 0

    return int(visit(value))


@dataclass(frozen=True)
class CPUParallelPlan:
    """One non-nested process-parallel plan for a workload level."""

    requested_mode: ParallelMode
    strategy: ParallelStrategy
    total_worker_budget: int
    context_workers: int
    q_workers: int
    context_count: int
    max_q_tasks_per_context: int
    estimated_context_bytes: int
    memory_budget_bytes: int
    memory_limited_context_workers: int
    max_context_workers: int
    q_parallel_supported: bool
    estimated_process_utilization: int
    estimated_peak_concurrent_context_bytes: int
    nested_process_pools: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_cpu_parallel_plan(
    *,
    mode: ParallelMode,
    requested_workers: int,
    context_count: int,
    max_q_tasks_per_context: int,
    estimated_context_bytes: int,
    memory_budget_gb: float = 0.0,
    max_context_workers: int = 0,
    q_parallel_supported: bool = True,
) -> CPUParallelPlan:
    """Choose q-parallel or material-context-parallel execution.

    Automatic mode prefers the axis that can occupy more processes.  Ties prefer
    q-parallelism because forked q workers share one readonly material cache.
    Context parallelism is capped by the configured memory budget.
    """

    requested_mode = str(mode)
    if requested_mode not in {"auto", "serial", "q", "context"}:
        raise ValueError("mode must be auto, serial, q, or context")
    contexts = int(context_count)
    q_tasks = int(max_q_tasks_per_context)
    estimate = int(estimated_context_bytes)
    context_limit = int(max_context_workers)
    if contexts <= 0:
        raise ValueError("context_count must be positive")
    if q_tasks <= 0:
        raise ValueError("max_q_tasks_per_context must be positive")
    if estimate <= 0:
        raise ValueError("estimated_context_bytes must be positive")
    if context_limit < 0:
        raise ValueError("max_context_workers must be non-negative")

    total = resolve_worker_budget(int(requested_workers))
    memory_budget = resolve_memory_budget_bytes(float(memory_budget_gb))
    memory_cap = contexts if memory_budget == 0 else max(memory_budget // estimate, 1)
    configured_context_cap = total if context_limit == 0 else min(context_limit, total)
    context_workers = max(
        min(contexts, total, configured_context_cap, int(memory_cap)),
        1,
    )
    q_workers = max(min(q_tasks, total), 1) if q_parallel_supported else 1

    if requested_mode == "serial" or total == 1:
        strategy: ParallelStrategy = "serial"
        context_workers = 1
        q_workers = 1
        reason = "serial mode requested or only one CPU is available"
    elif requested_mode == "q":
        if not q_parallel_supported:
            raise ValueError("q parallelism was requested but is not supported")
        strategy = "q" if q_workers > 1 else "serial"
        context_workers = 1
        reason = "q parallelism explicitly requested"
    elif requested_mode == "context":
        strategy = "context" if context_workers > 1 else "serial"
        q_workers = 1
        reason = "material-context parallelism explicitly requested"
    elif context_workers > q_workers:
        strategy = "context"
        q_workers = 1
        reason = (
            "context axis occupies more processes than q axis within the memory budget"
        )
    elif q_workers > 1:
        strategy = "q"
        context_workers = 1
        reason = (
            "q axis is at least as parallel and shares one readonly material cache"
        )
    elif context_workers > 1:
        strategy = "context"
        q_workers = 1
        reason = "q axis is serial but multiple material contexts fit in memory"
    else:
        strategy = "serial"
        context_workers = 1
        q_workers = 1
        reason = "neither workload axis exposes safe process parallelism"

    utilization = (
        context_workers
        if strategy == "context"
        else q_workers
        if strategy == "q"
        else 1
    )
    peak_context_bytes = (
        context_workers * estimate if strategy == "context" else estimate
    )
    return CPUParallelPlan(
        requested_mode=requested_mode,  # type: ignore[arg-type]
        strategy=strategy,
        total_worker_budget=int(total),
        context_workers=int(context_workers),
        q_workers=int(q_workers),
        context_count=contexts,
        max_q_tasks_per_context=q_tasks,
        estimated_context_bytes=estimate,
        memory_budget_bytes=int(memory_budget),
        memory_limited_context_workers=int(memory_cap),
        max_context_workers=int(configured_context_cap),
        q_parallel_supported=bool(q_parallel_supported),
        estimated_process_utilization=int(utilization),
        estimated_peak_concurrent_context_bytes=int(peak_context_bytes),
        nested_process_pools=False,
        reason=reason,
    )


__all__ = [
    "CPUParallelPlan",
    "ParallelMode",
    "affinity_cpu_count",
    "available_memory_bytes",
    "choose_cpu_parallel_plan",
    "estimate_context_bytes",
    "numpy_array_bytes",
    "resolve_memory_budget_bytes",
    "resolve_worker_budget",
]
