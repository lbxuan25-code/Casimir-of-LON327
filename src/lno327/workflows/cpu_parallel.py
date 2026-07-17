"""Resource-aware single-pool CPU parallel planning.

Every numerical process keeps BLAS/OpenMP single-threaded. The planner exposes
three non-nested execution shapes:

* ``q``: one readonly material context, many q tasks;
* ``context``: independent material contexts, one task per context;
* ``wave``: a memory-safe wave of parent-built readonly material contexts and one
  forked pool over flattened ``(context, q)`` work units.

Automatic mode chooses the shape with the highest estimated process utilization.
Ties prefer the simpler/smaller-memory shape in the order q, context, wave.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from typing import Any, Literal

import numpy as np

ParallelMode = Literal["auto", "serial", "q", "context", "wave"]
ParallelStrategy = Literal["serial", "q", "context", "wave"]

_GIB = 1024**3
_DEFAULT_MEMORY_FRACTION = 0.70


def affinity_cpu_count() -> int:
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
    requested = int(requested_workers)
    if requested < 0:
        raise ValueError("requested_workers must be non-negative")
    available = affinity_cpu_count()
    return available if requested == 0 else min(requested, available)


def resolve_memory_budget_bytes(memory_budget_gb: float) -> int:
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
    """Count unique NumPy storage reachable from an object graph."""

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
    requested_mode: ParallelMode
    strategy: ParallelStrategy
    total_worker_budget: int
    context_workers: int
    q_workers: int
    flat_workers: int
    context_count: int
    max_q_tasks_per_context: int
    total_flat_tasks: int
    estimated_context_bytes: int
    memory_budget_bytes: int
    memory_limited_context_workers: int
    max_context_workers: int
    wave_count: int
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
    total_flat_tasks: int | None = None,
    memory_budget_gb: float = 0.0,
    max_context_workers: int = 0,
    q_parallel_supported: bool = True,
) -> CPUParallelPlan:
    requested_mode = str(mode)
    if requested_mode not in {"auto", "serial", "q", "context", "wave"}:
        raise ValueError("mode must be auto, serial, q, context, or wave")

    contexts = int(context_count)
    q_tasks = int(max_q_tasks_per_context)
    estimate = int(estimated_context_bytes)
    context_limit = int(max_context_workers)
    flat_tasks = contexts * q_tasks if total_flat_tasks is None else int(total_flat_tasks)
    if contexts <= 0:
        raise ValueError("context_count must be positive")
    if q_tasks <= 0:
        raise ValueError("max_q_tasks_per_context must be positive")
    if flat_tasks <= 0:
        raise ValueError("total_flat_tasks must be positive")
    if estimate <= 0:
        raise ValueError("estimated_context_bytes must be positive")
    if context_limit < 0:
        raise ValueError("max_context_workers must be non-negative")

    total = resolve_worker_budget(int(requested_workers))
    memory_budget = resolve_memory_budget_bytes(float(memory_budget_gb))
    memory_cap = contexts if memory_budget == 0 else max(memory_budget // estimate, 1)
    configured_context_cap = total if context_limit == 0 else min(context_limit, total)
    contexts_per_wave = max(
        min(contexts, total, configured_context_cap, int(memory_cap)),
        1,
    )
    context_utilization = min(contexts_per_wave, total)

    # q mode evaluates one identical-frequency group at a time, so its useful
    # concurrency is the largest number of q labels inside any one group.
    q_utilization = min(q_tasks, total) if q_parallel_supported else 1

    # Wave mode flattens *all* groups in every live context into one pool. When
    # active frequencies differ by q label, a context can contain several groups
    # even though each individual group has only one q label. Estimate the live
    # wave capacity from total flattened tasks per context, not only the largest
    # same-frequency group. The executor still caps workers by the exact number of
    # tasks in each actual wave, so this estimate cannot oversubscribe work.
    flat_tasks_per_context = max(
        q_tasks,
        int(math.ceil(flat_tasks / contexts)),
    )
    wave_task_capacity = min(
        flat_tasks,
        contexts_per_wave * flat_tasks_per_context,
    )
    wave_utilization = min(wave_task_capacity, total) if q_parallel_supported else 1

    if requested_mode == "serial" or total == 1:
        strategy: ParallelStrategy = "serial"
        context_workers = 1
        q_workers = 1
        flat_workers = 1
        reason = "serial mode requested or only one CPU is available"
    elif requested_mode == "q":
        if not q_parallel_supported:
            raise ValueError("q parallelism was requested but fork is unavailable")
        strategy = "q" if q_utilization > 1 else "serial"
        context_workers = 1
        q_workers = q_utilization
        flat_workers = q_workers
        reason = "q parallelism explicitly requested"
    elif requested_mode == "context":
        strategy = "context" if context_utilization > 1 else "serial"
        context_workers = context_utilization
        q_workers = 1
        flat_workers = context_workers
        reason = "material-context parallelism explicitly requested"
    elif requested_mode == "wave":
        if not q_parallel_supported:
            raise ValueError("wave parallelism requires POSIX fork")
        strategy = "wave" if wave_utilization > 1 else "serial"
        context_workers = contexts_per_wave
        q_workers = wave_utilization
        flat_workers = wave_utilization
        reason = "single-pool context-wave/q-task parallelism explicitly requested"
    else:
        candidates: list[tuple[int, int, ParallelStrategy]] = [
            (q_utilization, 3, "q"),
            (context_utilization, 2, "context"),
            (wave_utilization, 1, "wave"),
            (1, 0, "serial"),
        ]
        _, _, strategy = max(candidates, key=lambda item: (item[0], item[1]))
        if strategy == "q":
            context_workers = 1
            q_workers = q_utilization
            flat_workers = q_workers
            reason = "q axis gives the best utilization and shares one readonly cache"
        elif strategy == "context":
            context_workers = context_utilization
            q_workers = 1
            flat_workers = context_workers
            reason = "context axis gives the best utilization within the memory budget"
        elif strategy == "wave":
            context_workers = contexts_per_wave
            q_workers = wave_utilization
            flat_workers = wave_utilization
            reason = (
                "flattened context-wave/q tasks occupy more processes than either "
                "axis alone within the memory budget"
            )
        else:
            context_workers = 1
            q_workers = 1
            flat_workers = 1
            reason = "the workload exposes no safe process parallelism"

    utilization = (
        flat_workers
        if strategy == "wave"
        else context_workers
        if strategy == "context"
        else q_workers
        if strategy == "q"
        else 1
    )
    live_contexts = context_workers if strategy in {"context", "wave"} else 1
    wave_count = (
        int(math.ceil(contexts / max(context_workers, 1)))
        if strategy == "wave"
        else 1
    )
    return CPUParallelPlan(
        requested_mode=requested_mode,  # type: ignore[arg-type]
        strategy=strategy,
        total_worker_budget=int(total),
        context_workers=int(context_workers),
        q_workers=int(q_workers),
        flat_workers=int(flat_workers),
        context_count=contexts,
        max_q_tasks_per_context=q_tasks,
        total_flat_tasks=flat_tasks,
        estimated_context_bytes=estimate,
        memory_budget_bytes=int(memory_budget),
        memory_limited_context_workers=int(memory_cap),
        max_context_workers=int(configured_context_cap),
        wave_count=int(wave_count),
        q_parallel_supported=bool(q_parallel_supported),
        estimated_process_utilization=int(utilization),
        estimated_peak_concurrent_context_bytes=int(live_contexts * estimate),
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
