"""Persistent q-level POSIX-fork execution for vector-adaptive arbitrary-q response."""
from __future__ import annotations

from dataclasses import replace
from multiprocessing import get_all_start_methods, get_context
from multiprocessing.pool import Pool
import os
import pickle
from threading import Lock
from time import perf_counter
from typing import Iterator, Sequence

import numpy as np

from lno327.workflows.arbitrary_q_parallel import (
    QLabAngleTask,
    QLabAngleTaskResult,
    _memory_snapshot,
    actual_threadpool_record,
    thread_environment,
    validate_single_thread_blas_environment,
)
from lno327.workflows.arbitrary_q_vector_adaptive import (
    ArbitraryQVectorAdaptiveOptions,
    ArbitraryQVectorAdaptiveResponseCache,
    HierarchicalMaterialNodeCache,
    integrate_two_plate_angle_batch_vector_adaptive,
    material_node_cache_delta,
    material_node_cache_snapshot,
    prewarm_initial_adaptive_nodes,
)

_ADAPTIVE_FORK_GUARD = Lock()
_ADAPTIVE_FORK_EVALUATOR: "ArbitraryQVectorAdaptiveParallelEvaluator | None" = None


def _fork_evaluate_adaptive_task(task: QLabAngleTask) -> QLabAngleTaskResult:
    evaluator = _ADAPTIVE_FORK_EVALUATOR
    if evaluator is None:
        raise RuntimeError("fork worker has no inherited vector-adaptive evaluator")
    return evaluator._evaluate_local(task)


class ArbitraryQVectorAdaptiveParallelEvaluator:
    """Persistent q-task pool with prewarmed copy-on-write material nodes."""

    def __init__(
        self,
        *,
        node_cache: HierarchicalMaterialNodeCache,
        spec: object,
        ansatz: object,
        pairing: object,
        xi_eV_values: Sequence[float] | np.ndarray,
        temperature_K: float,
        eta_eV: float,
        adaptive_options: ArbitraryQVectorAdaptiveOptions | None = None,
        process_workers: int = 1,
        require_thread_environment: bool = True,
        require_converged: bool = True,
        prewarm_before_fork: bool = True,
    ) -> None:
        self.node_cache = node_cache
        self.spec = spec
        self.ansatz = ansatz
        self.pairing = pairing
        self.xi_eV_values = np.asarray(xi_eV_values, dtype=float)
        self.temperature_K = float(temperature_K)
        self.eta_eV = float(eta_eV)
        self.adaptive_options = adaptive_options or ArbitraryQVectorAdaptiveOptions()
        self.adaptive_options.validate()
        self.process_workers = int(process_workers)
        self.require_converged = bool(require_converged)
        self.prewarm_before_fork = bool(prewarm_before_fork)
        self._pool: Pool | None = None
        self._guard_held = False
        self._closed = False
        self.pool_startup_seconds = 0.0
        self.pool_shutdown_seconds = 0.0
        self.last_evaluate_wall_seconds = 0.0
        self.last_parent_collection_overhead_seconds = 0.0
        self.last_payload_bytes = 0
        self.last_worker_threadpool_all_passed = False
        self.last_worker_threadpool_records: tuple[tuple[dict[str, object], ...], ...] = ()
        self.parent_prewarm: dict[str, object] = {
            "enabled": False,
            "seconds": 0.0,
            "point_requests": 0,
            "unique_nodes_after": int(node_cache.metadata().get("entries", 0)),
        }
        self.last_worker_cache_telemetry: tuple[dict[str, object], ...] = ()
        self.last_worker_cache_final_by_pid: dict[str, dict[str, object]] = {}
        if self.process_workers <= 0:
            raise ValueError("process_workers must be positive")
        if self.process_workers > 1:
            if require_thread_environment:
                validate_single_thread_blas_environment()
            if self.prewarm_before_fork:
                self.parent_prewarm = {
                    "enabled": True,
                    **prewarm_initial_adaptive_nodes(
                        self.node_cache, self.adaptive_options
                    ),
                }
            self._start_pool()

    def _start_pool(self) -> None:
        if "fork" not in get_all_start_methods():
            raise RuntimeError("vector-adaptive process execution requires POSIX fork")
        _ADAPTIVE_FORK_GUARD.acquire()
        self._guard_held = True
        global _ADAPTIVE_FORK_EVALUATOR
        if _ADAPTIVE_FORK_EVALUATOR is not None:
            self._guard_held = False
            _ADAPTIVE_FORK_GUARD.release()
            raise RuntimeError("another vector-adaptive fork evaluator is active")
        _ADAPTIVE_FORK_EVALUATOR = self
        started = perf_counter()
        try:
            self._pool = get_context("fork").Pool(processes=self.process_workers)
        except BaseException:
            _ADAPTIVE_FORK_EVALUATOR = None
            self._guard_held = False
            _ADAPTIVE_FORK_GUARD.release()
            raise
        self.pool_startup_seconds = float(perf_counter() - started)

    def _evaluate_local(self, task: QLabAngleTask) -> QLabAngleTaskResult:
        threadpools, threadpool_passed = actual_threadpool_record()
        if self.process_workers > 1 and not threadpool_passed:
            raise RuntimeError(
                "vector-adaptive worker has a non-single-thread BLAS runtime: "
                f"{threadpools}"
            )
        cache_before = material_node_cache_snapshot(self.node_cache)
        started = perf_counter()
        response_cache = ArbitraryQVectorAdaptiveResponseCache()
        result = integrate_two_plate_angle_batch_vector_adaptive(
            q_lab=task.q_lab,
            theta_1_rad=task.theta_1_rad,
            theta_2_rad_values=task.theta_2_rad_values,
            node_cache=self.node_cache,
            spec=self.spec,
            ansatz=self.ansatz,
            pairing=self.pairing,
            xi_eV_values=self.xi_eV_values,
            temperature_K=self.temperature_K,
            eta_eV=self.eta_eV,
            adaptive_options=self.adaptive_options,
            response_cache=response_cache,
            require_converged=self.require_converged,
        )
        worker_seconds = float(perf_counter() - started)
        cache_after = material_node_cache_snapshot(self.node_cache)
        telemetry = {
            "worker_pid": int(os.getpid()),
            "task_index": int(task.index),
            "cache_delta": material_node_cache_delta(cache_before, cache_after),
            "cache_snapshot_after_task": cache_after,
        }
        result = replace(
            result,
            response_cache_metadata={
                **dict(result.response_cache_metadata),
                "worker_cache_telemetry": telemetry,
            },
        )
        rss, pss = _memory_snapshot()
        provisional = QLabAngleTaskResult(
            index=int(task.index),
            result=result,
            worker_seconds=worker_seconds,
            worker_rss_bytes=rss,
            worker_pss_bytes=pss,
            worker_threadpools=threadpools,
            worker_threadpool_passed=threadpool_passed,
        )
        payload_bytes = len(pickle.dumps(provisional, protocol=pickle.HIGHEST_PROTOCOL))
        return QLabAngleTaskResult(
            index=provisional.index,
            result=provisional.result,
            worker_seconds=provisional.worker_seconds,
            payload_bytes=payload_bytes,
            worker_rss_bytes=rss,
            worker_pss_bytes=pss,
            worker_threadpools=threadpools,
            worker_threadpool_passed=threadpool_passed,
        )

    def evaluate_iter(
        self,
        tasks: Sequence[QLabAngleTask],
        *,
        chunksize: int = 1,
    ) -> Iterator[QLabAngleTaskResult]:
        if self._closed:
            raise RuntimeError("vector-adaptive evaluator is closed")
        ordered = tuple(tasks)
        indices = [task.index for task in ordered]
        if len(indices) != len(set(indices)):
            raise ValueError("q task indices must be unique")
        started = perf_counter()
        pool = self._pool
        iterator = (
            (self._evaluate_local(task) for task in ordered)
            if pool is None
            else pool.imap(_fork_evaluate_adaptive_task, ordered, chunksize=int(chunksize))
        )
        collected_worker_seconds = 0.0
        payload_bytes = 0
        threadpool_records: list[tuple[dict[str, object], ...]] = []
        threadpool_passed: list[bool] = []
        cache_rows: list[dict[str, object]] = []
        for expected_index, result in zip(indices, iterator, strict=True):
            if result.index != expected_index:
                raise RuntimeError("ordered vector-adaptive q-task result index mismatch")
            collected_worker_seconds += float(result.worker_seconds)
            payload_bytes += int(result.payload_bytes)
            threadpool_records.append(tuple(result.worker_threadpools))
            threadpool_passed.append(bool(result.worker_threadpool_passed))
            row = result.result.response_cache_metadata.get("worker_cache_telemetry")
            if isinstance(row, dict):
                cache_rows.append(dict(row))
            yield result
        wall = float(perf_counter() - started)
        self.last_evaluate_wall_seconds = wall
        self.last_payload_bytes = payload_bytes
        ideal_parallel = collected_worker_seconds / max(self.process_workers, 1)
        self.last_parent_collection_overhead_seconds = max(wall - ideal_parallel, 0.0)
        self.last_worker_threadpool_records = tuple(threadpool_records)
        self.last_worker_threadpool_all_passed = bool(threadpool_passed) and all(
            threadpool_passed
        )
        self.last_worker_cache_telemetry = tuple(cache_rows)
        final_by_pid: dict[str, dict[str, object]] = {}
        for row in cache_rows:
            pid = str(int(row["worker_pid"]))
            snapshot = dict(row["cache_snapshot_after_task"])
            current = final_by_pid.get(pid)
            if current is None or int(snapshot.get("entries", 0)) >= int(
                current.get("entries", 0)
            ):
                final_by_pid[pid] = snapshot
        self.last_worker_cache_final_by_pid = final_by_pid

    def evaluate(self, tasks: Sequence[QLabAngleTask]) -> tuple[QLabAngleTaskResult, ...]:
        return tuple(self.evaluate_iter(tasks))

    def close(self, *, terminate: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        started = perf_counter()
        pool = self._pool
        self._pool = None
        if pool is not None:
            if terminate:
                pool.terminate()
            else:
                pool.close()
            pool.join()
        self.pool_shutdown_seconds = float(perf_counter() - started)
        global _ADAPTIVE_FORK_EVALUATOR
        if _ADAPTIVE_FORK_EVALUATOR is self:
            _ADAPTIVE_FORK_EVALUATOR = None
        if self._guard_held:
            self._guard_held = False
            _ADAPTIVE_FORK_GUARD.release()

    def __enter__(self) -> "ArbitraryQVectorAdaptiveParallelEvaluator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(terminate=exc_type is not None)

    def metadata(self) -> dict[str, object]:
        return {
            "execution_strategy": (
                "persistent_posix_fork_q_tasks_vector_adaptive"
                if self.process_workers > 1
                else "serial_q_lab_angle_batch_tasks_vector_adaptive"
            ),
            "quadrature_backend": "vector_adaptive",
            "process_workers": int(self.process_workers),
            "pool_startup_seconds": float(self.pool_startup_seconds),
            "pool_shutdown_seconds": float(self.pool_shutdown_seconds),
            "parent_prewarm": dict(self.parent_prewarm),
            "last_evaluate_wall_seconds": float(self.last_evaluate_wall_seconds),
            "parent_collection_overhead_seconds": float(
                self.last_parent_collection_overhead_seconds
            ),
            "last_payload_bytes": int(self.last_payload_bytes),
            "thread_environment": thread_environment(),
            "worker_actual_threadpool_all_passed": bool(
                self.last_worker_threadpool_all_passed
            ),
            "worker_actual_threadpools": [
                [dict(item) for item in record]
                for record in self.last_worker_threadpool_records
            ],
            "worker_cache_telemetry": [
                dict(row) for row in self.last_worker_cache_telemetry
            ],
            "worker_cache_final_by_pid": {
                pid: dict(snapshot)
                for pid, snapshot in self.last_worker_cache_final_by_pid.items()
            },
            "material_node_cache_parent": self.node_cache.metadata(),
            "adaptive_options": self.adaptive_options.as_dict(),
        }


__all__ = ["ArbitraryQVectorAdaptiveParallelEvaluator"]
