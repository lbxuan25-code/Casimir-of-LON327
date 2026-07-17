"""Persistent POSIX-fork orchestration for arbitrary-q q/angle tasks."""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from multiprocessing.pool import Pool
import os
import pickle
import resource
from threading import Lock
from time import perf_counter
from typing import Iterator, Sequence

import numpy as np

from lno327.response.arbitrary_q_formal_policy import EXECUTION_STRATEGY
from lno327.response.arbitrary_q_material_cache import MaterialGridCache
from lno327.response.ward_validation import PrimitiveWardRHS
from lno327.workflows.arbitrary_q_matsubara import (
    ArbitraryQPeriodicBZResult,
    CrystalResponseCache,
    TwoPlateAngleBatchResult,
    integrate_two_plate_angle_batch,
)

_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def thread_environment() -> dict[str, str]:
    return {
        name: os.environ.get(name, "")
        for name in (*_THREAD_ENV, "OMP_DYNAMIC", "MKL_DYNAMIC")
    }


def validate_single_thread_blas_environment() -> None:
    invalid = {
        name: os.environ.get(name)
        for name in _THREAD_ENV
        if os.environ.get(name) != "1"
    }
    if os.environ.get("OMP_DYNAMIC", "FALSE").upper() not in {"FALSE", "0"}:
        invalid["OMP_DYNAMIC"] = os.environ.get("OMP_DYNAMIC")
    if os.environ.get("MKL_DYNAMIC", "FALSE").upper() not in {"FALSE", "0"}:
        invalid["MKL_DYNAMIC"] = os.environ.get("MKL_DYNAMIC")
    if invalid:
        raise RuntimeError(
            "process-parallel arbitrary-q execution requires single-thread BLAS/OMP; "
            f"invalid environment: {invalid}"
        )


def actual_threadpool_record() -> tuple[tuple[dict[str, object], ...], bool]:
    try:
        from threadpoolctl import threadpool_info  # type: ignore
    except ImportError:
        return (({"error": "threadpoolctl not installed", "num_threads": -1},), False)
    rows = tuple(dict(item) for item in threadpool_info())
    passed = bool(rows) and all(int(row.get("num_threads", -1)) == 1 for row in rows)
    return rows, passed


def _memory_snapshot() -> tuple[int, int]:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.name == "posix":
        rss *= 1024
    pss = 0
    try:
        import psutil  # type: ignore

        info = psutil.Process().memory_full_info()
        rss = max(rss, int(info.rss))
        pss = int(getattr(info, "pss", 0))
    except (ImportError, OSError, AttributeError):
        pass
    return rss, pss


def _response_payload(response: ArbitraryQPeriodicBZResult) -> tuple[object, ...]:
    rhs_payloads = tuple(
        (
            np.asarray(item.left),
            np.asarray(item.right),
            np.asarray(item.q_model),
            float(item.xi_eV),
            float(item.delta0_eV),
            dict(item.metadata),
        )
        for item in response.rhs
    )
    return (
        np.asarray(response.q_model),
        np.asarray(response.xi_eV_values),
        np.asarray(response.packed_primitives),
        response.components,
        rhs_payloads,
        response.operator_ward,
        response.profile,
        response.material_cache_fingerprint,
        dict(response.metadata),
    )


def _restore_response(payload: tuple[object, ...]) -> ArbitraryQPeriodicBZResult:
    (
        q_model,
        xi_values,
        packed,
        components,
        rhs_payloads,
        operator_ward,
        profile,
        fingerprint,
        metadata,
    ) = payload
    rhs = tuple(
        PrimitiveWardRHS(
            left=item[0],
            right=item[1],
            q_model=item[2],
            xi_eV=item[3],
            delta0_eV=item[4],
            metadata=item[5],
        )
        for item in rhs_payloads
    )
    return ArbitraryQPeriodicBZResult(
        q_model=q_model,
        xi_eV_values=xi_values,
        packed_primitives=packed,
        components=components,
        rhs=rhs,
        operator_ward=operator_ward,
        profile=profile,
        material_cache_fingerprint=fingerprint,
        metadata=metadata,
    )


def _restore_task_result(
    index: int,
    q_lab: np.ndarray,
    theta_1: float,
    theta_2: np.ndarray,
    plate_1_payload: tuple[object, ...],
    plate_2_payloads: tuple[tuple[object, ...], ...],
    cache_metadata: dict[str, int | str],
    worker_seconds: float,
    payload_bytes: int,
    worker_rss_bytes: int,
    worker_pss_bytes: int,
    worker_threadpools: tuple[dict[str, object], ...],
    worker_threadpool_passed: bool,
) -> "QLabAngleTaskResult":
    batch = TwoPlateAngleBatchResult(
        q_lab=q_lab,
        theta_1_rad=theta_1,
        theta_2_rad_values=theta_2,
        plate_1=_restore_response(plate_1_payload),
        plate_2=tuple(_restore_response(item) for item in plate_2_payloads),
        response_cache_metadata=cache_metadata,
    )
    return QLabAngleTaskResult(
        index=index,
        result=batch,
        worker_seconds=worker_seconds,
        payload_bytes=payload_bytes,
        worker_rss_bytes=worker_rss_bytes,
        worker_pss_bytes=worker_pss_bytes,
        worker_threadpools=worker_threadpools,
        worker_threadpool_passed=worker_threadpool_passed,
    )


@dataclass(frozen=True)
class QLabAngleTask:
    index: int
    q_lab: np.ndarray
    theta_1_rad: float
    theta_2_rad_values: np.ndarray

    def __post_init__(self) -> None:
        q = np.array(self.q_lab, dtype=float, copy=True)
        theta = np.array(self.theta_2_rad_values, dtype=float, copy=True)
        q.setflags(write=False)
        theta.setflags(write=False)
        object.__setattr__(self, "q_lab", q)
        object.__setattr__(self, "theta_2_rad_values", theta)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError("q_lab must be finite with shape (2,)")
        if theta.ndim != 1 or theta.size == 0 or not np.isfinite(theta).all():
            raise ValueError("theta_2_rad_values must be nonempty and finite")


@dataclass(frozen=True)
class QLabAngleTaskResult:
    index: int
    result: TwoPlateAngleBatchResult
    worker_seconds: float
    payload_bytes: int = 0
    worker_rss_bytes: int = 0
    worker_pss_bytes: int = 0
    worker_threadpools: tuple[dict[str, object], ...] = ()
    worker_threadpool_passed: bool = False

    def __reduce__(self):
        return (
            _restore_task_result,
            (
                int(self.index),
                np.asarray(self.result.q_lab),
                float(self.result.theta_1_rad),
                np.asarray(self.result.theta_2_rad_values),
                _response_payload(self.result.plate_1),
                tuple(_response_payload(item) for item in self.result.plate_2),
                dict(self.result.response_cache_metadata),
                float(self.worker_seconds),
                int(self.payload_bytes),
                int(self.worker_rss_bytes),
                int(self.worker_pss_bytes),
                tuple(dict(item) for item in self.worker_threadpools),
                bool(self.worker_threadpool_passed),
            ),
        )


_FORK_GUARD = Lock()
_FORK_EVALUATOR: "ArbitraryQParallelEvaluator | None" = None


def _fork_evaluate_task(task: QLabAngleTask) -> QLabAngleTaskResult:
    evaluator = _FORK_EVALUATOR
    if evaluator is None:
        raise RuntimeError("fork worker has no inherited arbitrary-q evaluator")
    return evaluator._evaluate_local(task)


class ArbitraryQParallelEvaluator:
    """One persistent q-level pool over an immutable material cache."""

    def __init__(
        self,
        *,
        material_cache: MaterialGridCache,
        spec: object,
        ansatz: object,
        pairing: object,
        xi_eV_values: Sequence[float] | np.ndarray,
        temperature_K: float,
        eta_eV: float,
        process_workers: int = 1,
        canonical_reduction_block_size: int = 4096,
        runtime_chunk_size: int = 16384,
        require_thread_environment: bool = True,
    ) -> None:
        self.material_cache = material_cache
        self.spec = spec
        self.ansatz = ansatz
        self.pairing = pairing
        self.xi_eV_values = np.asarray(xi_eV_values, dtype=float)
        self.temperature_K = float(temperature_K)
        self.eta_eV = float(eta_eV)
        self.process_workers = int(process_workers)
        self.canonical_reduction_block_size = int(canonical_reduction_block_size)
        self.runtime_chunk_size = int(runtime_chunk_size)
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
        if self.process_workers <= 0:
            raise ValueError("process_workers must be positive")
        if self.process_workers > 1:
            if require_thread_environment:
                validate_single_thread_blas_environment()
            self._start_pool()

    def _start_pool(self) -> None:
        if "fork" not in get_all_start_methods():
            raise RuntimeError("arbitrary-q process execution requires POSIX fork")
        _FORK_GUARD.acquire()
        self._guard_held = True
        global _FORK_EVALUATOR
        if _FORK_EVALUATOR is not None:
            self._guard_held = False
            _FORK_GUARD.release()
            raise RuntimeError("another arbitrary-q fork evaluator is active")
        _FORK_EVALUATOR = self
        started = perf_counter()
        try:
            self._pool = get_context("fork").Pool(processes=self.process_workers)
        except BaseException:
            _FORK_EVALUATOR = None
            self._guard_held = False
            _FORK_GUARD.release()
            raise
        self.pool_startup_seconds = float(perf_counter() - started)

    def _evaluate_local(self, task: QLabAngleTask) -> QLabAngleTaskResult:
        threadpools, threadpool_passed = actual_threadpool_record()
        if self.process_workers > 1 and not threadpool_passed:
            raise RuntimeError(
                "arbitrary-q worker has a non-single-thread BLAS runtime: "
                f"{threadpools}"
            )
        started = perf_counter()
        response_cache = CrystalResponseCache()
        result = integrate_two_plate_angle_batch(
            q_lab=task.q_lab,
            theta_1_rad=task.theta_1_rad,
            theta_2_rad_values=task.theta_2_rad_values,
            material_cache=self.material_cache,
            spec=self.spec,
            ansatz=self.ansatz,
            pairing=self.pairing,
            xi_eV_values=self.xi_eV_values,
            temperature_K=self.temperature_K,
            eta_eV=self.eta_eV,
            canonical_reduction_block_size=self.canonical_reduction_block_size,
            runtime_chunk_size=self.runtime_chunk_size,
            response_cache=response_cache,
        )
        worker_seconds = float(perf_counter() - started)
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
            raise RuntimeError("arbitrary-q evaluator is closed")
        ordered = tuple(tasks)
        indices = [task.index for task in ordered]
        if len(indices) != len(set(indices)):
            raise ValueError("q task indices must be unique")
        started = perf_counter()
        pool = self._pool
        iterator = (
            (self._evaluate_local(task) for task in ordered)
            if pool is None
            else pool.imap(_fork_evaluate_task, ordered, chunksize=int(chunksize))
        )
        collected_worker_seconds = 0.0
        payload_bytes = 0
        threadpool_records: list[tuple[dict[str, object], ...]] = []
        threadpool_passed: list[bool] = []
        for expected_index, result in zip(indices, iterator, strict=True):
            if result.index != expected_index:
                raise RuntimeError("ordered q-task result index mismatch")
            collected_worker_seconds += float(result.worker_seconds)
            payload_bytes += int(result.payload_bytes)
            threadpool_records.append(tuple(result.worker_threadpools))
            threadpool_passed.append(bool(result.worker_threadpool_passed))
            yield result
        wall = float(perf_counter() - started)
        self.last_evaluate_wall_seconds = wall
        self.last_payload_bytes = payload_bytes
        ideal_parallel = collected_worker_seconds / max(self.process_workers, 1)
        self.last_parent_collection_overhead_seconds = max(wall - ideal_parallel, 0.0)
        self.last_worker_threadpool_records = tuple(threadpool_records)
        self.last_worker_threadpool_all_passed = bool(threadpool_passed) and all(threadpool_passed)

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
        global _FORK_EVALUATOR
        if _FORK_EVALUATOR is self:
            _FORK_EVALUATOR = None
        if self._guard_held:
            self._guard_held = False
            _FORK_GUARD.release()

    def __enter__(self) -> "ArbitraryQParallelEvaluator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(terminate=exc_type is not None)

    def metadata(self) -> dict[str, object]:
        return {
            "execution_strategy": (
                EXECUTION_STRATEGY if self.process_workers > 1 else "serial_q_lab_angle_batch_tasks"
            ),
            "process_workers": int(self.process_workers),
            "pool_startup_seconds": float(self.pool_startup_seconds),
            "pool_shutdown_seconds": float(self.pool_shutdown_seconds),
            "last_evaluate_wall_seconds": float(self.last_evaluate_wall_seconds),
            "parent_collection_overhead_seconds": float(self.last_parent_collection_overhead_seconds),
            "parent_reduction_seconds": 0.0,
            "parent_reduction_note": "primitive reduction occurs deterministically inside each q worker",
            "last_payload_bytes": int(self.last_payload_bytes),
            "thread_environment": thread_environment(),
            "worker_actual_threadpool_all_passed": bool(self.last_worker_threadpool_all_passed),
            "worker_actual_threadpools": [
                [dict(item) for item in record]
                for record in self.last_worker_threadpool_records
            ],
            "material_cache_fingerprint": self.material_cache.fingerprint,
        }


__all__ = [
    "ArbitraryQParallelEvaluator",
    "QLabAngleTask",
    "QLabAngleTaskResult",
    "actual_threadpool_record",
    "thread_environment",
    "validate_single_thread_blas_environment",
]
