"""Reusable complete-orbit d-wave primitive evaluator with stage profiling."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from multiprocessing.pool import Pool
from threading import Lock
import time
from typing import Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.response.finite_q_material_workspace_batched import (
    precompute_finite_q_material_workspace_batched,
)
from lno327.response.finite_q_optimized import _vectorized_kubo_factors
from lno327.response.finite_q_q_workspace_batched import (
    precompute_finite_q_q_workspace_batched,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.dwave_positive_orbit_adaptive import _pack_orbit_primitives


@dataclass(frozen=True)
class DWaveOrbitEvaluatorProfile:
    callbacks: int
    complete_orbit_points: int
    material_workspace_implementation: str
    q_workspace_implementation: str
    material_workspace_seconds: float
    q_workspace_seconds: float
    kubo_factor_seconds: float
    kubo_contraction_seconds: float
    primitive_packing_seconds: float

    @property
    def total_seconds(self) -> float:
        """Return summed worker-seconds across all callbacks."""

        return float(
            self.material_workspace_seconds
            + self.q_workspace_seconds
            + self.kubo_factor_seconds
            + self.kubo_contraction_seconds
            + self.primitive_packing_seconds
        )

    @property
    def seconds_per_callback(self) -> float:
        return self.total_seconds / max(int(self.callbacks), 1)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "callbacks": int(self.callbacks),
            "complete_orbit_points": int(self.complete_orbit_points),
            "material_workspace_implementation": str(
                self.material_workspace_implementation
            ),
            "q_workspace_implementation": str(self.q_workspace_implementation),
            "material_workspace_seconds": float(self.material_workspace_seconds),
            "q_workspace_seconds": float(self.q_workspace_seconds),
            "kubo_factor_seconds": float(self.kubo_factor_seconds),
            "kubo_contraction_seconds": float(self.kubo_contraction_seconds),
            "primitive_packing_seconds": float(self.primitive_packing_seconds),
            "total_seconds": self.total_seconds,
            "seconds_per_callback": self.seconds_per_callback,
        }


@dataclass(frozen=True)
class _DWaveOrbitEvaluationMetrics:
    complete_orbit_points: int
    material_workspace_implementation: str
    q_workspace_implementation: str
    material_workspace_seconds: float
    q_workspace_seconds: float
    kubo_factor_seconds: float
    kubo_contraction_seconds: float
    primitive_packing_seconds: float


_FORK_PROCESS_GUARD = Lock()
_FORK_PROCESS_EVALUATOR: DWaveOrbitPrimitiveEvaluator | None = None


def _fork_process_evaluate(
    points: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, _DWaveOrbitEvaluationMetrics]:
    evaluator = _FORK_PROCESS_EVALUATOR
    if evaluator is None:
        raise RuntimeError("forked orbit worker has no inherited evaluator")
    return evaluator._evaluate_local(points, weights)


class DWaveOrbitPrimitiveEvaluator:
    """Evaluate complete orbits and aggregate stage profiling.

    Serial calls execute directly.  With ``process_workers > 1``, a POSIX-fork pool
    is created before the transverse submission threads start.  Each child therefore
    inherits the immutable model/evaluator configuration by copy-on-write and returns
    only the packed primitive vector plus lightweight timing metadata.  The parent
    remains responsible for profile aggregation and the Gauss integrator remains
    responsible for original-node-order Kahan reduction.
    """

    def __init__(
        self,
        *,
        spec: object,
        ansatz: object,
        pairing: object,
        xi_eV_values: Sequence[float] | np.ndarray,
        temperature_K: float,
        eta_eV: float,
        nk: int,
        mx: int,
        my: int,
        process_workers: int = 1,
    ) -> None:
        xi_values = np.asarray(xi_eV_values, dtype=float)
        if xi_values.ndim != 1 or xi_values.size == 0:
            raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
        if not np.isfinite(xi_values).all() or np.any(xi_values <= 0.0):
            raise ValueError("all xi_eV_values must be finite and positive")
        if getattr(ansatz, "name", None) != "dwave":
            raise ValueError("complete-orbit primitive evaluator is currently d-wave only")
        if getattr(ansatz, "phase_vertex", None) != "bond_endpoint_gauge":
            raise ValueError("d-wave primitive evaluator requires bond_endpoint_gauge")
        if int(nk) <= 0 or (int(mx) == 0 and int(my) == 0):
            raise ValueError("nk must be positive and q grid indices must be nonzero")
        workers = int(process_workers)
        if workers <= 0:
            raise ValueError("process_workers must be positive")

        self.spec = spec
        self.ansatz = ansatz
        self.pairing = pairing
        self.xi_values = np.array(xi_values, copy=True)
        self.xi_values.setflags(write=False)
        self.q_model = (2.0 * np.pi / float(nk)) * np.asarray(
            [int(mx), int(my)], dtype=float
        )
        self.q_model.setflags(write=False)
        self.base_config = KuboConfig.from_kelvin(
            omega_eV=float(self.xi_values[0]),
            temperature_K=float(temperature_K),
            eta_eV=float(eta_eV),
            output_si=False,
        )
        self.options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
        self.process_workers = workers

        self._profile_lock = Lock()
        self._submit_lock = Lock()
        self._callbacks = 0
        self._complete_orbit_points = 0
        self._material_workspace_implementation = "not_evaluated"
        self._q_workspace_implementation = "not_evaluated"
        self._material_workspace_seconds = 0.0
        self._q_workspace_seconds = 0.0
        self._kubo_factor_seconds = 0.0
        self._kubo_contraction_seconds = 0.0
        self._primitive_packing_seconds = 0.0
        self._process_pool: Pool | None = None
        self._fork_guard_held = False
        self._closed = False

        if workers > 1:
            self._start_process_pool()

    @property
    def parallel_execution_strategy(self) -> str:
        if self.process_workers > 1:
            return "fork_process_transverse_nodes_ordered_parent_reduction"
        return "serial_transverse_nodes"

    def _start_process_pool(self) -> None:
        if "fork" not in get_all_start_methods():
            raise RuntimeError(
                "multi-process d-wave orbit evaluation requires POSIX fork; "
                "use process_workers=1 on this platform"
            )

        _FORK_PROCESS_GUARD.acquire()
        self._fork_guard_held = True
        global _FORK_PROCESS_EVALUATOR
        if _FORK_PROCESS_EVALUATOR is not None:
            self._fork_guard_held = False
            _FORK_PROCESS_GUARD.release()
            raise RuntimeError("another forked orbit evaluator is already active")
        _FORK_PROCESS_EVALUATOR = self

        try:
            # multiprocessing.Pool starts all fork workers eagerly, before the generic
            # transverse ThreadPoolExecutor is created by the Gauss integrator.
            self._process_pool = get_context("fork").Pool(
                processes=self.process_workers
            )
        except BaseException:
            _FORK_PROCESS_EVALUATOR = None
            self._fork_guard_held = False
            _FORK_PROCESS_GUARD.release()
            raise

    def _evaluate_local(
        self,
        points: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, _DWaveOrbitEvaluationMetrics]:
        point_array = np.asarray(points, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        if point_array.ndim != 2 or point_array.shape[1] != 2:
            raise ValueError("complete-orbit points must have shape (n,2)")
        if weight_array.shape != (point_array.shape[0],):
            raise ValueError("complete-orbit weights have incompatible shape")

        started = time.perf_counter()
        material = precompute_finite_q_material_workspace_batched(
            self.spec,
            self.ansatz,
            point_array,
            weight_array,
            self.base_config,
            self.pairing,
            self.options,
        )
        material_seconds = time.perf_counter() - started
        material_implementation = str(
            material.metadata.get("material_workspace_implementation", "unknown")
        )

        started = time.perf_counter()
        workspace = precompute_finite_q_q_workspace_batched(material, self.q_model)
        q_workspace_seconds = time.perf_counter() - started
        q_implementation = str(
            workspace.metadata.get("q_workspace_implementation", "unknown")
        )

        started = time.perf_counter()
        raw_factors = _vectorized_kubo_factors(workspace, self.xi_values)
        kubo_factor_seconds = time.perf_counter() - started

        started = time.perf_counter()
        weighted = (
            0.5
            * workspace.material.k_weights[None, :, None, None]
            * raw_factors
        )
        blocks = np.einsum(
            "xkmn,kamn,kbmn->xab",
            weighted,
            workspace.left_vertices_band,
            np.conjugate(workspace.right_vertices_band),
            optimize=True,
        )
        kubo_contraction_seconds = time.perf_counter() - started

        started = time.perf_counter()
        packed = _pack_orbit_primitives(workspace=workspace, blocks=blocks)
        primitive_packing_seconds = time.perf_counter() - started

        metrics = _DWaveOrbitEvaluationMetrics(
            complete_orbit_points=int(point_array.shape[0]),
            material_workspace_implementation=material_implementation,
            q_workspace_implementation=q_implementation,
            material_workspace_seconds=float(material_seconds),
            q_workspace_seconds=float(q_workspace_seconds),
            kubo_factor_seconds=float(kubo_factor_seconds),
            kubo_contraction_seconds=float(kubo_contraction_seconds),
            primitive_packing_seconds=float(primitive_packing_seconds),
        )
        return np.asarray(packed, dtype=complex), metrics

    def _record_metrics(self, metrics: _DWaveOrbitEvaluationMetrics) -> None:
        with self._profile_lock:
            if self._material_workspace_implementation not in {
                "not_evaluated",
                metrics.material_workspace_implementation,
            }:
                raise RuntimeError(
                    "material workspace implementation changed across callbacks"
                )
            if self._q_workspace_implementation not in {
                "not_evaluated",
                metrics.q_workspace_implementation,
            }:
                raise RuntimeError("q workspace implementation changed across callbacks")
            self._material_workspace_implementation = (
                metrics.material_workspace_implementation
            )
            self._q_workspace_implementation = metrics.q_workspace_implementation
            self._material_workspace_seconds += metrics.material_workspace_seconds
            self._q_workspace_seconds += metrics.q_workspace_seconds
            self._kubo_factor_seconds += metrics.kubo_factor_seconds
            self._kubo_contraction_seconds += metrics.kubo_contraction_seconds
            self._primitive_packing_seconds += metrics.primitive_packing_seconds
            self._callbacks += 1
            self._complete_orbit_points += metrics.complete_orbit_points

    def __call__(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if self._closed:
            raise RuntimeError("orbit evaluator is closed")

        point_array = np.asarray(points, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        pool = self._process_pool
        if pool is None:
            packed, metrics = self._evaluate_local(point_array, weight_array)
        else:
            # Pool submission is serialized only for the queue operation.  Worker
            # execution and waiting remain concurrent across transverse submitters.
            with self._submit_lock:
                pending = pool.apply_async(
                    _fork_process_evaluate,
                    (point_array, weight_array),
                )
            packed, metrics = pending.get()

        self._record_metrics(metrics)
        return packed

    def close(self, *, terminate: bool = False) -> None:
        if self._closed:
            return
        self._closed = True

        pool = self._process_pool
        self._process_pool = None
        if pool is not None:
            if terminate:
                pool.terminate()
            else:
                pool.close()
            pool.join()

        global _FORK_PROCESS_EVALUATOR
        if _FORK_PROCESS_EVALUATOR is self:
            _FORK_PROCESS_EVALUATOR = None
        if self._fork_guard_held:
            self._fork_guard_held = False
            _FORK_PROCESS_GUARD.release()

    def __enter__(self) -> DWaveOrbitPrimitiveEvaluator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(terminate=exc_type is not None)

    def profile_snapshot(self) -> DWaveOrbitEvaluatorProfile:
        with self._profile_lock:
            return DWaveOrbitEvaluatorProfile(
                callbacks=int(self._callbacks),
                complete_orbit_points=int(self._complete_orbit_points),
                material_workspace_implementation=str(
                    self._material_workspace_implementation
                ),
                q_workspace_implementation=str(self._q_workspace_implementation),
                material_workspace_seconds=float(self._material_workspace_seconds),
                q_workspace_seconds=float(self._q_workspace_seconds),
                kubo_factor_seconds=float(self._kubo_factor_seconds),
                kubo_contraction_seconds=float(self._kubo_contraction_seconds),
                primitive_packing_seconds=float(self._primitive_packing_seconds),
            )


__all__ = [
    "DWaveOrbitEvaluatorProfile",
    "DWaveOrbitPrimitiveEvaluator",
]
