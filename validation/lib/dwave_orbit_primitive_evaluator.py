"""Reusable complete-orbit d-wave primitive evaluator with stage profiling."""

from __future__ import annotations

from dataclasses import dataclass
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
        """Return summed worker CPU-seconds across all callbacks."""

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


class DWaveOrbitPrimitiveEvaluator:
    """Evaluate complete orbits with thread-safe aggregate stage profiling.

    The model, ansatz, pairing parameters, q vector, and Matsubara batch are immutable
    after construction. Independent complete-orbit callbacks may therefore run in
    separate threads. Only the lightweight profile counters are protected by a lock.
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

        self._profile_lock = Lock()
        self._callbacks = 0
        self._complete_orbit_points = 0
        self._material_workspace_implementation = "not_evaluated"
        self._q_workspace_implementation = "not_evaluated"
        self._material_workspace_seconds = 0.0
        self._q_workspace_seconds = 0.0
        self._kubo_factor_seconds = 0.0
        self._kubo_contraction_seconds = 0.0
        self._primitive_packing_seconds = 0.0

    def __call__(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
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

        with self._profile_lock:
            if self._material_workspace_implementation not in {
                "not_evaluated",
                material_implementation,
            }:
                raise RuntimeError(
                    "material workspace implementation changed across callbacks"
                )
            if self._q_workspace_implementation not in {
                "not_evaluated",
                q_implementation,
            }:
                raise RuntimeError("q workspace implementation changed across callbacks")
            self._material_workspace_implementation = material_implementation
            self._q_workspace_implementation = q_implementation
            self._material_workspace_seconds += material_seconds
            self._q_workspace_seconds += q_workspace_seconds
            self._kubo_factor_seconds += kubo_factor_seconds
            self._kubo_contraction_seconds += kubo_contraction_seconds
            self._primitive_packing_seconds += primitive_packing_seconds
            self._callbacks += 1
            self._complete_orbit_points += int(point_array.shape[0])

        return packed

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
