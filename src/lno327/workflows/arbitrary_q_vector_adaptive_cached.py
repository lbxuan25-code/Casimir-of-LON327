"""Reusable hierarchical material cache for arbitrary-q vector-adaptive cubature.

The adaptive controller lives in :mod:`arbitrary_q_vector_adaptive`. This module
adds production-shaped reuse: q-independent midpoint eigensystems and the q=0
Goldstone counterterm integrand are built once per exact adaptive node and reused
across q, angle and Matsubara tasks.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from lno327.response.finite_q_optimized import _vectorized_kubo_factors
from lno327.response.finite_q_q_workspace_batched import (
    precompute_finite_q_q_workspace_batched,
)
from lno327.workflows.arbitrary_q_matsubara import TwoPlateAngleBatchResult
from lno327.workflows.arbitrary_q_vector_adaptive import (
    ArbitraryQVectorAdaptiveOptions,
    ArbitraryQVectorAdaptiveResponseCache,
    HierarchicalMaterialNodeCache,
    build_hierarchical_material_node_cache,
    integrate_arbitrary_q_vector_adaptive,
    integrate_two_plate_angle_batch_vector_adaptive,
)


class ReusableHierarchicalMaterialNodeCache(HierarchicalMaterialNodeCache):
    """Hierarchical cache with reusable Goldstone integrands and hard budgets."""

    def __init__(
        self,
        *,
        max_cache_nodes: int | None = None,
        max_cache_bytes: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if max_cache_nodes is not None and int(max_cache_nodes) <= 0:
            raise ValueError("max_cache_nodes must be positive when provided")
        if max_cache_bytes is not None and int(max_cache_bytes) <= 0:
            raise ValueError("max_cache_bytes must be positive when provided")
        self.max_cache_nodes = None if max_cache_nodes is None else int(max_cache_nodes)
        self.max_cache_bytes = None if max_cache_bytes is None else int(max_cache_bytes)
        self._phase_bubble_integrands: dict[str, complex] = {}
        self.counterterm_node_hits = 0
        self.counterterm_node_misses = 0
        self.counterterm_q0_workspace_build_count = 0
        self.counterterm_shifted_eigh_call_count = 0
        self.counterterm_q0_workspace_seconds = 0.0

    def estimated_cache_bytes(self) -> int:
        node_bytes = sum(
            int(value.energies.nbytes + value.states.nbytes + value.occupations.nbytes)
            for value in self._nodes.values()
        )
        counterterm_bytes = len(self._phase_bubble_integrands) * np.dtype(complex).itemsize
        return int(node_bytes + counterterm_bytes)

    def _missing_node_count(self, points: np.ndarray) -> int:
        keys = {self._point_key(point) for point in np.asarray(points, dtype=float)}
        return sum(key not in self._nodes for key in keys)

    def _check_node_budget_before_build(self, points: np.ndarray) -> None:
        missing = self._missing_node_count(points)
        if self.max_cache_nodes is not None and len(self._nodes) + missing > self.max_cache_nodes:
            raise MemoryError(
                "adaptive material-node cache would exceed max_cache_nodes: "
                f"current={len(self._nodes)}, missing={missing}, limit={self.max_cache_nodes}"
            )

    def _check_byte_budget_after_build(self) -> None:
        estimated = self.estimated_cache_bytes()
        if self.max_cache_bytes is not None and estimated > self.max_cache_bytes:
            raise MemoryError(
                "adaptive material-node cache exceeded max_cache_bytes: "
                f"estimated={estimated}, limit={self.max_cache_bytes}"
            )

    def _ensure_phase_bubble_integrands(
        self,
        points: np.ndarray,
        material_workspace,
    ) -> None:
        point_array = np.asarray(points, dtype=float)
        missing_indices = [
            index
            for index, point in enumerate(point_array)
            if self._point_key(point) not in self._phase_bubble_integrands
        ]
        if not missing_indices:
            self.counterterm_node_hits += int(point_array.shape[0])
            return

        indices = np.asarray(missing_indices, dtype=np.int64)
        missing_material = replace(
            material_workspace,
            k_points=np.asarray(material_workspace.k_points)[indices],
            k_weights=np.full(indices.size, 1.0 / indices.size, dtype=float),
            midpoint_energies=np.asarray(material_workspace.midpoint_energies)[indices],
            midpoint_states=np.asarray(material_workspace.midpoint_states)[indices],
            midpoint_occupations=np.asarray(material_workspace.midpoint_occupations)[indices],
            collective_counterterm_matrix=np.zeros((2, 2), dtype=complex),
            metadata={
                **dict(material_workspace.metadata),
                "workspace_kind": "adaptive_counterterm_node_view",
                "counterterm_included": False,
            },
        )
        from time import perf_counter

        started = perf_counter()
        q0_workspace = precompute_finite_q_q_workspace_batched(
            missing_material,
            np.zeros(2, dtype=float),
            operator_diagnostics=False,
        )
        self.counterterm_q0_workspace_seconds += perf_counter() - started
        self.counterterm_q0_workspace_build_count += int(
            q0_workspace.metadata.get("q_workspace_build_count", 1)
        )
        shifted = int(q0_workspace.metadata.get("shifted_eigh_call_count", 0))
        self.counterterm_shifted_eigh_call_count += shifted
        if shifted != 0:
            raise RuntimeError("q=0 counterterm cache rebuilt shifted eigensystems")

        factors = _vectorized_kubo_factors(q0_workspace, np.asarray([0.0], dtype=float))[0]
        phase_left = np.asarray(q0_workspace.left_vertices_band)[:, 4]
        phase_right = np.asarray(q0_workspace.right_vertices_band)[:, 4]
        local = 0.5 * np.einsum(
            "kmn,kmn,kmn->k",
            factors,
            phase_left,
            np.conjugate(phase_right),
            optimize=True,
        )
        for local_index, source_index in enumerate(missing_indices):
            self._phase_bubble_integrands[
                self._point_key(point_array[source_index])
            ] = complex(local[local_index])
        self.counterterm_node_misses += len(missing_indices)
        self.counterterm_node_hits += int(point_array.shape[0] - len(missing_indices))
        self._check_byte_budget_after_build()

    def material_workspace(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        *,
        include_counterterm: bool = False,
    ):
        self._check_node_budget_before_build(points)
        workspace = super().material_workspace(points, weights, include_counterterm=False)
        self._check_byte_budget_after_build()
        self._ensure_phase_bubble_integrands(points, workspace)
        counterterm = (
            self.counterterm(points, weights)
            if include_counterterm
            else np.zeros((2, 2), dtype=complex)
        )
        return replace(
            workspace,
            collective_counterterm_matrix=np.asarray(counterterm, dtype=complex),
            metadata={
                **dict(workspace.metadata),
                "workspace_kind": "reusable_hierarchical_adaptive_material_node_view",
                "counterterm_included": bool(include_counterterm),
                "counterterm_uses_cached_midpoint_eigensystems": True,
            },
        )

    def counterterm(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        point_array = np.asarray(points, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        if point_array.ndim != 2 or point_array.shape[1] != 2 or point_array.shape[0] == 0:
            raise ValueError("counterterm points must have shape (n,2)")
        if weight_array.shape != (point_array.shape[0],):
            raise ValueError("counterterm weights must have shape (n,)")
        keys = [self._point_key(point) for point in point_array]
        missing = [key for key in keys if key not in self._phase_bubble_integrands]
        if missing:
            self._check_node_budget_before_build(point_array)
            workspace = super().material_workspace(
                point_array, weight_array, include_counterterm=False
            )
            self._ensure_phase_bubble_integrands(point_array, workspace)
        phase_bubble = sum(
            float(weight) * self._phase_bubble_integrands[key]
            for weight, key in zip(weight_array, keys, strict=True)
        )
        return -complex(phase_bubble) * np.eye(2, dtype=complex)

    def metadata(self) -> dict[str, object]:
        return {
            **super().metadata(),
            "cache_extension": "ReusableHierarchicalMaterialNodeCache-v2",
            "counterterm_node_entries": len(self._phase_bubble_integrands),
            "counterterm_node_hits": int(self.counterterm_node_hits),
            "counterterm_node_misses": int(self.counterterm_node_misses),
            "counterterm_q0_workspace_build_count": int(
                self.counterterm_q0_workspace_build_count
            ),
            "counterterm_shifted_eigh_call_count": int(
                self.counterterm_shifted_eigh_call_count
            ),
            "counterterm_q0_workspace_seconds": float(
                self.counterterm_q0_workspace_seconds
            ),
            "counterterm_uses_cached_midpoint_eigensystems": True,
            "estimated_cache_bytes": self.estimated_cache_bytes(),
            "max_cache_nodes": self.max_cache_nodes,
            "max_cache_bytes": self.max_cache_bytes,
            "cache_budget_policy": "fail_closed_no_eviction",
        }


def build_reusable_hierarchical_material_node_cache(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    temperature_K: float,
    eta_eV: float,
    max_cache_nodes: int | None = None,
    max_cache_bytes: int | None = None,
) -> ReusableHierarchicalMaterialNodeCache:
    base = build_hierarchical_material_node_cache(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
    )
    return ReusableHierarchicalMaterialNodeCache(
        spec=base.spec,
        ansatz=base.ansatz,
        pairing=base.pairing,
        config=base.config,
        options=base.options,
        max_cache_nodes=max_cache_nodes,
        max_cache_bytes=max_cache_bytes,
    )


def integrate_arbitrary_q_vector_adaptive_cached(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    xi_eV_values: Sequence[float] | np.ndarray,
    temperature_K: float,
    eta_eV: float,
    q_model: np.ndarray,
    adaptive_options: ArbitraryQVectorAdaptiveOptions | None = None,
    node_cache: ReusableHierarchicalMaterialNodeCache | None = None,
    response_cache: ArbitraryQVectorAdaptiveResponseCache | None = None,
    operator_ward_atol: float = 512.0 * np.finfo(float).eps,
    operator_ward_rtol: float = 512.0 * np.finfo(float).eps,
    require_converged: bool = True,
    max_cache_nodes: int | None = None,
    max_cache_bytes: int | None = None,
):
    cache = node_cache or build_reusable_hierarchical_material_node_cache(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        max_cache_nodes=max_cache_nodes,
        max_cache_bytes=max_cache_bytes,
    )
    return integrate_arbitrary_q_vector_adaptive(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_eV_values,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        q_model=q_model,
        adaptive_options=adaptive_options,
        node_cache=cache,
        response_cache=response_cache,
        operator_ward_atol=operator_ward_atol,
        operator_ward_rtol=operator_ward_rtol,
        require_converged=require_converged,
    )


def integrate_two_plate_angle_batch_vector_adaptive_cached(
    *,
    q_lab: np.ndarray,
    theta_1_rad: float,
    theta_2_rad_values: Sequence[float] | np.ndarray,
    node_cache: ReusableHierarchicalMaterialNodeCache,
    spec: object,
    ansatz: object,
    pairing: object,
    xi_eV_values: Sequence[float] | np.ndarray,
    temperature_K: float,
    eta_eV: float,
    adaptive_options: ArbitraryQVectorAdaptiveOptions | None = None,
    response_cache: ArbitraryQVectorAdaptiveResponseCache | None = None,
    operator_ward_atol: float = 512.0 * np.finfo(float).eps,
    operator_ward_rtol: float = 512.0 * np.finfo(float).eps,
    require_converged: bool = True,
) -> TwoPlateAngleBatchResult:
    return integrate_two_plate_angle_batch_vector_adaptive(
        q_lab=q_lab,
        theta_1_rad=theta_1_rad,
        theta_2_rad_values=theta_2_rad_values,
        node_cache=node_cache,
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_eV_values,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        adaptive_options=adaptive_options,
        response_cache=response_cache,
        operator_ward_atol=operator_ward_atol,
        operator_ward_rtol=operator_ward_rtol,
        require_converged=require_converged,
    )


__all__ = [
    "ReusableHierarchicalMaterialNodeCache",
    "build_reusable_hierarchical_material_node_cache",
    "integrate_arbitrary_q_vector_adaptive_cached",
    "integrate_two_plate_angle_batch_vector_adaptive_cached",
]
