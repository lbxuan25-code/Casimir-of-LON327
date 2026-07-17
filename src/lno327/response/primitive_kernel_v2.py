"""Shared primitive kernel with q-workspace-integrated operator diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from lno327.response.finite_q_optimized import (
    FiniteQMaterialWorkspace,
    _vectorized_kubo_factors,
)
from lno327.response.finite_q_q_workspace_batched import (
    _integrated_linear_terms_from_workspace_slice,
    precompute_finite_q_q_workspace_batched,
)
from lno327.response.primitive_kernel import (
    OperatorWardReport,
    PrimitiveBatchMetrics,
    PrimitiveBatchResult,
    counterterm_primitive_vector,
    pack_integrated_primitives,
    primitive_vector_width,
    unpack_integrated_primitives,
)

_FLOAT_EPS = np.finfo(float).eps


def operator_ward_report_from_workspace(
    workspace: object,
    *,
    atol: float = 512.0 * _FLOAT_EPS,
    rtol: float = 512.0 * _FLOAT_EPS,
) -> OperatorWardReport:
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol <= 0.0 or rtol <= 0.0:
        raise ValueError("operator Ward tolerances must be finite and positive")
    metadata = workspace.metadata
    if metadata.get("operator_diagnostics_enabled") is not True:
        raise ValueError("q workspace was not built with operator diagnostics")
    delta = np.asarray(metadata.get("operator_identity_delta_norms"), dtype=float)
    scale = np.asarray(metadata.get("operator_identity_scales"), dtype=float)
    if delta.ndim != 1 or scale.shape != delta.shape or delta.size == 0:
        raise ValueError("q workspace does not contain integrated operator diagnostics")
    threshold = float(atol) + float(rtol) * scale
    mixed = delta / np.maximum(threshold, np.finfo(float).tiny)
    relative = delta / np.maximum(scale, np.finfo(float).tiny)
    finite = np.isfinite(mixed)
    failed = (~finite) | (mixed > 1.0)
    return OperatorWardReport(
        point_count=int(delta.size),
        max_absolute_error=float(np.max(delta)),
        max_relative_error=float(np.max(relative)),
        max_mixed_ratio=float(np.max(mixed)),
        failed_points=int(np.count_nonzero(failed)),
        atol=float(atol),
        rtol=float(rtol),
        passed=bool(not np.any(failed)),
    )


@dataclass(frozen=True)
class RuntimeChunkPrimitiveResult:
    packed_canonical_blocks: tuple[np.ndarray, ...]
    operator_ward: OperatorWardReport
    k_point_count: int
    canonical_block_count: int
    q_workspace_build_count: int
    shifted_eigh_call_count: int
    q_workspace_seconds: float
    kubo_factor_seconds: float
    kubo_contraction_seconds: float
    primitive_pack_seconds: float

    @property
    def total_seconds(self) -> float:
        return float(
            self.q_workspace_seconds
            + self.kubo_factor_seconds
            + self.kubo_contraction_seconds
            + self.primitive_pack_seconds
        )


def _validate_frequencies(values: Sequence[float] | np.ndarray) -> np.ndarray:
    xi_values = np.asarray(values, dtype=float)
    if xi_values.ndim != 1 or xi_values.size == 0:
        raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
    if not np.isfinite(xi_values).all() or np.any(xi_values < 0.0):
        raise ValueError("xi_eV_values must be finite and non-negative")
    if np.count_nonzero(xi_values == 0.0) > 1:
        raise ValueError("exact zero may appear at most once")
    return xi_values


def _canonical_workspace_view(
    workspace: object,
    start: int,
    stop: int,
) -> object:
    direct, phase_plus, ward_rhs = _integrated_linear_terms_from_workspace_slice(
        workspace, start, stop
    )
    zero_counterterm = np.zeros((2, 2), dtype=complex)
    material = SimpleNamespace(collective_counterterm_matrix=zero_counterterm)
    return SimpleNamespace(
        direct_contact_contribution=direct,
        phase_phase_direct_plus=phase_plus,
        phase_phase_direct_minus=-phase_plus,
        ward_rhs_vector=ward_rhs,
        material=material,
    )


def evaluate_runtime_chunk_canonical_primitives(
    material: FiniteQMaterialWorkspace | object,
    q_model: np.ndarray,
    xi_eV_values: Sequence[float] | np.ndarray,
    *,
    canonical_reduction_block_size: int,
    operator_ward_atol: float = 512.0 * _FLOAT_EPS,
    operator_ward_rtol: float = 512.0 * _FLOAT_EPS,
) -> RuntimeChunkPrimitiveResult:
    """Build one runtime q workspace and emit deterministic canonical blocks."""

    xi_values = _validate_frequencies(xi_eV_values)
    block_size = int(canonical_reduction_block_size)
    if block_size <= 0 or block_size % 2:
        raise ValueError("canonical reduction block size must be positive and even")

    started = perf_counter()
    workspace = precompute_finite_q_q_workspace_batched(
        material, q_model, operator_diagnostics=True
    )
    q_workspace_seconds = perf_counter() - started
    operator_ward = operator_ward_report_from_workspace(
        workspace, atol=operator_ward_atol, rtol=operator_ward_rtol
    )

    started = perf_counter()
    raw_factors = _vectorized_kubo_factors(workspace, xi_values)
    kubo_factor_seconds = perf_counter() - started

    packed_blocks: list[np.ndarray] = []
    contraction_seconds = 0.0
    pack_seconds = 0.0
    nk = int(material.k_points.shape[0])
    weights = np.asarray(workspace.material.k_weights, dtype=float)
    for start in range(0, nk, block_size):
        stop = min(start + block_size, nk)
        timer = perf_counter()
        weighted = (
            0.5
            * weights[None, start:stop, None, None]
            * raw_factors[:, start:stop]
        )
        blocks = np.einsum(
            "xkmn,kamn,kbmn->xab",
            weighted,
            workspace.left_vertices_band[start:stop],
            np.conjugate(workspace.right_vertices_band[start:stop]),
            optimize=True,
        )
        contraction_seconds += perf_counter() - timer
        timer = perf_counter()
        packed = pack_integrated_primitives(
            workspace=_canonical_workspace_view(workspace, start, stop),
            blocks=blocks,
            include_counterterm=False,
        )
        packed_blocks.append(np.asarray(packed, dtype=complex))
        pack_seconds += perf_counter() - timer

    return RuntimeChunkPrimitiveResult(
        packed_canonical_blocks=tuple(packed_blocks),
        operator_ward=operator_ward,
        k_point_count=nk,
        canonical_block_count=len(packed_blocks),
        q_workspace_build_count=int(workspace.metadata.get("q_workspace_build_count", 1)),
        shifted_eigh_call_count=int(workspace.metadata.get("shifted_eigh_call_count", 0)),
        q_workspace_seconds=float(q_workspace_seconds),
        kubo_factor_seconds=float(kubo_factor_seconds),
        kubo_contraction_seconds=float(contraction_seconds),
        primitive_pack_seconds=float(pack_seconds),
    )


def evaluate_primitive_batch_from_material(
    material: FiniteQMaterialWorkspace | object,
    q_model: np.ndarray,
    xi_eV_values: Sequence[float] | np.ndarray,
    *,
    include_counterterm: bool = True,
    operator_ward_atol: float = 512.0 * _FLOAT_EPS,
    operator_ward_rtol: float = 512.0 * _FLOAT_EPS,
) -> PrimitiveBatchResult:
    """Evaluate one complete k batch for exact q and all frequencies."""

    xi_values = _validate_frequencies(xi_eV_values)
    started = perf_counter()
    workspace = precompute_finite_q_q_workspace_batched(
        material, q_model, operator_diagnostics=True
    )
    q_workspace_seconds = perf_counter() - started
    operator_ward = operator_ward_report_from_workspace(
        workspace, atol=operator_ward_atol, rtol=operator_ward_rtol
    )

    started = perf_counter()
    raw_factors = _vectorized_kubo_factors(workspace, xi_values)
    kubo_factor_seconds = perf_counter() - started
    started = perf_counter()
    weighted = (
        0.5
        * np.asarray(workspace.material.k_weights, dtype=float)[None, :, None, None]
        * raw_factors
    )
    blocks = np.einsum(
        "xkmn,kamn,kbmn->xab",
        weighted,
        workspace.left_vertices_band,
        np.conjugate(workspace.right_vertices_band),
        optimize=True,
    )
    kubo_contraction_seconds = perf_counter() - started
    started = perf_counter()
    packed = pack_integrated_primitives(
        workspace=workspace, blocks=blocks, include_counterterm=include_counterterm
    )
    primitive_pack_seconds = perf_counter() - started
    metrics = PrimitiveBatchMetrics(
        k_point_count=int(material.k_points.shape[0]),
        frequency_count=int(xi_values.size),
        q_workspace_seconds=float(q_workspace_seconds),
        kubo_factor_seconds=float(kubo_factor_seconds),
        kubo_contraction_seconds=float(kubo_contraction_seconds),
        primitive_pack_seconds=float(primitive_pack_seconds),
        shifted_eigensystem_build_count=int(
            workspace.metadata.get("shifted_eigh_call_count", 0)
        ),
        q_workspace_implementation="batched_model_capability",
    )
    return PrimitiveBatchResult(
        packed=np.asarray(packed, dtype=complex),
        operator_ward=operator_ward,
        metrics=metrics,
    )


__all__ = [
    "OperatorWardReport",
    "PrimitiveBatchMetrics",
    "PrimitiveBatchResult",
    "RuntimeChunkPrimitiveResult",
    "counterterm_primitive_vector",
    "evaluate_primitive_batch_from_material",
    "evaluate_runtime_chunk_canonical_primitives",
    "operator_ward_report_from_workspace",
    "pack_integrated_primitives",
    "primitive_vector_width",
    "unpack_integrated_primitives",
]
