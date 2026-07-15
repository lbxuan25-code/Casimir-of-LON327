"""Shared-cell vector cubature primitives for exact-static d-wave response.

The adaptive controller evaluates paired, non-embedded tensor-Gauss rules on
every Brillouin-zone cell. All electromagnetic, collective, contact and Ward-RHS
quantities are kept as one complex primitive vector. Cell refinement is driven
by the complete microscopic contract, and accepted cell primitives are summed
before the single amplitude/phase Schur complement.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.finite_q_bdg import _finalize_components
from lno327.response.finite_q_optimized import FiniteQQWorkspace
from lno327.response.ward_validation import PrimitiveWardRHS, primitive_ward_vectors_xy


@dataclass(frozen=True, order=True)
class DWaveCubatureCell:
    """One half-open rectangular Brillouin-zone cell."""

    x0: float
    x1: float
    y0: float
    y1: float
    level: int = 0

    @property
    def area_fraction(self) -> float:
        return float((self.x1 - self.x0) * (self.y1 - self.y0) / (2.0 * np.pi) ** 2)


@dataclass(frozen=True)
class DWaveVectorAdaptiveOptions:
    coarse_grid: int = 6
    low_order: int = 2
    high_order: int = 3
    relative_tolerance: float = 1e-3
    absolute_tolerance: float = 1e-9
    max_level: int = 5
    max_iterations: int = 8
    refine_fraction: float = 0.15
    min_refine_cells: int = 4
    max_cells: int = 4000
    max_evaluation_points: int = 60_000


def validate_vector_adaptive_options(options: DWaveVectorAdaptiveOptions) -> None:
    if int(options.coarse_grid) <= 0:
        raise ValueError("coarse_grid must be positive")
    if int(options.low_order) <= 0 or int(options.high_order) <= int(options.low_order):
        raise ValueError("high_order must be greater than positive low_order")
    for name in ("relative_tolerance", "absolute_tolerance"):
        value = float(getattr(options, name))
        if value < 0.0 or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
    if int(options.max_level) < 0 or int(options.max_iterations) < 0:
        raise ValueError("max_level and max_iterations must be non-negative")
    if not 0.0 < float(options.refine_fraction) <= 1.0:
        raise ValueError("refine_fraction must lie in (0, 1]")
    if int(options.min_refine_cells) <= 0:
        raise ValueError("min_refine_cells must be positive")
    if int(options.max_cells) <= 0 or int(options.max_evaluation_points) <= 0:
        raise ValueError("max_cells and max_evaluation_points must be positive")


def initial_cubature_cells(coarse_grid: int) -> list[DWaveCubatureCell]:
    count = int(coarse_grid)
    if count <= 0:
        raise ValueError("coarse_grid must be positive")
    edges = np.linspace(-np.pi, np.pi, count + 1, dtype=float)
    return [
        DWaveCubatureCell(
            float(edges[ix]), float(edges[ix + 1]),
            float(edges[iy]), float(edges[iy + 1]), 0,
        )
        for ix in range(count)
        for iy in range(count)
    ]


def subdivide_cubature_cell(cell: DWaveCubatureCell) -> tuple[DWaveCubatureCell, ...]:
    xm = 0.5 * (cell.x0 + cell.x1)
    ym = 0.5 * (cell.y0 + cell.y1)
    level = int(cell.level) + 1
    return (
        DWaveCubatureCell(cell.x0, xm, cell.y0, ym, level),
        DWaveCubatureCell(cell.x0, xm, ym, cell.y1, level),
        DWaveCubatureCell(xm, cell.x1, cell.y0, ym, level),
        DWaveCubatureCell(xm, cell.x1, ym, cell.y1, level),
    )


@lru_cache(maxsize=None)
def _tensor_gauss_reference(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return readonly reference-square points and product weights for one order."""

    count = int(order)
    if count <= 0:
        raise ValueError("order must be positive")
    nodes, one_d_weights = np.polynomial.legendre.leggauss(count)
    gx, gy = np.meshgrid(nodes, nodes, indexing="ij")
    wx, wy = np.meshgrid(one_d_weights, one_d_weights, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel()])
    weights = (wx * wy).ravel()
    points.setflags(write=False)
    weights.setflags(write=False)
    return points, weights


def cubature_cell_gauss_rule(
    cell: DWaveCubatureCell,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return points and full-BZ-normalized tensor-Gauss weights for one cell."""

    reference_points, reference_weights = _tensor_gauss_reference(int(order))
    xm = 0.5 * (cell.x0 + cell.x1)
    ym = 0.5 * (cell.y0 + cell.y1)
    xh = 0.5 * (cell.x1 - cell.x0)
    yh = 0.5 * (cell.y1 - cell.y0)
    points = np.empty_like(reference_points)
    points[:, 0] = xm + xh * reference_points[:, 0]
    points[:, 1] = ym + yh * reference_points[:, 1]
    weights = np.asarray(
        reference_weights * xh * yh / (2.0 * np.pi) ** 2, dtype=float
    )
    if not np.isclose(np.sum(weights), cell.area_fraction, rtol=0.0, atol=2e-15):
        raise RuntimeError("cell Gauss weights do not reproduce the cell area")
    return points, weights


def primitive_component_vector(
    components: BdGFiniteQResponseComponents,
    rhs: PrimitiveWardRHS,
) -> np.ndarray:
    phase_plus = complex(components.metadata["phase_phase_direct_plus_convention"])
    phase_minus = complex(components.metadata["phase_phase_direct_minus_convention"])
    fields = (
        components.bare_bubble,
        components.direct,
        components.collective_bubble,
        components.collective_counterterm,
        components.em_collective_left,
        components.collective_em_right,
        np.asarray([phase_plus, phase_minus], dtype=complex),
        rhs.left,
        rhs.right,
    )
    return np.concatenate([np.asarray(value, dtype=complex).ravel() for value in fields])


def primitive_ward_residual_vector(
    components: BdGFiniteQResponseComponents,
    rhs: PrimitiveWardRHS,
) -> np.ndarray:
    u_left, u_right, w_left, w_right = primitive_ward_vectors_xy(
        rhs.xi_eV, rhs.q_model, rhs.delta0_eV
    )
    k_ss = np.asarray(components.bare_total, dtype=complex)
    k_seta = np.asarray(components.em_collective_left, dtype=complex)
    k_etas = np.asarray(components.collective_em_right, dtype=complex)
    left = u_left @ k_ss + w_left @ k_etas - rhs.left
    right = k_ss @ u_right + k_seta @ w_right - rhs.right
    return np.concatenate([left, right])


def merge_cell_components_before_schur(
    components: Sequence[BdGFiniteQResponseComponents],
    rhs_values: Sequence[PrimitiveWardRHS],
    template_workspace: FiniteQQWorkspace,
    *,
    omega_eV: float = 0.0,
) -> tuple[BdGFiniteQResponseComponents, PrimitiveWardRHS]:
    if not components or len(components) != len(rhs_values):
        raise ValueError("components and rhs_values must be nonempty and have equal length")

    def array_sum(name: str) -> np.ndarray:
        return np.sum(
            np.stack([np.asarray(getattr(item, name), dtype=complex) for item in components]),
            axis=0,
        )

    material = template_workspace.material
    config = replace(material.config, omega_eV=float(omega_eV))
    q = np.asarray(template_workspace.q_model, dtype=float)
    delta0 = float(material.pairing_params.delta0_eV)
    bubble = array_sum("bare_bubble")
    direct = array_sum("direct")
    collective_bubble = array_sum("collective_bubble")
    collective_counterterm = array_sum("collective_counterterm")
    em_collective_left = array_sum("em_collective_left")
    collective_em_right = array_sum("collective_em_right")
    phase_left = delta0 * em_collective_left[:, 1]
    phase_right = delta0 * collective_em_right[1, :]
    phase_bubble = np.asarray(
        [[delta0 * delta0 * collective_bubble[1, 1]]], dtype=complex
    )
    phase_plus = sum(
        complex(item.metadata["phase_phase_direct_plus_convention"]) for item in components
    )
    phase_minus = sum(
        complex(item.metadata["phase_phase_direct_minus_convention"]) for item in components
    )
    merged = _finalize_components(
        ansatz=material.ansatz,
        opts=material.options,
        shared_eigenbasis_q0=template_workspace.shared_eigenbasis_q0,
        shared_eigenbasis_q0_tolerance=1e-14,
        collective_mode=material.collective_mode,
        collective_mode_disabled_reason=material.collective_mode_disabled_reason,
        bubble=bubble,
        direct=direct,
        phase_left=phase_left,
        phase_right=phase_right,
        phase_phase_bubble_matrix=phase_bubble,
        phase_phase_direct_plus=phase_plus,
        phase_phase_direct_minus=phase_minus,
        collective_bubble=collective_bubble,
        collective_counterterm_matrix=collective_counterterm,
        em_collective_left=em_collective_left,
        collective_em_right=collective_em_right,
        config=config,
        q=q,
        workspace_evaluation=True,
    )
    metadata = dict(merged.metadata)
    metadata.update(
        {
            "vector_adaptive_cells_merged_before_schur": True,
            "num_accepted_cells": len(components),
            "per_cell_schur_results_discarded": True,
        }
    )
    merged = replace(merged, metadata=metadata)
    left = np.sum(np.stack([np.asarray(item.left) for item in rhs_values]), axis=0)
    right = np.sum(np.stack([np.asarray(item.right) for item in rhs_values]), axis=0)
    merged_rhs = PrimitiveWardRHS(
        left=left,
        right=right,
        q_model=q,
        xi_eV=float(omega_eV),
        delta0_eV=delta0,
        metadata={
            "source": "sum of accepted vector-adaptive cell Ward RHS values",
            "vector_adaptive_cells_merged_before_ward_validation": True,
            "num_accepted_cells": len(rhs_values),
        },
    )
    return merged, merged_rhs


_PACKED_HEADER_WIDTH = 18
_PACKED_FREQUENCY_WIDTH = 25
_PACKED_HEADER_GROUP_WIDTHS = (9, 4, 2, 3)
_PACKED_FREQUENCY_GROUP_WIDTHS = (9, 4, 6, 6)


def _primitive_error_groups(width: int) -> tuple[slice, ...]:
    """Return stable physical block slices, with a whole-vector fallback."""

    size = int(width)
    if size <= 0:
        raise ValueError("primitive vectors must be nonempty")
    dynamic = size - _PACKED_HEADER_WIDTH
    if dynamic < 0 or dynamic % _PACKED_FREQUENCY_WIDTH:
        return (slice(0, size),)

    groups: list[slice] = []
    cursor = 0
    for group_width in _PACKED_HEADER_GROUP_WIDTHS:
        groups.append(slice(cursor, cursor + group_width))
        cursor += group_width
    frequency_count = dynamic // _PACKED_FREQUENCY_WIDTH
    for _ in range(frequency_count):
        for group_width in _PACKED_FREQUENCY_GROUP_WIDTHS:
            groups.append(slice(cursor, cursor + group_width))
            cursor += group_width
    if cursor != size:
        raise RuntimeError("primitive error groups do not cover the complete vector")
    return tuple(groups)


def _max_abs(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=complex)
    return float(np.max(np.abs(array))) if array.size else 0.0


def vector_error_metrics(
    low_vectors: Sequence[np.ndarray],
    high_vectors: Sequence[np.ndarray],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    low_ward_vectors: Sequence[np.ndarray] | None = None,
    high_ward_vectors: Sequence[np.ndarray] | None = None,
    ward_threshold: float | None = None,
) -> dict[str, Any]:
    """Measure global low/high convergence while retaining local refinement scores.

    The hard convergence metric is formed only after summing all active cells.
    Stable physical blocks are compared with mixed absolute/relative max norms, so
    cell-to-cell cancellation and symmetry-forced near-zero components do not become
    artificial global errors. Per-cell low/high differences remain only as a ranking
    signal for deciding which cells to refine next.
    """

    if not low_vectors or len(low_vectors) != len(high_vectors):
        raise ValueError("low_vectors and high_vectors must be nonempty and aligned")
    rtol = float(relative_tolerance)
    atol = float(absolute_tolerance)
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("adaptive tolerances must be finite and non-negative")

    low = np.stack([np.asarray(value, dtype=complex) for value in low_vectors])
    high = np.stack([np.asarray(value, dtype=complex) for value in high_vectors])
    if low.shape != high.shape or low.ndim != 2:
        raise ValueError("low/high primitive vectors must have equal two-dimensional shape")

    delta = high - low
    low_total = np.sum(low, axis=0)
    high_total = np.sum(high, axis=0)
    global_delta = high_total - low_total
    groups = _primitive_error_groups(high.shape[1])
    tiny = np.finfo(float).tiny

    group_ratios: list[float] = []
    group_thresholds: list[float] = []
    local_group_ratios: list[np.ndarray] = []
    local_absolute_ratios: list[float] = []
    for group in groups:
        scale = max(_max_abs(low_total[group]), _max_abs(high_total[group]))
        threshold = max(atol + rtol * scale, tiny)
        group_thresholds.append(float(threshold))
        group_ratios.append(_max_abs(global_delta[group]) / threshold)
        local = np.max(np.abs(delta[:, group]), axis=1) / threshold
        local_group_ratios.append(np.asarray(local, dtype=float))
        local_absolute_ratios.append(float(np.sum(local)))

    scores = np.max(np.stack(local_group_ratios, axis=1), axis=1)
    global_ratio = float(max(group_ratios, default=0.0))
    local_absolute_ratio = float(max(local_absolute_ratios, default=0.0))

    ward_global_ratio = float("nan")
    ward_local_absolute_ratio = float("nan")
    if low_ward_vectors is not None or high_ward_vectors is not None:
        if low_ward_vectors is None or high_ward_vectors is None:
            raise ValueError("both low and high Ward vectors are required")
        if len(low_ward_vectors) != len(high_ward_vectors) or len(low_ward_vectors) != len(low_vectors):
            raise ValueError("Ward vectors must align with primitive cell vectors")
        low_ward = np.stack(
            [np.asarray(value, dtype=complex) for value in low_ward_vectors]
        )
        high_ward = np.stack(
            [np.asarray(value, dtype=complex) for value in high_ward_vectors]
        )
        if low_ward.shape != high_ward.shape or low_ward.ndim != 2:
            raise ValueError("low/high Ward vectors must have equal two-dimensional shape")
        ward_delta = high_ward - low_ward
        low_ward_total = np.sum(low_ward, axis=0)
        high_ward_total = np.sum(high_ward, axis=0)
        ward_atol = atol if ward_threshold is None else float(ward_threshold)
        if not np.isfinite(ward_atol) or ward_atol < 0.0:
            raise ValueError("ward_threshold must be finite and non-negative")
        ward_scale = max(_max_abs(low_ward_total), _max_abs(high_ward_total))
        mixed_ward_threshold = max(ward_atol + rtol * ward_scale, tiny)
        ward_global_ratio = (
            _max_abs(high_ward_total - low_ward_total) / mixed_ward_threshold
        )
        ward_local = np.max(np.abs(ward_delta), axis=1) / mixed_ward_threshold
        scores = np.maximum(scores, ward_local)
        ward_local_absolute_ratio = float(np.sum(ward_local))

    return {
        "cell_scores": np.asarray(scores, dtype=float),
        # Backward-compatible field names now carry the correct global signed
        # mixed-error semantics. The explicit v2 names below remove ambiguity.
        "conservative_error_ratio_max": global_ratio,
        "signed_error_ratio_max": global_ratio,
        "ward_error_ratio_conservative": ward_global_ratio,
        "global_group_error_ratio_max": global_ratio,
        "local_absolute_error_ratio_max": local_absolute_ratio,
        "ward_global_error_ratio": ward_global_ratio,
        "ward_local_absolute_error_ratio": ward_local_absolute_ratio,
        "group_error_ratios": np.asarray(group_ratios, dtype=float),
        "group_thresholds": np.asarray(group_thresholds, dtype=float),
        "error_estimator_contract": "global_signed_group_mixed_v2",
        "global_high_vector_norm": float(np.linalg.norm(high_total)),
    }


__all__ = [
    "DWaveCubatureCell",
    "DWaveVectorAdaptiveOptions",
    "_tensor_gauss_reference",
    "cubature_cell_gauss_rule",
    "initial_cubature_cells",
    "merge_cell_components_before_schur",
    "primitive_component_vector",
    "primitive_ward_residual_vector",
    "subdivide_cubature_cell",
    "validate_vector_adaptive_options",
    "vector_error_metrics",
]
