"""Shared primitive kernel with q-workspace-integrated operator diagnostics."""
from __future__ import annotations

from time import perf_counter
from typing import Sequence

import numpy as np

from lno327.response.finite_q_optimized import (
    FiniteQMaterialWorkspace,
    _vectorized_kubo_factors,
)
from lno327.response.finite_q_q_workspace_batched_operator import (
    precompute_finite_q_q_workspace_batched_operator,
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
    if (
        not np.isfinite(atol)
        or not np.isfinite(rtol)
        or atol <= 0.0
        or rtol <= 0.0
    ):
        raise ValueError("operator Ward tolerances must be finite and positive")
    metadata = workspace.metadata
    delta = np.asarray(
        metadata.get("operator_identity_delta_norms"),
        dtype=float,
    )
    scale = np.asarray(
        metadata.get("operator_identity_scales"),
        dtype=float,
    )
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


def evaluate_primitive_batch_from_material(
    material: FiniteQMaterialWorkspace | object,
    q_model: np.ndarray,
    xi_eV_values: Sequence[float] | np.ndarray,
    *,
    include_counterterm: bool = True,
    operator_ward_atol: float = 512.0 * _FLOAT_EPS,
    operator_ward_rtol: float = 512.0 * _FLOAT_EPS,
) -> PrimitiveBatchResult:
    """Evaluate exact q and all Matsubara frequencies without duplicate operator work."""

    xi_values = np.asarray(xi_eV_values, dtype=float)
    if xi_values.ndim != 1 or xi_values.size == 0:
        raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
    if not np.isfinite(xi_values).all() or np.any(xi_values < 0.0):
        raise ValueError("xi_eV_values must be finite and non-negative")
    if np.count_nonzero(xi_values == 0.0) > 1:
        raise ValueError("exact zero may appear at most once")

    started = perf_counter()
    workspace = precompute_finite_q_q_workspace_batched_operator(
        material,
        q_model,
    )
    q_workspace_seconds = perf_counter() - started
    operator_ward = operator_ward_report_from_workspace(
        workspace,
        atol=operator_ward_atol,
        rtol=operator_ward_rtol,
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
        workspace=workspace,
        blocks=blocks,
        include_counterterm=include_counterterm,
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
        # Preserve the established public profile label used by complete-orbit
        # qualification.  The operator-integrated implementation is recorded in
        # the q-workspace metadata, not by changing this stable contract field.
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
    "counterterm_primitive_vector",
    "evaluate_primitive_batch_from_material",
    "operator_ward_report_from_workspace",
    "pack_integrated_primitives",
    "primitive_vector_width",
    "unpack_integrated_primitives",
]
