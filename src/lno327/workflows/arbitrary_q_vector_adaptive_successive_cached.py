"""Reusable-cache entry points for successive-high arbitrary-q adaptive cubature."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from lno327.workflows.arbitrary_q_matsubara import (
    TwoPlateAngleBatchResult,
    rotate_lab_q_to_crystal,
)
from lno327.workflows.arbitrary_q_vector_adaptive_cached import (
    ReusableHierarchicalMaterialNodeCache,
    build_reusable_hierarchical_material_node_cache,
)
from lno327.workflows.arbitrary_q_vector_adaptive_successive import (
    ArbitraryQVectorAdaptiveSuccessiveOptions,
    ArbitraryQVectorAdaptiveSuccessiveResponseCache,
    integrate_arbitrary_q_vector_adaptive_successive,
)


def _normalize_timing_profile(
    result,
    *,
    settings: ArbitraryQVectorAdaptiveSuccessiveOptions,
    response_cache: ArbitraryQVectorAdaptiveSuccessiveResponseCache | None,
):
    if bool(result.metadata.get("successive_timing_profile_normalized", False)):
        return result
    profile = result.profile
    primitive_seconds = max(
        float(profile.primitive_integration_seconds)
        - float(profile.iteration_postprocess_seconds),
        0.0,
    )
    normalized_profile = replace(
        profile,
        primitive_integration_seconds=primitive_seconds,
    )
    metadata = {
        **dict(result.metadata),
        "successive_timing_profile_normalized": True,
        "primitive_time_excludes_iteration_ward_postprocess": True,
        "accumulation_profile": normalized_profile.as_dict(),
    }
    normalized = replace(result, profile=normalized_profile, metadata=metadata)
    if response_cache is not None:
        response_cache.put(normalized, settings)
    return normalized


def integrate_arbitrary_q_vector_adaptive_successive_cached(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    xi_eV_values: Sequence[float] | np.ndarray,
    temperature_K: float,
    eta_eV: float,
    q_model: np.ndarray,
    adaptive_options: ArbitraryQVectorAdaptiveSuccessiveOptions | None = None,
    node_cache: ReusableHierarchicalMaterialNodeCache | None = None,
    response_cache: ArbitraryQVectorAdaptiveSuccessiveResponseCache | None = None,
    operator_ward_atol: float = 512.0 * np.finfo(float).eps,
    operator_ward_rtol: float = 512.0 * np.finfo(float).eps,
    require_converged: bool = True,
    max_cache_nodes: int | None = None,
    max_cache_bytes: int | None = None,
):
    settings = adaptive_options or ArbitraryQVectorAdaptiveSuccessiveOptions()
    cache = node_cache or build_reusable_hierarchical_material_node_cache(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        max_cache_nodes=max_cache_nodes,
        max_cache_bytes=max_cache_bytes,
    )
    result = integrate_arbitrary_q_vector_adaptive_successive(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_eV_values,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        q_model=q_model,
        adaptive_options=settings,
        node_cache=cache,
        response_cache=response_cache,
        operator_ward_atol=operator_ward_atol,
        operator_ward_rtol=operator_ward_rtol,
        require_converged=require_converged,
    )
    return _normalize_timing_profile(
        result,
        settings=settings,
        response_cache=response_cache,
    )


def integrate_two_plate_angle_batch_vector_adaptive_successive_cached(
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
    adaptive_options: ArbitraryQVectorAdaptiveSuccessiveOptions | None = None,
    response_cache: ArbitraryQVectorAdaptiveSuccessiveResponseCache | None = None,
    operator_ward_atol: float = 512.0 * np.finfo(float).eps,
    operator_ward_rtol: float = 512.0 * np.finfo(float).eps,
    require_converged: bool = True,
) -> TwoPlateAngleBatchResult:
    q = np.asarray(q_lab, dtype=float)
    angles = np.asarray(theta_2_rad_values, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_lab must be a finite vector with shape (2,)")
    if angles.ndim != 1 or angles.size == 0 or not np.isfinite(angles).all():
        raise ValueError("theta_2_rad_values must be a nonempty finite vector")
    settings = adaptive_options or ArbitraryQVectorAdaptiveSuccessiveOptions()
    local_cache = response_cache or ArbitraryQVectorAdaptiveSuccessiveResponseCache()
    common = dict(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_eV_values,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        adaptive_options=settings,
        node_cache=node_cache,
        response_cache=local_cache,
        operator_ward_atol=operator_ward_atol,
        operator_ward_rtol=operator_ward_rtol,
        require_converged=require_converged,
    )
    plate_1 = integrate_arbitrary_q_vector_adaptive_successive_cached(
        q_model=rotate_lab_q_to_crystal(q, float(theta_1_rad)),
        **common,
    )
    plate_2 = tuple(
        integrate_arbitrary_q_vector_adaptive_successive_cached(
            q_model=rotate_lab_q_to_crystal(q, float(theta)),
            **common,
        )
        for theta in angles
    )
    return TwoPlateAngleBatchResult(
        q_lab=q,
        theta_1_rad=float(theta_1_rad),
        theta_2_rad_values=angles,
        plate_1=plate_1,
        plate_2=plate_2,
        response_cache_metadata=local_cache.metadata(),
    )


__all__ = [
    "ReusableHierarchicalMaterialNodeCache",
    "build_reusable_hierarchical_material_node_cache",
    "integrate_arbitrary_q_vector_adaptive_successive_cached",
    "integrate_two_plate_angle_batch_vector_adaptive_successive_cached",
]
