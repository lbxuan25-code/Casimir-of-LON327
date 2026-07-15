"""Reusable-cache entry points for successive-high arbitrary-q adaptive cubature."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from lno327.workflows.arbitrary_q_matsubara import TwoPlateAngleBatchResult
from lno327.workflows.arbitrary_q_vector_adaptive_cached import (
    ReusableHierarchicalMaterialNodeCache,
    build_reusable_hierarchical_material_node_cache,
)
from lno327.workflows.arbitrary_q_vector_adaptive_successive import (
    ArbitraryQVectorAdaptiveSuccessiveOptions,
    ArbitraryQVectorAdaptiveSuccessiveResponseCache,
    integrate_arbitrary_q_vector_adaptive_successive,
    integrate_two_plate_angle_batch_vector_adaptive_successive,
)


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
    cache = node_cache or build_reusable_hierarchical_material_node_cache(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        max_cache_nodes=max_cache_nodes,
        max_cache_bytes=max_cache_bytes,
    )
    return integrate_arbitrary_q_vector_adaptive_successive(
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
    return integrate_two_plate_angle_batch_vector_adaptive_successive(
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
    "integrate_arbitrary_q_vector_adaptive_successive_cached",
    "integrate_two_plate_angle_batch_vector_adaptive_successive_cached",
]
