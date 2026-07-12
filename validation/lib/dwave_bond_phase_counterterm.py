"""Validation compatibility wrapper for the core d-wave bond phase metric."""

from __future__ import annotations

import numpy as np

from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.phase_hessian import (
    PhaseHessianApplication,
    apply_phase_hessian_policy_to_components,
    nearest_neighbor_dwave_bond_metric,
)

DWaveBondPhaseCountertermApplication = PhaseHessianApplication


def apply_nearest_neighbor_dwave_phase_counterterm(
    components: BdGFiniteQResponseComponents,
    q_model: np.ndarray,
    *,
    condition_threshold: float = 1e12,
) -> tuple[BdGFiniteQResponseComponents, DWaveBondPhaseCountertermApplication]:
    """Apply the opt-in core bond phase metric to one validation response."""

    model_input = components.metadata.get("model_input_layer")
    if not isinstance(model_input, dict):
        raise ValueError("components metadata is missing model_input_layer")

    class _MetadataAnsatz:
        name = model_input.get("name")
        phase_vertex = model_input.get("phase_vertex")

    return apply_phase_hessian_policy_to_components(
        components,
        _MetadataAnsatz(),
        q_model,
        "nearest_neighbor_bond_metric",
        condition_threshold=condition_threshold,
    )


__all__ = [
    "DWaveBondPhaseCountertermApplication",
    "apply_nearest_neighbor_dwave_phase_counterterm",
    "nearest_neighbor_dwave_bond_metric",
]
