"""Validation compatibility wrapper for the core d-wave bond phase metric."""

from __future__ import annotations

from dataclasses import replace

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

    corrected, application = apply_phase_hessian_policy_to_components(
        components,
        _MetadataAnsatz(),
        q_model,
        "nearest_neighbor_bond_metric",
        condition_threshold=condition_threshold,
    )
    metadata = dict(corrected.metadata)
    metadata.update(
        {
            "diagnostic_phase_counterterm_policy": (
                "nearest_neighbor_dwave_bond_phase_metric"
            ),
            "diagnostic_phase_counterterm_multiplier": metadata[
                "phase_hessian_multiplier"
            ],
            "diagnostic_phase_counterterm_base_22": metadata[
                "phase_hessian_base_counterterm_22"
            ],
            "diagnostic_phase_counterterm_applied_22": metadata[
                "phase_hessian_applied_counterterm_22"
            ],
            "diagnostic_phase_counterterm_delta_22": metadata[
                "phase_hessian_counterterm_delta_22"
            ],
            "diagnostic_phase_counterterm_changed_only_22": metadata[
                "phase_hessian_changed_only_22"
            ],
            "finite_q_phase_hessian_source": metadata["phase_hessian_source"],
        }
    )
    return replace(corrected, metadata=metadata), application


__all__ = [
    "DWaveBondPhaseCountertermApplication",
    "apply_nearest_neighbor_dwave_phase_counterterm",
    "nearest_neighbor_dwave_bond_metric",
]
