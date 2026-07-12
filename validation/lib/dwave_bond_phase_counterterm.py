"""Diagnostic finite-q d-wave phase counterterm from the bond gauge metric.

This module deliberately lives under ``validation``.  It rebuilds only the
collective phase diagonal and the resulting amplitude/phase Schur complement;
the production response engine and its default q-independent Goldstone
counterterm remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from lno327.collective.schur import apply_amplitude_phase_schur
from lno327.response.finite_q import BdGFiniteQResponseComponents


@dataclass(frozen=True)
class DWaveBondPhaseCountertermApplication:
    """Audit record for one diagnostic phase-counterterm replacement."""

    multiplier: float
    base_counterterm: np.ndarray
    applied_counterterm: np.ndarray
    phase_counterterm_delta: complex
    schur_condition_number: float | None
    schur_inverse_method: str


def nearest_neighbor_dwave_bond_metric(q_model: np.ndarray) -> float:
    """Return ``[cos(qx/2)^2 + cos(qy/2)^2] / 2`` for one finite q."""

    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")
    return float(0.5 * (np.cos(0.5 * q[0]) ** 2 + np.cos(0.5 * q[1]) ** 2))


def _require_supported_dwave_vertex(metadata: Mapping[str, Any]) -> None:
    model_input = metadata.get("model_input_layer")
    if not isinstance(model_input, Mapping):
        raise ValueError("components metadata is missing model_input_layer")
    if model_input.get("name") != "dwave":
        raise ValueError("bond phase metric is defined here only for the d-wave ansatz")
    if model_input.get("phase_vertex") != "bond_endpoint_gauge":
        raise ValueError(
            "bond phase metric requires phase_vertex='bond_endpoint_gauge'"
        )


def apply_nearest_neighbor_dwave_phase_counterterm(
    components: BdGFiniteQResponseComponents,
    q_model: np.ndarray,
    *,
    condition_threshold: float = 1e12,
) -> tuple[BdGFiniteQResponseComponents, DWaveBondPhaseCountertermApplication]:
    """Replace only ``K_eta2_eta2^HS`` and rebuild the full collective Schur kernel.

    The input response must already contain the complete amplitude/phase blocks.
    ``K_11`` and both amplitude--phase off-diagonal counterterms are preserved
    exactly.  The returned response remains diagnostic-only and invalid for
    Casimir input.
    """

    metadata = dict(components.metadata)
    _require_supported_dwave_vertex(metadata)

    threshold = float(condition_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("condition_threshold must be finite and positive")

    base = np.asarray(components.collective_counterterm, dtype=complex)
    bubble = np.asarray(components.collective_bubble, dtype=complex)
    if base.shape != (2, 2) or bubble.shape != (2, 2):
        raise ValueError("collective counterterm and bubble must both have shape (2, 2)")
    if not np.isfinite(base.real).all() or not np.isfinite(base.imag).all():
        raise ValueError("collective counterterm must be finite")

    multiplier = nearest_neighbor_dwave_bond_metric(q_model)
    applied = np.array(base, dtype=complex, copy=True)
    applied[1, 1] = multiplier * base[1, 1]
    collective_total = bubble + applied

    schur = apply_amplitude_phase_schur(
        components.bare_total,
        components.em_collective_left,
        collective_total,
        components.collective_em_right,
        condition_threshold=threshold,
    )

    phase_delta = complex(applied[1, 1] - base[1, 1])
    metadata.update(
        {
            "diagnostic_only": True,
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
            "casimir_gating_status": (
                "diagnostic_nearest_neighbor_dwave_bond_phase_counterterm_not_promoted"
            ),
            "diagnostic_phase_counterterm_policy": (
                "nearest_neighbor_dwave_bond_phase_metric"
            ),
            "diagnostic_phase_counterterm_multiplier": multiplier,
            "diagnostic_phase_counterterm_base_22": complex(base[1, 1]),
            "diagnostic_phase_counterterm_applied_22": complex(applied[1, 1]),
            "diagnostic_phase_counterterm_delta_22": phase_delta,
            "diagnostic_phase_counterterm_changed_only_22": bool(
                np.array_equal(applied[0:1, :], base[0:1, :])
                and applied[1, 0] == base[1, 0]
            ),
            "finite_q_phase_hessian_source": (
                "pullback_of_isotropic_nearest_neighbor_xy_bond_metric"
            ),
            "collective_total_condition_number": schur.condition_number,
            "collective_inverse_method": schur.inverse_method,
            "amplitude_phase_schur_status": schur.status,
            "selected_gauge_restored": "amplitude_phase_schur",
            "gauge_restored_selected": "amplitude_phase_schur",
            "phase_correction_applied": True,
            "phase_correction_status": "diagnostic_bond_phase_metric_applied",
            "goldstone_counterterm_Cg": complex(applied[1, 1]),
            "warning": schur.warning,
        }
    )

    corrected = replace(
        components,
        collective_counterterm=applied,
        collective_total=collective_total,
        amplitude_phase_schur=schur.corrected_response,
        gauge_restored=schur.corrected_response,
        metadata=metadata,
    )
    application = DWaveBondPhaseCountertermApplication(
        multiplier=multiplier,
        base_counterterm=np.array(base, copy=True),
        applied_counterterm=np.array(applied, copy=True),
        phase_counterterm_delta=phase_delta,
        schur_condition_number=schur.condition_number,
        schur_inverse_method=schur.inverse_method,
    )
    return corrected, application


__all__ = [
    "DWaveBondPhaseCountertermApplication",
    "apply_nearest_neighbor_dwave_phase_counterterm",
    "nearest_neighbor_dwave_bond_metric",
]
