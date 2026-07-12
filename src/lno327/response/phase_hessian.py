"""Opt-in finite-q collective phase-Hessian policies.

The q=0 Goldstone gap equation fixes the scalar phase counterterm at the
uniform saddle.  A nonlocal bond pairing vertex can embed the scalar phase
coordinate into a q-dependent bond-space gauge tangent.  This module applies
that geometric pullback after the fermionic response blocks are assembled.

The default policy is deliberately q-independent for backward compatibility.
The nearest-neighbour bond metric is currently supported only for a d-wave
ansatz using the bond-endpoint gauge vertex.  All corrected responses remain
fail-closed and invalid for Casimir input until the production validation grid
is completed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, Mapping

import numpy as np

from lno327.collective.schur import apply_amplitude_phase_schur
from lno327.response.finite_q import BdGFiniteQResponseComponents

PhaseHessianPolicy = Literal["q_independent", "nearest_neighbor_bond_metric"]
SUPPORTED_PHASE_HESSIAN_POLICIES: tuple[PhaseHessianPolicy, ...] = (
    "q_independent",
    "nearest_neighbor_bond_metric",
)


class PhaseHessianApplication(dict[str, Any]):
    """JSON-serializable audit record for one phase-Hessian policy application."""

    def __init__(
        self,
        *,
        policy: PhaseHessianPolicy,
        multiplier: float,
        base_counterterm: np.ndarray,
        applied_counterterm: np.ndarray,
        phase_counterterm_delta: complex,
        schur_condition_number: float | None,
        schur_inverse_method: str,
    ) -> None:
        super().__init__(
            policy=str(policy),
            multiplier=float(multiplier),
            base_counterterm=np.array(base_counterterm, dtype=complex, copy=True),
            applied_counterterm=np.array(applied_counterterm, dtype=complex, copy=True),
            phase_counterterm_delta=complex(phase_counterterm_delta),
            schur_condition_number=(
                None if schur_condition_number is None else float(schur_condition_number)
            ),
            schur_inverse_method=str(schur_inverse_method),
        )

    @property
    def policy(self) -> str:
        return str(self["policy"])

    @property
    def multiplier(self) -> float:
        return float(self["multiplier"])

    @property
    def base_counterterm(self) -> np.ndarray:
        return np.asarray(self["base_counterterm"], dtype=complex)

    @property
    def applied_counterterm(self) -> np.ndarray:
        return np.asarray(self["applied_counterterm"], dtype=complex)

    @property
    def phase_counterterm_delta(self) -> complex:
        return complex(self["phase_counterterm_delta"])

    @property
    def schur_condition_number(self) -> float | None:
        value = self["schur_condition_number"]
        return None if value is None else float(value)

    @property
    def schur_inverse_method(self) -> str:
        return str(self["schur_inverse_method"])


def validate_phase_hessian_policy(value: str) -> PhaseHessianPolicy:
    """Validate and normalize one finite-q phase-Hessian policy name."""

    policy = str(value)
    if policy not in SUPPORTED_PHASE_HESSIAN_POLICIES:
        choices = ", ".join(SUPPORTED_PHASE_HESSIAN_POLICIES)
        raise ValueError(f"phase_hessian_policy must be one of: {choices}")
    return policy  # type: ignore[return-value]


def phase_hessian_policy_from_options(options: object | None) -> PhaseHessianPolicy:
    """Read the policy from an options object with a backward-compatible default."""

    return validate_phase_hessian_policy(
        getattr(options, "phase_hessian_policy", "q_independent")
    )


def nearest_neighbor_dwave_bond_metric(q_model: np.ndarray) -> float:
    """Return ``[cos(qx/2)^2 + cos(qy/2)^2] / 2`` for one finite q."""

    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")
    return float(0.5 * (np.cos(0.5 * q[0]) ** 2 + np.cos(0.5 * q[1]) ** 2))


def _require_supported_bond_metric_ansatz(ansatz: object) -> None:
    name = getattr(ansatz, "name", None)
    phase_vertex = getattr(ansatz, "phase_vertex", None)
    if name != "dwave":
        raise ValueError(
            "nearest_neighbor_bond_metric is supported only for the d-wave ansatz"
        )
    if phase_vertex != "bond_endpoint_gauge":
        raise ValueError(
            "nearest_neighbor_bond_metric requires "
            "phase_vertex='bond_endpoint_gauge'"
        )


def _require_collective_blocks(components: BdGFiniteQResponseComponents) -> None:
    metadata = components.metadata
    if metadata.get("collective_mode") != "amplitude_phase":
        raise ValueError(
            "phase-Hessian policies require collective_mode='amplitude_phase'"
        )
    base = np.asarray(components.collective_counterterm, dtype=complex)
    bubble = np.asarray(components.collective_bubble, dtype=complex)
    if base.shape != (2, 2) or bubble.shape != (2, 2):
        raise ValueError(
            "collective counterterm and bubble must both have shape (2, 2)"
        )
    for name, matrix in (("collective counterterm", base), ("collective bubble", bubble)):
        if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
            raise ValueError(f"{name} must be finite")


def apply_phase_hessian_policy_to_components(
    components: BdGFiniteQResponseComponents,
    ansatz: object,
    q_model: np.ndarray,
    policy: str = "q_independent",
    *,
    condition_threshold: float = 1e12,
) -> tuple[BdGFiniteQResponseComponents, PhaseHessianApplication]:
    """Apply one phase-Hessian policy and rebuild the full collective Schur kernel.

    ``q_independent`` is a numerical no-op.  ``nearest_neighbor_bond_metric``
    changes only the phase diagonal ``K_eta2_eta2^HS``.  The amplitude diagonal
    and both amplitude--phase counterterm entries are preserved exactly.
    """

    selected = validate_phase_hessian_policy(policy)
    threshold = float(condition_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("condition_threshold must be finite and positive")

    _require_collective_blocks(components)
    base = np.asarray(components.collective_counterterm, dtype=complex)
    applied = np.array(base, dtype=complex, copy=True)

    if selected == "q_independent":
        multiplier = 1.0
    else:
        _require_supported_bond_metric_ansatz(ansatz)
        multiplier = nearest_neighbor_dwave_bond_metric(q_model)
        applied[1, 1] = multiplier * base[1, 1]

    collective_total = np.asarray(components.collective_bubble, dtype=complex) + applied
    schur = apply_amplitude_phase_schur(
        components.bare_total,
        components.em_collective_left,
        collective_total,
        components.collective_em_right,
        condition_threshold=threshold,
    )

    changed_only_22 = bool(
        np.array_equal(applied[0:1, :], base[0:1, :])
        and applied[1, 0] == base[1, 0]
    )
    phase_delta = complex(applied[1, 1] - base[1, 1])
    metadata = dict(components.metadata)
    selected_gauge = str(metadata.get("selected_gauge_restored", ""))
    gauge_restored = (
        schur.corrected_response
        if selected_gauge == "amplitude_phase_schur"
        else np.asarray(components.gauge_restored, dtype=complex)
    )
    metadata.update(
        {
            "phase_hessian_policy": selected,
            "phase_hessian_policy_opt_in": selected != "q_independent",
            "phase_hessian_multiplier": multiplier,
            "phase_hessian_base_counterterm_22": complex(base[1, 1]),
            "phase_hessian_applied_counterterm_22": complex(applied[1, 1]),
            "phase_hessian_counterterm_delta_22": phase_delta,
            "phase_hessian_changed_only_22": changed_only_22,
            "phase_hessian_source": (
                "q_independent_goldstone_scalar"
                if selected == "q_independent"
                else "pullback_of_isotropic_nearest_neighbor_xy_bond_metric"
            ),
            "collective_total_condition_number": schur.condition_number,
            "collective_inverse_method": schur.inverse_method,
            "amplitude_phase_schur_status": schur.status,
            "goldstone_counterterm_Cg": complex(applied[1, 1]),
            "diagnostic_only": True,
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
            "casimir_gating_status": (
                "diagnostic_finite_q_phase_hessian_policy_not_production_validated"
                if selected != "q_independent"
                else metadata.get(
                    "casimir_gating_status",
                    "diagnostic_finite_q_response_not_unit_converted_or_ward_validated",
                )
            ),
        }
    )

    corrected = replace(
        components,
        collective_counterterm=applied,
        collective_total=collective_total,
        amplitude_phase_schur=schur.corrected_response,
        gauge_restored=gauge_restored,
        metadata=metadata,
    )
    application = PhaseHessianApplication(
        policy=selected,
        multiplier=multiplier,
        base_counterterm=base,
        applied_counterterm=applied,
        phase_counterterm_delta=phase_delta,
        schur_condition_number=schur.condition_number,
        schur_inverse_method=schur.inverse_method,
    )
    return corrected, application


def finite_q_bdg_response_from_model_ansatz_with_phase_hessian(
    spec: object,
    ansatz: object,
    omega_eV: float,
    q_model: np.ndarray,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: object,
    pairing_params: object | None = None,
    options: object | None = None,
) -> BdGFiniteQResponseComponents:
    """Evaluate the direct engine and apply its explicit phase-Hessian policy."""

    from lno327.response.finite_q_bdg import (
        _DefaultFiniteQOptions,
        finite_q_bdg_response_from_model_ansatz,
    )

    opts = options or _DefaultFiniteQOptions()
    base = finite_q_bdg_response_from_model_ansatz(
        spec,
        ansatz,
        omega_eV,
        q_model,
        k_points,
        k_weights,
        config,
        pairing_params,
        opts,
    )
    corrected, _ = apply_phase_hessian_policy_to_components(
        base,
        ansatz,
        q_model,
        phase_hessian_policy_from_options(opts),
    )
    return corrected


__all__ = [
    "PhaseHessianApplication",
    "PhaseHessianPolicy",
    "SUPPORTED_PHASE_HESSIAN_POLICIES",
    "apply_phase_hessian_policy_to_components",
    "finite_q_bdg_response_from_model_ansatz_with_phase_hessian",
    "nearest_neighbor_dwave_bond_metric",
    "phase_hessian_policy_from_options",
    "validate_phase_hessian_policy",
]
