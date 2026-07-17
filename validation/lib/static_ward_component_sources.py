"""Component-level source localization for the static Ward contract audit.

The base audit separates the observed longitudinal contraction into the effective
finite-quadrature RHS and the primitive closure residual.  This module further
splits those terms into bubble/direct/contact and collective bubble/counterterm
sources using ``BdGFiniteQResponseComponents``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from lno327.response.effective_kernel import EffectiveEMKernel
from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.ward_validation import PrimitiveWardRHS
from validation.lib.static_ward_contract_audit import audit_static_ward_contract


def _norm(value: Any) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _left_lt(vector: np.ndarray, transform: np.ndarray, q_norm: float) -> np.ndarray:
    return np.asarray(vector, dtype=complex) @ transform.T / q_norm


def _right_lt(vector: np.ndarray, transform: np.ndarray, q_norm: float) -> np.ndarray:
    return transform @ np.asarray(vector, dtype=complex) / q_norm


def _source_payload(
    sources: Mapping[str, np.ndarray],
    *,
    transform: np.ndarray,
    q_norm: float,
    orientation: str,
) -> dict[str, Any]:
    if orientation == "left":
        to_lt = _left_lt
    elif orientation == "right":
        to_lt = _right_lt
    else:
        raise ValueError(f"unknown orientation {orientation!r}")
    norms = {name: _norm(vector) for name, vector in sources.items()}
    largest = max(norms, key=norms.get) if norms else "none"
    maximum = max(norms.values(), default=0.0)
    total = sum((np.asarray(vector, dtype=complex) for vector in sources.values()), np.zeros(3, dtype=complex))
    return {
        "sources": dict(sources),
        "sources_lt_over_q": {
            name: to_lt(vector, transform, q_norm) for name, vector in sources.items()
        },
        "source_norms": norms,
        "source_norms_over_q": {name: value / q_norm for name, value in norms.items()},
        "largest_source": largest,
        "largest_source_norm": maximum,
        "sum": total,
        "sum_norm": _norm(total),
        "cancellation_ratio": _norm(total) / max(maximum, 1e-30),
    }


def _side_component_sources(
    *,
    orientation: str,
    audit_side: dict[str, Any],
    u: np.ndarray,
    w: np.ndarray,
    components: BdGFiniteQResponseComponents,
    kernel: EffectiveEMKernel,
    transform: np.ndarray,
    q_norm: float,
    rhs_pieces: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    bubble = np.asarray(components.bare_bubble, dtype=complex)
    direct = np.asarray(components.direct, dtype=complex)
    collective_bubble = np.asarray(components.collective_bubble, dtype=complex)
    collective_counterterm = np.asarray(components.collective_counterterm, dtype=complex)
    inverse = np.linalg.inv(np.asarray(kernel.k_etaeta, dtype=complex))

    translation_without_contact = (
        np.asarray(rhs_pieces["equal_forward"], dtype=complex)
        + np.asarray(rhs_pieces["minus_delta_v_mid"], dtype=complex)
    )
    contact_rhs = np.asarray(rhs_pieces["qM_mid"], dtype=complex)

    if orientation == "left":
        primitive_bubble = u @ bubble
        primitive_direct = u @ direct
        primitive_collective = w @ kernel.k_etas
        collective_em = u @ kernel.k_seta
        rotation_bubble = w @ collective_bubble
        rotation_counterterm = w @ collective_counterterm

        def project(defect: np.ndarray) -> np.ndarray:
            return defect @ inverse @ kernel.k_etas

        to_lt = _left_lt
    elif orientation == "right":
        primitive_bubble = bubble @ u
        primitive_direct = direct @ u
        primitive_collective = kernel.k_seta @ w
        collective_em = kernel.k_etas @ u
        rotation_bubble = collective_bubble @ w
        rotation_counterterm = collective_counterterm @ w

        def project(defect: np.ndarray) -> np.ndarray:
            return kernel.k_seta @ inverse @ defect

        to_lt = _right_lt
    else:
        raise ValueError(f"unknown orientation {orientation!r}")

    bubble_translation_residual = (
        primitive_bubble + primitive_collective - translation_without_contact
    )
    contact_residual = primitive_direct - contact_rhs
    primitive_split_sum = bubble_translation_residual + contact_residual
    primitive_split_error = (
        np.asarray(audit_side["primitive_residual"], dtype=complex) - primitive_split_sum
    )

    collective_parts = {
        "em_collective_contraction": collective_em,
        "phase_rotation_bubble": rotation_bubble,
        "phase_rotation_counterterm": rotation_counterterm,
    }
    collective_sum = sum(
        (np.asarray(value, dtype=complex) for value in collective_parts.values()),
        np.zeros(2, dtype=complex),
    )
    collective_split_error = (
        np.asarray(audit_side["collective_defect"], dtype=complex) - collective_sum
    )

    projection_parts = {name: project(value) for name, value in collective_parts.items()}
    projection_sum = sum(
        (np.asarray(value, dtype=complex) for value in projection_parts.values()),
        np.zeros(3, dtype=complex),
    )
    projection_split_error = (
        np.asarray(audit_side["collective_projection"], dtype=complex) - projection_sum
    )

    effective_sources = {
        "equal_forward": np.asarray(rhs_pieces["equal_forward"], dtype=complex),
        "minus_delta_v_mid": np.asarray(rhs_pieces["minus_delta_v_mid"], dtype=complex),
        "qM_mid": contact_rhs,
        "minus_projection_em_collective": -projection_parts[
            "em_collective_contraction"
        ],
        "minus_projection_phase_bubble": -projection_parts["phase_rotation_bubble"],
        "minus_projection_phase_counterterm": -projection_parts[
            "phase_rotation_counterterm"
        ],
    }
    effective_payload = _source_payload(
        effective_sources,
        transform=transform,
        q_norm=q_norm,
        orientation=orientation,
    )
    effective_source_error = (
        np.asarray(audit_side["effective_predicted"], dtype=complex)
        - np.asarray(effective_payload["sum"], dtype=complex)
    )

    return {
        "orientation": orientation,
        "primitive_bubble_contraction": primitive_bubble,
        "primitive_direct_contraction": primitive_direct,
        "primitive_collective_contraction": primitive_collective,
        "translation_without_contact_rhs": translation_without_contact,
        "contact_rhs": contact_rhs,
        "bubble_translation_residual": bubble_translation_residual,
        "contact_residual": contact_residual,
        "primitive_split_sum": primitive_split_sum,
        "primitive_split_error": primitive_split_error,
        "primitive_split_error_norm": _norm(primitive_split_error),
        "bubble_translation_residual_norm_over_q": _norm(
            bubble_translation_residual
        )
        / q_norm,
        "contact_residual_norm_over_q": _norm(contact_residual) / q_norm,
        "bubble_translation_residual_lt_over_q": to_lt(
            bubble_translation_residual, transform, q_norm
        ),
        "contact_residual_lt_over_q": to_lt(contact_residual, transform, q_norm),
        "collective_defect_parts": collective_parts,
        "collective_defect_part_norms": {
            name: _norm(value) for name, value in collective_parts.items()
        },
        "collective_defect_part_norms_over_q": {
            name: _norm(value) / q_norm for name, value in collective_parts.items()
        },
        "collective_defect_sum": collective_sum,
        "collective_split_error": collective_split_error,
        "collective_split_error_norm": _norm(collective_split_error),
        "collective_projection_parts": projection_parts,
        "collective_projection_part_norms": {
            name: _norm(value) for name, value in projection_parts.items()
        },
        "collective_projection_part_norms_over_q": {
            name: _norm(value) / q_norm for name, value in projection_parts.items()
        },
        "collective_projection_parts_lt_over_q": {
            name: to_lt(value, transform, q_norm)
            for name, value in projection_parts.items()
        },
        "collective_projection_sum": projection_sum,
        "projection_split_error": projection_split_error,
        "projection_split_error_norm": _norm(projection_split_error),
        "effective_predicted_sources": effective_payload,
        "effective_source_error": effective_source_error,
        "effective_source_error_norm": _norm(effective_source_error),
    }


def audit_static_ward_contract_with_components(
    kernel: EffectiveEMKernel,
    rhs: PrimitiveWardRHS,
    components: BdGFiniteQResponseComponents,
    *,
    energy_scale_eV: float = 1.0,
) -> dict[str, Any]:
    """Return the base audit enriched with component-level source decompositions."""

    audit = audit_static_ward_contract(
        kernel, rhs, energy_scale_eV=energy_scale_eV
    )
    q = np.asarray(kernel.q_model, dtype=float)
    q_norm = float(audit["q_norm"])
    transform = np.eye(3, dtype=float)
    transform[1:3, 1:3] = np.asarray(
        [
            [q[0] / q_norm, q[1] / q_norm],
            [-q[1] / q_norm, q[0] / q_norm],
        ],
        dtype=float,
    )
    rhs_pieces = audit["left"]["rhs_pieces"]
    left = _side_component_sources(
        orientation="left",
        audit_side=audit["left"],
        u=np.asarray(audit["u_left"], dtype=complex),
        w=np.asarray(audit["w_left"], dtype=complex),
        components=components,
        kernel=kernel,
        transform=transform,
        q_norm=q_norm,
        rhs_pieces=rhs_pieces,
    )
    right = _side_component_sources(
        orientation="right",
        audit_side=audit["right"],
        u=np.asarray(audit["u_right"], dtype=complex),
        w=np.asarray(audit["w_right"], dtype=complex),
        components=components,
        kernel=kernel,
        transform=transform,
        q_norm=q_norm,
        rhs_pieces=rhs_pieces,
    )
    audit["component_sources"] = {"left": left, "right": right}
    audit["component_source_consistency_max"] = max(
        float(left["primitive_split_error_norm"]),
        float(right["primitive_split_error_norm"]),
        float(left["collective_split_error_norm"]),
        float(right["collective_split_error_norm"]),
        float(left["projection_split_error_norm"]),
        float(right["projection_split_error_norm"]),
        float(left["effective_source_error_norm"]),
        float(right["effective_source_error_norm"]),
    )
    audit["schema"] = "static_ward_contract_component_source_audit_v1"
    return audit


__all__ = ["audit_static_ward_contract_with_components"]
