"""Diagnostic decomposition of the exact-static Ward and longitudinal residuals.

This module does not alter the response kernel, Ward gate, longitudinal policy, or
Casimir readiness.  It expands the already-integrated primitive blocks into the
terms appearing in the analytic static gauge contract:

    u K_eff = R_S - C_eta K_etaeta^{-1} K_etaS + r_primitive.

The same decomposition is evaluated on the right.  At exact zero Matsubara
frequency the contractions divided by |q| are the longitudinal row and column of
``K_eff`` in the local ``(A0, L, T)`` basis.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from lno327.electrodynamics.basis import xy_to_lt_rotation
from lno327.response.effective_kernel import EffectiveEMKernel
from lno327.response.ward_validation import PrimitiveWardRHS, primitive_ward_vectors_xy


COLLECTIVE_LABELS = ("amplitude_eta1", "phase_eta2")
EM_LT_LABELS = ("A0", "L", "T")


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=complex)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array.real).all() or not np.isfinite(array.imag).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _norm(value: Any) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1e-30)


def _rhs_piece(metadata: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in metadata:
        raise ValueError(
            f"PrimitiveWardRHS.metadata is missing {name!r}; the source audit "
            "requires separately integrated equal, delta-v, and qM pieces"
        )
    return _array(metadata[name], (3,), f"rhs.metadata[{name!r}]")


def _lt_transform(q_model: np.ndarray) -> tuple[np.ndarray, float]:
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        raise ValueError("static Ward contract audit requires nonzero q_model")
    transform = np.eye(3, dtype=float)
    transform[1:3, 1:3] = xy_to_lt_rotation(float(q[0]), float(q[1]))
    return transform, q_norm


def _left_to_lt(vector: np.ndarray, transform: np.ndarray, q_norm: float) -> np.ndarray:
    return np.asarray(vector, dtype=complex) @ transform.T / q_norm


def _right_to_lt(vector: np.ndarray, transform: np.ndarray, q_norm: float) -> np.ndarray:
    return transform @ np.asarray(vector, dtype=complex) / q_norm


def _source_classification(effective_predicted_norm: float, primitive_residual_norm: float) -> str:
    if effective_predicted_norm <= 1e-14 and primitive_residual_norm <= 1e-14:
        return "closed_to_absolute_floor"
    if effective_predicted_norm >= 10.0 * max(primitive_residual_norm, 1e-30):
        return "external_collective_mismatch_dominant"
    if primitive_residual_norm >= 10.0 * max(effective_predicted_norm, 1e-30):
        return "primitive_closure_residual_dominant"
    return "mixed_external_collective_and_primitive"


def _external_collective_classification(
    rhs_norm: float,
    projection_norm: float,
    predicted_norm: float,
) -> str:
    if rhs_norm <= 1e-14 and projection_norm <= 1e-14:
        return "both_at_absolute_floor"
    if rhs_norm >= 10.0 * max(projection_norm, 1e-30):
        return "external_rhs_dominant"
    if projection_norm >= 10.0 * max(rhs_norm, 1e-30):
        return "collective_projection_dominant"
    cancellation = predicted_norm / max(rhs_norm, projection_norm, 1e-30)
    if cancellation <= 0.1:
        return "large_external_collective_cancellation"
    return "external_collective_same_scale_without_strong_cancellation"


def _side_payload(
    *,
    orientation: str,
    primitive_em: np.ndarray,
    primitive_collective: np.ndarray,
    primitive_rhs: np.ndarray,
    collective_defect: np.ndarray,
    inverse: np.ndarray,
    k_mixed: np.ndarray,
    effective_direct: np.ndarray,
    transform: np.ndarray,
    q_norm: float,
    rhs_pieces: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    primitive_total = primitive_em + primitive_collective
    primitive_residual = primitive_total - primitive_rhs

    if orientation == "left":
        solved_coefficients = collective_defect @ inverse
        projection_by_channel = solved_coefficients[:, None] * k_mixed
        collective_projection = np.sum(projection_by_channel, axis=0)
        to_lt = _left_to_lt
    elif orientation == "right":
        solved_coefficients = inverse @ collective_defect
        projection_by_channel = (k_mixed * solved_coefficients[None, :]).T
        collective_projection = np.sum(projection_by_channel, axis=0)
        to_lt = _right_to_lt
    else:
        raise ValueError(f"unknown orientation {orientation!r}")

    effective_predicted = primitive_rhs - collective_projection
    effective_residual = effective_direct - effective_predicted
    reconstructed_direct = effective_predicted + primitive_residual
    reconstruction_error = effective_direct - reconstructed_direct
    residual_identity_error = effective_residual - primitive_residual

    rhs_piece_lt = {
        name: to_lt(vector, transform, q_norm) for name, vector in rhs_pieces.items()
    }
    projection_by_channel_lt = np.stack(
        [to_lt(vector, transform, q_norm) for vector in projection_by_channel], axis=0
    )

    norms = {
        "primitive_em": _norm(primitive_em),
        "primitive_collective": _norm(primitive_collective),
        "primitive_total": _norm(primitive_total),
        "primitive_rhs": _norm(primitive_rhs),
        "primitive_residual": _norm(primitive_residual),
        "collective_defect": _norm(collective_defect),
        "collective_projection": _norm(collective_projection),
        "effective_predicted": _norm(effective_predicted),
        "effective_direct": _norm(effective_direct),
        "effective_residual": _norm(effective_residual),
        "reconstruction_error": _norm(reconstruction_error),
        "residual_identity_error": _norm(residual_identity_error),
    }
    q_normalized = {
        name: value / q_norm
        for name, value in norms.items()
        if name
        in {
            "primitive_rhs",
            "primitive_residual",
            "collective_defect",
            "collective_projection",
            "effective_predicted",
            "effective_direct",
            "effective_residual",
        }
    }

    return {
        "orientation": orientation,
        "primitive_em": primitive_em,
        "primitive_collective": primitive_collective,
        "primitive_total": primitive_total,
        "primitive_rhs": primitive_rhs,
        "primitive_residual": primitive_residual,
        "collective_defect": collective_defect,
        "collective_solved_coefficients": solved_coefficients,
        "collective_projection_by_channel": projection_by_channel,
        "collective_projection": collective_projection,
        "effective_predicted": effective_predicted,
        "effective_direct": effective_direct,
        "effective_residual": effective_residual,
        "reconstructed_effective_direct": reconstructed_direct,
        "reconstruction_error": reconstruction_error,
        "residual_identity_error": residual_identity_error,
        "primitive_rhs_lt_over_q": to_lt(primitive_rhs, transform, q_norm),
        "primitive_residual_lt_over_q": to_lt(primitive_residual, transform, q_norm),
        "collective_projection_lt_over_q": to_lt(
            collective_projection, transform, q_norm
        ),
        "collective_projection_by_channel_lt_over_q": projection_by_channel_lt,
        "effective_predicted_lt_over_q": to_lt(
            effective_predicted, transform, q_norm
        ),
        "effective_direct_lt_over_q": to_lt(effective_direct, transform, q_norm),
        "effective_residual_lt_over_q": to_lt(
            effective_residual, transform, q_norm
        ),
        "rhs_pieces": dict(rhs_pieces),
        "rhs_pieces_lt_over_q": rhs_piece_lt,
        "norms": norms,
        "q_normalized_norms": q_normalized,
        "source_classification": _source_classification(
            norms["effective_predicted"], norms["primitive_residual"]
        ),
        "external_collective_classification": _external_collective_classification(
            norms["primitive_rhs"],
            norms["collective_projection"],
            norms["effective_predicted"],
        ),
        "external_collective_cancellation_ratio": _safe_ratio(
            norms["effective_predicted"],
            max(norms["primitive_rhs"], norms["collective_projection"]),
        ),
        "projection_over_collective_defect": _safe_ratio(
            norms["collective_projection"], norms["collective_defect"]
        ),
    }


def audit_static_ward_contract(
    kernel: EffectiveEMKernel,
    rhs: PrimitiveWardRHS,
    *,
    energy_scale_eV: float = 1.0,
) -> dict[str, Any]:
    """Return a fail-closed diagnostic decomposition for one exact-static kernel.

    The audit is purely observational.  It does not project the kernel and never
    marks a result as a production reference or valid Casimir input.
    """

    if float(kernel.xi_eV) != 0.0 or float(rhs.xi_eV) != 0.0:
        raise ValueError("static Ward contract audit requires xi_eV == 0 exactly")
    if not np.allclose(kernel.q_model, rhs.q_model, rtol=0.0, atol=1e-14):
        raise ValueError("kernel and Ward RHS q_model do not match")
    energy = float(energy_scale_eV)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy_scale_eV must be finite and positive")

    transform, q_norm = _lt_transform(kernel.q_model)
    inverse = np.linalg.inv(np.asarray(kernel.k_etaeta, dtype=complex))
    u_left, u_right, w_left, w_right = primitive_ward_vectors_xy(
        kernel.xi_eV, kernel.q_model, rhs.delta0_eV
    )

    equal = _rhs_piece(rhs.metadata, "equal_forward")
    delta_v = _rhs_piece(rhs.metadata, "delta_v_mid")
    contact = _rhs_piece(rhs.metadata, "qM_mid")
    reconstructed_rhs = equal - delta_v + contact
    rhs_metadata_error_left = reconstructed_rhs - np.asarray(rhs.left, dtype=complex)
    rhs_metadata_error_right = reconstructed_rhs - np.asarray(rhs.right, dtype=complex)
    rhs_pieces = {
        "equal_forward": equal,
        "minus_delta_v_mid": -delta_v,
        "qM_mid": contact,
    }

    left = _side_payload(
        orientation="left",
        primitive_em=u_left @ kernel.k_ss,
        primitive_collective=w_left @ kernel.k_etas,
        primitive_rhs=np.asarray(rhs.left, dtype=complex),
        collective_defect=u_left @ kernel.k_seta + w_left @ kernel.k_etaeta,
        inverse=inverse,
        k_mixed=np.asarray(kernel.k_etas, dtype=complex),
        effective_direct=u_left @ kernel.k_eff,
        transform=transform,
        q_norm=q_norm,
        rhs_pieces=rhs_pieces,
    )
    right = _side_payload(
        orientation="right",
        primitive_em=kernel.k_ss @ u_right,
        primitive_collective=kernel.k_seta @ w_right,
        primitive_rhs=np.asarray(rhs.right, dtype=complex),
        collective_defect=kernel.k_etas @ u_right + kernel.k_etaeta @ w_right,
        inverse=inverse,
        k_mixed=np.asarray(kernel.k_seta, dtype=complex),
        effective_direct=kernel.k_eff @ u_right,
        transform=transform,
        q_norm=q_norm,
        rhs_pieces=rhs_pieces,
    )

    kernel_lt = transform @ np.asarray(kernel.k_eff, dtype=complex) @ transform.T
    scaled = kernel_lt.copy()
    scaled[0, 0] *= energy
    scaled[1:3, 1:3] /= energy
    scale = max(float(np.linalg.norm(scaled.real)), 1.0)
    longitudinal_entries = np.asarray(
        [scaled[0, 1], scaled[1, 0], scaled[1, 1], scaled[1, 2], scaled[2, 1]],
        dtype=complex,
    )
    relative_longitudinal = _norm(longitudinal_entries) / scale

    left_direct_lt = np.asarray(left["effective_direct_lt_over_q"], dtype=complex)
    right_direct_lt = np.asarray(right["effective_direct_lt_over_q"], dtype=complex)
    lt_mapping_error = np.asarray(
        [
            left_direct_lt[0] - kernel_lt[1, 0],
            right_direct_lt[0] - kernel_lt[0, 1],
            left_direct_lt[1] - kernel_lt[1, 1],
            right_direct_lt[1] - kernel_lt[1, 1],
            left_direct_lt[2] - kernel_lt[1, 2],
            right_direct_lt[2] - kernel_lt[2, 1],
        ],
        dtype=complex,
    )

    return {
        "schema": "static_ward_contract_source_audit_v1",
        "status": {
            "diagnostic_run_completed": True,
            "analytic_continuum_target": "zero_longitudinal",
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
        "q_model": np.asarray(kernel.q_model, dtype=float),
        "q_norm": q_norm,
        "xi_eV": float(kernel.xi_eV),
        "delta0_eV": float(rhs.delta0_eV),
        "u_left": u_left,
        "u_right": u_right,
        "w_left": w_left,
        "w_right": w_right,
        "collective_labels": COLLECTIVE_LABELS,
        "lt_labels": EM_LT_LABELS,
        "k_etaeta_inverse": inverse,
        "schur_condition_number": float(np.linalg.cond(kernel.k_etaeta)),
        "schur_inverse_method": str(kernel.schur_inverse_method),
        "rhs_reconstructed_from_pieces": reconstructed_rhs,
        "rhs_metadata_error_left": rhs_metadata_error_left,
        "rhs_metadata_error_right": rhs_metadata_error_right,
        "rhs_metadata_error_norm": max(
            _norm(rhs_metadata_error_left), _norm(rhs_metadata_error_right)
        ),
        "left": left,
        "right": right,
        "kernel_lt": kernel_lt,
        "dimensionless_kernel_lt": scaled,
        "dimensionless_reference_scale": scale,
        "longitudinal_entries_order": ("K_0L", "K_L0", "K_LL", "K_LT", "K_TL"),
        "longitudinal_entries": longitudinal_entries,
        "relative_longitudinal_gauge_residual": relative_longitudinal,
        "lt_contraction_mapping_error": lt_mapping_error,
        "lt_contraction_mapping_error_norm": _norm(lt_mapping_error),
        "max_effective_direct_over_q": max(
            left["q_normalized_norms"]["effective_direct"],
            right["q_normalized_norms"]["effective_direct"],
        ),
        "max_effective_predicted_over_q": max(
            left["q_normalized_norms"]["effective_predicted"],
            right["q_normalized_norms"]["effective_predicted"],
        ),
        "max_primitive_residual_over_q": max(
            left["q_normalized_norms"]["primitive_residual"],
            right["q_normalized_norms"]["primitive_residual"],
        ),
        "max_external_rhs_over_q": max(
            left["q_normalized_norms"]["primitive_rhs"],
            right["q_normalized_norms"]["primitive_rhs"],
        ),
        "max_collective_projection_over_q": max(
            left["q_normalized_norms"]["collective_projection"],
            right["q_normalized_norms"]["collective_projection"],
        ),
        "max_collective_defect_over_q": max(
            left["q_normalized_norms"]["collective_defect"],
            right["q_normalized_norms"]["collective_defect"],
        ),
    }


__all__ = [
    "COLLECTIVE_LABELS",
    "EM_LT_LABELS",
    "audit_static_ward_contract",
]
