"""Diagnostic-only audit of implemented vs Ward-required contact terms.

This module does not fit or accept a production convention. It compares the
current primitive EM contact contraction against the contact contraction
required by the fixed finite-q Matsubara primitive Ward identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..adapters.bubble_adapter import TargetBareBlocks, compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .collective_schur_factors import collective_order_from_ansatz
from .contact_ablation import _shifted_payload, scaled_contact_blocks
from .extended_ward_kernel import PRIMITIVE_ORDER, complex_vector_payload
from .primitive_response_closure_suite import DEFAULT_CANDIDATE
from .primitive_response_ward_audit import primitive_schur_effective, primitive_ward_candidate_vectors
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .ward_basis_convention_scan import primitive_blocks_from_baseline

SCHEMA_VERSION = "finite_q_tmte_contact_formula_audit_v1"
RATIO_EPS = 1e-14


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-30) -> float:
    return float(numerator) / max(float(denominator), float(eps))


def _vec(values: np.ndarray, labels: Sequence[str]) -> list[dict[str, Any]]:
    return complex_vector_payload(np.asarray(values, dtype=complex).reshape(-1), labels)


def _candidate_lookup(xi_eV: float, q_norm: float, delta0_eV: float) -> dict[str, dict[str, Any]]:
    return {row["candidate"]: row for row in primitive_ward_candidate_vectors(xi_eV, q_norm, delta0_eV)}


def scalar_projection(required: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    """Return alpha minimizing ||required - alpha*current||."""

    r = np.asarray(required, dtype=complex).reshape(-1)
    c = np.asarray(current, dtype=complex).reshape(-1)
    denom = np.vdot(c, c)
    if abs(denom) < 1e-30:
        alpha = 0.0 + 0.0j
    else:
        alpha = np.vdot(c, r) / denom
    residual = r - alpha * c
    return {
        "alpha_required_over_current": complex(alpha),
        "residual_norm": _norm(residual),
        "required_norm": _norm(r),
        "current_norm": _norm(c),
        "residual_over_required_norm": _safe_ratio(_norm(residual), _norm(r)),
        "residual_over_current_norm": _safe_ratio(_norm(residual), _norm(c)),
        "valid_for_casimir_input": False,
    }


def componentwise_ratio(required: np.ndarray, current: np.ndarray, labels: Sequence[str]) -> list[dict[str, Any]]:
    r = np.asarray(required, dtype=complex).reshape(-1)
    c = np.asarray(current, dtype=complex).reshape(-1)
    rows: list[dict[str, Any]] = []
    for label, required_value, current_value in zip(labels, r, c, strict=True):
        if abs(current_value) > RATIO_EPS:
            ratio = required_value / current_value
            ratio_defined = True
        else:
            ratio = 0.0 + 0.0j
            ratio_defined = False
        rows.append(
            {
                "label": label,
                "required": complex(required_value),
                "current": complex(current_value),
                "required_minus_current": complex(required_value - current_value),
                "required_over_current": complex(ratio),
                "ratio_defined": ratio_defined,
                "current_abs": float(abs(current_value)),
                "required_abs": float(abs(required_value)),
                "valid_for_casimir_input": False,
            }
        )
    return rows


def parallelism(required: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    r = np.asarray(required, dtype=complex).reshape(-1)
    c = np.asarray(current, dtype=complex).reshape(-1)
    denom = max(_norm(r) * _norm(c), 1e-30)
    overlap = np.vdot(c, r) / denom
    return {
        "normalized_overlap": complex(overlap),
        "abs_overlap": float(abs(overlap)),
        "real_overlap": float(np.real(overlap)),
        "imag_overlap": float(np.imag(overlap)),
        "required_norm": _norm(r),
        "current_norm": _norm(c),
        "valid_for_casimir_input": False,
    }


def side_contact_audit(
    *,
    side: str,
    current: np.ndarray,
    required: np.ndarray,
    bubble: np.ndarray,
    mixed: np.ndarray,
    labels: Sequence[str] = PRIMITIVE_ORDER,
) -> dict[str, Any]:
    current = np.asarray(current, dtype=complex).reshape(-1)
    required = np.asarray(required, dtype=complex).reshape(-1)
    bubble = np.asarray(bubble, dtype=complex).reshape(-1)
    mixed = np.asarray(mixed, dtype=complex).reshape(-1)
    residual_with_current = bubble + current + mixed
    alpha = scalar_projection(required, current)
    fitted_residual = required - complex(alpha["alpha_required_over_current"]) * current
    return {
        "side": side,
        "ward_formula": "contact_required = -bubble - mixed",
        "bubble_contribution": {"values": _vec(bubble, labels), "norm": _norm(bubble)},
        "mixed_contribution": {"values": _vec(mixed, labels), "norm": _norm(mixed)},
        "contact_current": {"values": _vec(current, labels), "norm": _norm(current)},
        "contact_required": {"values": _vec(required, labels), "norm": _norm(required)},
        "current_minus_required": {"values": _vec(current - required, labels), "norm": _norm(current - required)},
        "ward_residual_with_current_contact": {"values": _vec(residual_with_current, labels), "norm": _norm(residual_with_current)},
        "required_over_current_scalar_projection": alpha,
        "residual_after_scalar_projection": {"values": _vec(fitted_residual, labels), "norm": _norm(fitted_residual)},
        "componentwise_required_over_current": componentwise_ratio(required, current, labels),
        "parallelism": parallelism(required, current),
        "accepted_convention": False,
        "valid_for_casimir_input": False,
    }


def analyze_contact_formula(
    *,
    primitive: dict[str, Any],
    candidate: dict[str, Any],
    collective_order: tuple[str, ...],
) -> dict[str, Any]:
    k_ss_bubble = np.asarray(primitive["k_ss_bubble"], dtype=complex)
    k_ss_contact = np.asarray(primitive["k_ss_contact"], dtype=complex)
    k_seta = np.asarray(primitive["k_seta"], dtype=complex)
    k_etas = np.asarray(primitive["k_etas"], dtype=complex)
    u_left = np.asarray(candidate["left_u"], dtype=complex).reshape(3)
    u_right = np.asarray(candidate["right_u"], dtype=complex).reshape(3)
    w_left = np.asarray(candidate["left_w"], dtype=complex).reshape(len(collective_order))
    w_right = np.asarray(candidate["right_w"], dtype=complex).reshape(len(collective_order))

    left_bubble = u_left @ k_ss_bubble
    left_mixed = w_left @ k_etas
    left_current = u_left @ k_ss_contact
    left_required = -left_bubble - left_mixed

    right_bubble = k_ss_bubble @ u_right
    right_mixed = k_seta @ w_right
    right_current = k_ss_contact @ u_right
    right_required = -right_bubble - right_mixed

    left = side_contact_audit(
        side="left_observable_row",
        current=left_current,
        required=left_required,
        bubble=left_bubble,
        mixed=left_mixed,
    )
    right = side_contact_audit(
        side="right_source_column",
        current=right_current,
        required=right_required,
        bubble=right_bubble,
        mixed=right_mixed,
    )
    alpha_left = complex(left["required_over_current_scalar_projection"]["alpha_required_over_current"])
    alpha_right = complex(right["required_over_current_scalar_projection"]["alpha_required_over_current"])
    return {
        "candidate": str(candidate["candidate"]),
        "description": str(candidate["description"]),
        "left_primitive_vector": _vec(u_left, PRIMITIVE_ORDER),
        "right_primitive_vector": _vec(u_right, PRIMITIVE_ORDER),
        "left_collective_vector": _vec(w_left, collective_order),
        "right_collective_vector": _vec(w_right, collective_order),
        "left": left,
        "right": right,
        "left_right_scalar_consistency": {
            "alpha_left": alpha_left,
            "alpha_right": alpha_right,
            "alpha_left_minus_right": alpha_left - alpha_right,
            "abs_difference": float(abs(alpha_left - alpha_right)),
            "valid_for_casimir_input": False,
        },
        "interpretation_guardrails": {
            "not_a_fit_fix": True,
            "production_contact_coefficient_must_be_derived_not_fitted": True,
            "if_parallelism_high_and_alpha_not_one": "implemented contact contraction has the right direction but wrong magnitude in this Ward direction",
            "if_component_ratios_disagree": "scalar projection is only a directional artifact; inspect tensor/projection/endpoint routing",
            "valid_for_casimir_input": False,
        },
        "accepted_convention": False,
        "valid_for_casimir_input": False,
    }


def block_norms(primitive: dict[str, Any], effective: np.ndarray, schur_correction: np.ndarray) -> dict[str, Any]:
    return {
        "K_SS_bubble_norm": _norm(primitive["k_ss_bubble"]),
        "K_SS_contact_norm": _norm(primitive["k_ss_contact"]),
        "K_SS_total_norm": _norm(primitive["k_ss"]),
        "K_Seta_norm": _norm(primitive["k_seta"]),
        "K_etaS_norm": _norm(primitive["k_etas"]),
        "K_etaeta_bubble_norm": _norm(primitive["k_etaeta_bubble"]),
        "K_etaeta_counterterm_norm": _norm(primitive["k_etaeta_counterterm"]),
        "K_etaeta_total_norm": _norm(primitive["k_etaeta"]),
        "Schur_correction_norm": _norm(schur_correction),
        "K_eff_norm": _norm(effective),
        "valid_for_casimir_input": False,
    }


def run_contact_formula_audit(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_value: float,
    nk: int,
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    contact_scale: float = 1.0,
    candidate_name: str = DEFAULT_CANDIDATE,
) -> dict[str, Any]:
    xi_eV = matsubara_xi_eV(matsubara_index, temperature_K)
    inputs = build_model_scan_inputs(
        model_name=model_name,
        pairing_name=pairing_name,
        xi_eV=xi_eV,
        nk=nk,
        delta0_eV=delta0_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
    )
    q_model = np.asarray([float(q_value), 0.0], dtype=float)
    shifts = shift_pairs_from_fractions(shift_fractions)
    scaled_blocks: list[TargetBareBlocks] = []
    for sx, sy in shifts:
        points = shifted_uniform_bz_mesh(nk, sx, sy)
        weights = weights_for_points(points)
        blocks = compute_target_bare_blocks(
            spec=inputs.spec,
            ansatz=inputs.ansatz,
            q_model=q_model,
            xi_eV=xi_eV,
            k_points=points,
            weights=weights,
            config=inputs.config,
            pairing_params=inputs.pairing_params,
        )
        scaled_blocks.append(scaled_contact_blocks(blocks, contact_scale))
    baseline = average_bare_blocks_then_schur(scaled_blocks).bare_blocks
    collective_order, raw_names = collective_order_from_ansatz(inputs.ansatz, baseline.k_etaeta.shape[0])
    primitive = primitive_blocks_from_baseline(baseline)
    effective, schur_correction, schur = primitive_schur_effective(primitive)
    delta0 = float(getattr(inputs.pairing_params, "delta0_eV", 0.0))
    candidates = _candidate_lookup(xi_eV, float(abs(q_value)), delta0)
    if candidate_name not in candidates:
        raise ValueError(f"unknown primitive Ward candidate {candidate_name!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "requires_analytic_derivation_before_convention_change": True,
            "valid_for_casimir_input": False,
            "reason": "contact_formula_audit_not_production_convention",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "frequency": frequency_payload(matsubara_index, temperature_K),
        "debug_parameters": {
            "q_model_convention": "q_along_x_only",
            "q_value": float(q_value),
            "nk": int(nk),
            "shift_fractions": [float(value) for value in shift_fractions],
            "num_shifted_meshes": len(shifts),
            "contact_scale": float(contact_scale),
            "candidate_name": candidate_name,
            "shifted_mesh_average": _shifted_payload(shift_fractions, shifts),
            "note": "Compares current contact contraction against Ward-required contact contraction. The scalar ratio is diagnostic only.",
            "valid_for_casimir_input": False,
        },
        "collective_order": list(collective_order),
        "raw_ansatz_channel_names": list(raw_names) if raw_names is not None else None,
        "primitive_metadata": primitive["metadata"],
        "block_norms": block_norms(primitive, effective, schur_correction),
        "schur": schur,
        "contact_formula_analysis": analyze_contact_formula(
            primitive=primitive,
            candidate=candidates[candidate_name],
            collective_order=collective_order,
        ),
        "valid_for_casimir_input": False,
    }


def run_and_write_contact_formula_audit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_contact_formula_audit(**kwargs)
    write_json(Path(output_dir) / "contact_formula_audit.json", payload)
    return payload
