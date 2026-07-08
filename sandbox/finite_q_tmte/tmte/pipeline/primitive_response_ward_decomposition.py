"""Diagnostic-only decomposition of primitive response-level Ward residuals."""

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
from .primitive_response_ward_audit import primitive_schur_effective, primitive_ward_candidate_vectors
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .ward_basis_convention_scan import primitive_blocks_from_baseline

SCHEMA_VERSION = "finite_q_tmte_primitive_response_ward_decomposition_v1"
DEFAULT_CANDIDATES = ("matrix_inferred_matsubara_i_asymmetric",)


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-30) -> float:
    return float(numerator) / max(float(denominator), float(eps))


def _vec(values: np.ndarray, labels: Sequence[str]) -> list[dict[str, Any]]:
    return complex_vector_payload(np.asarray(values, dtype=complex).reshape(-1), labels)


def term_payload(name: str, values: np.ndarray, labels: Sequence[str], reference_norm: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=complex).reshape(-1)
    return {
        "term": name,
        "values": _vec(values, labels),
        "norm": _norm(values),
        "norm_over_reference": _safe_ratio(_norm(values), reference_norm),
        "valid_for_casimir_input": False,
    }


def sum_terms(terms: Sequence[dict[str, Any]]) -> np.ndarray:
    arrays = []
    for term in terms:
        arrays.append(np.asarray([complex(item["value"]) for item in term["values"]], dtype=complex))
    if not arrays:
        raise ValueError("cannot sum empty term list")
    return np.sum(arrays, axis=0)


def _candidate_lookup(xi_eV: float, q_norm: float, delta0_eV: float) -> dict[str, dict[str, Any]]:
    return {row["candidate"]: row for row in primitive_ward_candidate_vectors(xi_eV, q_norm, delta0_eV)}


def decompose_candidate(
    *,
    primitive: dict[str, Any],
    effective: np.ndarray,
    schur_correction: np.ndarray,
    candidate: dict[str, Any],
    collective_order: tuple[str, ...],
    block_reference_norm: float,
    effective_reference_norm: float,
) -> dict[str, Any]:
    k_ss_bubble = np.asarray(primitive["k_ss_bubble"], dtype=complex)
    k_ss_contact = np.asarray(primitive["k_ss_contact"], dtype=complex)
    k_seta = np.asarray(primitive["k_seta"], dtype=complex)
    k_etas = np.asarray(primitive["k_etas"], dtype=complex)
    k_etaeta_bubble = np.asarray(primitive["k_etaeta_bubble"], dtype=complex)
    k_etaeta_counterterm = np.asarray(primitive["k_etaeta_counterterm"], dtype=complex)
    k_etaeta = np.asarray(primitive["k_etaeta"], dtype=complex)
    u_left = np.asarray(candidate["left_u"], dtype=complex).reshape(3)
    u_right = np.asarray(candidate["right_u"], dtype=complex).reshape(3)
    w_left = np.asarray(candidate["left_w"], dtype=complex).reshape(k_etaeta.shape[0])
    w_right = np.asarray(candidate["right_w"], dtype=complex).reshape(k_etaeta.shape[0])

    left_em_terms = [
        term_payload("u_L @ K_SS_bubble", u_left @ k_ss_bubble, PRIMITIVE_ORDER, block_reference_norm),
        term_payload("u_L @ K_SS_contact", u_left @ k_ss_contact, PRIMITIVE_ORDER, block_reference_norm),
        term_payload("W_L @ K_etaS", w_left @ k_etas, PRIMITIVE_ORDER, block_reference_norm),
    ]
    left_collective_terms = [
        term_payload("u_L @ K_Seta", u_left @ k_seta, collective_order, block_reference_norm),
        term_payload("W_L @ K_etaeta_bubble", w_left @ k_etaeta_bubble, collective_order, block_reference_norm),
        term_payload("W_L @ K_etaeta_counterterm", w_left @ k_etaeta_counterterm, collective_order, block_reference_norm),
    ]
    right_em_terms = [
        term_payload("K_SS_bubble @ u_R", k_ss_bubble @ u_right, PRIMITIVE_ORDER, block_reference_norm),
        term_payload("K_SS_contact @ u_R", k_ss_contact @ u_right, PRIMITIVE_ORDER, block_reference_norm),
        term_payload("K_Seta @ W_R", k_seta @ w_right, PRIMITIVE_ORDER, block_reference_norm),
    ]
    right_collective_terms = [
        term_payload("K_etaS @ u_R", k_etas @ u_right, collective_order, block_reference_norm),
        term_payload("K_etaeta_bubble @ W_R", k_etaeta_bubble @ w_right, collective_order, block_reference_norm),
        term_payload("K_etaeta_counterterm @ W_R", k_etaeta_counterterm @ w_right, collective_order, block_reference_norm),
    ]

    left_em_total = sum_terms(left_em_terms)
    left_collective_total = sum_terms(left_collective_terms)
    right_em_total = sum_terms(right_em_terms)
    right_collective_total = sum_terms(right_collective_terms)
    left_schur_terms = [
        term_payload("u_L @ K_SS_bubble", u_left @ k_ss_bubble, PRIMITIVE_ORDER, effective_reference_norm),
        term_payload("u_L @ K_SS_contact", u_left @ k_ss_contact, PRIMITIVE_ORDER, effective_reference_norm),
        term_payload("- u_L @ Schur_correction", -(u_left @ schur_correction), PRIMITIVE_ORDER, effective_reference_norm),
    ]
    right_schur_terms = [
        term_payload("K_SS_bubble @ u_R", k_ss_bubble @ u_right, PRIMITIVE_ORDER, effective_reference_norm),
        term_payload("K_SS_contact @ u_R", k_ss_contact @ u_right, PRIMITIVE_ORDER, effective_reference_norm),
        term_payload("- Schur_correction @ u_R", -(schur_correction @ u_right), PRIMITIVE_ORDER, effective_reference_norm),
    ]
    left_schur_total = sum_terms(left_schur_terms)
    right_schur_total = sum_terms(right_schur_terms)

    return {
        "candidate": str(candidate["candidate"]),
        "description": str(candidate["description"]),
        "left_primitive_vector": _vec(u_left, PRIMITIVE_ORDER),
        "right_primitive_vector": _vec(u_right, PRIMITIVE_ORDER),
        "left_collective_vector": _vec(w_left, collective_order),
        "right_collective_vector": _vec(w_right, collective_order),
        "left_em_decomposition": {
            "terms": left_em_terms,
            "total": term_payload("left_em_total", left_em_total, PRIMITIVE_ORDER, block_reference_norm),
            "valid_for_casimir_input": False,
        },
        "left_collective_decomposition": {
            "terms": left_collective_terms,
            "total": term_payload("left_collective_total", left_collective_total, collective_order, block_reference_norm),
            "valid_for_casimir_input": False,
        },
        "right_em_decomposition": {
            "terms": right_em_terms,
            "total": term_payload("right_em_total", right_em_total, PRIMITIVE_ORDER, block_reference_norm),
            "valid_for_casimir_input": False,
        },
        "right_collective_decomposition": {
            "terms": right_collective_terms,
            "total": term_payload("right_collective_total", right_collective_total, collective_order, block_reference_norm),
            "valid_for_casimir_input": False,
        },
        "schur_effective_decomposition": {
            "left_terms": left_schur_terms,
            "left_total": term_payload("u_L @ K_eff_total", left_schur_total, PRIMITIVE_ORDER, effective_reference_norm),
            "right_terms": right_schur_terms,
            "right_total": term_payload("K_eff_total @ u_R", right_schur_total, PRIMITIVE_ORDER, effective_reference_norm),
            "direct_left_effective_check": term_payload("u_L @ K_eff_direct", u_left @ np.asarray(effective, dtype=complex), PRIMITIVE_ORDER, effective_reference_norm),
            "direct_right_effective_check": term_payload("K_eff_direct @ u_R", np.asarray(effective, dtype=complex) @ u_right, PRIMITIVE_ORDER, effective_reference_norm),
            "valid_for_casimir_input": False,
        },
        "norm_summary": {
            "left_em_total_norm": _norm(left_em_total),
            "left_collective_total_norm": _norm(left_collective_total),
            "right_em_total_norm": _norm(right_em_total),
            "right_collective_total_norm": _norm(right_collective_total),
            "left_schur_total_norm": _norm(left_schur_total),
            "right_schur_total_norm": _norm(right_schur_total),
            "left_extended_total_norm": float(np.sqrt(_norm(left_em_total) ** 2 + _norm(left_collective_total) ** 2)),
            "right_extended_total_norm": float(np.sqrt(_norm(right_em_total) ** 2 + _norm(right_collective_total) ** 2)),
            "left_schur_over_effective_norm": _safe_ratio(_norm(left_schur_total), effective_reference_norm),
            "right_schur_over_effective_norm": _safe_ratio(_norm(right_schur_total), effective_reference_norm),
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
        "schur_correction_norm": _norm(schur_correction),
        "K_eff_norm": _norm(effective),
        "valid_for_casimir_input": False,
    }


def run_primitive_response_ward_decomposition(
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
    candidate_names: Sequence[str] = DEFAULT_CANDIDATES,
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
    norms = block_norms(primitive, effective, schur_correction)
    block_reference_norm = max(
        norms["K_SS_total_norm"], norms["K_Seta_norm"], norms["K_etaS_norm"], norms["K_etaeta_total_norm"], 1e-30
    )
    effective_reference_norm = max(norms["K_eff_norm"], 1e-30)
    candidates = _candidate_lookup(xi_eV, float(abs(q_value)), delta0)
    missing = [name for name in candidate_names if name not in candidates]
    if missing:
        raise ValueError(f"unknown primitive Ward candidate(s): {missing}")
    decompositions = [
        decompose_candidate(
            primitive=primitive,
            effective=effective,
            schur_correction=schur_correction,
            candidate=candidates[name],
            collective_order=collective_order,
            block_reference_norm=block_reference_norm,
            effective_reference_norm=effective_reference_norm,
        )
        for name in candidate_names
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "requires_analytic_derivation_before_convention_change": True,
            "valid_for_casimir_input": False,
            "reason": "primitive_response_ward_decomposition_not_production_convention",
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
            "candidate_names": list(candidate_names),
            "shifted_mesh_average": _shifted_payload(shift_fractions, shifts),
            "note": "Decomposes primitive response-level Ward residuals into bubble/contact/mixed/counterterm contributions. No convention is accepted.",
            "valid_for_casimir_input": False,
        },
        "collective_order": list(collective_order),
        "raw_ansatz_channel_names": list(raw_names) if raw_names is not None else None,
        "primitive_metadata": primitive["metadata"],
        "block_norms": norms,
        "schur": schur,
        "candidate_decompositions": decompositions,
        "valid_for_casimir_input": False,
    }


def run_and_write_primitive_response_ward_decomposition(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_primitive_response_ward_decomposition(**kwargs)
    write_json(Path(output_dir) / "primitive_response_ward_decomposition.json", payload)
    return payload
