"""Diagnostic-only primitive response-level Ward audit."""

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
from .nk_sweep import RATIO_EPS
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .ward_basis_convention_scan import primitive_blocks_from_baseline

SCHEMA_VERSION = "finite_q_tmte_primitive_response_ward_audit_v1"
SCHUR_CONDITION_THRESHOLD = 1e12


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _safe_ratio(numerator: float, denominator: float, eps: float = RATIO_EPS) -> float:
    return float(numerator) / max(float(denominator), float(eps))


def _complex_vector(values: np.ndarray, labels: Sequence[str]) -> list[dict[str, Any]]:
    return complex_vector_payload(np.asarray(values, dtype=complex).reshape(-1), labels)


def primitive_schur_effective(primitive: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    k_etaeta = np.asarray(primitive["k_etaeta"], dtype=complex)
    k_seta = np.asarray(primitive["k_seta"], dtype=complex)
    k_etas = np.asarray(primitive["k_etas"], dtype=complex)
    condition = float(np.linalg.cond(k_etaeta))
    if not np.isfinite(condition) or condition > SCHUR_CONDITION_THRESHOLD:
        action = np.linalg.pinv(k_etaeta) @ k_etas
        method = "pinv_diagnostic"
    else:
        action = np.linalg.solve(k_etaeta, k_etas)
        method = "solve"
    correction = k_seta @ action
    effective = np.asarray(primitive["k_ss"], dtype=complex) - correction
    return effective, correction, {
        "solve_method": method,
        "etaeta_condition_number": condition,
        "condition_threshold": float(SCHUR_CONDITION_THRESHOLD),
        "numerically_suspect": method == "pinv_diagnostic",
        "valid_for_casimir_input": False,
    }


def primitive_ward_candidate_vectors(xi_eV: float, q_norm: float, delta0_eV: float) -> list[dict[str, Any]]:
    """Return diagnostic primitive Ward vector candidates inferred from matrix-level checks."""

    xi = float(xi_eV)
    q = float(q_norm)
    d = float(delta0_eV)
    return [
        {
            "candidate": "matrix_inferred_matsubara_i_asymmetric",
            "description": "Observable uses +i xi A0 + q L - 2i Delta eta2; source uses -i xi A0 + q L - 2i Delta eta2 as read from inverse-Green audit.",
            "left_u": np.asarray([1j * xi, q, 0.0], dtype=complex),
            "left_w": np.asarray([0.0 + 0.0j, -2j * d], dtype=complex),
            "right_u": np.asarray([-1j * xi, q, 0.0], dtype=complex),
            "right_w": np.asarray([0.0 + 0.0j, -2j * d], dtype=complex),
        },
        {
            "candidate": "matrix_inferred_source_overall_flipped",
            "description": "Same source null vector with an overall right-side sign flip; equivalent if the response-level mapping is perfectly linear.",
            "left_u": np.asarray([1j * xi, q, 0.0], dtype=complex),
            "left_w": np.asarray([0.0 + 0.0j, -2j * d], dtype=complex),
            "right_u": np.asarray([1j * xi, -q, 0.0], dtype=complex),
            "right_w": np.asarray([0.0 + 0.0j, 2j * d], dtype=complex),
        },
        {
            "candidate": "legacy_validation_like",
            "description": "Historical physical EM contraction: left i xi rho + q j, right i xi rho - q j, with opposite phase-generator signs.",
            "left_u": np.asarray([1j * xi, q, 0.0], dtype=complex),
            "left_w": np.asarray([0.0 + 0.0j, -2j * d], dtype=complex),
            "right_u": np.asarray([1j * xi, -q, 0.0], dtype=complex),
            "right_w": np.asarray([0.0 + 0.0j, 2j * d], dtype=complex),
        },
        {
            "candidate": "baseline_real_target_like",
            "description": "Current real target-basis-like diagnostic retained as a negative control.",
            "left_u": np.asarray([xi, q, 0.0], dtype=complex),
            "left_w": np.asarray([0.0 + 0.0j, -2j * d], dtype=complex),
            "right_u": np.asarray([xi, q, 0.0], dtype=complex),
            "right_w": np.asarray([0.0 + 0.0j, -2j * d], dtype=complex),
        },
        {
            "candidate": "baseline_real_validation_spatial_sign",
            "description": "Real frequency coefficient with validation-style right spatial sign and opposite phase sign; debug negative control.",
            "left_u": np.asarray([xi, q, 0.0], dtype=complex),
            "left_w": np.asarray([0.0 + 0.0j, -2j * d], dtype=complex),
            "right_u": np.asarray([xi, -q, 0.0], dtype=complex),
            "right_w": np.asarray([0.0 + 0.0j, 2j * d], dtype=complex),
        },
    ]


def _ensure_collective_vector(vector: np.ndarray, n_eta: int) -> np.ndarray:
    values = np.asarray(vector, dtype=complex).reshape(-1)
    if values.shape[0] != n_eta:
        raise ValueError("collective Ward vector length does not match K_etaeta")
    return values


def evaluate_primitive_ward_candidate(
    *,
    primitive: dict[str, Any],
    effective: np.ndarray,
    candidate: dict[str, Any],
    primitive_norm: float,
    effective_norm: float,
    collective_order: tuple[str, ...],
) -> dict[str, Any]:
    k_ss = np.asarray(primitive["k_ss"], dtype=complex)
    k_seta = np.asarray(primitive["k_seta"], dtype=complex)
    k_etas = np.asarray(primitive["k_etas"], dtype=complex)
    k_etaeta = np.asarray(primitive["k_etaeta"], dtype=complex)
    u_left = np.asarray(candidate["left_u"], dtype=complex).reshape(3)
    u_right = np.asarray(candidate["right_u"], dtype=complex).reshape(3)
    w_left = _ensure_collective_vector(candidate["left_w"], k_etaeta.shape[0])
    w_right = _ensure_collective_vector(candidate["right_w"], k_etaeta.shape[0])

    left_em = u_left @ k_ss + w_left @ k_etas
    left_collective = u_left @ k_seta + w_left @ k_etaeta
    right_em = k_ss @ u_right + k_seta @ w_right
    right_collective = k_etas @ u_right + k_etaeta @ w_right
    left_effective = u_left @ np.asarray(effective, dtype=complex)
    right_effective = np.asarray(effective, dtype=complex) @ u_right

    left_em_norm = _norm(left_em)
    left_collective_norm = _norm(left_collective)
    right_em_norm = _norm(right_em)
    right_collective_norm = _norm(right_collective)
    left_effective_norm = _norm(left_effective)
    right_effective_norm = _norm(right_effective)
    return {
        "candidate": str(candidate["candidate"]),
        "description": str(candidate["description"]),
        "left_primitive_vector": _complex_vector(u_left, PRIMITIVE_ORDER),
        "right_primitive_vector": _complex_vector(u_right, PRIMITIVE_ORDER),
        "left_collective_vector": _complex_vector(w_left, collective_order),
        "right_collective_vector": _complex_vector(w_right, collective_order),
        "bare_block_residuals": {
            "left_em": _complex_vector(left_em, PRIMITIVE_ORDER),
            "left_collective": _complex_vector(left_collective, collective_order),
            "right_em": _complex_vector(right_em, PRIMITIVE_ORDER),
            "right_collective": _complex_vector(right_collective, collective_order),
            "valid_for_casimir_input": False,
        },
        "effective_schur_residuals": {
            "left_effective": _complex_vector(left_effective, PRIMITIVE_ORDER),
            "right_effective": _complex_vector(right_effective, PRIMITIVE_ORDER),
            "valid_for_casimir_input": False,
        },
        "norms": {
            "left_em_norm": left_em_norm,
            "left_collective_norm": left_collective_norm,
            "right_em_norm": right_em_norm,
            "right_collective_norm": right_collective_norm,
            "left_total_extended_norm": float(np.sqrt(left_em_norm**2 + left_collective_norm**2)),
            "right_total_extended_norm": float(np.sqrt(right_em_norm**2 + right_collective_norm**2)),
            "left_effective_norm": left_effective_norm,
            "right_effective_norm": right_effective_norm,
            "left_effective_over_effective_norm": _safe_ratio(left_effective_norm, effective_norm),
            "right_effective_over_effective_norm": _safe_ratio(right_effective_norm, effective_norm),
            "left_total_over_primitive_norm": _safe_ratio(float(np.sqrt(left_em_norm**2 + left_collective_norm**2)), primitive_norm),
            "right_total_over_primitive_norm": _safe_ratio(float(np.sqrt(right_em_norm**2 + right_collective_norm**2)), primitive_norm),
            "valid_for_casimir_input": False,
        },
        "accepted_convention": False,
        "valid_for_casimir_input": False,
    }


def block_norm_payload(primitive: dict[str, Any], effective: np.ndarray, correction: np.ndarray) -> dict[str, Any]:
    return {
        "primitive_K_SS_norm": _norm(primitive["k_ss"]),
        "primitive_K_Seta_norm": _norm(primitive["k_seta"]),
        "primitive_K_etaS_norm": _norm(primitive["k_etas"]),
        "primitive_K_etaeta_norm": _norm(primitive["k_etaeta"]),
        "primitive_schur_correction_norm": _norm(correction),
        "primitive_K_eff_norm": _norm(effective),
        "valid_for_casimir_input": False,
    }


def run_primitive_response_ward_audit(
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
    ratio_eps: float = RATIO_EPS,
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
    effective, correction, schur = primitive_schur_effective(primitive)
    delta0 = float(getattr(inputs.pairing_params, "delta0_eV", 0.0))
    norms = block_norm_payload(primitive, effective, correction)
    primitive_norm = max(norms["primitive_K_SS_norm"], norms["primitive_K_Seta_norm"], norms["primitive_K_etaS_norm"], norms["primitive_K_etaeta_norm"], ratio_eps)
    effective_norm = max(norms["primitive_K_eff_norm"], ratio_eps)
    candidates = [
        evaluate_primitive_ward_candidate(
            primitive=primitive,
            effective=effective,
            candidate=candidate,
            primitive_norm=primitive_norm,
            effective_norm=effective_norm,
            collective_order=collective_order,
        )
        for candidate in primitive_ward_candidate_vectors(xi_eV, float(abs(q_value)), delta0)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "requires_analytic_derivation_before_convention_change": True,
            "valid_for_casimir_input": False,
            "reason": "primitive_response_ward_audit_not_production_convention",
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
            "ratio_eps": float(ratio_eps),
            "shifted_mesh_average": _shifted_payload(shift_fractions, shifts),
            "primitive_basis_source_order": list(PRIMITIVE_ORDER),
            "note": "Primitive response-level Ward contractions use asymmetric vectors inferred from inverse-Green matrix audit. No candidate is accepted without analytic derivation.",
            "valid_for_casimir_input": False,
        },
        "collective_order": list(collective_order),
        "raw_ansatz_channel_names": list(raw_names) if raw_names is not None else None,
        "primitive_metadata": primitive["metadata"],
        "block_norms": norms,
        "schur": schur,
        "candidate_results": candidates,
        "valid_for_casimir_input": False,
    }


def run_and_write_primitive_response_ward_audit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_primitive_response_ward_audit(**kwargs)
    write_json(Path(output_dir) / "primitive_response_ward_audit.json", payload)
    return payload
