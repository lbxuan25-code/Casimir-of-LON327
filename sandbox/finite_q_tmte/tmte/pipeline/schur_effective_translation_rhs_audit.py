"""Diagnostic-only Schur-effective translation RHS audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..adapters.bubble_adapter import compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .contact_ablation import _shifted_payload, scaled_contact_blocks
from .extended_ward_kernel import PRIMITIVE_ORDER, complex_vector_payload
from .primitive_em_translation_rhs_audit import _accumulate as accumulate_translation_blocks
from .primitive_em_translation_rhs_audit import _average as average_translation_blocks
from .primitive_response_ward_audit import primitive_ward_candidate_vectors
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .ward_basis_convention_scan import primitive_blocks_from_baseline

SCHEMA_VERSION = "finite_q_tmte_schur_effective_translation_rhs_audit_v1"
DEFAULT_CANDIDATE = "matrix_inferred_matsubara_i_asymmetric"
SCHUR_CONDITION_THRESHOLD = 1e12
COLLECTIVE_ORDER = ("amplitude", "phase_eta2")


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=complex)))


def _ratio(a: float, b: float, eps: float = 1e-30) -> float:
    return float(a) / max(float(b), eps)


def _vec(values: np.ndarray, labels: Sequence[str]) -> list[dict[str, Any]]:
    return complex_vector_payload(np.asarray(values, dtype=complex).reshape(-1), labels)


def _term(values: np.ndarray, labels: Sequence[str], reference_norm: float | None = None) -> dict[str, Any]:
    arr = np.asarray(values, dtype=complex).reshape(-1)
    ref = _norm(arr) if reference_norm is None else float(reference_norm)
    return {
        "values": _vec(arr, labels),
        "norm": _norm(arr),
        "norm_over_reference": _ratio(_norm(arr), ref),
        "valid_for_casimir_input": False,
    }


def _matrix_payload(matrix: np.ndarray, rows: Sequence[str], cols: Sequence[str]) -> list[dict[str, Any]]:
    arr = np.asarray(matrix, dtype=complex)
    return [{"row": row, "values": _vec(arr[i, :], cols)} for i, row in enumerate(rows)]


def _candidate_lookup(xi_eV: float, q_value: float, delta0_eV: float) -> dict[str, dict[str, Any]]:
    return {row["candidate"]: row for row in primitive_ward_candidate_vectors(xi_eV, abs(float(q_value)), delta0_eV)}


def _solve_etaeta(k_etaeta: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    eta = np.asarray(k_etaeta, dtype=complex)
    b = np.asarray(rhs, dtype=complex)
    condition = float(np.linalg.cond(eta))
    if not np.isfinite(condition) or condition > SCHUR_CONDITION_THRESHOLD:
        result = np.linalg.pinv(eta) @ b
        method = "pinv_diagnostic"
    else:
        result = np.linalg.solve(eta, b)
        method = "solve"
    return result, {
        "solve_method": method,
        "etaeta_condition_number": condition,
        "condition_threshold": float(SCHUR_CONDITION_THRESHOLD),
        "numerically_suspect": bool(method == "pinv_diagnostic"),
        "valid_for_casimir_input": False,
    }


def _translation_rhs(translation: dict[str, Any], contact_scale: float) -> dict[str, np.ndarray]:
    equal = np.asarray(translation["equal_forward"], dtype=complex).reshape(3)
    delta = np.asarray(translation["delta_v_mid"], dtype=complex).reshape(3)
    qm = float(contact_scale) * np.asarray(translation["qM_mid"], dtype=complex).reshape(3)
    forward = equal - delta
    return {
        "equal_forward": equal,
        "delta_v_mid": delta,
        "qM_mid": qm,
        "translation_forward": forward,
        "translation_plus_qM": forward + qm,
        "minus_translation_plus_qM": -forward - qm,
    }


def _left_schur_payload(
    *,
    k_ss: np.ndarray,
    k_seta: np.ndarray,
    k_etas: np.ndarray,
    k_etaeta: np.ndarray,
    k_eff: np.ndarray,
    action: np.ndarray,
    u_left: np.ndarray,
    w_left: np.ndarray,
    rhs_s: np.ndarray,
) -> dict[str, Any]:
    left_em = u_left @ k_ss
    left_mixed = w_left @ k_etas
    left_s_total = left_em + left_mixed
    left_s_residual = left_s_total - rhs_s
    eta_total = u_left @ k_seta + w_left @ k_etaeta
    eta_projected = eta_total @ action
    effective_direct = u_left @ k_eff
    effective_predicted = rhs_s - eta_projected
    effective_residual = effective_direct - effective_predicted
    ref = max(_norm(effective_direct), _norm(effective_predicted), _norm(rhs_s), 1e-30)
    return {
        "em_total": _term(left_em, PRIMITIVE_ORDER, max(_norm(left_em), 1e-30)),
        "mixed_collective": _term(left_mixed, PRIMITIVE_ORDER, max(_norm(left_em), 1e-30)),
        "s_channel_total": _term(left_s_total, PRIMITIVE_ORDER, max(_norm(rhs_s), 1e-30)),
        "s_channel_rhs": _term(rhs_s, PRIMITIVE_ORDER, max(_norm(rhs_s), 1e-30)),
        "s_channel_residual": _term(left_s_residual, PRIMITIVE_ORDER, max(_norm(rhs_s), 1e-30)),
        "eta_channel_total_C_eta": _term(eta_total, COLLECTIVE_ORDER, max(_norm(eta_total), 1e-30)),
        "eta_projected_to_external": _term(eta_projected, PRIMITIVE_ORDER, ref),
        "effective_direct": _term(effective_direct, PRIMITIVE_ORDER, ref),
        "effective_rhs_predicted": _term(effective_predicted, PRIMITIVE_ORDER, ref),
        "effective_residual": _term(effective_residual, PRIMITIVE_ORDER, ref),
        "effective_residual_over_reference": _ratio(_norm(effective_residual), ref),
        "eta_projection_over_rhs_s": _ratio(_norm(eta_projected), _norm(rhs_s)),
        "u_vector": _term(u_left, PRIMITIVE_ORDER, max(_norm(u_left), 1e-30)),
        "w_vector": _term(w_left, COLLECTIVE_ORDER, max(_norm(w_left), 1e-30)),
        "valid_for_casimir_input": False,
    }


def _right_schur_payload(
    *,
    k_ss: np.ndarray,
    k_seta: np.ndarray,
    k_etas: np.ndarray,
    k_etaeta: np.ndarray,
    k_eff: np.ndarray,
    u_right: np.ndarray,
    w_right: np.ndarray,
    rhs_s: np.ndarray,
) -> dict[str, Any]:
    right_em = k_ss @ u_right
    right_mixed = k_seta @ w_right
    right_s_total = right_em + right_mixed
    right_s_residual = right_s_total - rhs_s
    eta_total = k_etas @ u_right + k_etaeta @ w_right
    eta_solution, solve_meta = _solve_etaeta(k_etaeta, eta_total)
    eta_projected = k_seta @ eta_solution
    effective_direct = k_eff @ u_right
    effective_predicted = rhs_s - eta_projected
    effective_residual = effective_direct - effective_predicted
    ref = max(_norm(effective_direct), _norm(effective_predicted), _norm(rhs_s), 1e-30)
    return {
        "em_total": _term(right_em, PRIMITIVE_ORDER, max(_norm(right_em), 1e-30)),
        "mixed_collective": _term(right_mixed, PRIMITIVE_ORDER, max(_norm(right_em), 1e-30)),
        "s_channel_total": _term(right_s_total, PRIMITIVE_ORDER, max(_norm(rhs_s), 1e-30)),
        "s_channel_rhs": _term(rhs_s, PRIMITIVE_ORDER, max(_norm(rhs_s), 1e-30)),
        "s_channel_residual": _term(right_s_residual, PRIMITIVE_ORDER, max(_norm(rhs_s), 1e-30)),
        "eta_channel_total_C_eta": _term(eta_total, COLLECTIVE_ORDER, max(_norm(eta_total), 1e-30)),
        "eta_solution_Ketaeta_inverse_Ceta": _term(eta_solution, COLLECTIVE_ORDER, max(_norm(eta_solution), 1e-30)),
        "eta_projected_to_external": _term(eta_projected, PRIMITIVE_ORDER, ref),
        "effective_direct": _term(effective_direct, PRIMITIVE_ORDER, ref),
        "effective_rhs_predicted": _term(effective_predicted, PRIMITIVE_ORDER, ref),
        "effective_residual": _term(effective_residual, PRIMITIVE_ORDER, ref),
        "effective_residual_over_reference": _ratio(_norm(effective_residual), ref),
        "eta_projection_over_rhs_s": _ratio(_norm(eta_projected), _norm(rhs_s)),
        "eta_solve_metadata": solve_meta,
        "u_vector": _term(u_right, PRIMITIVE_ORDER, max(_norm(u_right), 1e-30)),
        "w_vector": _term(w_right, COLLECTIVE_ORDER, max(_norm(w_right), 1e-30)),
        "valid_for_casimir_input": False,
    }


def run_schur_effective_translation_rhs_audit(
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
    if nk <= 0:
        raise ValueError("nk must be positive")
    if float(q_value) <= 0.0:
        raise ValueError("q must be positive")
    xi = matsubara_xi_eV(matsubara_index, temperature_K)
    inputs = build_model_scan_inputs(model_name=model_name, pairing_name=pairing_name, xi_eV=xi, nk=nk, delta0_eV=delta0_eV, temperature_K=temperature_K, eta_eV=eta_eV)
    q = np.asarray([float(q_value), 0.0], dtype=float)
    shifts = shift_pairs_from_fractions(shift_fractions)
    target_blocks = []
    translation_parts = []
    for sx, sy in shifts:
        points = shifted_uniform_bz_mesh(nk, sx, sy)
        weights = weights_for_points(points)
        target_blocks.append(
            compute_target_bare_blocks(
                spec=inputs.spec,
                ansatz=inputs.ansatz,
                q_model=q,
                xi_eV=xi,
                k_points=points,
                weights=weights,
                config=inputs.config,
                pairing_params=inputs.pairing_params,
            )
        )
        translation_parts.append(
            accumulate_translation_blocks(
                spec=inputs.spec,
                ansatz=inputs.ansatz,
                pairing_params=inputs.pairing_params,
                q=q,
                xi_eV=xi,
                points=points,
                weights=weights,
                config=inputs.config,
            )
        )
    if float(contact_scale) == 1.0:
        bare = average_bare_blocks_then_schur(target_blocks).bare_blocks
    else:
        bare = average_bare_blocks_then_schur([scaled_contact_blocks(block, contact_scale) for block in target_blocks]).bare_blocks
    primitive = primitive_blocks_from_baseline(bare)
    translation = _translation_rhs(average_translation_blocks(translation_parts), contact_scale)
    delta0 = float(getattr(inputs.pairing_params, "delta0_eV", 0.0))
    candidates = _candidate_lookup(xi, q_value, delta0)
    if candidate_name not in candidates:
        raise ValueError(f"unknown primitive Ward candidate {candidate_name!r}")
    candidate = candidates[candidate_name]
    u_left = np.asarray(candidate["left_u"], dtype=complex).reshape(3)
    u_right = np.asarray(candidate["right_u"], dtype=complex).reshape(3)
    w_left = np.asarray(candidate["left_w"], dtype=complex).reshape(-1)
    w_right = np.asarray(candidate["right_w"], dtype=complex).reshape(-1)
    k_ss = np.asarray(primitive["k_ss"], dtype=complex)
    k_seta = np.asarray(primitive["k_seta"], dtype=complex)
    k_etas = np.asarray(primitive["k_etas"], dtype=complex)
    k_etaeta = np.asarray(primitive["k_etaeta"], dtype=complex)
    action, solve_meta = _solve_etaeta(k_etaeta, k_etas)
    schur_correction = k_seta @ action
    k_eff = k_ss - schur_correction
    rhs_s = translation["translation_plus_qM"]
    left = _left_schur_payload(k_ss=k_ss, k_seta=k_seta, k_etas=k_etas, k_etaeta=k_etaeta, k_eff=k_eff, action=action, u_left=u_left, w_left=w_left, rhs_s=rhs_s)
    right = _right_schur_payload(k_ss=k_ss, k_seta=k_seta, k_etas=k_etas, k_etaeta=k_etaeta, k_eff=k_eff, u_right=u_right, w_right=w_right, rhs_s=rhs_s)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {"diagnostic_run_completed": True, "diagnostic_only_not_a_fix": True, "accepted_convention": False, "valid_for_casimir_input": False, "reason": "schur_effective_translation_rhs_audit_not_production_convention"},
        "model": {"name": model_name, "pairing": pairing_name, "delta0_eV": delta0, "valid_for_casimir_input": False},
        "frequency": frequency_payload(matsubara_index, temperature_K),
        "debug_parameters": {"q_value": float(q_value), "nk": int(nk), "eta_eV": float(eta_eV), "shift_fractions": [float(v) for v in shift_fractions], "shifted_mesh_average": _shifted_payload(shift_fractions, shifts), "contact_scale": float(contact_scale), "candidate_name": candidate_name, "primitive_order": list(PRIMITIVE_ORDER), "collective_order": list(COLLECTIVE_ORDER), "valid_for_casimir_input": False},
        "block_norms": {"K_SS_norm": _norm(k_ss), "K_Seta_norm": _norm(k_seta), "K_etaS_norm": _norm(k_etas), "K_etaeta_norm": _norm(k_etaeta), "Schur_correction_norm": _norm(schur_correction), "K_eff_norm": _norm(k_eff), "valid_for_casimir_input": False},
        "schur_solve_metadata": solve_meta,
        "schur_action_Ketaeta_inverse_KetaS": {"matrix": _matrix_payload(action, COLLECTIVE_ORDER, PRIMITIVE_ORDER), "norm": _norm(action), "valid_for_casimir_input": False},
        "rhs_vectors": {name: _term(vector, PRIMITIVE_ORDER, max(_norm(rhs_s), 1e-30)) for name, vector in translation.items()},
        "ward_decomposition": {"left": left, "right": right, "valid_for_casimir_input": False},
        "summary": {"left_effective_residual_over_reference": left["effective_residual_over_reference"], "right_effective_residual_over_reference": right["effective_residual_over_reference"], "left_eta_projection_over_rhs_s": left["eta_projection_over_rhs_s"], "right_eta_projection_over_rhs_s": right["eta_projection_over_rhs_s"], "left_s_channel_residual_over_rhs_s": left["s_channel_residual"]["norm_over_reference"], "right_s_channel_residual_over_rhs_s": right["s_channel_residual"]["norm_over_reference"], "valid_for_casimir_input": False},
        "interpretation_guardrails": {"identity_tested": "u K_eff = R_S - R_eta K_etaeta^{-1} K_etaS, with R_eta measured as C_eta", "rhs_s_is_translation_forward_plus_qM": True, "diagnostic_only_not_a_fix": True, "valid_for_casimir_input": False},
        "valid_for_casimir_input": False,
    }


def run_and_write_schur_effective_translation_rhs_audit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_schur_effective_translation_rhs_audit(**kwargs)
    write_json(Path(output_dir) / "schur_effective_translation_rhs_audit.json", payload)
    return payload
