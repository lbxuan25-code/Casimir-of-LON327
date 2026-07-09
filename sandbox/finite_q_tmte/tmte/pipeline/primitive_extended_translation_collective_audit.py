"""Diagnostic-only primitive Ward audit with translation RHS and collective mixed term."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..adapters.bubble_adapter import compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .contact_ablation import _shifted_payload
from .extended_ward_kernel import PRIMITIVE_ORDER, complex_vector_payload
from .primitive_em_translation_rhs_audit import _accumulate as accumulate_translation_blocks
from .primitive_em_translation_rhs_audit import _average as average_translation_blocks
from .primitive_em_translation_rhs_audit import _fit, _match
from .primitive_response_ward_audit import primitive_ward_candidate_vectors
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .ward_basis_convention_scan import primitive_blocks_from_baseline

SCHEMA_VERSION = "finite_q_tmte_primitive_extended_translation_collective_audit_v1"
DEFAULT_CANDIDATE = "matrix_inferred_matsubara_i_asymmetric"


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=complex)))


def _ratio(a: float, b: float, eps: float = 1e-30) -> float:
    return float(a) / max(float(b), eps)


def _vec(x: np.ndarray) -> list[dict[str, Any]]:
    return complex_vector_payload(np.asarray(x, dtype=complex).reshape(-1), PRIMITIVE_ORDER)


def _candidate_lookup(xi_eV: float, q_value: float, delta0_eV: float) -> dict[str, dict[str, Any]]:
    return {row["candidate"]: row for row in primitive_ward_candidate_vectors(xi_eV, abs(float(q_value)), delta0_eV)}


def _term(values: np.ndarray) -> dict[str, Any]:
    a = np.asarray(values, dtype=complex).reshape(3)
    return {"values": _vec(a), "norm": _norm(a), "valid_for_casimir_input": False}


def _rank_translation_vectors(target: np.ndarray, translation: dict[str, Any]) -> list[dict[str, Any]]:
    delta = np.asarray(translation["delta_v_mid"], dtype=complex)
    equal = np.asarray(translation["equal_forward"], dtype=complex)
    qm = np.asarray(translation["qM_mid"], dtype=complex)
    trans = equal - delta
    vectors = {
        "translation_forward": trans,
        "minus_translation_forward": -trans,
        "equal_forward": equal,
        "minus_equal_forward": -equal,
        "delta_v_mid": delta,
        "minus_delta_v_mid": -delta,
        "qM_mid": qm,
        "minus_qM_mid": -qm,
        "delta_v_plus_qM": delta + qm,
        "minus_delta_v_plus_qM": -delta - qm,
        "translation_plus_qM": trans + qm,
        "minus_translation_plus_qM": -trans - qm,
    }
    return sorted((_match(name, vector, target) for name, vector in vectors.items()), key=lambda row: float(row["difference_over_target_norm"]))


def run_primitive_extended_translation_collective_audit(
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
    bare = average_bare_blocks_then_schur(target_blocks).bare_blocks
    if float(contact_scale) != 1.0:
        from .contact_ablation import scaled_contact_blocks

        bare = average_bare_blocks_then_schur([scaled_contact_blocks(block, contact_scale) for block in target_blocks]).bare_blocks
    primitive = primitive_blocks_from_baseline(bare)
    translation = average_translation_blocks(translation_parts)
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
    k_etas = np.asarray(primitive["k_etas"], dtype=complex)
    k_seta = np.asarray(primitive["k_seta"], dtype=complex)
    if k_etas.shape[0] != w_left.shape[0] or k_seta.shape[1] != w_right.shape[0]:
        raise ValueError("collective Ward vector length does not match primitive collective blocks")
    left_em = u_left @ k_ss
    right_em = k_ss @ u_right
    left_mixed = w_left @ k_etas
    right_mixed = k_seta @ w_right
    left_extended = left_em + left_mixed
    right_extended = right_em + right_mixed
    left_ranked = _rank_translation_vectors(-left_extended, translation)
    right_ranked = _rank_translation_vectors(-right_extended, translation)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {"diagnostic_run_completed": True, "diagnostic_only_not_a_fix": True, "accepted_convention": False, "valid_for_casimir_input": False, "reason": "primitive_extended_translation_collective_audit_not_production_convention"},
        "model": {"name": model_name, "pairing": pairing_name, "delta0_eV": delta0, "valid_for_casimir_input": False},
        "frequency": frequency_payload(matsubara_index, temperature_K),
        "debug_parameters": {"q_value": float(q_value), "nk": int(nk), "eta_eV": float(eta_eV), "shift_fractions": [float(v) for v in shift_fractions], "shifted_mesh_average": _shifted_payload(shift_fractions, shifts), "contact_scale": float(contact_scale), "candidate_name": candidate_name, "primitive_order": list(PRIMITIVE_ORDER), "valid_for_casimir_input": False},
        "block_norms": {"K_SS_norm": _norm(k_ss), "K_etaS_norm": _norm(k_etas), "K_Seta_norm": _norm(k_seta), "valid_for_casimir_input": False},
        "ward_decomposition": {
            "left": {"em_total": _term(left_em), "mixed_collective": _term(left_mixed), "extended_total": _term(left_extended), "missing_to_close": _term(-left_extended), "em_to_extended_reduction": _ratio(_norm(left_extended), _norm(left_em)), "u_vector": _vec(u_left), "w_vector": complex_vector_payload(w_left, ["amplitude", "phase_eta2"]), "valid_for_casimir_input": False},
            "right": {"em_total": _term(right_em), "mixed_collective": _term(right_mixed), "extended_total": _term(right_extended), "missing_to_close": _term(-right_extended), "em_to_extended_reduction": _ratio(_norm(right_extended), _norm(right_em)), "u_vector": _vec(u_right), "w_vector": complex_vector_payload(w_right, ["amplitude", "phase_eta2"]), "valid_for_casimir_input": False},
            "valid_for_casimir_input": False,
        },
        "left_translation_candidates_ranked": left_ranked,
        "right_translation_candidates_ranked": right_ranked,
        "raw_translation_vectors": {name: {"values": _vec(vector), "norm": _norm(vector)} for name, vector in {"equal_forward": translation["equal_forward"], "delta_v_mid": translation["delta_v_mid"], "translation_forward": translation["equal_forward"] - translation["delta_v_mid"], "qM_mid": translation["qM_mid"]}.items()},
        "interpretation_guardrails": {"includes_collective_mixed_terms": True, "excludes_schur_etaeta_closure": True, "not_a_fit_fix": True, "if_translation_matches_extended": "primitive EM plus W K_etaS residual is a finite-q translation RHS", "if_no_match": "remaining residual needs BdG/pairing/collective equal-time structure beyond this RHS", "valid_for_casimir_input": False},
        "valid_for_casimir_input": False,
    }


def run_and_write_primitive_extended_translation_collective_audit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_primitive_extended_translation_collective_audit(**kwargs)
    write_json(Path(output_dir) / "primitive_extended_translation_collective_audit.json", payload)
    return payload
