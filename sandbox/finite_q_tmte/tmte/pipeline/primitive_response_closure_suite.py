"""Diagnostic-only primitive response closure suite.

This audit fixes the matrix-inferred Matsubara primitive Ward vector and tests
whether the remaining response residual can be explained by contact, mixed,
Schur, or collective counterterm normalization/sign choices. It is not a
production convention proposal.
"""

from __future__ import annotations

from itertools import product
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

SCHEMA_VERSION = "finite_q_tmte_primitive_response_closure_suite_v1"
DEFAULT_CANDIDATE = "matrix_inferred_matsubara_i_asymmetric"
GRID_COEFFICIENTS = (1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j)


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-30) -> float:
    return float(numerator) / max(float(denominator), float(eps))


def _vec(values: np.ndarray, labels: Sequence[str]) -> list[dict[str, Any]]:
    return complex_vector_payload(np.asarray(values, dtype=complex).reshape(-1), labels)


def _candidate_lookup(xi_eV: float, q_norm: float, delta0_eV: float) -> dict[str, dict[str, Any]]:
    return {row["candidate"]: row for row in primitive_ward_candidate_vectors(xi_eV, q_norm, delta0_eV)}


def fit_one_scale(target: np.ndarray, basis: np.ndarray, *, real_only: bool) -> dict[str, Any]:
    """Fit coefficient c minimizing ||target + c*basis||."""

    t = np.asarray(target, dtype=complex).reshape(-1)
    b = np.asarray(basis, dtype=complex).reshape(-1)
    denom = np.vdot(b, b)
    if abs(denom) < 1e-30:
        coeff = 0.0 + 0.0j
    elif real_only:
        coeff = complex(-float(np.real(np.vdot(b, t))) / float(np.real(denom)), 0.0)
    else:
        coeff = -np.vdot(b, t) / denom
    residual = t + coeff * b
    return {
        "coefficient": complex(coeff),
        "residual_norm": _norm(residual),
        "target_norm": _norm(t),
        "basis_norm": _norm(b),
        "improvement_factor_vs_target": _safe_ratio(_norm(t), _norm(residual)),
        "real_only": bool(real_only),
        "valid_for_casimir_input": False,
    }


def fit_two_scales(fixed: np.ndarray, first: np.ndarray, second: np.ndarray, *, real_only: bool) -> dict[str, Any]:
    """Fit c1,c2 minimizing ||fixed + c1*first + c2*second||."""

    f = np.asarray(fixed, dtype=complex).reshape(-1)
    a = np.vstack([np.asarray(first, dtype=complex).reshape(-1), np.asarray(second, dtype=complex).reshape(-1)]).T
    if real_only:
        a_real = np.vstack([np.real(a), np.imag(a)])
        f_real = np.concatenate([np.real(f), np.imag(f)])
        coeffs_real, *_ = np.linalg.lstsq(a_real, -f_real, rcond=None)
        coeffs = coeffs_real.astype(complex)
    else:
        coeffs, *_ = np.linalg.lstsq(a, -f, rcond=None)
    residual = f + a @ coeffs
    return {
        "coefficients": [complex(value) for value in coeffs],
        "residual_norm": _norm(residual),
        "fixed_norm": _norm(f),
        "first_basis_norm": _norm(first),
        "second_basis_norm": _norm(second),
        "improvement_factor_vs_fixed": _safe_ratio(_norm(f), _norm(residual)),
        "real_only": bool(real_only),
        "valid_for_casimir_input": False,
    }


def best_phase_grid(fixed: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Try phase/sign grid for fixed + c1*first + c2*second."""

    f = np.asarray(fixed, dtype=complex).reshape(-1)
    a = np.asarray(first, dtype=complex).reshape(-1)
    b = np.asarray(second, dtype=complex).reshape(-1)
    best: tuple[float, complex, complex, np.ndarray] | None = None
    for c1, c2 in product(GRID_COEFFICIENTS, repeat=2):
        residual = f + c1 * a + c2 * b
        score = _norm(residual)
        if best is None or score < best[0]:
            best = (score, c1, c2, residual)
    assert best is not None
    return {
        "first_coefficient": complex(best[1]),
        "second_coefficient": complex(best[2]),
        "residual_norm": best[0],
        "fixed_norm": _norm(f),
        "improvement_factor_vs_fixed": _safe_ratio(_norm(f), best[0]),
        "grid_coefficients": [complex(value) for value in GRID_COEFFICIENTS],
        "valid_for_casimir_input": False,
    }


def cancellation_angle(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Report normalized overlap and cancellation quality between two vectors."""

    a = np.asarray(first, dtype=complex).reshape(-1)
    b = np.asarray(second, dtype=complex).reshape(-1)
    denom = max(_norm(a) * _norm(b), 1e-30)
    overlap = np.vdot(a, b) / denom
    return {
        "first_norm": _norm(a),
        "second_norm": _norm(b),
        "sum_norm": _norm(a + b),
        "difference_norm": _norm(a - b),
        "normalized_overlap": complex(overlap),
        "real_overlap": float(np.real(overlap)),
        "imag_overlap": float(np.imag(overlap)),
        "sum_over_first_plus_second_norms": _safe_ratio(_norm(a + b), _norm(a) + _norm(b)),
        "valid_for_casimir_input": False,
    }


def sector_terms(primitive: dict[str, Any], schur_correction: np.ndarray, candidate: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    k_ss_bubble = np.asarray(primitive["k_ss_bubble"], dtype=complex)
    k_ss_contact = np.asarray(primitive["k_ss_contact"], dtype=complex)
    k_seta = np.asarray(primitive["k_seta"], dtype=complex)
    k_etas = np.asarray(primitive["k_etas"], dtype=complex)
    k_etaeta_bubble = np.asarray(primitive["k_etaeta_bubble"], dtype=complex)
    k_etaeta_counterterm = np.asarray(primitive["k_etaeta_counterterm"], dtype=complex)
    u_left = np.asarray(candidate["left_u"], dtype=complex).reshape(3)
    u_right = np.asarray(candidate["right_u"], dtype=complex).reshape(3)
    w_left = np.asarray(candidate["left_w"], dtype=complex).reshape(k_etaeta_bubble.shape[0])
    w_right = np.asarray(candidate["right_w"], dtype=complex).reshape(k_etaeta_bubble.shape[0])
    return {
        "left_em": {
            "bubble": u_left @ k_ss_bubble,
            "contact": u_left @ k_ss_contact,
            "mixed": w_left @ k_etas,
        },
        "right_em": {
            "bubble": k_ss_bubble @ u_right,
            "contact": k_ss_contact @ u_right,
            "mixed": k_seta @ w_right,
        },
        "left_schur": {
            "bubble": u_left @ k_ss_bubble,
            "contact": u_left @ k_ss_contact,
            "schur": -(u_left @ np.asarray(schur_correction, dtype=complex)),
        },
        "right_schur": {
            "bubble": k_ss_bubble @ u_right,
            "contact": k_ss_contact @ u_right,
            "schur": -(np.asarray(schur_correction, dtype=complex) @ u_right),
        },
        "left_collective": {
            "mixed": u_left @ k_seta,
            "etaeta_bubble": w_left @ k_etaeta_bubble,
            "etaeta_counterterm": w_left @ k_etaeta_counterterm,
        },
        "right_collective": {
            "mixed": k_etas @ u_right,
            "etaeta_bubble": k_etaeta_bubble @ w_right,
            "etaeta_counterterm": k_etaeta_counterterm @ w_right,
        },
    }


def vector_payload(name: str, values: np.ndarray, labels: Sequence[str], reference_norm: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=complex).reshape(-1)
    return {
        "name": name,
        "values": _vec(values, labels),
        "norm": _norm(values),
        "norm_over_reference": _safe_ratio(_norm(values), reference_norm),
        "valid_for_casimir_input": False,
    }


def analyze_three_term_sector(
    *,
    sector_name: str,
    terms: dict[str, np.ndarray],
    labels: Sequence[str],
    reference_norm: float,
    first_name: str,
    second_name: str,
    fixed_name: str,
) -> dict[str, Any]:
    fixed = terms[fixed_name]
    first = terms[first_name]
    second = terms[second_name]
    total = fixed + first + second
    return {
        "sector": sector_name,
        "fixed_term": vector_payload(fixed_name, fixed, labels, reference_norm),
        "first_scalable_term": vector_payload(first_name, first, labels, reference_norm),
        "second_scalable_term": vector_payload(second_name, second, labels, reference_norm),
        "current_total": vector_payload("current_total", total, labels, reference_norm),
        "one_scale_fits": {
            f"fit_{first_name}": {
                "complex": fit_one_scale(fixed + second, first, real_only=False),
                "real": fit_one_scale(fixed + second, first, real_only=True),
            },
            f"fit_{second_name}": {
                "complex": fit_one_scale(fixed + first, second, real_only=False),
                "real": fit_one_scale(fixed + first, second, real_only=True),
            },
            f"fit_{fixed_name}": {
                "complex": fit_one_scale(first + second, fixed, real_only=False),
                "real": fit_one_scale(first + second, fixed, real_only=True),
            },
            "valid_for_casimir_input": False,
        },
        "two_scale_fits": {
            f"fit_{first_name}_and_{second_name}": {
                "complex": fit_two_scales(fixed, first, second, real_only=False),
                "real": fit_two_scales(fixed, first, second, real_only=True),
            },
            "valid_for_casimir_input": False,
        },
        "phase_sign_grid": best_phase_grid(fixed, first, second),
        "cancellation": {
            f"{first_name}_vs_{second_name}": cancellation_angle(first, second),
            f"{fixed_name}_vs_{first_name}_plus_{second_name}": cancellation_angle(fixed, first + second),
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def analyze_candidate(
    *,
    primitive: dict[str, Any],
    schur_correction: np.ndarray,
    candidate: dict[str, Any],
    collective_order: tuple[str, ...],
    block_reference_norm: float,
    effective_reference_norm: float,
) -> dict[str, Any]:
    terms = sector_terms(primitive, schur_correction, candidate)
    return {
        "candidate": str(candidate["candidate"]),
        "description": str(candidate["description"]),
        "left_primitive_vector": _vec(candidate["left_u"], PRIMITIVE_ORDER),
        "right_primitive_vector": _vec(candidate["right_u"], PRIMITIVE_ORDER),
        "left_collective_vector": _vec(candidate["left_w"], collective_order),
        "right_collective_vector": _vec(candidate["right_w"], collective_order),
        "em_balance": {
            "left": analyze_three_term_sector(
                sector_name="left_em",
                terms=terms["left_em"],
                labels=PRIMITIVE_ORDER,
                reference_norm=block_reference_norm,
                fixed_name="bubble",
                first_name="contact",
                second_name="mixed",
            ),
            "right": analyze_three_term_sector(
                sector_name="right_em",
                terms=terms["right_em"],
                labels=PRIMITIVE_ORDER,
                reference_norm=block_reference_norm,
                fixed_name="bubble",
                first_name="contact",
                second_name="mixed",
            ),
            "valid_for_casimir_input": False,
        },
        "schur_balance": {
            "left": analyze_three_term_sector(
                sector_name="left_schur",
                terms=terms["left_schur"],
                labels=PRIMITIVE_ORDER,
                reference_norm=effective_reference_norm,
                fixed_name="bubble",
                first_name="contact",
                second_name="schur",
            ),
            "right": analyze_three_term_sector(
                sector_name="right_schur",
                terms=terms["right_schur"],
                labels=PRIMITIVE_ORDER,
                reference_norm=effective_reference_norm,
                fixed_name="bubble",
                first_name="contact",
                second_name="schur",
            ),
            "valid_for_casimir_input": False,
        },
        "collective_balance": {
            "left": analyze_three_term_sector(
                sector_name="left_collective",
                terms=terms["left_collective"],
                labels=collective_order,
                reference_norm=block_reference_norm,
                fixed_name="mixed",
                first_name="etaeta_bubble",
                second_name="etaeta_counterterm",
            ),
            "right": analyze_three_term_sector(
                sector_name="right_collective",
                terms=terms["right_collective"],
                labels=collective_order,
                reference_norm=block_reference_norm,
                fixed_name="mixed",
                first_name="etaeta_bubble",
                second_name="etaeta_counterterm",
            ),
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


def run_primitive_response_closure_suite(
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
    norms = block_norms(primitive, effective, schur_correction)
    block_reference_norm = max(norms["K_SS_total_norm"], norms["K_Seta_norm"], norms["K_etaS_norm"], norms["K_etaeta_total_norm"], 1e-30)
    effective_reference_norm = max(norms["K_eff_norm"], 1e-30)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "requires_analytic_derivation_before_convention_change": True,
            "valid_for_casimir_input": False,
            "reason": "primitive_response_closure_suite_not_production_convention",
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
            "note": "Fixed matrix-inferred primitive Ward vector. Fits/scans response-sector balances only; no convention is accepted.",
            "valid_for_casimir_input": False,
        },
        "collective_order": list(collective_order),
        "raw_ansatz_channel_names": list(raw_names) if raw_names is not None else None,
        "primitive_metadata": primitive["metadata"],
        "block_norms": norms,
        "schur": schur,
        "candidate_analysis": analyze_candidate(
            primitive=primitive,
            schur_correction=schur_correction,
            candidate=candidates[candidate_name],
            collective_order=collective_order,
            block_reference_norm=block_reference_norm,
            effective_reference_norm=effective_reference_norm,
        ),
        "valid_for_casimir_input": False,
    }


def run_and_write_primitive_response_closure_suite(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_primitive_response_closure_suite(**kwargs)
    write_json(Path(output_dir) / "primitive_response_closure_suite.json", payload)
    return payload
