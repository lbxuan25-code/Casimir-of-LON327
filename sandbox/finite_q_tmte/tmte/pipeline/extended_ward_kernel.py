"""Debug-only extended Ward-kernel audit for finite-q TM/TE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..adapters.bubble_adapter import TargetBareBlocks, compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.conventions import require_diagnostic_source_order
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .collective_schur_factors import collective_order_from_ansatz, solve_collective_action
from .contact_ablation import _shifted_payload, scaled_contact_blocks
from .nk_sweep import RATIO_EPS
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .signed_decomposition import decomposition_ratios

SCHEMA_VERSION = "finite_q_tmte_extended_ward_kernel_v1"
EXTENDED_WARD_TOLERANCE = 1e-8
SCHUR_CONDITION_THRESHOLD = 1e12


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _safe_ratio(numerator: float, denominator: float, eps: float = RATIO_EPS) -> float:
    return float(numerator) / max(float(denominator), float(eps))


def complex_vector_payload(values: np.ndarray, labels: Sequence[str]) -> list[dict[str, Any]]:
    vector = np.asarray(values, dtype=complex).reshape(-1)
    labels_tuple = tuple(labels)
    if vector.shape[0] != len(labels_tuple):
        raise ValueError("label count does not match complex vector length")
    return [{"label": str(label), "value": complex(value)} for label, value in zip(labels_tuple, vector, strict=True)]


def solve_left_collective_vector(k_etaeta: np.ndarray, k_geta: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return W_left satisfying K_Geta + W_left K_etaeta = 0."""

    etaeta = np.asarray(k_etaeta, dtype=complex)
    geta = np.asarray(k_geta, dtype=complex).reshape(-1)
    condition = float(np.linalg.cond(etaeta))
    if not np.isfinite(condition) or condition > SCHUR_CONDITION_THRESHOLD:
        w_left = -geta @ np.linalg.pinv(etaeta)
        method = "pinv_diagnostic"
    else:
        w_left = np.linalg.solve(etaeta.T, -geta.T).T
        method = "solve"
    return np.asarray(w_left, dtype=complex), {
        "solve_method": method,
        "etaeta_condition_number": condition,
        "condition_threshold": float(SCHUR_CONDITION_THRESHOLD),
        "numerically_suspect": method == "pinv_diagnostic",
        "valid_for_casimir_input": False,
    }


def solve_right_collective_vector(k_etaeta: np.ndarray, k_etag: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return W_right satisfying K_etaG + K_etaeta W_right = 0."""

    etaeta = np.asarray(k_etaeta, dtype=complex)
    etag = np.asarray(k_etag, dtype=complex).reshape(-1)
    condition = float(np.linalg.cond(etaeta))
    if not np.isfinite(condition) or condition > SCHUR_CONDITION_THRESHOLD:
        w_right = -np.linalg.pinv(etaeta) @ etag
        method = "pinv_diagnostic"
    else:
        w_right = np.linalg.solve(etaeta, -etag)
        method = "solve"
    return np.asarray(w_right, dtype=complex), {
        "solve_method": method,
        "etaeta_condition_number": condition,
        "condition_threshold": float(SCHUR_CONDITION_THRESHOLD),
        "numerically_suspect": method == "pinv_diagnostic",
        "valid_for_casimir_input": False,
    }


def schur_effective_from_blocks(blocks: TargetBareBlocks) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_action, schur = solve_collective_action(blocks.k_etaeta, blocks.k_etas)
    correction = np.asarray(blocks.k_seta, dtype=complex) @ x_action
    effective = np.asarray(blocks.k_ss, dtype=complex) - correction
    return effective, correction, schur


def em_source_decomposition(
    *,
    blocks: TargetBareBlocks,
    w_eta_left: np.ndarray,
    w_eta_right: np.ndarray,
    left_em: np.ndarray,
    right_em: np.ndarray,
) -> dict[str, Any]:
    """Decompose fitted/analytic EM residuals into bubble, contact, and mixed collective pieces."""

    require_diagnostic_source_order(blocks.source_order)
    source_order = blocks.source_order
    g = source_order.index("G")
    w_left = np.asarray(w_eta_left, dtype=complex).reshape(-1)
    w_right = np.asarray(w_eta_right, dtype=complex).reshape(-1)
    left_bubble = np.asarray(blocks.k_ss_bubble, dtype=complex)[g, :]
    left_contact = np.asarray(blocks.k_ss_contact, dtype=complex)[g, :]
    left_collective = w_left @ np.asarray(blocks.k_etas, dtype=complex)
    left_total = left_bubble + left_contact + left_collective
    right_bubble = np.asarray(blocks.k_ss_bubble, dtype=complex)[:, g]
    right_contact = np.asarray(blocks.k_ss_contact, dtype=complex)[:, g]
    right_collective = np.asarray(blocks.k_seta, dtype=complex) @ w_right
    right_total = right_bubble + right_contact + right_collective
    return {
        "left": {
            "bubble": complex_vector_payload(left_bubble, source_order),
            "contact": complex_vector_payload(left_contact, source_order),
            "collective_mixed": complex_vector_payload(left_collective, source_order),
            "total": complex_vector_payload(left_total, source_order),
            "reconstruction_error_norm": float(np.linalg.norm(left_total - np.asarray(left_em, dtype=complex))),
            "valid_for_casimir_input": False,
        },
        "right": {
            "bubble": complex_vector_payload(right_bubble, source_order),
            "contact": complex_vector_payload(right_contact, source_order),
            "collective_mixed": complex_vector_payload(right_collective, source_order),
            "total": complex_vector_payload(right_total, source_order),
            "reconstruction_error_norm": float(np.linalg.norm(right_total - np.asarray(right_em, dtype=complex))),
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def collective_mixed_channel_decomposition(
    *,
    blocks: TargetBareBlocks,
    w_eta_left: np.ndarray,
    w_eta_right: np.ndarray,
    collective_order: tuple[str, ...],
) -> dict[str, Any]:
    """Decompose W_eta K_etaS and K_Seta W_eta into per-collective-channel products."""

    require_diagnostic_source_order(blocks.source_order)
    source_order = blocks.source_order
    n_eta = int(np.asarray(blocks.k_etaeta).shape[0])
    if len(collective_order) != n_eta:
        raise ValueError("collective_order length must match collective channel count")
    w_left = np.asarray(w_eta_left, dtype=complex).reshape(-1)
    w_right = np.asarray(w_eta_right, dtype=complex).reshape(-1)
    k_etas = np.asarray(blocks.k_etas, dtype=complex)
    k_seta = np.asarray(blocks.k_seta, dtype=complex)
    if w_left.shape[0] != n_eta or w_right.shape[0] != n_eta:
        raise ValueError("W_eta vector length must match collective channel count")

    left_entries = []
    left_totals = []
    for source_index, source_label in enumerate(source_order):
        values = w_left * k_etas[:, source_index]
        total = complex(np.sum(values))
        left_totals.append(total)
        left_entries.append(
            {
                "source_label": source_label,
                "contributions": complex_vector_payload(values, collective_order),
                "total": total,
                "valid_for_casimir_input": False,
            }
        )

    right_entries = []
    right_totals = []
    for source_index, source_label in enumerate(source_order):
        values = k_seta[source_index, :] * w_right
        total = complex(np.sum(values))
        right_totals.append(total)
        right_entries.append(
            {
                "source_label": source_label,
                "contributions": complex_vector_payload(values, collective_order),
                "total": total,
                "valid_for_casimir_input": False,
            }
        )

    left_direct = w_left @ k_etas
    right_direct = k_seta @ w_right
    return {
        "collective_order": list(collective_order),
        "left": left_entries,
        "right": right_entries,
        "left_reconstruction_error_norm": float(np.linalg.norm(np.asarray(left_totals, dtype=complex) - left_direct)),
        "right_reconstruction_error_norm": float(np.linalg.norm(np.asarray(right_totals, dtype=complex) - right_direct)),
        "valid_for_casimir_input": False,
    }


def extended_ward_candidate_result(
    *,
    name: str,
    description: str,
    blocks: TargetBareBlocks,
    w_eta_left: np.ndarray,
    w_eta_right: np.ndarray,
    collective_order: tuple[str, ...],
    physical_matrix_norm: float,
    etaeta_norm: float,
    solve_metadata: dict[str, Any] | None = None,
    tolerance: float = EXTENDED_WARD_TOLERANCE,
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    """Evaluate one candidate extended Ward vector."""

    require_diagnostic_source_order(blocks.source_order)
    source_order = blocks.source_order
    k_ss = np.asarray(blocks.k_ss, dtype=complex)
    k_seta = np.asarray(blocks.k_seta, dtype=complex)
    k_etas = np.asarray(blocks.k_etas, dtype=complex)
    k_etaeta = np.asarray(blocks.k_etaeta, dtype=complex)
    g = source_order.index("G")
    w_left = np.asarray(w_eta_left, dtype=complex).reshape(-1)
    w_right = np.asarray(w_eta_right, dtype=complex).reshape(-1)
    if w_left.shape[0] != k_etaeta.shape[0] or w_right.shape[0] != k_etaeta.shape[0]:
        raise ValueError("W_eta vector length must match collective channel count")

    left_em = k_ss[g, :] + w_left @ k_etas
    left_collective = k_seta[g, :] + w_left @ k_etaeta
    right_em = k_ss[:, g] + k_seta @ w_right
    right_collective = k_etas[:, g] + k_etaeta @ w_right

    left_em_norm = _norm(left_em)
    left_collective_norm = _norm(left_collective)
    right_em_norm = _norm(right_em)
    right_collective_norm = _norm(right_collective)
    left_total = float(np.sqrt(left_em_norm**2 + left_collective_norm**2))
    right_total = float(np.sqrt(right_em_norm**2 + right_collective_norm**2))

    return {
        "candidate": str(name),
        "description": str(description),
        "W_eta_left": complex_vector_payload(w_left, collective_order),
        "W_eta_right": complex_vector_payload(w_right, collective_order),
        "left_em_residual": complex_vector_payload(left_em, source_order),
        "left_collective_residual": complex_vector_payload(left_collective, collective_order),
        "right_em_residual": complex_vector_payload(right_em, source_order),
        "right_collective_residual": complex_vector_payload(right_collective, collective_order),
        "em_source_decomposition": em_source_decomposition(
            blocks=blocks,
            w_eta_left=w_left,
            w_eta_right=w_right,
            left_em=left_em,
            right_em=right_em,
        ),
        "collective_mixed_channel_decomposition": collective_mixed_channel_decomposition(
            blocks=blocks,
            w_eta_left=w_left,
            w_eta_right=w_right,
            collective_order=collective_order,
        ),
        "norms": {
            "left_em_norm": left_em_norm,
            "left_collective_norm": left_collective_norm,
            "left_total_extended_norm": left_total,
            "right_em_norm": right_em_norm,
            "right_collective_norm": right_collective_norm,
            "right_total_extended_norm": right_total,
            "left_em_norm_over_physical": _safe_ratio(left_em_norm, physical_matrix_norm, ratio_eps),
            "right_em_norm_over_physical": _safe_ratio(right_em_norm, physical_matrix_norm, ratio_eps),
            "left_collective_norm_over_etaeta": _safe_ratio(left_collective_norm, etaeta_norm, ratio_eps),
            "right_collective_norm_over_etaeta": _safe_ratio(right_collective_norm, etaeta_norm, ratio_eps),
            "valid_for_casimir_input": False,
        },
        "flags": {
            "left_total_below_tolerance": left_total < float(tolerance),
            "right_total_below_tolerance": right_total < float(tolerance),
            "valid_for_casimir_input": False,
        },
        "solve": {**(solve_metadata or {}), "valid_for_casimir_input": False},
        "valid_for_casimir_input": False,
    }


def extended_ward_candidates(
    *,
    blocks: TargetBareBlocks,
    delta0_eV: float,
    collective_order: tuple[str, ...],
    tolerance: float = EXTENDED_WARD_TOLERANCE,
    ratio_eps: float = RATIO_EPS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return candidate diagnostics plus Schur consistency data."""

    k_eff, schur_correction, schur = schur_effective_from_blocks(blocks)
    ratios = decomposition_ratios(k_eff, eps=ratio_eps, source_order=blocks.source_order)
    physical_norm = float(ratios["physical_matrix_norm"])
    etaeta_norm = _norm(blocks.k_etaeta)
    n_eta = int(np.asarray(blocks.k_etaeta).shape[0])
    if n_eta != len(collective_order):
        raise ValueError("collective_order length must match K_etaeta size")
    phase_index = collective_order.index("phase_eta2") if "phase_eta2" in collective_order else n_eta - 1

    zero = np.zeros(n_eta, dtype=complex)
    analytic_left = np.zeros(n_eta, dtype=complex)
    analytic_right = np.zeros(n_eta, dtype=complex)
    analytic_left[phase_index] = 2j * float(delta0_eV)
    analytic_right[phase_index] = -2j * float(delta0_eV)
    analytic_same_negative_left = np.zeros(n_eta, dtype=complex)
    analytic_same_negative_right = np.zeros(n_eta, dtype=complex)
    analytic_same_negative_left[phase_index] = -2j * float(delta0_eV)
    analytic_same_negative_right[phase_index] = -2j * float(delta0_eV)
    analytic_same_positive_left = np.zeros(n_eta, dtype=complex)
    analytic_same_positive_right = np.zeros(n_eta, dtype=complex)
    analytic_same_positive_left[phase_index] = 2j * float(delta0_eV)
    analytic_same_positive_right[phase_index] = 2j * float(delta0_eV)
    legacy_left = np.zeros(n_eta, dtype=complex)
    legacy_right = np.zeros(n_eta, dtype=complex)
    legacy_left[phase_index] = 2.0 * float(delta0_eV)
    legacy_right[phase_index] = 2.0 * float(delta0_eV)

    g = blocks.source_order.index("G")
    fitted_left, fitted_left_meta = solve_left_collective_vector(blocks.k_etaeta, np.asarray(blocks.k_seta)[g, :])
    fitted_right, fitted_right_meta = solve_right_collective_vector(blocks.k_etaeta, np.asarray(blocks.k_etas)[:, g])

    base_kwargs = {
        "blocks": blocks,
        "collective_order": collective_order,
        "physical_matrix_norm": physical_norm,
        "etaeta_norm": etaeta_norm,
        "tolerance": tolerance,
        "ratio_eps": ratio_eps,
    }
    candidates = [
        extended_ward_candidate_result(
            name="zero_collective",
            description="No order-parameter gauge vector; reproduces raw G row/column residual of K_SS.",
            w_eta_left=zero,
            w_eta_right=zero,
            **base_kwargs,
        ),
        extended_ward_candidate_result(
            name="analytic_imaginary_left_right_opposite",
            description="Analytic BdG eta2 convention: W_left=[0,+2i delta0], W_right=[0,-2i delta0].",
            w_eta_left=analytic_left,
            w_eta_right=analytic_right,
            **base_kwargs,
        ),
        extended_ward_candidate_result(
            name="analytic_imaginary_same_negative",
            description="Same-sign negative imaginary diagnostic: W_left=[0,-2i delta0], W_right=[0,-2i delta0].",
            w_eta_left=analytic_same_negative_left,
            w_eta_right=analytic_same_negative_right,
            **base_kwargs,
        ),
        extended_ward_candidate_result(
            name="analytic_imaginary_same_positive",
            description="Same-sign positive imaginary diagnostic: W_left=[0,+2i delta0], W_right=[0,+2i delta0].",
            w_eta_left=analytic_same_positive_left,
            w_eta_right=analytic_same_positive_right,
            **base_kwargs,
        ),
        extended_ward_candidate_result(
            name="analytic_real_same_sign_legacy",
            description="Legacy real same-sign comparison vector retained for diagnostics only.",
            w_eta_left=legacy_left,
            w_eta_right=legacy_right,
            **base_kwargs,
        ),
        extended_ward_candidate_result(
            name="fitted_left_collective_equation",
            description="W_left fitted so K_Geta + W_left K_etaeta = 0.",
            w_eta_left=fitted_left,
            w_eta_right=zero,
            solve_metadata=fitted_left_meta,
            **base_kwargs,
        ),
        extended_ward_candidate_result(
            name="fitted_right_collective_equation",
            description="W_right fitted so K_etaG + K_etaeta W_right = 0.",
            w_eta_left=zero,
            w_eta_right=fitted_right,
            solve_metadata=fitted_right_meta,
            **base_kwargs,
        ),
        extended_ward_candidate_result(
            name="fitted_both_independent",
            description="Independent left/right W_eta vectors fitted from their own collective equations.",
            w_eta_left=fitted_left,
            w_eta_right=fitted_right,
            solve_metadata={"left": fitted_left_meta, "right": fitted_right_meta, "valid_for_casimir_input": False},
            **base_kwargs,
        ),
    ]

    by_name = {row["candidate"]: row for row in candidates}
    left_em = np.asarray([item["value"] for item in by_name["fitted_left_collective_equation"]["left_em_residual"]], dtype=complex)
    right_em = np.asarray([item["value"] for item in by_name["fitted_right_collective_equation"]["right_em_residual"]], dtype=complex)
    consistency = {
        "fitted_left_em_minus_schur_g_row_norm": float(np.linalg.norm(left_em - k_eff[g, :])),
        "fitted_right_em_minus_schur_g_col_norm": float(np.linalg.norm(right_em - k_eff[:, g])),
        "schur": schur,
        "valid_for_casimir_input": False,
    }
    return candidates, {
        "K_eff": k_eff,
        "Schur_correction": schur_correction,
        "ratios": ratios,
        "schur_consistency": consistency,
        "valid_for_casimir_input": False,
    }


def interpretation_flags(candidate_results: Sequence[dict[str, Any]], *, tolerance: float = EXTENDED_WARD_TOLERANCE) -> dict[str, Any]:
    by_name = {str(item["candidate"]): item for item in candidate_results}
    analytic = by_name.get("analytic_imaginary_left_right_opposite")
    analytic_same_negative = by_name.get("analytic_imaginary_same_negative")
    analytic_same_positive = by_name.get("analytic_imaginary_same_positive")
    fitted_left = by_name.get("fitted_left_collective_equation")
    fitted_right = by_name.get("fitted_right_collective_equation")

    def norm(item: dict[str, Any] | None, key: str) -> float:
        return float("nan") if item is None else float(item["norms"][key])

    return {
        "analytic_extended_ward_left_closed": bool(analytic is not None and norm(analytic, "left_total_extended_norm") < float(tolerance)),
        "analytic_extended_ward_right_closed": bool(analytic is not None and norm(analytic, "right_total_extended_norm") < float(tolerance)),
        "analytic_same_negative_left_closed": bool(analytic_same_negative is not None and norm(analytic_same_negative, "left_total_extended_norm") < float(tolerance)),
        "analytic_same_negative_right_closed": bool(analytic_same_negative is not None and norm(analytic_same_negative, "right_total_extended_norm") < float(tolerance)),
        "analytic_same_positive_left_closed": bool(analytic_same_positive is not None and norm(analytic_same_positive, "left_total_extended_norm") < float(tolerance)),
        "analytic_same_positive_right_closed": bool(analytic_same_positive is not None and norm(analytic_same_positive, "right_total_extended_norm") < float(tolerance)),
        "fitted_collective_left_closes_but_em_fails": bool(
            fitted_left is not None and norm(fitted_left, "left_collective_norm") < float(tolerance) and norm(fitted_left, "left_em_norm") >= float(tolerance)
        ),
        "fitted_collective_right_closes_but_em_fails": bool(
            fitted_right is not None and norm(fitted_right, "right_collective_norm") < float(tolerance) and norm(fitted_right, "right_em_norm") >= float(tolerance)
        ),
        "tolerance": float(tolerance),
        "valid_for_casimir_input": False,
    }


def block_norms(blocks: TargetBareBlocks) -> dict[str, Any]:
    g = blocks.source_order.index("G")
    return {
        "K_SS_norm": _norm(blocks.k_ss),
        "K_Seta_norm": _norm(blocks.k_seta),
        "K_etaS_norm": _norm(blocks.k_etas),
        "K_etaeta_norm": _norm(blocks.k_etaeta),
        "K_SS_G_row_norm": _norm(np.asarray(blocks.k_ss)[g, :]),
        "K_SS_G_col_norm": _norm(np.asarray(blocks.k_ss)[:, g]),
        "valid_for_casimir_input": False,
    }


def extended_ward_kernel_payload(
    *,
    model_name: str,
    pairing_name: str,
    frequency: dict[str, Any],
    debug_parameters: dict[str, Any],
    source_order: tuple[str, ...],
    collective_order: tuple[str, ...],
    raw_ansatz_channel_names: tuple[str, ...] | None,
    block_norm_payload: dict[str, Any],
    candidate_results: Sequence[dict[str, Any]],
    schur_consistency: dict[str, Any],
    flags: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "valid_for_casimir_input": False,
            "reason": "extended_ward_kernel_debug_not_casimir_input",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "frequency": frequency,
        "debug_parameters": {
            **debug_parameters,
            "debug_only_extended_ward_kernel": True,
            "average_order": "average_blocks_then_schur",
            "valid_for_casimir_input": False,
        },
        "source_order_diagnostic": list(source_order),
        "collective_order": list(collective_order),
        "raw_ansatz_channel_names": list(raw_ansatz_channel_names) if raw_ansatz_channel_names is not None else None,
        "block_norms": block_norm_payload,
        "candidate_results": list(candidate_results),
        "schur_consistency": schur_consistency,
        "interpretation_flags": flags,
        "valid_for_casimir_input": False,
    }


def run_extended_ward_kernel(
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
    tolerance: float = EXTENDED_WARD_TOLERANCE,
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    """Run q-along-x debug-only extended Ward-kernel audit."""

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

    response = average_bare_blocks_then_schur(scaled_blocks)
    averaged = response.bare_blocks
    collective_order, raw_names = collective_order_from_ansatz(inputs.ansatz, averaged.k_etaeta.shape[0])
    delta0 = float(getattr(inputs.pairing_params, "delta0_eV", 0.0))
    candidates, derived = extended_ward_candidates(
        blocks=averaged,
        delta0_eV=delta0,
        collective_order=collective_order,
        tolerance=tolerance,
        ratio_eps=ratio_eps,
    )
    shifted = _shifted_payload(shift_fractions, shifts)
    return extended_ward_kernel_payload(
        model_name=model_name,
        pairing_name=pairing_name,
        frequency=frequency_payload(matsubara_index, temperature_K),
        debug_parameters={
            "q_model_convention": "q_along_x_only",
            "q_value": float(q_value),
            "nk": int(nk),
            "shift_fractions": [float(value) for value in shift_fractions],
            "num_shifted_meshes": len(shifts),
            "contact_scale": float(contact_scale),
            "ratio_eps": float(ratio_eps),
            "tolerance": float(tolerance),
            "shifted_mesh_average": shifted,
            "valid_for_casimir_input": False,
        },
        source_order=averaged.source_order,
        collective_order=collective_order,
        raw_ansatz_channel_names=raw_names,
        block_norm_payload=block_norms(averaged),
        candidate_results=candidates,
        schur_consistency=derived["schur_consistency"],
        flags=interpretation_flags(candidates, tolerance=tolerance),
    )


def run_and_write_extended_ward_kernel(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_extended_ward_kernel(**kwargs)
    write_json(Path(output_dir) / "extended_ward_kernel.json", payload)
    return payload
