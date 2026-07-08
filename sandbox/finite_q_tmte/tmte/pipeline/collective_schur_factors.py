"""Debug-only collective Schur factor decomposition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..adapters.bubble_adapter import TargetBareBlocks, compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.conventions import SOURCE_ORDER_DIAGNOSTIC, require_diagnostic_source_order
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .contact_ablation import _shifted_payload, scaled_contact_blocks
from .nk_sweep import RATIO_EPS
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .signed_decomposition import ENTRY_SPECS, decomposition_ratios

SCHEMA_VERSION = "finite_q_tmte_collective_schur_factors_v1"
COLLECTIVE_ORDER = ("eta0", "eta1")
SCHUR_CONDITION_THRESHOLD = 1e12


def solve_collective_action(
    k_etaeta: np.ndarray,
    k_etas: np.ndarray,
    *,
    condition_threshold: float = SCHUR_CONDITION_THRESHOLD,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return X = solve(K_etaeta, K_etaS) using the sandbox Schur fallback convention."""

    etaeta = np.asarray(k_etaeta, dtype=complex)
    etas = np.asarray(k_etas, dtype=complex)
    condition = float(np.linalg.cond(etaeta))
    if not np.isfinite(condition) or condition > condition_threshold:
        x_action = np.linalg.pinv(etaeta) @ etas
        solve_method = "pinv_diagnostic"
    else:
        x_action = np.linalg.solve(etaeta, etas)
        solve_method = "solve"
    return x_action, {
        "solve_method": solve_method,
        "etaeta_condition_number": condition,
        "condition_threshold": float(condition_threshold),
        "numerically_suspect": solve_method == "pinv_diagnostic",
        "valid_for_casimir_input": False,
    }


def schur_factor_decomposition(
    k_seta: np.ndarray,
    x_action: np.ndarray,
    *,
    source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC,
    collective_order: tuple[str, ...] = COLLECTIVE_ORDER,
) -> dict[str, Any]:
    """Decompose Schur entries into per-collective-channel products."""

    require_diagnostic_source_order(source_order)
    seta = np.asarray(k_seta, dtype=complex)
    x = np.asarray(x_action, dtype=complex)
    if seta.shape[1] != x.shape[0]:
        raise ValueError("K_Seta and X collective dimensions must match")
    if len(collective_order) != seta.shape[1]:
        raise ValueError("collective_order length must match K_Seta collective dimension")

    entries: dict[str, Any] = {}
    for entry_name, row_label, col_label in ENTRY_SPECS:
        row = source_order.index(row_label)
        col = source_order.index(col_label)
        contributions = []
        products = []
        for alpha, channel in enumerate(collective_order):
            left = complex(seta[row, alpha])
            right = complex(x[alpha, col])
            product = left * right
            products.append(product)
            contributions.append(
                {
                    "collective_channel": channel,
                    "K_Seta": left,
                    "X": right,
                    "product": product,
                }
            )
        total = sum(products, 0.0 + 0.0j)
        entries[entry_name] = {
            "total": total,
            "contributions": contributions,
            "reconstruction_error": float(abs(total - sum(products, 0.0 + 0.0j))),
        }
    return entries


def consistency_diagnostics(
    *,
    k_eff: np.ndarray,
    schur_correction: np.ndarray,
    k_seta: np.ndarray,
    k_etas: np.ndarray,
    k_etaeta: np.ndarray,
    source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC,
) -> dict[str, Any]:
    """Return diagnostic-only consistency checks, without pass/fail physics claims."""

    require_diagnostic_source_order(source_order)
    eff = np.asarray(k_eff, dtype=complex)
    correction = np.asarray(schur_correction, dtype=complex)
    seta = np.asarray(k_seta, dtype=complex)
    etas = np.asarray(k_etas, dtype=complex)
    g = source_order.index("G")
    tm = source_order.index("TM")

    eff_difference = complex(eff[g, tm] - np.conj(eff[tm, g]))
    schur_difference = complex(correction[g, tm] - np.conj(correction[tm, g]))
    seta_shape = seta.shape
    etas_dagger_shape = etas.conj().T.shape
    if seta_shape == etas_dagger_shape:
        seta_etas_difference = seta - etas.conj().T
        seta_etas_payload: dict[str, Any] = {
            "available": True,
            "difference": seta_etas_difference,
            "frobenius_norm": float(np.linalg.norm(seta_etas_difference)),
            "max_abs": float(np.max(np.abs(seta_etas_difference))),
        }
    else:
        seta_etas_payload = {
            "available": False,
            "reason": "K_Seta shape does not match K_etaS dagger shape",
            "K_Seta_shape": list(seta_shape),
            "K_etaS_dagger_shape": list(etas_dagger_shape),
        }

    return {
        "diagnostic_only": True,
        "K_eff_GTM_minus_conj_TMG": {"difference": eff_difference, "abs": float(abs(eff_difference))},
        "Schur_GTM_minus_conj_TMG": {"difference": schur_difference, "abs": float(abs(schur_difference))},
        "K_Seta_minus_K_etaS_dagger": seta_etas_payload,
        "etaeta_condition_number": float(np.linalg.cond(np.asarray(k_etaeta, dtype=complex))),
        "valid_for_casimir_input": False,
    }


def collective_schur_factors_from_blocks(
    *,
    blocks: TargetBareBlocks,
    contact_scale: float,
    shifted_payload: dict[str, Any],
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    """Build the debug collective Schur factor payload pieces from averaged blocks."""

    scaled = scaled_contact_blocks(blocks, contact_scale)
    response = average_bare_blocks_then_schur([scaled])
    averaged = response.bare_blocks
    x_action, schur = solve_collective_action(averaged.k_etaeta, averaged.k_etas, condition_threshold=response.schur.condition_threshold)
    schur_correction = averaged.k_seta @ x_action
    k_eff = averaged.k_ss - schur_correction
    return {
        "q_model": averaged.conventions.q,
        "q_norm": averaged.conventions.q_norm,
        "shifted_mesh_average": shifted_payload,
        "matrices": {
            "K_Seta": averaged.k_seta,
            "K_etaS": averaged.k_etas,
            "K_etaeta": averaged.k_etaeta,
            "K_etaeta_inverse_or_solver_action": {
                "kind": "solve_action_X",
                "description": "X = solve(K_etaeta, K_etaS); no explicit inverse is used for Schur construction",
                "X": x_action,
                "diagnostic_inverse_only": False,
                "valid_for_casimir_input": False,
            },
            "Schur_correction": schur_correction,
            "K_eff": k_eff,
        },
        "schur_factor_decomposition": schur_factor_decomposition(averaged.k_seta, x_action, source_order=averaged.source_order),
        "consistency_diagnostics": consistency_diagnostics(
            k_eff=k_eff,
            schur_correction=schur_correction,
            k_seta=averaged.k_seta,
            k_etas=averaged.k_etas,
            k_etaeta=averaged.k_etaeta,
            source_order=averaged.source_order,
        ),
        "ratios": decomposition_ratios(k_eff, eps=ratio_eps, source_order=averaged.source_order),
        "schur": schur,
    }


def collective_schur_factors_payload(
    *,
    model_name: str,
    pairing_name: str,
    frequency: dict[str, Any],
    debug_parameters: dict[str, Any],
    matrices: dict[str, Any],
    schur_factor_decomposition_payload: dict[str, Any],
    consistency_diagnostics_payload: dict[str, Any],
    ratios: dict[str, Any],
    schur: dict[str, Any],
    collective_order: tuple[str, ...] = COLLECTIVE_ORDER,
) -> dict[str, Any]:
    """Build the top-level collective Schur factor JSON payload."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "valid_for_casimir_input": False,
            "reason": "collective_schur_factors_debug_not_casimir_input",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "frequency": frequency,
        "debug_parameters": {
            **debug_parameters,
            "debug_only_collective_schur_factors": True,
            "average_order": "average_blocks_then_schur",
            "valid_for_casimir_input": False,
        },
        "source_order_diagnostic": list(SOURCE_ORDER_DIAGNOSTIC),
        "collective_order": list(collective_order),
        "matrices": matrices,
        "schur_factor_decomposition": schur_factor_decomposition_payload,
        "consistency_diagnostics": {**consistency_diagnostics_payload, "diagnostic_only": True, "valid_for_casimir_input": False},
        "ratios": {**ratios, "valid_for_casimir_input": False},
        "schur": {**schur, "valid_for_casimir_input": False},
        "valid_for_casimir_input": False,
    }


def run_collective_schur_factors(
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
    """Run q-along-x debug-only collective Schur factor diagnostics."""

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

    shifted = _shifted_payload(shift_fractions, shifts)
    response = average_bare_blocks_then_schur(scaled_blocks)
    averaged = response.bare_blocks
    x_action, schur = solve_collective_action(averaged.k_etaeta, averaged.k_etas, condition_threshold=response.schur.condition_threshold)
    schur_correction = averaged.k_seta @ x_action
    k_eff = averaged.k_ss - schur_correction

    return collective_schur_factors_payload(
        model_name=model_name,
        pairing_name=pairing_name,
        frequency=frequency_payload(matsubara_index, temperature_K),
        debug_parameters={
            "debug_only_collective_schur_factors": True,
            "q_model_convention": "q_along_x_only",
            "q_value": float(q_value),
            "nk": int(nk),
            "shift_fractions": [float(value) for value in shift_fractions],
            "num_shifted_meshes": len(shifts),
            "contact_scale": float(contact_scale),
            "ratio_eps": float(ratio_eps),
            "average_order": "average_blocks_then_schur",
            "valid_for_casimir_input": False,
        },
        matrices={
            "K_Seta": averaged.k_seta,
            "K_etaS": averaged.k_etas,
            "K_etaeta": averaged.k_etaeta,
            "K_etaeta_inverse_or_solver_action": {
                "kind": "solve_action_X",
                "description": "X = solve(K_etaeta, K_etaS); no explicit inverse is used for Schur construction",
                "X": x_action,
                "diagnostic_inverse_only": False,
                "valid_for_casimir_input": False,
            },
            "Schur_correction": schur_correction,
            "K_eff": k_eff,
        },
        schur_factor_decomposition_payload=schur_factor_decomposition(averaged.k_seta, x_action, source_order=averaged.source_order),
        consistency_diagnostics_payload=consistency_diagnostics(
            k_eff=k_eff,
            schur_correction=schur_correction,
            k_seta=averaged.k_seta,
            k_etas=averaged.k_etas,
            k_etaeta=averaged.k_etaeta,
            source_order=averaged.source_order,
        ),
        ratios=decomposition_ratios(k_eff, eps=ratio_eps, source_order=averaged.source_order),
        schur=schur,
    )


def run_and_write_collective_schur_factors(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_collective_schur_factors(**kwargs)
    write_json(Path(output_dir) / "collective_schur_factors.json", payload)
    return payload
