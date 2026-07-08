"""Debug-only eta-channel Schur ablation diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..adapters.bubble_adapter import TargetBareBlocks, compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.conventions import SOURCE_ORDER_DIAGNOSTIC, require_diagnostic_source_order
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .collective_schur_factors import COLLECTIVE_ORDER, solve_collective_action
from .contact_ablation import _shifted_payload, scaled_contact_blocks
from .nk_sweep import RATIO_EPS
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .signed_decomposition import ENTRY_SPECS, decomposition_ratios, signed_entry

SCHEMA_VERSION = "finite_q_tmte_eta_channel_ablation_v1"
SCHUR_ENTRY_NAMES = ("GG", "GTM", "TMG", "TMTM")


def _entries(
    matrix: np.ndarray,
    entry_names: tuple[str, ...],
    *,
    source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC,
) -> dict[str, complex]:
    specs = {name: (row, col) for name, row, col in ENTRY_SPECS}
    return {name: signed_entry(matrix, specs[name][0], specs[name][1], source_order) for name in entry_names}


def eta_channel_mode_result(
    *,
    mode: str,
    k_ss_scaled: np.ndarray,
    k_seta: np.ndarray,
    k_etas: np.ndarray,
    k_etaeta: np.ndarray,
    channel_indices: tuple[int, ...],
    source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC,
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    """Build one debug-only eta-channel ablation mode result."""

    require_diagnostic_source_order(source_order)
    kss = np.asarray(k_ss_scaled, dtype=complex)
    if mode == "no_schur":
        schur_correction = np.zeros_like(kss, dtype=complex)
        schur = {
            "solve_method": "none",
            "etaeta_condition_number": None,
            "condition_threshold": None,
            "numerically_suspect": False,
            "valid_for_casimir_input": False,
        }
    else:
        if not channel_indices:
            raise ValueError("channel_indices must not be empty for Schur modes")
        selector = np.asarray(channel_indices, dtype=int)
        seta = np.asarray(k_seta, dtype=complex)[:, selector]
        etas = np.asarray(k_etas, dtype=complex)[selector, :]
        etaeta = np.asarray(k_etaeta, dtype=complex)[np.ix_(selector, selector)]
        x_action, schur = solve_collective_action(etaeta, etas)
        schur_correction = seta @ x_action

    k_eff = kss - schur_correction
    ratios = decomposition_ratios(k_eff, eps=ratio_eps, source_order=source_order)
    diagnostics = {
        "gauge_row_norm": ratios["gauge_row_norm"],
        "gauge_col_norm": ratios["gauge_col_norm"],
        "gauge_gg_norm": ratios["gauge_gg_norm"],
        "physical_matrix_norm": ratios["physical_matrix_norm"],
        "gauge_over_physical": ratios["gauge_over_physical"],
        "gauge_over_tm_abs": ratios["gauge_over_tm_abs"],
        "gauge_gg_over_tm_abs": ratios["gauge_gg_over_tm_abs"],
        "ratio_eps": ratios["ratio_eps"],
        "etaeta_condition_number": schur["etaeta_condition_number"],
        "schur_solve_method": schur["solve_method"],
        "schur_numerically_suspect": schur["numerically_suspect"],
        "valid_for_casimir_input": False,
    }
    return {
        "mode": mode,
        "included_collective_channels": [COLLECTIVE_ORDER[index] for index in channel_indices],
        "Schur_correction_entries": _entries(schur_correction, SCHUR_ENTRY_NAMES, source_order=source_order),
        "K_eff_entries": _entries(k_eff, tuple(name for name, _, _ in ENTRY_SPECS), source_order=source_order),
        "diagnostics": diagnostics,
        "schur": schur,
        "valid_for_casimir_input": False,
    }


def eta_channel_mode_results(
    *,
    blocks: TargetBareBlocks,
    ratio_eps: float = RATIO_EPS,
) -> list[dict[str, Any]]:
    """Return no/one-channel/full Schur mode results from averaged scaled blocks."""

    return [
        eta_channel_mode_result(
            mode="no_schur",
            k_ss_scaled=blocks.k_ss,
            k_seta=blocks.k_seta,
            k_etas=blocks.k_etas,
            k_etaeta=blocks.k_etaeta,
            channel_indices=(),
            source_order=blocks.source_order,
            ratio_eps=ratio_eps,
        ),
        eta_channel_mode_result(
            mode="eta0_only",
            k_ss_scaled=blocks.k_ss,
            k_seta=blocks.k_seta,
            k_etas=blocks.k_etas,
            k_etaeta=blocks.k_etaeta,
            channel_indices=(0,),
            source_order=blocks.source_order,
            ratio_eps=ratio_eps,
        ),
        eta_channel_mode_result(
            mode="eta1_only",
            k_ss_scaled=blocks.k_ss,
            k_seta=blocks.k_seta,
            k_etas=blocks.k_etas,
            k_etaeta=blocks.k_etaeta,
            channel_indices=(1,),
            source_order=blocks.source_order,
            ratio_eps=ratio_eps,
        ),
        eta_channel_mode_result(
            mode="eta_all",
            k_ss_scaled=blocks.k_ss,
            k_seta=blocks.k_seta,
            k_etas=blocks.k_etas,
            k_etaeta=blocks.k_etaeta,
            channel_indices=tuple(range(blocks.k_etaeta.shape[0])),
            source_order=blocks.source_order,
            ratio_eps=ratio_eps,
        ),
    ]


def eta_channel_ablation_payload(
    *,
    model_name: str,
    pairing_name: str,
    frequency: dict[str, Any],
    debug_parameters: dict[str, Any],
    mode_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build the top-level eta-channel ablation payload."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "valid_for_casimir_input": False,
            "reason": "eta_channel_ablation_debug_not_casimir_input",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "frequency": frequency,
        "debug_parameters": {
            **debug_parameters,
            "debug_only_eta_channel_ablation": True,
            "average_order": "average_blocks_then_schur",
            "valid_for_casimir_input": False,
        },
        "source_order_diagnostic": list(SOURCE_ORDER_DIAGNOSTIC),
        "collective_order": list(COLLECTIVE_ORDER),
        "mode_results": list(mode_results),
        "valid_for_casimir_input": False,
    }


def eta_channel_ablation_from_blocks(
    *,
    blocks: TargetBareBlocks,
    contact_scale: float,
    shifted_payload: dict[str, Any],
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    """Build eta-channel ablation pieces from an already computed block set."""

    response = average_bare_blocks_then_schur([scaled_contact_blocks(blocks, contact_scale)])
    return {
        "q_model": response.bare_blocks.conventions.q,
        "q_norm": response.bare_blocks.conventions.q_norm,
        "shifted_mesh_average": shifted_payload,
        "mode_results": eta_channel_mode_results(blocks=response.bare_blocks, ratio_eps=ratio_eps),
        "valid_for_casimir_input": False,
    }


def run_eta_channel_ablation(
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
    """Run q-along-x debug-only eta-channel Schur ablation diagnostics."""

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
    return eta_channel_ablation_payload(
        model_name=model_name,
        pairing_name=pairing_name,
        frequency=frequency_payload(matsubara_index, temperature_K),
        debug_parameters={
            "debug_only_eta_channel_ablation": True,
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
        mode_results=eta_channel_mode_results(blocks=response.bare_blocks, ratio_eps=ratio_eps),
    )


def run_and_write_eta_channel_ablation(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_eta_channel_ablation(**kwargs)
    write_json(Path(output_dir) / "eta_channel_ablation.json", payload)
    return payload
