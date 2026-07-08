"""Debug-only spatial-contact ablation diagnostics for finite-q TM/TE."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..adapters.bubble_adapter import TargetBareBlocks, compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.conventions import SOURCE_ORDER_DIAGNOSTIC, require_diagnostic_source_order
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .block_builder import EffectiveTargetResponse, build_effective_from_blocks
from .nk_sweep import RATIO_EPS, selected_matrix_elements
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions

SCHEMA_VERSION = "finite_q_tmte_contact_ablation_v1"


def scaled_contact_blocks(blocks: TargetBareBlocks, contact_scale: float) -> TargetBareBlocks:
    """Return a debug copy with only the spatial K_SS contact scaled."""

    scale = float(contact_scale)
    scaled_contact = scale * np.asarray(blocks.k_ss_contact, dtype=complex)
    return replace(
        blocks,
        k_ss_contact=scaled_contact,
        k_ss=np.asarray(blocks.k_ss_bubble, dtype=complex) + scaled_contact,
        metadata={
            **blocks.metadata,
            "debug_only_contact_ablation": True,
            "contact_scale": scale,
            "valid_for_casimir_input": False,
        },
    )


def _matrix_diagnostics(matrix: np.ndarray, *, source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC) -> dict[str, Any]:
    require_diagnostic_source_order(source_order)
    array = np.asarray(matrix, dtype=complex)
    if array.shape != (3, 3):
        raise ValueError("diagnostic matrix must have shape (3, 3)")
    g = source_order.index("G")
    tm = source_order.index("TM")
    te = source_order.index("TE")
    return {
        "G_row_norm": float(np.linalg.norm(array[g, :])),
        "G_col_norm": float(np.linalg.norm(array[:, g])),
        "GG_abs": float(abs(array[g, g])),
        "GTM_abs": float(abs(array[g, tm])),
        "TMG_abs": float(abs(array[tm, g])),
        "TMTM_real": float(np.real(array[tm, tm])),
        "TETE_real": float(np.real(array[te, te])),
        "valid_for_casimir_input": False,
    }


def bare_diagnostics(blocks: TargetBareBlocks) -> dict[str, Any]:
    """Return diagnostics for bubble, scaled contact, and scaled bare K_SS."""

    return {
        "K_SS_bubble": _matrix_diagnostics(blocks.k_ss_bubble, source_order=blocks.source_order),
        "K_SS_contact_scaled": _matrix_diagnostics(blocks.k_ss_contact, source_order=blocks.source_order),
        "K_SS_scaled": _matrix_diagnostics(blocks.k_ss, source_order=blocks.source_order),
        "valid_for_casimir_input": False,
    }


def effective_diagnostics(response: EffectiveTargetResponse) -> dict[str, Any]:
    """Return compact diagnostics for K_eff = K_GTMTE_eff."""

    diagnostics = _matrix_diagnostics(response.schur.effective, source_order=response.bare_blocks.source_order)
    diagnostics.update(
        {
            "physical_matrix_norm": float(response.diagnostics["physical_matrix_norm"]),
            "etaeta_condition_number": float(response.schur.etaeta_condition_number),
            "schur_solve_method": response.schur.solve_method,
            "schur_numerically_suspect": bool(response.schur.numerically_suspect),
            "valid_for_casimir_input": False,
        }
    )
    return diagnostics


def ablation_ratios(eff_diagnostics: dict[str, Any], elements: dict[str, complex], *, eps: float = RATIO_EPS) -> dict[str, Any]:
    denominator_eps = float(eps)
    tm_abs = float(abs(elements["K_TMTM"]))
    return {
        "gauge_over_physical": float(eff_diagnostics["G_row_norm"]) / max(float(eff_diagnostics["physical_matrix_norm"]), denominator_eps),
        "gauge_over_tm_abs": float(eff_diagnostics["G_row_norm"]) / max(tm_abs, denominator_eps),
        "gauge_gg_over_tm_abs": float(eff_diagnostics["GG_abs"]) / max(tm_abs, denominator_eps),
        "ratio_eps": denominator_eps,
        "valid_for_casimir_input": False,
    }


def contact_scale_result(
    *,
    contact_scale: float,
    response: EffectiveTargetResponse,
    shifted_payload: dict[str, Any],
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    eff_diag = effective_diagnostics(response)
    elements = selected_matrix_elements(response.schur.effective)
    return {
        "debug_only_contact_ablation": True,
        "contact_scale": float(contact_scale),
        "q_model": response.bare_blocks.conventions.q,
        "q_norm": response.bare_blocks.conventions.q_norm,
        "shifted_mesh_average": shifted_payload,
        "bare_diagnostics": bare_diagnostics(response.bare_blocks),
        "effective_diagnostics": eff_diag,
        "schur": {
            "solve_method": response.schur.solve_method,
            "etaeta_condition_number": response.schur.etaeta_condition_number,
            "condition_threshold": response.schur.condition_threshold,
            "numerically_suspect": response.schur.numerically_suspect,
            "valid_for_casimir_input": False,
        },
        "selected_matrix_elements": elements,
        "ratios": ablation_ratios(eff_diag, elements, eps=ratio_eps),
        "valid_for_casimir_input": False,
    }


def _shifted_payload(shift_fractions: Sequence[float], shift_pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    return {
        "enabled": len(tuple(shift_pairs)) > 1 or any(abs(sx) > 0.0 or abs(sy) > 0.0 for sx, sy in shift_pairs),
        "average_order": "average_blocks_then_schur",
        "shift_fractions": [float(value) for value in shift_fractions],
        "shift_pairs": [[float(sx), float(sy)] for sx, sy in shift_pairs],
        "num_shifted_meshes": len(tuple(shift_pairs)),
        "valid_for_casimir_input": False,
    }


def run_contact_ablation(
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
    contact_scales: Sequence[float] = (1.0, 0.0, -1.0),
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    """Run q-along-x debug-only contact ablation diagnostics."""

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
    per_shift_blocks: list[TargetBareBlocks] = []
    for sx, sy in shifts:
        points = shifted_uniform_bz_mesh(nk, sx, sy)
        weights = weights_for_points(points)
        per_shift_blocks.append(
            compute_target_bare_blocks(
                spec=inputs.spec,
                ansatz=inputs.ansatz,
                q_model=q_model,
                xi_eV=xi_eV,
                k_points=points,
                weights=weights,
                config=inputs.config,
                pairing_params=inputs.pairing_params,
            )
        )

    shifted = _shifted_payload(shift_fractions, shifts)
    results = []
    for scale in contact_scales:
        scaled_blocks = [scaled_contact_blocks(blocks, float(scale)) for blocks in per_shift_blocks]
        response = average_bare_blocks_then_schur(scaled_blocks) if len(scaled_blocks) > 1 else build_effective_from_blocks(scaled_blocks[0])
        results.append(contact_scale_result(contact_scale=float(scale), response=response, shifted_payload=shifted, ratio_eps=ratio_eps))

    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "valid_for_casimir_input": False,
            "reason": "contact_ablation_debug_not_casimir_input",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "frequency": frequency_payload(matsubara_index, temperature_K),
        "debug_parameters": {
            "debug_only_contact_ablation": True,
            "q_model_convention": "q_along_x_only",
            "q_value": float(q_value),
            "nk": int(nk),
            "shift_fractions": [float(value) for value in shift_fractions],
            "contact_scales": [float(value) for value in contact_scales],
            "ratio_eps": float(ratio_eps),
            "average_order": "average_blocks_then_schur",
            "valid_for_casimir_input": False,
        },
        "contact_scale_results": results,
        "valid_for_casimir_input": False,
    }


def run_and_write_contact_ablation(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_contact_ablation(**kwargs)
    write_json(Path(output_dir) / "contact_ablation.json", payload)
    return payload

