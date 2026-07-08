"""Debug-only phase_eta2 convention-transform diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..adapters.bubble_adapter import TargetBareBlocks, compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.conventions import SOURCE_ORDER_DIAGNOSTIC
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .collective_schur_factors import collective_order_from_ansatz, solve_collective_action
from .contact_ablation import _shifted_payload, scaled_contact_blocks
from .eta_channel_ablation import _entries
from .nk_sweep import RATIO_EPS
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .signed_decomposition import ENTRY_SPECS, decomposition_ratios

SCHEMA_VERSION = "finite_q_tmte_phase_eta2_convention_v1"
PHASE_ETA2_INDEX = 1
DEFAULT_TRANSFORMS = (
    "identity",
    "phase_eta2_seta_sign_flip",
    "phase_eta2_etas_sign_flip",
    "phase_eta2_both_sign_flip",
    "phase_eta2_seta_times_i",
    "phase_eta2_seta_times_minus_i",
    "phase_eta2_etas_times_i",
    "phase_eta2_etas_times_minus_i",
    "phase_eta2_seta_conjugate",
    "phase_eta2_etas_conjugate",
    "phase_eta2_both_conjugate",
    "phase_eta2_kernel_sign_flip",
)

TRANSFORM_DESCRIPTIONS = {
    "identity": "No debug convention transform.",
    "phase_eta2_seta_sign_flip": "Multiply K_Seta[:, phase_eta2] by -1.",
    "phase_eta2_etas_sign_flip": "Multiply K_etaS[phase_eta2, :] by -1.",
    "phase_eta2_both_sign_flip": "Multiply K_Seta[:, phase_eta2] and K_etaS[phase_eta2, :] by -1.",
    "phase_eta2_seta_times_i": "Multiply K_Seta[:, phase_eta2] by 1j.",
    "phase_eta2_seta_times_minus_i": "Multiply K_Seta[:, phase_eta2] by -1j.",
    "phase_eta2_etas_times_i": "Multiply K_etaS[phase_eta2, :] by 1j.",
    "phase_eta2_etas_times_minus_i": "Multiply K_etaS[phase_eta2, :] by -1j.",
    "phase_eta2_seta_conjugate": "Conjugate K_Seta[:, phase_eta2].",
    "phase_eta2_etas_conjugate": "Conjugate K_etaS[phase_eta2, :].",
    "phase_eta2_both_conjugate": "Conjugate K_Seta[:, phase_eta2] and K_etaS[phase_eta2, :].",
    "phase_eta2_kernel_sign_flip": "Multiply the phase_eta2 row and column of K_etaeta by -1.",
}


def apply_phase_eta2_transform(
    *,
    transform: str,
    k_seta: np.ndarray,
    k_etas: np.ndarray,
    k_etaeta: np.ndarray,
    phase_index: int = PHASE_ETA2_INDEX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Apply one debug-only phase_eta2 transform to matrix copies."""

    seta = np.array(k_seta, dtype=complex, copy=True)
    etas = np.array(k_etas, dtype=complex, copy=True)
    etaeta = np.array(k_etaeta, dtype=complex, copy=True)
    changed: list[str] = []

    if transform == "identity":
        return seta, etas, etaeta, changed
    if transform == "phase_eta2_seta_sign_flip":
        seta[:, phase_index] *= -1
        changed.append("K_Seta")
    elif transform == "phase_eta2_etas_sign_flip":
        etas[phase_index, :] *= -1
        changed.append("K_etaS")
    elif transform == "phase_eta2_both_sign_flip":
        seta[:, phase_index] *= -1
        etas[phase_index, :] *= -1
        changed.extend(["K_Seta", "K_etaS"])
    elif transform == "phase_eta2_seta_times_i":
        seta[:, phase_index] *= 1j
        changed.append("K_Seta")
    elif transform == "phase_eta2_seta_times_minus_i":
        seta[:, phase_index] *= -1j
        changed.append("K_Seta")
    elif transform == "phase_eta2_etas_times_i":
        etas[phase_index, :] *= 1j
        changed.append("K_etaS")
    elif transform == "phase_eta2_etas_times_minus_i":
        etas[phase_index, :] *= -1j
        changed.append("K_etaS")
    elif transform == "phase_eta2_seta_conjugate":
        seta[:, phase_index] = np.conj(seta[:, phase_index])
        changed.append("K_Seta")
    elif transform == "phase_eta2_etas_conjugate":
        etas[phase_index, :] = np.conj(etas[phase_index, :])
        changed.append("K_etaS")
    elif transform == "phase_eta2_both_conjugate":
        seta[:, phase_index] = np.conj(seta[:, phase_index])
        etas[phase_index, :] = np.conj(etas[phase_index, :])
        changed.extend(["K_Seta", "K_etaS"])
    elif transform == "phase_eta2_kernel_sign_flip":
        etaeta[phase_index, :] *= -1
        etaeta[:, phase_index] *= -1
        changed.append("K_etaeta")
    else:
        raise ValueError(f"unknown phase_eta2 transform: {transform}")
    return seta, etas, etaeta, changed


def phase_transform_result(
    *,
    transform: str,
    blocks: TargetBareBlocks,
    identity_diagnostics: dict[str, Any] | None = None,
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    """Return one debug-only transform result from averaged blocks."""

    seta, etas, etaeta, changed = apply_phase_eta2_transform(
        transform=transform,
        k_seta=blocks.k_seta,
        k_etas=blocks.k_etas,
        k_etaeta=blocks.k_etaeta,
    )
    x_action, schur = solve_collective_action(etaeta, etas)
    schur_correction = seta @ x_action
    k_eff = np.asarray(blocks.k_ss, dtype=complex) - schur_correction
    ratios = decomposition_ratios(k_eff, eps=ratio_eps, source_order=blocks.source_order)
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
    if identity_diagnostics is not None:
        diagnostics.update(
            {
                "delta_gauge_over_tm_abs_vs_identity": diagnostics["gauge_over_tm_abs"] - identity_diagnostics["gauge_over_tm_abs"],
                "delta_gauge_gg_over_tm_abs_vs_identity": diagnostics["gauge_gg_over_tm_abs"] - identity_diagnostics["gauge_gg_over_tm_abs"],
                "delta_gauge_row_norm_vs_identity": diagnostics["gauge_row_norm"] - identity_diagnostics["gauge_row_norm"],
                "delta_gauge_gg_norm_vs_identity": diagnostics["gauge_gg_norm"] - identity_diagnostics["gauge_gg_norm"],
            }
        )
    return {
        "transform": transform,
        "description": TRANSFORM_DESCRIPTIONS[transform],
        "changed_blocks": changed,
        "Schur_correction_entries": _entries(schur_correction, tuple(name for name, _, _ in ENTRY_SPECS), source_order=blocks.source_order),
        "K_eff_entries": _entries(k_eff, tuple(name for name, _, _ in ENTRY_SPECS), source_order=blocks.source_order),
        "diagnostics": diagnostics,
        "schur": schur,
        "valid_for_casimir_input": False,
    }


def phase_transform_results(
    *,
    blocks: TargetBareBlocks,
    transforms: Sequence[str] = DEFAULT_TRANSFORMS,
    ratio_eps: float = RATIO_EPS,
) -> list[dict[str, Any]]:
    """Return transform results in requested order."""

    requested = tuple(str(transform) for transform in transforms)
    identity = phase_transform_result(transform="identity", blocks=blocks, ratio_eps=ratio_eps)
    identity_diagnostics = identity["diagnostics"]
    results = []
    for transform in requested:
        if transform == "identity":
            identity["diagnostics"].update(
                {
                    "delta_gauge_over_tm_abs_vs_identity": 0.0,
                    "delta_gauge_gg_over_tm_abs_vs_identity": 0.0,
                    "delta_gauge_row_norm_vs_identity": 0.0,
                    "delta_gauge_gg_norm_vs_identity": 0.0,
                }
            )
            results.append(identity)
        else:
            results.append(phase_transform_result(transform=transform, blocks=blocks, identity_diagnostics=identity_diagnostics, ratio_eps=ratio_eps))
    return results


def phase_eta2_convention_payload(
    *,
    model_name: str,
    pairing_name: str,
    frequency: dict[str, Any],
    debug_parameters: dict[str, Any],
    collective_order: tuple[str, ...],
    raw_ansatz_channel_names: tuple[str, ...] | None,
    transform_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "valid_for_casimir_input": False,
            "reason": "phase_eta2_convention_debug_not_casimir_input",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "frequency": frequency,
        "debug_parameters": {
            **debug_parameters,
            "debug_only_phase_eta2_convention": True,
            "average_order": "average_blocks_then_schur",
            "valid_for_casimir_input": False,
        },
        "source_order_diagnostic": list(SOURCE_ORDER_DIAGNOSTIC),
        "collective_order": list(collective_order),
        "raw_ansatz_channel_names": list(raw_ansatz_channel_names) if raw_ansatz_channel_names is not None else None,
        "transform_results": list(transform_results),
        "valid_for_casimir_input": False,
    }


def run_phase_eta2_convention(
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
    transforms: Sequence[str] = DEFAULT_TRANSFORMS,
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    """Run q-along-x debug-only phase_eta2 convention diagnostics."""

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
    collective_order, raw_names = collective_order_from_ansatz(inputs.ansatz, response.bare_blocks.k_etaeta.shape[0])
    return phase_eta2_convention_payload(
        model_name=model_name,
        pairing_name=pairing_name,
        frequency=frequency_payload(matsubara_index, temperature_K),
        debug_parameters={
            "debug_only_phase_eta2_convention": True,
            "q_model_convention": "q_along_x_only",
            "q_value": float(q_value),
            "nk": int(nk),
            "shift_fractions": [float(value) for value in shift_fractions],
            "num_shifted_meshes": len(shifts),
            "contact_scale": float(contact_scale),
            "transforms": [str(transform) for transform in transforms],
            "ratio_eps": float(ratio_eps),
            "average_order": "average_blocks_then_schur",
            "shifted_mesh_average": _shifted_payload(shift_fractions, shifts),
            "valid_for_casimir_input": False,
        },
        collective_order=collective_order,
        raw_ansatz_channel_names=raw_names,
        transform_results=phase_transform_results(blocks=response.bare_blocks, transforms=transforms, ratio_eps=ratio_eps),
    )


def run_and_write_phase_eta2_convention(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_phase_eta2_convention(**kwargs)
    write_json(Path(output_dir) / "phase_eta2_convention.json", payload)
    return payload
