"""Debug-only signed complex matrix-entry decomposition."""

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

SCHEMA_VERSION = "finite_q_tmte_signed_decomposition_v1"
ENTRY_SPECS = (
    ("GG", "G", "G"),
    ("GTM", "G", "TM"),
    ("TMG", "TM", "G"),
    ("TMTM", "TM", "TM"),
    ("GTE", "G", "TE"),
    ("TEG", "TE", "G"),
)


def signed_entry(matrix: np.ndarray, row_label: str, col_label: str, source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC) -> complex:
    """Extract one signed complex entry by source labels."""

    require_diagnostic_source_order(source_order)
    array = np.asarray(matrix, dtype=complex)
    if array.shape != (3, 3):
        raise ValueError("signed decomposition matrix must have shape (3, 3)")
    return complex(array[source_order.index(row_label), source_order.index(col_label)])


def signed_entries(matrices: dict[str, np.ndarray], source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC) -> dict[str, dict[str, complex]]:
    """Extract all signed decomposition entries from named matrices."""

    return {
        entry_name: {
            matrix_name: signed_entry(matrix, row_label, col_label, source_order)
            for matrix_name, matrix in matrices.items()
        }
        for entry_name, row_label, col_label in ENTRY_SPECS
    }


def decomposition_ratios(k_eff: np.ndarray, *, eps: float = RATIO_EPS, source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC) -> dict[str, Any]:
    """Return gauge diagnostics and ratios from K_eff."""

    require_diagnostic_source_order(source_order)
    matrix = np.asarray(k_eff, dtype=complex)
    g = source_order.index("G")
    tm = source_order.index("TM")
    te = source_order.index("TE")
    gauge_row_norm = float(np.linalg.norm(matrix[g, :]))
    gauge_col_norm = float(np.linalg.norm(matrix[:, g]))
    gauge_gg_norm = float(abs(matrix[g, g]))
    physical_matrix_norm = float(np.linalg.norm(matrix[np.ix_([tm, te], [tm, te])]))
    tm_abs = float(abs(matrix[tm, tm]))
    denominator_eps = float(eps)
    return {
        "gauge_row_norm": gauge_row_norm,
        "gauge_col_norm": gauge_col_norm,
        "gauge_gg_norm": gauge_gg_norm,
        "physical_matrix_norm": physical_matrix_norm,
        "gauge_over_physical": gauge_row_norm / max(physical_matrix_norm, denominator_eps),
        "gauge_over_tm_abs": gauge_row_norm / max(tm_abs, denominator_eps),
        "gauge_gg_over_tm_abs": gauge_gg_norm / max(tm_abs, denominator_eps),
        "ratio_eps": denominator_eps,
        "valid_for_casimir_input": False,
    }


def signed_decomposition_from_blocks(
    *,
    blocks: TargetBareBlocks,
    contact_scale: float,
    shifted_payload: dict[str, Any],
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    """Build signed decomposition from averaged, contact-scaled bare blocks."""

    response = average_bare_blocks_then_schur([blocks])
    matrices = {
        "K_SS_bubble": response.bare_blocks.k_ss_bubble,
        "K_SS_contact_scaled": response.bare_blocks.k_ss_contact,
        "K_SS_scaled": response.bare_blocks.k_ss,
        "Schur_correction": response.schur.correction,
        "K_eff": response.schur.effective,
    }
    return {
        "entries": signed_entries(matrices, response.bare_blocks.source_order),
        "schur": {
            "solve_method": response.schur.solve_method,
            "etaeta_condition_number": response.schur.etaeta_condition_number,
            "condition_threshold": response.schur.condition_threshold,
            "numerically_suspect": response.schur.numerically_suspect,
            "valid_for_casimir_input": False,
        },
        "ratios": decomposition_ratios(response.schur.effective, eps=ratio_eps, source_order=response.bare_blocks.source_order),
        "q_model": response.bare_blocks.conventions.q,
        "q_norm": response.bare_blocks.conventions.q_norm,
        "contact_scale": float(contact_scale),
        "shifted_mesh_average": shifted_payload,
    }


def run_signed_decomposition(
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
    """Run q-along-x debug-only signed decomposition diagnostics."""

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
    scaled_blocks = []
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
    matrices = {
        "K_SS_bubble": response.bare_blocks.k_ss_bubble,
        "K_SS_contact_scaled": response.bare_blocks.k_ss_contact,
        "K_SS_scaled": response.bare_blocks.k_ss,
        "Schur_correction": response.schur.correction,
        "K_eff": response.schur.effective,
    }
    return signed_decomposition_payload(
        model_name=model_name,
        pairing_name=pairing_name,
        frequency=frequency_payload(matsubara_index, temperature_K),
        debug_parameters={
            "debug_only_signed_decomposition": True,
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
        entries=signed_entries(matrices, response.bare_blocks.source_order),
        schur={
            "solve_method": response.schur.solve_method,
            "etaeta_condition_number": response.schur.etaeta_condition_number,
            "condition_threshold": response.schur.condition_threshold,
            "numerically_suspect": response.schur.numerically_suspect,
            "valid_for_casimir_input": False,
        },
        ratios=decomposition_ratios(response.schur.effective, eps=ratio_eps, source_order=response.bare_blocks.source_order),
    )


def signed_decomposition_payload(
    *,
    model_name: str,
    pairing_name: str,
    frequency: dict[str, Any],
    debug_parameters: dict[str, Any],
    entries: dict[str, dict[str, complex]],
    schur: dict[str, Any],
    ratios: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "valid_for_casimir_input": False,
            "reason": "signed_decomposition_debug_not_casimir_input",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "frequency": frequency,
        "debug_parameters": {**debug_parameters, "debug_only_signed_decomposition": True, "valid_for_casimir_input": False},
        "source_order_diagnostic": list(SOURCE_ORDER_DIAGNOSTIC),
        "entries": entries,
        "schur": {**schur, "valid_for_casimir_input": False},
        "ratios": {**ratios, "valid_for_casimir_input": False},
        "valid_for_casimir_input": False,
    }


def run_and_write_signed_decomposition(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_signed_decomposition(**kwargs)
    write_json(Path(output_dir) / "signed_decomposition.json", payload)
    return payload
