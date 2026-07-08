"""Debug-only Ward-basis convention fingerprint scan.

This module intentionally does not define or accept a production convention.
It only re-expresses already-computed finite-q target blocks in primitive
(A0, L, T) variables and probes candidate linear target bases as diagnostic
fingerprints for a possible Matsubara time/space convention issue.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..adapters.bubble_adapter import TargetBareBlocks, compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .collective_schur_factors import collective_order_from_ansatz, solve_collective_action
from .contact_ablation import _shifted_payload, scaled_contact_blocks
from .extended_ward_kernel import PRIMITIVE_ORDER, complex_vector_payload, primitive_mixed_blocks, solve_left_collective_vector, solve_right_collective_vector
from .nk_sweep import RATIO_EPS
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .signed_decomposition import decomposition_ratios

SCHEMA_VERSION = "finite_q_tmte_ward_basis_convention_scan_v1"
SOURCE_ORDER = ("G", "TM", "TE")


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=complex)))


def _matrix_payload(matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str]) -> list[dict[str, Any]]:
    array = np.asarray(matrix, dtype=complex)
    if array.shape != (len(row_labels), len(col_labels)):
        raise ValueError("matrix shape does not match labels")
    return [
        {
            "row": str(row_label),
            "values": complex_vector_payload(array[row_index, :], col_labels),
        }
        for row_index, row_label in enumerate(row_labels)
    ]


def target_transform_matrix(name: str, *, xi_eV: float, q_norm: float) -> np.ndarray:
    """Return candidate target transform T mapping primitive [A0,L,T] to [G,TM,TE]."""

    xi = float(xi_eV)
    q = float(q_norm)
    candidates: dict[str, np.ndarray] = {
        "baseline_real": np.asarray([[xi, q, 0.0], [-q, xi, 0.0], [0.0, 0.0, 1.0]], dtype=complex),
        "temporal_i_plus": np.asarray([[1j * xi, q, 0.0], [-q, 1j * xi, 0.0], [0.0, 0.0, 1.0]], dtype=complex),
        "temporal_i_minus": np.asarray([[-1j * xi, q, 0.0], [-q, -1j * xi, 0.0], [0.0, 0.0, 1.0]], dtype=complex),
        "spatial_i_plus": np.asarray([[xi, 1j * q, 0.0], [-1j * q, xi, 0.0], [0.0, 0.0, 1.0]], dtype=complex),
        "spatial_i_minus": np.asarray([[xi, -1j * q, 0.0], [1j * q, xi, 0.0], [0.0, 0.0, 1.0]], dtype=complex),
        "all_i_plus": 1j * np.asarray([[xi, q, 0.0], [-q, xi, 0.0], [0.0, 0.0, -1j]], dtype=complex),
        "a0_i_only_plus": np.asarray([[1j * xi, q, 0.0], [-1j * q, xi, 0.0], [0.0, 0.0, 1.0]], dtype=complex),
        "a0_i_only_minus": np.asarray([[-1j * xi, q, 0.0], [1j * q, xi, 0.0], [0.0, 0.0, 1.0]], dtype=complex),
    }
    if name not in candidates:
        raise ValueError(f"unknown Ward basis candidate {name!r}")
    matrix = candidates[name]
    if abs(np.linalg.det(matrix)) <= 1e-30:
        raise ValueError(f"candidate transform {name!r} is singular")
    return matrix


def default_candidate_names() -> tuple[str, ...]:
    return (
        "baseline_real",
        "temporal_i_plus",
        "temporal_i_minus",
        "spatial_i_plus",
        "spatial_i_minus",
        "a0_i_only_plus",
        "a0_i_only_minus",
    )


def primitive_blocks_from_baseline(blocks: TargetBareBlocks) -> dict[str, Any]:
    """Invert baseline target blocks to primitive A0/L/T blocks using current real transform."""

    xi = float(blocks.conventions.g0)
    q = float(blocks.conventions.gL)
    baseline = target_transform_matrix("baseline_real", xi_eV=xi, q_norm=q)
    inv_baseline = np.linalg.inv(baseline)
    k_ss = inv_baseline @ np.asarray(blocks.k_ss, dtype=complex) @ inv_baseline.T
    k_ss_bubble = inv_baseline @ np.asarray(blocks.k_ss_bubble, dtype=complex) @ inv_baseline.T
    k_ss_contact = inv_baseline @ np.asarray(blocks.k_ss_contact, dtype=complex) @ inv_baseline.T
    k_seta = inv_baseline @ np.asarray(blocks.k_seta, dtype=complex)
    k_etas = np.asarray(blocks.k_etas, dtype=complex) @ inv_baseline.T
    return {
        "transform_baseline": baseline,
        "inverse_baseline": inv_baseline,
        "k_ss": k_ss,
        "k_ss_bubble": k_ss_bubble,
        "k_ss_contact": k_ss_contact,
        "k_seta": k_seta,
        "k_etas": k_etas,
        "k_etaeta_bubble": np.asarray(blocks.k_etaeta_bubble, dtype=complex),
        "k_etaeta_counterterm": np.asarray(blocks.k_etaeta_counterterm, dtype=complex),
        "k_etaeta": np.asarray(blocks.k_etaeta, dtype=complex),
        "metadata": {
            "primitive_order": list(PRIMITIVE_ORDER),
            "source_order_baseline": list(blocks.source_order),
            "baseline_transform_convention": "G=xi*A0+q*L; TM=-q*A0+xi*L; TE=T",
            "valid_for_casimir_input": False,
        },
    }


def target_blocks_from_primitive(base: TargetBareBlocks, primitive: dict[str, Any], transform: np.ndarray) -> TargetBareBlocks:
    """Apply a candidate transform to primitive blocks and return target-like blocks."""

    t = np.asarray(transform, dtype=complex)
    k_ss = t @ primitive["k_ss"] @ t.T
    k_ss_bubble = t @ primitive["k_ss_bubble"] @ t.T
    k_ss_contact = t @ primitive["k_ss_contact"] @ t.T
    k_seta = t @ primitive["k_seta"]
    k_etas = primitive["k_etas"] @ t.T
    return replace(
        base,
        k_ss_bubble=k_ss_bubble,
        k_ss_contact=k_ss_contact,
        k_ss=k_ss,
        k_seta=k_seta,
        k_etas=k_etas,
        metadata={**base.metadata, "ward_basis_convention_scan_candidate": True, "valid_for_casimir_input": False},
    )


def _ward_residuals(blocks: TargetBareBlocks, *, delta0_eV: float, collective_order: tuple[str, ...], ratio_eps: float = RATIO_EPS) -> dict[str, Any]:
    k_ss = np.asarray(blocks.k_ss, dtype=complex)
    k_seta = np.asarray(blocks.k_seta, dtype=complex)
    k_etas = np.asarray(blocks.k_etas, dtype=complex)
    k_etaeta = np.asarray(blocks.k_etaeta, dtype=complex)
    k_eff, _, schur = _schur_effective(blocks)
    ratios = decomposition_ratios(k_eff, eps=ratio_eps, source_order=blocks.source_order)
    g = blocks.source_order.index("G")
    phase_index = collective_order.index("phase_eta2") if "phase_eta2" in collective_order else len(collective_order) - 1
    zero = np.zeros(len(collective_order), dtype=complex)
    same_negative = np.zeros(len(collective_order), dtype=complex)
    same_negative[phase_index] = -2j * float(delta0_eV)
    fitted_left, fitted_left_meta = solve_left_collective_vector(k_etaeta, k_seta[g, :])
    fitted_right, fitted_right_meta = solve_right_collective_vector(k_etaeta, k_etas[:, g])

    def candidate_payload(name: str, w_left: np.ndarray, w_right: np.ndarray, solve: dict[str, Any] | None = None) -> dict[str, Any]:
        left_em = k_ss[g, :] + w_left @ k_etas
        left_eta = k_seta[g, :] + w_left @ k_etaeta
        right_em = k_ss[:, g] + k_seta @ w_right
        right_eta = k_etas[:, g] + k_etaeta @ w_right
        return {
            "candidate": name,
            "W_eta_left": complex_vector_payload(w_left, collective_order),
            "W_eta_right": complex_vector_payload(w_right, collective_order),
            "left_em_residual": complex_vector_payload(left_em, blocks.source_order),
            "right_em_residual": complex_vector_payload(right_em, blocks.source_order),
            "left_collective_residual": complex_vector_payload(left_eta, collective_order),
            "right_collective_residual": complex_vector_payload(right_eta, collective_order),
            "norms": {
                "left_em_norm": _norm(left_em),
                "right_em_norm": _norm(right_em),
                "left_collective_norm": _norm(left_eta),
                "right_collective_norm": _norm(right_eta),
                "left_total_extended_norm": float(np.sqrt(_norm(left_em) ** 2 + _norm(left_eta) ** 2)),
                "right_total_extended_norm": float(np.sqrt(_norm(right_em) ** 2 + _norm(right_eta) ** 2)),
                "valid_for_casimir_input": False,
            },
            "solve": {**(solve or {}), "valid_for_casimir_input": False},
            "valid_for_casimir_input": False,
        }

    return {
        "effective_metrics": {
            "K_eff_GG": complex(k_eff[g, g]),
            "K_eff_GTM": complex(k_eff[g, blocks.source_order.index("TM")]),
            "K_eff_TMG": complex(k_eff[blocks.source_order.index("TM"), g]),
            "K_eff_TMTM": complex(k_eff[blocks.source_order.index("TM"), blocks.source_order.index("TM")]),
            "gauge_row_norm": float(ratios["gauge_row_norm"]),
            "gauge_gg_norm": float(ratios["gauge_gg_norm"]),
            "physical_matrix_norm": float(ratios["physical_matrix_norm"]),
            "gauge_over_tm_abs": float(ratios["gauge_over_tm_abs"]),
            "gauge_gg_over_tm_abs": float(ratios["gauge_gg_over_tm_abs"]),
            "valid_for_casimir_input": False,
        },
        "extended_ward_candidates": [
            candidate_payload("zero_collective", zero, zero),
            candidate_payload("analytic_same_negative", same_negative, same_negative),
            candidate_payload("fitted_both_independent", fitted_left, fitted_right, {"left": fitted_left_meta, "right": fitted_right_meta}),
        ],
        "schur": schur,
        "valid_for_casimir_input": False,
    }


def _schur_effective(blocks: TargetBareBlocks) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    action, schur = solve_collective_action(np.asarray(blocks.k_etaeta, dtype=complex), np.asarray(blocks.k_etas, dtype=complex))
    correction = np.asarray(blocks.k_seta, dtype=complex) @ action
    return np.asarray(blocks.k_ss, dtype=complex) - correction, correction, schur


def phase_eta2_primitive_fingerprint(blocks: TargetBareBlocks, collective_order: tuple[str, ...]) -> dict[str, Any]:
    primitive_etas, primitive_seta, metadata = primitive_mixed_blocks(blocks)
    phase_index = collective_order.index("phase_eta2") if "phase_eta2" in collective_order else len(collective_order) - 1
    return {
        **metadata,
        "collective_order": list(collective_order),
        "left_K_etaS_phase_primitive": complex_vector_payload(primitive_etas[phase_index, :], PRIMITIVE_ORDER),
        "right_K_Seta_primitive_phase": complex_vector_payload(primitive_seta[:, phase_index], PRIMITIVE_ORDER),
        "valid_for_casimir_input": False,
    }


def scan_candidate(
    *,
    name: str,
    baseline_blocks: TargetBareBlocks,
    primitive: dict[str, Any],
    delta0_eV: float,
    collective_order: tuple[str, ...],
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    xi = float(baseline_blocks.conventions.g0)
    q = float(baseline_blocks.conventions.gL)
    transform = target_transform_matrix(name, xi_eV=xi, q_norm=q)
    transformed_blocks = target_blocks_from_primitive(baseline_blocks, primitive, transform)
    residuals = _ward_residuals(transformed_blocks, delta0_eV=delta0_eV, collective_order=collective_order, ratio_eps=ratio_eps)
    return {
        "basis_candidate": name,
        "status": {
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "requires_analytic_derivation_before_convention_change": True,
            "valid_for_casimir_input": False,
        },
        "transform_matrix_primitive_to_target": _matrix_payload(transform, SOURCE_ORDER, PRIMITIVE_ORDER),
        "transform_determinant": complex(np.linalg.det(transform)),
        "phase_eta2_primitive_fingerprint": phase_eta2_primitive_fingerprint(transformed_blocks, collective_order),
        **residuals,
        "valid_for_casimir_input": False,
    }


def ward_basis_convention_scan_payload(
    *,
    model_name: str,
    pairing_name: str,
    frequency: dict[str, Any],
    debug_parameters: dict[str, Any],
    collective_order: tuple[str, ...],
    raw_ansatz_channel_names: tuple[str, ...] | None,
    primitive_metadata: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "requires_analytic_derivation_before_convention_change": True,
            "valid_for_casimir_input": False,
            "reason": "ward_basis_convention_fingerprint_scan_not_production_convention",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "frequency": frequency,
        "debug_parameters": {
            **debug_parameters,
            "scan_purpose": "fingerprint_possible_matsubara_A0_L_relative_i_convention",
            "not_a_residual_minimization_fix": True,
            "valid_for_casimir_input": False,
        },
        "collective_order": list(collective_order),
        "raw_ansatz_channel_names": list(raw_ansatz_channel_names) if raw_ansatz_channel_names is not None else None,
        "primitive_metadata": primitive_metadata,
        "basis_candidates": list(candidates),
        "valid_for_casimir_input": False,
    }


def run_ward_basis_convention_scan(
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
    candidate_names: Sequence[str] = default_candidate_names(),
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
    response = average_bare_blocks_then_schur(scaled_blocks)
    baseline = response.bare_blocks
    collective_order, raw_names = collective_order_from_ansatz(inputs.ansatz, baseline.k_etaeta.shape[0])
    primitive = primitive_blocks_from_baseline(baseline)
    delta0 = float(getattr(inputs.pairing_params, "delta0_eV", 0.0))
    candidates = [
        scan_candidate(
            name=name,
            baseline_blocks=baseline,
            primitive=primitive,
            delta0_eV=delta0,
            collective_order=collective_order,
            ratio_eps=ratio_eps,
        )
        for name in candidate_names
    ]
    return ward_basis_convention_scan_payload(
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
            "candidate_names": list(candidate_names),
            "shifted_mesh_average": _shifted_payload(shift_fractions, shifts),
            "valid_for_casimir_input": False,
        },
        collective_order=collective_order,
        raw_ansatz_channel_names=raw_names,
        primitive_metadata=primitive["metadata"],
        candidates=candidates,
    )


def run_and_write_ward_basis_convention_scan(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_ward_basis_convention_scan(**kwargs)
    write_json(Path(output_dir) / "ward_basis_convention_scan.json", payload)
    return payload
