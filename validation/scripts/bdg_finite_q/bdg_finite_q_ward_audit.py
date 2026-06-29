#!/usr/bin/env python3
"""Compact superconducting BdG finite-q Ward residual audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from lno327.conductivity import KuboConfig, k_weights  # noqa: E402
from lno327.finite_q_engine import FiniteQEngineOptions, finite_q_bdg_response_from_ansatz  # noqa: E402
from lno327.pairing import PairingAmplitudes, build_pairing_ansatz  # noqa: E402
from lno327.ward_response import normal_physical_density_current_response_components_imag_axis  # noqa: E402
from normal_finite_q_ward_audit import (  # noqa: E402
    DIRECTION_VECTORS,
    _print_progress,
    actual_twist_offsets,
    uniform_bz_mesh_twisted,
)
from lno327.finite_q_primitives import density_vertex, phase_vertex  # noqa: E402
from lno327.pairing import bdg_hamiltonian  # noqa: E402

WARD_COMPONENT_LABELS = ("density", "current_x", "current_y")
MAX_JSON_SIZE_MB = 50.0


def _physical_ward_residuals(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    response = np.asarray(matrix, dtype=complex)
    qx, qy = float(q[0]), float(q[1])
    left = 1j * omega_eV * response[0, :] + qx * response[1, :] + qy * response[2, :]
    right = 1j * omega_eV * response[:, 0] - qx * response[:, 1] - qy * response[:, 2]
    return left, right


def _nested_twist_mesh(nk: int, actual_twist_count: int, twist_mode: str) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    offsets = actual_twist_offsets(actual_twist_count, twist_mode)
    meshes = [uniform_bz_mesh_twisted(nk, offset) for offset in offsets]
    points = np.vstack(meshes)
    weights = k_weights(points)
    weight_sum = float(np.sum(weights))
    if abs(weight_sum - 1.0) >= 1e-12:
        raise ValueError(f"twist quadrature weights sum to {weight_sum}, not 1")
    return points, weights, [[float(x), float(y)] for x, y in offsets]


def _norm(value: np.ndarray | complex) -> float:
    return float(np.linalg.norm(value))


def _response_norms(matrix: np.ndarray) -> dict[str, float]:
    response = np.asarray(matrix, dtype=complex)
    return {
        "total_response_norm": _norm(response),
        "density_density_norm": float(abs(response[0, 0])),
        "density_current_block_norm": _norm(response[0:1, 1:3]),
        "current_density_block_norm": _norm(response[1:3, 0:1]),
        "current_current_block_norm": _norm(response[1:3, 1:3]),
    }


def _ward_norms(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, float]:
    left, right = _physical_ward_residuals(matrix, omega_eV, q)
    q_norm = float(np.linalg.norm(q))
    return {
        "left_ward_residual_norm": _norm(left),
        "right_ward_residual_norm": _norm(right),
        "left_ward_residual_over_q_norm": float(_norm(left) / q_norm),
        "right_ward_residual_over_q_norm": float(_norm(right) / q_norm),
        "density_column_residual_norm": float(abs(right[0])),
        "current_x_column_residual_norm": float(abs(right[1])),
        "current_y_column_residual_norm": float(abs(right[2])),
        "density_row_residual_norm": float(abs(left[0])),
        "current_x_row_residual_norm": float(abs(left[1])),
        "current_y_row_residual_norm": float(abs(left[2])),
    }


def _normal_like_em_ward_diagnostic(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    values = _ward_norms(matrix, omega_eV, q)
    return {
        "normal_like_em_left_residual_norm": values["left_ward_residual_norm"],
        "normal_like_em_right_residual_norm": values["right_ward_residual_norm"],
        "normal_like_em_left_residual_over_q_norm": values["left_ward_residual_over_q_norm"],
        "normal_like_em_right_residual_over_q_norm": values["right_ward_residual_over_q_norm"],
        "normal_like_em_residual_interpretation": (
            "bare_em_block_only_not_expected_to_close_in_superconducting_state; "
            "normal-state electromagnetic Ward contraction is diagnostic-only for BdG"
        ),
    }


def _complex_scalar(value: complex) -> dict[str, float]:
    z = complex(value)
    return {"real": float(np.real(z)), "imag": float(np.imag(z)), "abs": float(abs(z))}


def _complex_vector(value: np.ndarray, labels: tuple[str, ...]) -> list[dict[str, float | str]]:
    vector = np.asarray(value, dtype=complex).reshape(-1)
    if vector.shape[0] != len(labels):
        raise ValueError("complex vector label count does not match vector length")
    return [{"label": label, **_complex_scalar(component)} for label, component in zip(labels, vector, strict=True)]


def _em_ward_vectors(omega_eV: float, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qx, qy = float(q[0]), float(q[1])
    left = np.asarray([1j * float(omega_eV), qx, qy], dtype=complex)
    right = np.asarray([1j * float(omega_eV), -qx, -qy], dtype=complex)
    return left, right


def _collective_channel_count(response: Any, variant: str) -> int:
    if variant == "phase_schur":
        return 1
    if variant == "amplitude_phase_schur":
        return int(np.asarray(response.collective_total).shape[0])
    return 0


def _collective_blocks(response: Any, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], str] | None:
    if variant == "phase_schur":
        left = np.asarray(response.phase_coupling_left, dtype=complex).reshape(3, 1)
        right = np.asarray(response.phase_coupling_right, dtype=complex).reshape(1, 3)
        kernel = np.asarray([[response.phase_phase_total]], dtype=complex)
        return left, right, kernel, ("theta",), "phase_only"
    elif variant == "amplitude_phase_schur":
        left = np.asarray(response.em_collective_left, dtype=complex)
        right = np.asarray(response.collective_em_right, dtype=complex)
        kernel = np.asarray(response.collective_total, dtype=complex)
        return left, right, kernel, ("amplitude_eta1", "phase_eta2"), "amplitude_phase"
    return None


def _mixed_norms(response: Any, variant: str) -> tuple[float, float, float]:
    blocks = _collective_blocks(response, variant)
    if blocks is None:
        left = np.zeros((3, 0), dtype=complex)
        right = np.zeros((0, 3), dtype=complex)
        kernel = np.zeros((0, 0), dtype=complex)
    else:
        left, right, kernel, _, _ = blocks
    return _norm(left), _norm(right), _norm(kernel)


def _order_parameter_gauge_vector_candidates(
    pairing_name: str,
    collective_mode: str,
    delta0_eV: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if collective_mode == "phase_only":
        labels = ("theta",)
        legacy = np.asarray([2.0 + 0.0j], dtype=complex)
        analytic_left = np.asarray([2.0j], dtype=complex)
        analytic_right = np.asarray([-2.0j], dtype=complex)
        metadata = {
            "collective_basis_labels": list(labels),
            "order_parameter_phase_normalization": "dimensionless theta; phase vertex is dH_BdG/dtheta = delta0 * dH_BdG/deta2",
            "order_parameter_gauge_vector_rule": "analytic BdG phase-vertex convention gives W_theta_left=+2j and W_theta_right=-2j",
            "phase_tangent_source": (
                f"{pairing_name} PairingAnsatz.phase_pairing_matrix with bond_endpoint_gauge form factor"
            ),
            "amplitude_gauge_weight": None,
            "dwave_form_factor_tangent_used": pairing_name == "dwave",
        }
    elif collective_mode == "amplitude_phase":
        labels = ("amplitude_eta1", "phase_eta2")
        legacy = np.asarray([0.0 + 0.0j, 2.0 * float(delta0_eV) + 0.0j], dtype=complex)
        analytic_left = np.asarray([0.0 + 0.0j, 2.0j * float(delta0_eV)], dtype=complex)
        analytic_right = np.asarray([0.0 + 0.0j, -2.0j * float(delta0_eV)], dtype=complex)
        metadata = {
            "collective_basis_labels": list(labels),
            "order_parameter_phase_normalization": "energy-like eta2 coordinate with eta2 = delta0 * theta",
            "order_parameter_gauge_vector_rule": "analytic BdG eta2 convention gives W_eta2_left=+2j*delta0 and W_eta2_right=-2j*delta0",
            "phase_tangent_source": f"{pairing_name} PairingAnsatz.collective_vertices()[1] using bond_endpoint_gauge form factor",
            "amplitude_gauge_weight": 0.0,
            "dwave_form_factor_tangent_used": pairing_name == "dwave",
        }
    else:
        raise ValueError(f"unsupported collective_mode for W_eta: {collective_mode}")
    candidates = {
        "real_same_sign_legacy": {
            "convention_name": "real_same_sign_legacy",
            "derived_from": "previous diagnostic convention retained only for comparison; not selected by residual size",
            "left": legacy,
            "right": legacy,
        },
        "imaginary_left_right_opposite": {
            "convention_name": "imaginary_left_right_opposite",
            "derived_from": "BdG phase vertex Gamma_theta=[[0,iDelta],[-iDelta^dagger,0]] and eta2=delta0*theta",
            "left": analytic_left,
            "right": analytic_right,
        },
    }
    return candidates, metadata


def _solve_kernel(kernel: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.linalg.solve(np.asarray(kernel, dtype=complex), np.asarray(rhs, dtype=complex))


def _schur_from_blocks(
    bare: np.ndarray,
    k_aeta: np.ndarray,
    k_etaeta: np.ndarray,
    k_etaa: np.ndarray,
) -> np.ndarray:
    return np.asarray(bare, dtype=complex) - np.asarray(k_aeta, dtype=complex) @ _solve_kernel(k_etaeta, k_etaa)


def _evaluate_extended_candidate(
    *,
    bare: np.ndarray,
    reported: np.ndarray,
    k_aeta: np.ndarray,
    k_etaa: np.ndarray,
    k_etaeta: np.ndarray,
    wa_left: np.ndarray,
    wa_right: np.ndarray,
    weta_left: np.ndarray,
    weta_right: np.ndarray,
    labels: tuple[str, ...],
    omega_eV: float,
    q: np.ndarray,
    convention_name: str,
    derived_from: str,
) -> dict[str, Any]:
    schur = _schur_from_blocks(bare, k_aeta, k_etaeta, k_etaa)
    schur_left, schur_right = _physical_ward_residuals(schur, omega_eV, q)
    extended_left_em = wa_left @ bare + weta_left @ k_etaa
    extended_left_collective = wa_left @ k_aeta + weta_left @ k_etaeta
    extended_right_em = bare @ wa_right + k_aeta @ weta_right
    extended_right_collective = k_etaa @ wa_right + k_etaeta @ weta_right
    left_total = np.concatenate([extended_left_em.reshape(-1), extended_left_collective.reshape(-1)])
    right_total = np.concatenate([extended_right_em.reshape(-1), extended_right_collective.reshape(-1)])
    left_schur_from_extended = extended_left_em - extended_left_collective @ _solve_kernel(k_etaeta, k_etaa)
    right_schur_from_extended = extended_right_em - k_aeta @ _solve_kernel(k_etaeta, extended_right_collective)
    return {
        "convention_name": convention_name,
        "derived_from": derived_from,
        "left_vector": _complex_vector(weta_left, labels),
        "right_vector": _complex_vector(weta_right, labels),
        "extended_left_em_equation_norm": _norm(extended_left_em),
        "extended_left_collective_equation_norm": _norm(extended_left_collective),
        "extended_left_total_residual_norm": _norm(left_total),
        "extended_right_em_equation_norm": _norm(extended_right_em),
        "extended_right_collective_equation_norm": _norm(extended_right_collective),
        "extended_right_total_residual_norm": _norm(right_total),
        "schur_from_blocks_left_residual_norm": _norm(schur_left),
        "schur_from_blocks_right_residual_norm": _norm(schur_right),
        "schur_from_blocks_left_residual_over_q_norm": float(_norm(schur_left) / float(np.linalg.norm(q))),
        "schur_from_blocks_right_residual_over_q_norm": float(_norm(schur_right) / float(np.linalg.norm(q))),
        "schur_from_blocks_response_norm": _norm(schur),
        "schur_from_blocks_minus_reported_response_norm": _norm(schur - reported),
        "extended_to_schur_left_consistency_norm": _norm(schur_left - left_schur_from_extended),
        "extended_to_schur_right_consistency_norm": _norm(schur_right - right_schur_from_extended),
    }


def _offdiagonal_bdg_block(matrix: np.ndarray) -> np.ndarray:
    value = np.zeros_like(np.asarray(matrix, dtype=complex))
    value[:4, 4:] = matrix[:4, 4:]
    value[4:, :4] = matrix[4:, :4]
    return value


def _operator_level_weta_audit_row(
    pairing_name: str,
    q_direction_name: str,
    q: np.ndarray,
    delta0_eV: float,
) -> dict[str, Any]:
    representative_k = np.asarray([0.37, -0.23], dtype=float)
    kx, ky = float(representative_k[0]), float(representative_k[1])
    qx, qy = float(q[0]), float(q[1])
    amp = PairingAmplitudes(delta0_eV=float(delta0_eV))
    ansatz = build_pairing_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    delta_theta = ansatz.phase_pairing_matrix(kx, ky, qx, qy, amp)
    phi = ansatz.collective_form_factor(kx, ky, qx, qy, amp)
    gamma_theta = phase_vertex(delta_theta)
    gamma_eta2 = phase_vertex(phi)

    theta_eps = 1e-6
    h_theta_plus = bdg_hamiltonian(kx, ky, delta_theta * np.exp(1j * theta_eps))
    h_theta_minus = bdg_hamiltonian(kx, ky, delta_theta * np.exp(-1j * theta_eps))
    gamma_theta_fd = (h_theta_plus - h_theta_minus) / (2.0 * theta_eps)
    eta_eps = 1e-6 * max(1.0, abs(float(delta0_eV)))
    h_eta_plus = bdg_hamiltonian(kx, ky, delta_theta + 1j * eta_eps * phi)
    h_eta_minus = bdg_hamiltonian(kx, ky, delta_theta - 1j * eta_eps * phi)
    gamma_eta2_fd = (h_eta_plus - h_eta_minus) / (2.0 * eta_eps)

    rho = density_vertex()
    h_mid = bdg_hamiltonian(kx, ky, delta_theta)
    left_pairing_anomalous = _offdiagonal_bdg_block(rho @ h_mid - h_mid @ rho)
    right_pairing_anomalous = _offdiagonal_bdg_block(h_mid @ rho - rho @ h_mid)
    theta_left_residual = left_pairing_anomalous + 2.0j * gamma_theta
    theta_right_residual = right_pairing_anomalous - 2.0j * gamma_theta
    eta2_left_residual = left_pairing_anomalous + (2.0j * float(delta0_eV)) * gamma_eta2
    eta2_right_residual = right_pairing_anomalous - (2.0j * float(delta0_eV)) * gamma_eta2
    legacy_left_residual = left_pairing_anomalous + 2.0 * gamma_theta
    legacy_right_residual = right_pairing_anomalous + 2.0 * gamma_theta

    theta_norm = max(_norm(gamma_theta), 1e-300)
    eta2_norm = max(_norm(gamma_eta2), 1e-300)
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "pairing_name": str(pairing_name),
        "q_direction": str(q_direction_name),
        "q_norm": float(np.linalg.norm(q)),
        "representative_k": [float(kx), float(ky)],
        "phase_vertex_finite_difference_error_norm": _norm(gamma_theta_fd - gamma_theta),
        "phase_vertex_finite_difference_relative_error": float(_norm(gamma_theta_fd - gamma_theta) / theta_norm),
        "eta2_vertex_finite_difference_error_norm": _norm(gamma_eta2_fd - gamma_eta2),
        "eta2_vertex_finite_difference_relative_error": float(_norm(gamma_eta2_fd - gamma_eta2) / eta2_norm),
        "left_anomalous_weta_operator_residual_norm": _norm(theta_left_residual),
        "right_anomalous_weta_operator_residual_norm": _norm(theta_right_residual),
        "left_anomalous_eta2_weta_operator_residual_norm": _norm(eta2_left_residual),
        "right_anomalous_eta2_weta_operator_residual_norm": _norm(eta2_right_residual),
        "legacy_real_weta_left_operator_residual_norm": _norm(legacy_left_residual),
        "legacy_real_weta_right_operator_residual_norm": _norm(legacy_right_residual),
        "operator_audit_interpretation": (
            "representative-k operator check only; verifies phase/eta2 tangent convention and anomalous "
            "block sign without fitting W_eta or modifying response"
        ),
    }


def _empty_extended_ward_diagnostic(response: Any, variant: str, matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    gi_left, gi_right = _physical_ward_residuals(matrix, omega_eV, q)
    q_norm = float(np.linalg.norm(q))
    mixed_left_norm, mixed_right_norm, kernel_norm = _mixed_norms(response, variant)
    if variant == "bare_bdg":
        unsupported_reason = "bare electromagnetic block alone is not a closed superconducting BdG Ward object"
    else:
        unsupported_reason = "cannot determine W_eta normalization from collective vertex convention"
    return {
        "extended_ward_formula": "K_ext=[[K_AA,K_Aeta],[K_etaA,K_etaeta]] with W_A and W_eta contractions",
        "em_ward_left_convention": "iomega*K[0,:] + qx*K[1,:] + qy*K[2,:]",
        "em_ward_right_convention": "iomega*K[:,0] - qx*K[:,1] - qy*K[:,2]",
        "collective_basis_labels": [],
        "order_parameter_gauge_vector_left": [],
        "order_parameter_gauge_vector_right": [],
        "order_parameter_phase_normalization": "not_applicable",
        "order_parameter_gauge_vector_rule": "not_applicable",
        "order_parameter_gauge_vector_convention": "not_applicable",
        "order_parameter_gauge_vector_fitted": False,
        "old_real_weta_kept_for_diagnostic": False,
        "main_weta_convention": "not_applicable",
        "legacy_real_weta_available_for_comparison": False,
        "phase_tangent_source": "not_applicable",
        "amplitude_gauge_weight": None,
        "extended_left_residual_norm": None,
        "extended_right_residual_norm": None,
        "extended_left_residual_over_q_norm": None,
        "extended_right_residual_over_q_norm": None,
        "extended_left_em_equation_norm": None,
        "extended_left_collective_equation_norm": None,
        "extended_left_total_residual_norm": None,
        "extended_right_em_equation_norm": None,
        "extended_right_collective_equation_norm": None,
        "extended_right_total_residual_norm": None,
        "em_collective_mixed_left_norm": mixed_left_norm,
        "em_collective_mixed_right_norm": mixed_right_norm,
        "collective_kernel_ward_norm": kernel_norm,
        "schur_gauge_invariant_left_residual_norm": _norm(gi_left),
        "schur_gauge_invariant_right_residual_norm": _norm(gi_right),
        "schur_gauge_invariant_left_residual_over_q_norm": float(_norm(gi_left) / q_norm),
        "schur_gauge_invariant_right_residual_over_q_norm": float(_norm(gi_right) / q_norm),
        "schur_from_blocks_left_residual_norm": None,
        "schur_from_blocks_right_residual_norm": None,
        "schur_from_blocks_left_residual_over_q_norm": None,
        "schur_from_blocks_right_residual_over_q_norm": None,
        "schur_from_blocks_response_norm": None,
        "schur_from_blocks_minus_reported_response_norm": None,
        "extended_to_schur_left_consistency_norm": None,
        "extended_to_schur_right_consistency_norm": None,
        "extended_to_schur_consistency_interpretation": "not_applicable_without_collective_block",
        "weta_convention_comparison": [],
        "extended_ward_implemented": False,
        "order_parameter_tangent_available": False,
        "order_parameter_tangent_rule": "not_applicable_for_bare_em_block",
        "phase_tangent_included": False,
        "amplitude_tangent_included": False,
        "dwave_form_factor_tangent_used": False,
        "unsupported_reason": unsupported_reason,
    }


def _extended_ward_diagnostic(
    response: Any,
    variant: str,
    matrix: np.ndarray,
    omega_eV: float,
    q: np.ndarray,
    pairing_name: str,
    delta0_eV: float,
) -> dict[str, Any]:
    if variant == "bare_bdg":
        return _empty_extended_ward_diagnostic(response, variant, matrix, omega_eV, q)

    blocks = _collective_blocks(response, variant)
    if blocks is None:
        return _empty_extended_ward_diagnostic(response, variant, matrix, omega_eV, q)

    k_aeta, k_etaa, k_etaeta, labels, collective_mode = blocks
    q_norm = float(np.linalg.norm(q))
    wa_left, wa_right = _em_ward_vectors(omega_eV, q)
    weta_candidates, weta_metadata = _order_parameter_gauge_vector_candidates(pairing_name, collective_mode, delta0_eV)
    bare = np.asarray(response.bare_total, dtype=complex)
    reported = np.asarray(matrix, dtype=complex)

    try:
        comparison = [
            _evaluate_extended_candidate(
                bare=bare,
                reported=reported,
                k_aeta=k_aeta,
                k_etaa=k_etaa,
                k_etaeta=k_etaeta,
                wa_left=wa_left,
                wa_right=wa_right,
                weta_left=np.asarray(candidate["left"], dtype=complex),
                weta_right=np.asarray(candidate["right"], dtype=complex),
                labels=labels,
                omega_eV=omega_eV,
                q=q,
                convention_name=str(candidate["convention_name"]),
                derived_from=str(candidate["derived_from"]),
            )
            for candidate in weta_candidates.values()
        ]
        main = next(row for row in comparison if row["convention_name"] == "imaginary_left_right_opposite")
        implementation_error: str | None = None
        implemented = True
    except np.linalg.LinAlgError as exc:
        empty = _empty_extended_ward_diagnostic(response, variant, matrix, omega_eV, q)
        empty.update(
            {
                "collective_basis_labels": list(labels),
                "order_parameter_tangent_available": True,
                "unsupported_reason": f"collective kernel solve failed: {exc}",
            }
        )
        return empty

    schur_reported_left, schur_reported_right = _physical_ward_residuals(reported, omega_eV, q)
    collective_kernel_ward_norm = max(
        float(main["extended_left_collective_equation_norm"]),
        float(main["extended_right_collective_equation_norm"]),
    )
    left_total_norm = float(main["extended_left_total_residual_norm"])
    right_total_norm = float(main["extended_right_total_residual_norm"])
    return {
        "extended_ward_formula": (
            "left: W_A^L K_AA + W_eta^L K_etaA, W_A^L K_Aeta + W_eta^L K_etaeta; "
            "right: K_AA W_A^R + K_Aeta W_eta^R, K_etaA W_A^R + K_etaeta W_eta^R"
        ),
        "em_ward_left_convention": "iomega*K[0,:] + qx*K[1,:] + qy*K[2,:]",
        "em_ward_right_convention": "iomega*K[:,0] - qx*K[:,1] - qy*K[:,2]",
        "collective_basis_labels": list(labels),
        "order_parameter_gauge_vector_left": main["left_vector"],
        "order_parameter_gauge_vector_right": main["right_vector"],
        "order_parameter_phase_normalization": weta_metadata["order_parameter_phase_normalization"],
        "order_parameter_gauge_vector_rule": weta_metadata["order_parameter_gauge_vector_rule"],
        "order_parameter_gauge_vector_convention": "analytic_imaginary_left_right_opposite",
        "order_parameter_gauge_vector_fitted": False,
        "old_real_weta_kept_for_diagnostic": True,
        "main_weta_convention": "imaginary_left_right_opposite",
        "legacy_real_weta_available_for_comparison": True,
        "phase_tangent_source": weta_metadata["phase_tangent_source"],
        "amplitude_gauge_weight": weta_metadata["amplitude_gauge_weight"],
        "extended_left_em_equation_norm": main["extended_left_em_equation_norm"],
        "extended_left_collective_equation_norm": main["extended_left_collective_equation_norm"],
        "extended_left_total_residual_norm": left_total_norm,
        "extended_right_em_equation_norm": main["extended_right_em_equation_norm"],
        "extended_right_collective_equation_norm": main["extended_right_collective_equation_norm"],
        "extended_right_total_residual_norm": right_total_norm,
        "extended_left_residual_norm": left_total_norm,
        "extended_right_residual_norm": right_total_norm,
        "extended_left_residual_over_q_norm": float(left_total_norm / q_norm),
        "extended_right_residual_over_q_norm": float(right_total_norm / q_norm),
        "em_collective_mixed_left_norm": _norm(k_aeta),
        "em_collective_mixed_right_norm": _norm(k_etaa),
        "collective_kernel_ward_norm": collective_kernel_ward_norm,
        "schur_gauge_invariant_left_residual_norm": _norm(schur_reported_left),
        "schur_gauge_invariant_right_residual_norm": _norm(schur_reported_right),
        "schur_gauge_invariant_left_residual_over_q_norm": float(_norm(schur_reported_left) / q_norm),
        "schur_gauge_invariant_right_residual_over_q_norm": float(_norm(schur_reported_right) / q_norm),
        "schur_from_blocks_left_residual_norm": main["schur_from_blocks_left_residual_norm"],
        "schur_from_blocks_right_residual_norm": main["schur_from_blocks_right_residual_norm"],
        "schur_from_blocks_left_residual_over_q_norm": main["schur_from_blocks_left_residual_over_q_norm"],
        "schur_from_blocks_right_residual_over_q_norm": main["schur_from_blocks_right_residual_over_q_norm"],
        "schur_from_blocks_response_norm": main["schur_from_blocks_response_norm"],
        "schur_from_blocks_minus_reported_response_norm": main["schur_from_blocks_minus_reported_response_norm"],
        "extended_to_schur_left_consistency_norm": main["extended_to_schur_left_consistency_norm"],
        "extended_to_schur_right_consistency_norm": main["extended_to_schur_right_consistency_norm"],
        "extended_to_schur_consistency_interpretation": (
            "Schur algebra consistency only; it is insensitive to some common W_eta convention errors because "
            "W_eta terms cancel algebraically between EM and collective equations. Physical W_eta is checked by "
            "operator-level tangent audit and extended residuals."
        ),
        "weta_convention_comparison": comparison,
        "extended_ward_implemented": implemented,
        "order_parameter_tangent_available": True,
        "order_parameter_tangent_rule": (
            "W_eta is derived from global U(1) transformation of the PairingAnsatz collective phase tangent; "
            "it is not fit from Ward residuals"
        ),
        "phase_tangent_included": True,
        "amplitude_tangent_included": collective_mode == "amplitude_phase",
        "dwave_form_factor_tangent_used": weta_metadata["dwave_form_factor_tangent_used"],
        "unsupported_reason": implementation_error,
    }


def _collective_metadata(response: Any, variant: str, matrix: np.ndarray) -> dict[str, Any]:
    metadata = response.metadata
    if variant == "bare_bdg":
        correction = np.zeros_like(matrix)
        return {
            "collective_mode": "none",
            "schur_applied": False,
            "schur_sign_convention": "not_applied",
            "schur_denominator_norm": 0.0,
            "schur_condition_estimate": None,
            "collective_block_norm": _norm(response.collective_total),
            "collective_correction_norm": _norm(correction),
        }
    if variant == "phase_schur":
        correction = response.bare_total - response.minus_schur
        return {
            "collective_mode": "phase_only",
            "schur_applied": metadata.get("phase_only_schur_status") not in {None, "skipped_zero_phase_kernel"},
            "schur_sign_convention": "minus",
            "schur_denominator_norm": float(abs(response.phase_phase_total)),
            "schur_condition_estimate": None,
            "collective_block_norm": float(abs(response.phase_phase_total)),
            "collective_correction_norm": _norm(correction),
        }
    correction = response.bare_total - response.amplitude_phase_schur
    return {
        "collective_mode": "amplitude_phase",
        "schur_applied": str(metadata.get("amplitude_phase_schur_status")) not in {"not_used", "skipped"},
        "schur_sign_convention": "minus_matrix_schur",
        "schur_denominator_norm": _norm(response.collective_total),
        "schur_condition_estimate": metadata.get("collective_total_condition_number"),
        "collective_block_norm": _norm(response.collective_total),
        "collective_correction_norm": _norm(correction),
    }


def _variant_matrix(response: Any, variant: str) -> tuple[np.ndarray | None, bool, str | None]:
    if variant == "bare_bdg":
        return response.bare_total, True, None
    if variant == "phase_schur":
        return response.minus_schur, True, None
    if variant == "amplitude_phase_schur":
        if response.amplitude_phase_schur is None:
            return None, False, "amplitude_phase_schur_missing"
        return response.amplitude_phase_schur, True, None
    return None, False, f"unsupported response variant: {variant}"


def _case_worker(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        pairing_name,
        nk,
        actual_twist_count,
        twist_mode,
        q_direction_name,
        q_value,
        direction,
        response_variants,
        omega_eV,
        temperature_K,
        eta_eV,
        delta0_eV,
    ) = args
    started = time.perf_counter()
    q = float(q_value) * np.asarray(direction, dtype=float) / float(np.linalg.norm(direction))
    points, weights, offsets = _nested_twist_mesh(int(nk), int(actual_twist_count), str(twist_mode))
    config = KuboConfig.from_kelvin(
        omega_eV=float(omega_eV),
        temperature_K=float(temperature_K),
        eta_eV=float(eta_eV),
        output_si=False,
    )
    ansatz = build_pairing_ansatz(str(pairing_name), phase_vertex="bond_endpoint_gauge")
    pairing_params = PairingAmplitudes(delta0_eV=float(delta0_eV))
    options = FiniteQEngineOptions(
        current_vertex="peierls",
        collective_mode="amplitude_phase",
        collective_counterterm="goldstone_gap_equation",
        include_phase_phase_direct=True,
    )
    response = finite_q_bdg_response_from_ansatz(
        ansatz,
        float(omega_eV),
        q,
        points,
        weights,
        config,
        pairing_params,
        options,
    )
    operator_audit_row = _operator_level_weta_audit_row(str(pairing_name), str(q_direction_name), q, float(delta0_eV))
    rows: list[dict[str, Any]] = []
    matrices_by_variant: dict[str, np.ndarray] = {}
    for variant in response_variants:
        matrix, supported, reason = _variant_matrix(response, str(variant))
        if matrix is None:
            rows.append(
                {
                    "pairing_name": str(pairing_name),
                    "response_variant": str(variant),
                    "supported": False,
                    "unsupported_reason": reason,
                    "valid_for_casimir_input": False,
                }
            )
            continue
        matrices_by_variant[str(variant)] = matrix
        row = {
            "pairing_name": str(pairing_name),
            "delta0_eV": float(delta0_eV),
            "temperature_K": float(temperature_K),
            "omega_eV": float(omega_eV),
            "eta_eV": float(eta_eV),
            "nk": int(nk),
            "q_direction": str(q_direction_name),
            "q_norm": float(np.linalg.norm(q)),
            "actual_twist_count": int(actual_twist_count),
            "twist_mode": str(twist_mode),
            "adaptive_mode": "none",
            "response_variant": str(variant),
            "supported": True,
            "effective_total_nodes": int(points.shape[0]),
            "weight_sum": float(np.sum(weights)),
            "abs_weight_sum_minus_one": float(abs(np.sum(weights) - 1.0)),
            "twist_offset_rule": (
                "q-independent nested symmetry-preserving equal-weight twist quadrature; "
                "offsets are not fitted to Ward residuals"
            ),
            "twist_nested_family": "halton_orbit_prefix_24_32_48",
            "twist_q_independent": True,
            "twist_equal_weight": True,
            "twist_residual_fitted": False,
            "twist_symmetry_inversion": True,
            "twist_symmetry_xy_exchange": True,
            "valid_for_casimir_input": False,
        }
        normal_like = _ward_norms(matrix, float(omega_eV), q)
        row.update(_normal_like_em_ward_diagnostic(matrix, float(omega_eV), q))
        row.update(
            _extended_ward_diagnostic(
                response,
                str(variant),
                matrix,
                float(omega_eV),
                q,
                str(pairing_name),
                float(delta0_eV),
            )
        )
        row.update(
            {
                "density_column_residual_norm": normal_like["density_column_residual_norm"],
                "current_x_column_residual_norm": normal_like["current_x_column_residual_norm"],
                "current_y_column_residual_norm": normal_like["current_y_column_residual_norm"],
                "density_row_residual_norm": normal_like["density_row_residual_norm"],
                "current_x_row_residual_norm": normal_like["current_x_row_residual_norm"],
                "current_y_row_residual_norm": normal_like["current_y_row_residual_norm"],
            }
        )
        row.update(_response_norms(matrix))
        row.update(_collective_metadata(response, str(variant), matrix))
        rows.append(row)
    normal_components = normal_physical_density_current_response_components_imag_axis(points, config, q, weights)
    normal_total = normal_components["total"]
    normal_left, normal_right = _physical_ward_residuals(normal_total, float(omega_eV), q)
    comparison_rows = []
    for corrected_variant in ("bare_bdg", "phase_schur", "amplitude_phase_schur"):
        if corrected_variant not in matrices_by_variant:
            continue
        sc_matrix = matrices_by_variant[corrected_variant]
        sc_left, sc_right = _physical_ward_residuals(sc_matrix, float(omega_eV), q)
        comparison_rows.append(
            {
                "pairing_name": str(pairing_name),
                "response_variant": corrected_variant,
                "nk": int(nk),
                "q_direction": str(q_direction_name),
                "q_norm": float(np.linalg.norm(q)),
                "actual_twist_count": int(actual_twist_count),
                "normal_current_current_block_norm": _norm(normal_total[1:3, 1:3]),
                "superconducting_current_current_block_norm": _norm(sc_matrix[1:3, 1:3]),
                "normal_ward_residual_norm": float(max(_norm(normal_left), _norm(normal_right))),
                "superconducting_ward_residual_norm": float(max(_norm(sc_left), _norm(sc_right))),
                "valid_for_casimir_input": False,
            }
        )
    return {
        "rows": rows,
        "operator_audit_rows": [operator_audit_row],
        "comparison_rows": comparison_rows,
        "convergence_items": [
            {
                "pairing_name": str(pairing_name),
                "response_variant": variant,
                "q_direction": str(q_direction_name),
                "q_norm": float(np.linalg.norm(q)),
                "temperature_K": float(temperature_K),
                "nk": int(nk),
                "actual_twist_count": int(actual_twist_count),
                "effective_total_nodes": int(points.shape[0]),
                "matrix": matrix,
                "current_current_block": matrix[1:3, 1:3],
                "ward_residual_norm": float(max(_ward_norms(matrix, float(omega_eV), q)["left_ward_residual_norm"], _ward_norms(matrix, float(omega_eV), q)["right_ward_residual_norm"])),
            }
            for variant, matrix in matrices_by_variant.items()
        ],
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _convergence_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, float], list[dict[str, Any]]] = {}
    for item in items:
        key = (
            item["pairing_name"],
            item["response_variant"],
            item["q_direction"],
            f"{float(item['q_norm']):.16g}",
            float(item["temperature_K"]),
        )
        grouped.setdefault(key, []).append(item)
    rows = []
    for values in grouped.values():
        ordered = sorted(values, key=lambda row: (int(row["nk"]), int(row["actual_twist_count"])))
        for level_a, level_b in zip(ordered, ordered[1:], strict=False):
            response_b_norm = max(_norm(level_b["matrix"]), 1e-300)
            block_b_norm = max(_norm(level_b["current_current_block"]), 1e-300)
            rows.append(
                {
                    "diagnostic_only": True,
                    "valid_for_casimir_input": False,
                    "level_a": {
                        "nk": int(level_a["nk"]),
                        "actual_twist_count": int(level_a["actual_twist_count"]),
                    },
                    "level_b": {
                        "nk": int(level_b["nk"]),
                        "actual_twist_count": int(level_b["actual_twist_count"]),
                    },
                    "pairing_name": level_b["pairing_name"],
                    "response_variant": level_b["response_variant"],
                    "q_direction": level_b["q_direction"],
                    "q_norm": float(level_b["q_norm"]),
                    "response_relative_change_norm": float(_norm(level_b["matrix"] - level_a["matrix"]) / response_b_norm),
                    "current_current_block_relative_change_norm": float(
                        _norm(level_b["current_current_block"] - level_a["current_current_block"]) / block_b_norm
                    ),
                    "ward_residual_change_norm": float(abs(level_b["ward_residual_norm"] - level_a["ward_residual_norm"])),
                    "cost_ratio_effective_nodes": float(level_b["effective_total_nodes"] / max(level_a["effective_total_nodes"], 1)),
                }
            )
    return rows


def _bdg_weta_convention_audit() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "bdg_hamiltonian_convention": "H_BdG=[[h(k),Delta(k)],[Delta^dagger(k),-h^T(-k)]]",
        "density_vertex_convention": "rho=tau_z=diag(+I4,-I4)",
        "phase_vertex_convention": "Gamma_theta=[[0,i*Delta],[-i*Delta^dagger,0]]",
        "eta2_phase_vertex_convention": "Gamma_eta2=[[0,i*Phi],[-i*Phi^dagger,0]]",
        "eta2_phase_relation": "eta2 = delta0 * theta",
        "em_observable_source_convention": "observable_vertices=(rho,-Vx,-Vy), source_vertices=(rho,Vx,Vy)",
        "left_em_ward_vector": "W_A^L=(iomega,+qx,+qy)",
        "right_em_ward_vector": "W_A^R=(iomega,-qx,-qy)",
        "old_weta_convention": {
            "phase_only": "left=+2, right=+2",
            "amplitude_phase": "left=[0,+2*delta0], right=[0,+2*delta0]",
        },
        "analytic_weta_convention": {
            "phase_only": "left=+2j, right=-2j",
            "amplitude_phase": "left=[0,+2j*delta0], right=[0,-2j*delta0]",
        },
        "order_parameter_gauge_vector_convention": "analytic_imaginary_left_right_opposite",
        "old_real_weta_kept_for_diagnostic": True,
        "residual_fitted": False,
    }


def run_bdg_finite_q_ward_audit(
    *,
    pairings: tuple[str, ...],
    response_variants: tuple[str, ...],
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
    delta0_eV: float,
    nk_values: tuple[int, ...],
    actual_twist_counts: tuple[int, ...],
    twist_mode: str,
    q_values: tuple[float, ...],
    directions: tuple[str, ...],
    workers: int,
    progress_enabled: bool,
) -> dict[str, Any]:
    if twist_mode != "nested_symmetric":
        raise ValueError("BdG compact Ward audit currently supports --twist-mode nested_symmetric")
    tasks = [
        (
            pairing,
            nk,
            actual_twist_count,
            twist_mode,
            direction_name,
            q_value,
            DIRECTION_VECTORS[direction_name],
            response_variants,
            omega_eV,
            temperature_K,
            eta_eV,
            delta0_eV,
        )
        for pairing in pairings
        for nk in nk_values
        for actual_twist_count in actual_twist_counts
        for direction_name in directions
        for q_value in q_values
    ]
    progress_enabled = bool(progress_enabled and sys.stdout.isatty())
    _print_progress(0, len(tasks), enabled=progress_enabled)
    started = time.perf_counter()
    if workers > 1:
        results = []
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_case_worker, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                _print_progress(completed, len(tasks), enabled=progress_enabled)
        backend = "concurrent.futures.ProcessPoolExecutor"
    else:
        results = []
        for completed, task in enumerate(tasks, start=1):
            results.append(_case_worker(task))
            _print_progress(completed, len(tasks), enabled=progress_enabled)
        backend = "sequential"
    rows: list[dict[str, Any]] = []
    operator_audit_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    convergence_items: list[dict[str, Any]] = []
    worker_runtime = 0.0
    for result in results:
        rows.extend(result["rows"])
        operator_audit_rows.extend(result["operator_audit_rows"])
        comparison_rows.extend(result["comparison_rows"])
        convergence_items.extend(result["convergence_items"])
        worker_runtime += float(result["runtime_seconds"])
    return {
        "audit_name": "bdg_finite_q_ward_audit",
        "scope": "diagnostic_only_superconducting_bdg_finite_q_ward_residual_summary",
        "ward_formula_scope": (
            "superconducting BdG Ward closure requires extended electromagnetic + order-parameter collective "
            "kernel diagnostics; bare electromagnetic normal-like residual is not a closure criterion"
        ),
        "superconducting_bdg_closure_formula": "extended electromagnetic + order-parameter collective Ward identity",
        "bare_em_block_closure_criterion": False,
        "schur_response_closure_requires_extended_ward": True,
        "pairing_names": list(pairings),
        "response_variants": list(response_variants),
        "omega_eV": float(omega_eV),
        "temperature_K": float(temperature_K),
        "eta_eV": float(eta_eV),
        "delta0_eV": float(delta0_eV),
        "nk_values": [int(value) for value in nk_values],
        "actual_twist_counts": [int(value) for value in actual_twist_counts],
        "twist_mode": twist_mode,
        "adaptive_mode": "none",
        "component_labels": list(WARD_COMPONENT_LABELS),
        "bdg_weta_convention_audit": _bdg_weta_convention_audit(),
        "bdg_operator_level_weta_audit": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "summary": (
                "representative-k phase/eta2 finite-difference and anomalous-block W_eta convention audit; "
                "does not affect response assembly"
            ),
            "rows": operator_audit_rows,
        },
        "bdg_extended_ward_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "normal_like_em_ward_residual": (
                "retained as electromagnetic-block diagnostic only; not used as superconducting BdG closure"
            ),
            "rows": rows,
        },
        "bdg_ward_convergence_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "rows": _convergence_rows(convergence_items),
        },
        "normal_reference_comparison_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "interpretation": (
                "normal-state closure formula is only for normal response; this block is an integration-error "
                "reference and is not used to judge superconducting BdG Ward closure"
            ),
            "rows": comparison_rows,
        },
        "runtime_profile_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "total_runtime_seconds": float(time.perf_counter() - started),
            "worker_runtime_seconds_sum": float(worker_runtime),
            "workers": int(workers),
            "parallel_backend": backend,
        },
        "output_format": {
            "summary_only": True,
            "removed_large_fields": [
                "per_k_residuals",
                "full_response_matrices",
                "band_basis_matrix_dumps",
                "4x4_matrix_entries",
                "eigenvectors",
                "full_bdg_spectrum_dump",
                "full_collective_matrices_per_k",
            ],
            "max_expected_file_size_mb": 10.0,
            "github_safe_output": True,
        },
        "ward_identity_closed": False,
        "valid_for_casimir_input": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    size_mb = path.stat().st_size / (1024.0 * 1024.0)
    if size_mb > MAX_JSON_SIZE_MB:
        raise RuntimeError(f"BdG Ward audit JSON is {size_mb:.2f} MB, above {MAX_JSON_SIZE_MB:.1f} MB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 compact superconducting BdG finite-q Ward residual audit。")
    parser.add_argument("--temperature-K", type=float, default=30.0)
    parser.add_argument("--omega-eV", type=float, default=0.01)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--delta0-eV", type=float, default=0.04)
    parser.add_argument("--nk-values", nargs="+", type=int, default=[41])
    parser.add_argument("--actual-twist-counts", nargs="+", type=int, default=[32])
    parser.add_argument("--twist-mode", choices=("nested_symmetric",), default="nested_symmetric")
    parser.add_argument("--adaptive-mode", choices=("none",), default="none")
    parser.add_argument("--pairings", nargs="+", choices=("onsite_s", "spm", "dwave"), default=["onsite_s", "spm", "dwave"])
    parser.add_argument("--response-variants", nargs="+", choices=("bare_bdg", "phase_schur", "amplitude_phase_schur"), default=["bare_bdg", "phase_schur", "amplitude_phase_schur"])
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.001, 0.005])
    parser.add_argument("--directions", nargs="+", choices=tuple(DIRECTION_VECTORS), default=["x", "diagonal"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summary-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    payload = run_bdg_finite_q_ward_audit(
        pairings=tuple(args.pairings),
        response_variants=tuple(args.response_variants),
        omega_eV=args.omega_eV,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        delta0_eV=args.delta0_eV,
        nk_values=tuple(args.nk_values),
        actual_twist_counts=tuple(args.actual_twist_counts),
        twist_mode=args.twist_mode,
        q_values=tuple(args.q_values),
        directions=tuple(args.directions),
        workers=max(1, int(args.workers)),
        progress_enabled=not bool(args.no_progress),
    )
    if args.json_output is not None:
        _write_json(args.json_output, payload)
    print(
        "BdG finite-q Ward audit prepared: "
        f"pairings={payload['pairing_names']}, actual_twist_counts={payload['actual_twist_counts']}, "
        f"valid_for_casimir_input={payload['valid_for_casimir_input']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
