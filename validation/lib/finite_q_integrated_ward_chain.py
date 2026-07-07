"""Integrated finite-q Ward proof chain diagnostics.

This module checks the algebraic steps after the operator identities and before the
final Schur response.  It is diagnostic-only and does not define a Casimir gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EM_LABELS = ("density", "current_x", "current_y")
IDENTITY_VERSION = "finite_q_integrated_ward_chain_v1"


def _complex_entries(vector: np.ndarray, labels: tuple[str, ...]) -> list[dict[str, float | str]]:
    array = np.asarray(vector, dtype=complex)
    if array.shape != (len(labels),):
        raise ValueError("vector shape is incompatible with labels")
    return [
        {"component": label, "real": float(np.real(value)), "imag": float(np.imag(value))}
        for label, value in zip(labels, array, strict=True)
    ]


def _vector_payload(vector: np.ndarray, labels: tuple[str, ...] = EM_LABELS) -> dict[str, Any]:
    array = np.asarray(vector, dtype=complex)
    abs_values = np.abs(array)
    dominant_index = int(np.argmax(abs_values)) if abs_values.size else 0
    return {
        "labels": list(labels),
        "vector": _complex_entries(array, labels),
        "norm": float(np.linalg.norm(array)),
        "max_abs": float(np.max(abs_values)) if abs_values.size else 0.0,
        "dominant_component": labels[dominant_index] if abs_values.size else None,
        "valid_for_casimir_input": False,
    }


def _left_em_contract(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=complex)
    return 1j * float(omega_eV) * array[0, :] + float(q[0]) * array[1, :] + float(q[1]) * array[2, :]


def _denominator_cancelled_left_equal_time(workspace: Any) -> np.ndarray:
    equal_time = np.zeros(3, dtype=complex)
    for entry in workspace.entries:
        tau3_band = np.asarray(entry.source_vertices_band[0], dtype=complex)
        for m, occupation_minus in enumerate(entry.occupations_minus):
            for n, occupation_plus in enumerate(entry.occupations_plus):
                factor = 0.5 * float(entry.weight) * (float(occupation_minus) - float(occupation_plus))
                if factor == 0.0:
                    continue
                for nu, right in enumerate(entry.source_vertices_band):
                    equal_time[nu] += factor * tau3_band[m, n] * np.conjugate(right[m, n])
    return equal_time


def evaluate_integrated_ward_chain(*, workspace: Any, response: Any, delta0_eV: float) -> dict[str, Any]:
    """Check the left integrated Ward proof chain for the AA block.

    The diagnostic compares three quantities:

    1. the response-level contraction W_L(K_AA_bubble)+R_L K_etaA;
    2. the denominator-cancelled equal-time trace obtained from the same band
       vertices and occupations;
    3. the contact target -W_L(K_AA_direct).
    """

    q = np.asarray(workspace.q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("workspace.q_model must have shape (2,)")
    omega = float(workspace.config.omega_eV)
    bubble_left = _left_em_contract(response.bare_bubble, omega, q)
    r_left = np.asarray([0.0 + 0.0j, -2j * float(delta0_eV)], dtype=complex)
    collective_left = r_left @ np.asarray(response.collective_em_right, dtype=complex)
    bubble_collective = bubble_left + collective_left
    direct_left = _left_em_contract(response.direct, omega, q)
    contact_target = -direct_left
    equal_time = _denominator_cancelled_left_equal_time(workspace)
    bubble_to_equal_time_difference = bubble_collective - equal_time
    equal_time_to_contact_difference = equal_time - contact_target
    full_chain_residual = bubble_collective + direct_left
    return {
        "identity_version": IDENTITY_VERSION,
        "diagnostic_role": "integrated_ward_proof_chain_not_a_new_ward_criterion",
        "pairing_name": str(workspace.ansatz.name),
        "q_model": [float(q[0]), float(q[1])],
        "omega_eV": omega,
        "delta0_eV": float(delta0_eV),
        "left_chain": {
            "formula": "W_L(K_AA_bubble)+R_L K_etaA -> equal-time trace -> -W_L(K_AA_direct)",
            "bubble_collective_contraction": _vector_payload(bubble_collective),
            "denominator_cancelled_equal_time": _vector_payload(equal_time),
            "contact_target_minus_direct_contraction": _vector_payload(contact_target),
            "bubble_to_equal_time_difference": _vector_payload(bubble_to_equal_time_difference),
            "equal_time_to_contact_difference": _vector_payload(equal_time_to_contact_difference),
            "full_chain_residual": _vector_payload(full_chain_residual),
            "valid_for_casimir_input": False,
        },
        "max_bubble_to_equal_time_difference_norm": float(np.linalg.norm(bubble_to_equal_time_difference)),
        "max_equal_time_to_contact_difference_norm": float(np.linalg.norm(equal_time_to_contact_difference)),
        "max_full_chain_residual_norm": float(np.linalg.norm(full_chain_residual)),
        "valid_for_casimir_input": False,
    }
