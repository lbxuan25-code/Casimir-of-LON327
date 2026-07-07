"""Analytic block Ward localization for finite-q collective BdG response.

This module is intentionally algebra-first.  It does not introduce a new Ward
criterion.  It evaluates the four block identities required by the Schur proof
and checks that their residuals reconstruct the final Schur Ward residual.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from lno327.collective.ward import physical_ward_residuals

EM_LABELS = ("density", "current_x", "current_y")
COLLECTIVE_LABELS = ("amplitude_eta1", "phase_eta2")
BLOCK_IDENTITY_VERSION = "collective_block_v1"


def collective_generators(delta0_eV: float) -> tuple[np.ndarray, np.ndarray]:
    """Return left and right collective Ward generators.

    The production collective vertex convention is eta2 = delta0 * theta and
    Gamma_eta2^H = [[0, i phi], [-i phi^dagger, 0]].  With the code's Peierls
    source vertex satisfying q_i Gamma_i = H_+ tau3 - tau3 H_- in the diagonal
    sector, the operator identity is

        D Gamma_A - 2 i delta0 Gamma_eta2 = Q_+ tau3 - tau3 Q_-,

    where D is the source-space electromagnetic Ward contraction.  After the
    physical observable/source current mapping is absorbed into W, the block
    identities are still written as W(K) + R K = 0, but the generator is
    R_L=(0,-2i*delta0) and R_R=(0,+2i*delta0).
    """

    delta0 = float(delta0_eV)
    return np.asarray([0.0 + 0.0j, -2j * delta0], dtype=complex), np.asarray([0.0 + 0.0j, 2j * delta0], dtype=complex)


def left_em_contract(matrix: np.ndarray, omega_eV: float, q_model: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=complex)
    if array.shape[0] != 3:
        raise ValueError("left EM Ward contraction requires three EM rows")
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model must have shape (2,)")
    return 1j * float(omega_eV) * array[0, :] + float(q[0]) * array[1, :] + float(q[1]) * array[2, :]


def right_em_contract(matrix: np.ndarray, omega_eV: float, q_model: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=complex)
    if array.shape[1] != 3:
        raise ValueError("right EM Ward contraction requires three EM columns")
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model must have shape (2,)")
    return 1j * float(omega_eV) * array[:, 0] - float(q[0]) * array[:, 1] - float(q[1]) * array[:, 2]


def _complex_entries(vector: np.ndarray, labels: tuple[str, ...]) -> list[dict[str, float | str]]:
    array = np.asarray(vector, dtype=complex)
    if array.shape != (len(labels),):
        raise ValueError(f"vector shape {array.shape} is incompatible with labels {labels}")
    return [
        {"component": label, "real": float(np.real(value)), "imag": float(np.imag(value))}
        for label, value in zip(labels, array, strict=True)
    ]


def _complex_matrix_entries(matrix: np.ndarray) -> list[list[dict[str, float]]]:
    array = np.asarray(matrix, dtype=complex)
    return [[{"real": float(np.real(value)), "imag": float(np.imag(value))} for value in row] for row in array]


def _vector_payload(vector: np.ndarray, labels: tuple[str, ...]) -> dict[str, Any]:
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


def _inverted(matrix: np.ndarray) -> tuple[np.ndarray, str, float | None]:
    array = np.asarray(matrix, dtype=complex)
    try:
        return np.linalg.inv(array), "inv", float(np.linalg.cond(array))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(array), "pinv_diagnostic", None


def _real_inner_product(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.real(np.vdot(np.asarray(left, dtype=complex), np.asarray(right, dtype=complex))))


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(_real_inner_product(left, right) / (left_norm * right_norm))


def _cancellation_fraction(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    denominator = left_norm + right_norm
    if denominator == 0.0:
        return None
    return float(1.0 - np.linalg.norm(np.asarray(left, dtype=complex) + np.asarray(right, dtype=complex)) / denominator)


def _identity_decomposition_payload(
    *,
    labels: tuple[str, ...],
    identity_name: str,
    first_name: str,
    first: np.ndarray,
    second_name: str,
    second: np.ndarray,
) -> dict[str, Any]:
    first_array = np.asarray(first, dtype=complex)
    second_array = np.asarray(second, dtype=complex)
    residual = first_array + second_array
    first_norm = float(np.linalg.norm(first_array))
    return {
        "identity": identity_name,
        "algebraic_form": f"{first_name} + {second_name} = 0",
        "labels": list(labels),
        "terms": {
            first_name: _vector_payload(first_array, labels),
            second_name: _vector_payload(second_array, labels),
        },
        "residual": _vector_payload(residual, labels),
        "cosine_between_terms": _cosine(first_array, second_array),
        "cancellation_fraction": _cancellation_fraction(first_array, second_array),
        "norm_ratio_second_to_first": None if first_norm == 0.0 else float(np.linalg.norm(second_array) / first_norm),
        "valid_for_casimir_input": False,
    }


def schur_residual_reconstruction(
    *,
    aa_left_error: np.ndarray,
    aeta_left_error: np.ndarray,
    aa_right_error: np.ndarray,
    etaa_right_error: np.ndarray,
    k_aeta: np.ndarray,
    k_etaa: np.ndarray,
    k_etaeta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, float | None]:
    """Reconstruct W(K_AA - K_Aeta inv(K_etaeta) K_etaA) from block errors."""

    inverse, inverse_method, condition_number = _inverted(k_etaeta)
    predicted_left = np.asarray(aa_left_error, dtype=complex) - np.asarray(aeta_left_error, dtype=complex) @ inverse @ np.asarray(k_etaa, dtype=complex)
    predicted_right = np.asarray(aa_right_error, dtype=complex) - np.asarray(k_aeta, dtype=complex) @ inverse @ np.asarray(etaa_right_error, dtype=complex)
    return predicted_left, predicted_right, inverse_method, condition_number


def evaluate_collective_ward_blocks(
    *,
    pairing_name: str,
    q_model: Sequence[float] | np.ndarray,
    omega_eV: float,
    delta0_eV: float,
    k_aa_full: np.ndarray,
    k_aeta: np.ndarray,
    k_etaa: np.ndarray,
    k_etaeta: np.ndarray,
    schur_response: np.ndarray,
) -> dict[str, Any]:
    """Evaluate the four block identities used by the full-Hessian Schur proof."""

    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model must have shape (2,)")
    aa = np.asarray(k_aa_full, dtype=complex)
    aeta = np.asarray(k_aeta, dtype=complex)
    etaa = np.asarray(k_etaa, dtype=complex)
    etaeta = np.asarray(k_etaeta, dtype=complex)
    schur = np.asarray(schur_response, dtype=complex)
    if aa.shape != (3, 3):
        raise ValueError("k_aa_full must have shape (3, 3)")
    if aeta.shape != (3, 2):
        raise ValueError("k_aeta must have shape (3, 2)")
    if etaa.shape != (2, 3):
        raise ValueError("k_etaa must have shape (2, 3)")
    if etaeta.shape != (2, 2):
        raise ValueError("k_etaeta must have shape (2, 2)")
    if schur.shape != (3, 3):
        raise ValueError("schur_response must have shape (3, 3)")

    r_left, r_right = collective_generators(delta0_eV)
    aa_left_w = left_em_contract(aa, omega_eV, q)
    aa_left_generator = r_left @ etaa
    aeta_left_w = left_em_contract(aeta, omega_eV, q)
    aeta_left_generator = r_left @ etaeta
    aa_right_w = right_em_contract(aa, omega_eV, q)
    aa_right_generator = aeta @ r_right
    etaa_right_w = right_em_contract(etaa, omega_eV, q)
    etaa_right_generator = etaeta @ r_right

    aa_left = aa_left_w + aa_left_generator
    aeta_left = aeta_left_w + aeta_left_generator
    aa_right = aa_right_w + aa_right_generator
    etaa_right = etaa_right_w + etaa_right_generator

    inverse, inverse_method, condition_number = _inverted(etaeta)
    left_from_aa_identity = aa_left
    left_from_aeta_identity = -aeta_left @ inverse @ etaa
    right_from_aa_identity = aa_right
    right_from_etaa_identity = -aeta @ inverse @ etaa_right
    predicted_left = left_from_aa_identity + left_from_aeta_identity
    predicted_right = right_from_aa_identity + right_from_etaa_identity
    actual_left, actual_right = physical_ward_residuals(schur, omega_eV, q)
    left_difference = actual_left - predicted_left
    right_difference = actual_right - predicted_right

    blocks = {
        "aa_left": _vector_payload(aa_left, EM_LABELS),
        "aeta_left": _vector_payload(aeta_left, COLLECTIVE_LABELS),
        "aa_right": _vector_payload(aa_right, EM_LABELS),
        "etaa_right": _vector_payload(etaa_right, COLLECTIVE_LABELS),
    }
    block_decomposition = {
        "aa_left": _identity_decomposition_payload(
            labels=EM_LABELS,
            identity_name="W_L(K_AA_full) + R_L K_etaA = 0",
            first_name="W_L(K_AA_full)",
            first=aa_left_w,
            second_name="R_L K_etaA",
            second=aa_left_generator,
        ),
        "aeta_left": _identity_decomposition_payload(
            labels=COLLECTIVE_LABELS,
            identity_name="W_L(K_Aeta) + R_L K_etaeta = 0",
            first_name="W_L(K_Aeta)",
            first=aeta_left_w,
            second_name="R_L K_etaeta",
            second=aeta_left_generator,
        ),
        "aa_right": _identity_decomposition_payload(
            labels=EM_LABELS,
            identity_name="W_R(K_AA_full) + K_Aeta R_R = 0",
            first_name="W_R(K_AA_full)",
            first=aa_right_w,
            second_name="K_Aeta R_R",
            second=aa_right_generator,
        ),
        "etaa_right": _identity_decomposition_payload(
            labels=COLLECTIVE_LABELS,
            identity_name="W_R(K_etaA) + K_etaeta R_R = 0",
            first_name="W_R(K_etaA)",
            first=etaa_right_w,
            second_name="K_etaeta R_R",
            second=etaa_right_generator,
        ),
    }
    schur_contributions = {
        "left_from_aa_identity": _vector_payload(left_from_aa_identity, EM_LABELS),
        "left_from_aeta_identity": _vector_payload(left_from_aeta_identity, EM_LABELS),
        "right_from_aa_identity": _vector_payload(right_from_aa_identity, EM_LABELS),
        "right_from_etaa_identity": _vector_payload(right_from_etaa_identity, EM_LABELS),
    }
    ranked = sorted(
        ({"block": name, "norm": float(payload["norm"]), "max_abs": float(payload["max_abs"])} for name, payload in blocks.items()),
        key=lambda item: item["norm"],
        reverse=True,
    )
    ranked_schur_contributions = sorted(
        ({"contribution": name, "norm": float(payload["norm"]), "max_abs": float(payload["max_abs"])} for name, payload in schur_contributions.items()),
        key=lambda item: item["norm"],
        reverse=True,
    )
    return {
        "identity_version": BLOCK_IDENTITY_VERSION,
        "diagnostic_role": "algebraic_block_identity_localization_not_a_new_criterion",
        "pairing_name": str(pairing_name),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "omega_eV": float(omega_eV),
        "delta0_eV": float(delta0_eV),
        "generators": {
            "left": _complex_entries(r_left, COLLECTIVE_LABELS),
            "right": _complex_entries(r_right, COLLECTIVE_LABELS),
            "eta2_convention": "eta2 = delta0 * theta",
            "operator_identity_convention": "D Gamma_A - 2i delta0 Gamma_eta2 = Q_plus tau3 - tau3 Q_minus",
        },
        "formal_identities": {
            "aa_left": "W_L(K_AA_full) + R_L K_etaA = 0",
            "aeta_left": "W_L(K_Aeta) + R_L K_etaeta = 0",
            "aa_right": "W_R(K_AA_full) + K_Aeta R_R = 0",
            "etaa_right": "W_R(K_etaA) + K_etaeta R_R = 0",
            "schur_reconstruction_left": "W_L(K_GI)=E_AA_L - E_Aeta_L inv(K_etaeta) K_etaA",
            "schur_reconstruction_right": "W_R(K_GI)=E_AA_R - K_Aeta inv(K_etaeta) E_etaA_R",
        },
        "input_shapes": {
            "k_aa_full": list(aa.shape),
            "k_aeta": list(aeta.shape),
            "k_etaa": list(etaa.shape),
            "k_etaeta": list(etaeta.shape),
            "schur_response": list(schur.shape),
        },
        "k_etaeta_matrix": _complex_matrix_entries(etaeta),
        "block_residuals": blocks,
        "block_decomposition": block_decomposition,
        "ranked_block_residuals": ranked,
        "dominant_block_residual": ranked[0] if ranked else None,
        "schur_contribution_breakdown": {
            "contributions": schur_contributions,
            "ranked_contributions": ranked_schur_contributions,
            "left_cancellation_fraction": _cancellation_fraction(left_from_aa_identity, left_from_aeta_identity),
            "right_cancellation_fraction": _cancellation_fraction(right_from_aa_identity, right_from_etaa_identity),
            "left_cosine_between_contributions": _cosine(left_from_aa_identity, left_from_aeta_identity),
            "right_cosine_between_contributions": _cosine(right_from_aa_identity, right_from_etaa_identity),
            "valid_for_casimir_input": False,
        },
        "schur_reconstruction": {
            "inverse_method": inverse_method,
            "condition_number": condition_number,
            "actual_left": _vector_payload(actual_left, EM_LABELS),
            "actual_right": _vector_payload(actual_right, EM_LABELS),
            "predicted_left": _vector_payload(predicted_left, EM_LABELS),
            "predicted_right": _vector_payload(predicted_right, EM_LABELS),
            "left_difference": _vector_payload(left_difference, EM_LABELS),
            "right_difference": _vector_payload(right_difference, EM_LABELS),
            "max_difference_norm": float(max(np.linalg.norm(left_difference), np.linalg.norm(right_difference))),
        },
        "valid_for_casimir_input": False,
    }
