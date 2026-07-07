"""Analytic block Ward localization for finite-q collective BdG response."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from lno327.collective.ward import physical_ward_residuals

EM_LABELS = ("density", "current_x", "current_y")
COLLECTIVE_LABELS = ("amplitude_eta1", "phase_eta2")
BLOCK_IDENTITY_VERSION = "collective_block_v1"


def collective_generators(delta0_eV: float) -> tuple[np.ndarray, np.ndarray]:
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
    return [{"component": label, "real": float(np.real(value)), "imag": float(np.imag(value))} for label, value in zip(labels, array, strict=True)]


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


def _identity_decomposition_payload(*, labels: tuple[str, ...], identity_name: str, first_name: str, first: np.ndarray, second_name: str, second: np.ndarray) -> dict[str, Any]:
    first_array = np.asarray(first, dtype=complex)
    second_array = np.asarray(second, dtype=complex)
    residual = first_array + second_array
    first_norm = float(np.linalg.norm(first_array))
    return {
        "identity": identity_name,
        "algebraic_form": f"{first_name} + {second_name} = 0",
        "labels": list(labels),
        "terms": {first_name: _vector_payload(first_array, labels), second_name: _vector_payload(second_array, labels)},
        "residual": _vector_payload(residual, labels),
        "cosine_between_terms": _cosine(first_array, second_array),
        "cancellation_fraction": _cancellation_fraction(first_array, second_array),
        "norm_ratio_second_to_first": None if first_norm == 0.0 else float(np.linalg.norm(second_array) / first_norm),
        "valid_for_casimir_input": False,
    }


def _aa_terms_payload(*, side: str, terms: dict[str, np.ndarray], full_residual: np.ndarray) -> dict[str, Any]:
    arrays = {name: np.asarray(value, dtype=complex) for name, value in terms.items()}
    total = sum(arrays.values(), np.zeros(3, dtype=complex))
    full = np.asarray(full_residual, dtype=complex)
    difference = total - full
    return {
        "side": side,
        "diagnostic_role": "aa_full_term_accounting_only",
        "terms": {name: _vector_payload(value, EM_LABELS) for name, value in arrays.items()},
        "sum": _vector_payload(total, EM_LABELS),
        "full_identity_residual": _vector_payload(full, EM_LABELS),
        "sum_minus_full_identity_residual": _vector_payload(difference, EM_LABELS),
        "sum_minus_full_identity_residual_norm": float(np.linalg.norm(difference)),
        "valid_for_casimir_input": False,
    }


def schur_residual_reconstruction(*, aa_left_error: np.ndarray, aeta_left_error: np.ndarray, aa_right_error: np.ndarray, etaa_right_error: np.ndarray, k_aeta: np.ndarray, k_etaa: np.ndarray, k_etaeta: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, float | None]:
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
    k_aa_bubble: np.ndarray | None = None,
    k_aa_direct: np.ndarray | None = None,
) -> dict[str, Any]:
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
    if (k_aa_bubble is None) != (k_aa_direct is None):
        raise ValueError("k_aa_bubble and k_aa_direct must be provided together")
    aa_bubble = None if k_aa_bubble is None else np.asarray(k_aa_bubble, dtype=complex)
    aa_direct = None if k_aa_direct is None else np.asarray(k_aa_direct, dtype=complex)
    if aa_bubble is not None and aa_bubble.shape != (3, 3):
        raise ValueError("k_aa_bubble must have shape (3, 3)")
    if aa_direct is not None and aa_direct.shape != (3, 3):
        raise ValueError("k_aa_direct must have shape (3, 3)")

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

    aa_full_term_decomposition = None
    aa_full_minus_parts_norm = None
    if aa_bubble is not None and aa_direct is not None:
        aa_full_minus_parts_norm = float(np.linalg.norm(aa - (aa_bubble + aa_direct)))
        aa_full_term_decomposition = {
            "diagnostic_role": "aa_full_term_accounting_only",
            "aa_full_definition": "K_AA_full = K_AA_bubble + K_AA_direct",
            "aa_full_minus_bubble_plus_direct_norm": aa_full_minus_parts_norm,
            "left": _aa_terms_payload(
                side="left",
                terms={"W_L(K_AA_bubble)": left_em_contract(aa_bubble, omega_eV, q), "W_L(K_AA_direct)": left_em_contract(aa_direct, omega_eV, q), "R_L K_etaA": aa_left_generator},
                full_residual=aa_left,
            ),
            "right": _aa_terms_payload(
                side="right",
                terms={"W_R(K_AA_bubble)": right_em_contract(aa_bubble, omega_eV, q), "W_R(K_AA_direct)": right_em_contract(aa_direct, omega_eV, q), "K_Aeta R_R": aa_right_generator},
                full_residual=aa_right,
            ),
            "valid_for_casimir_input": False,
        }

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

    blocks = {"aa_left": _vector_payload(aa_left, EM_LABELS), "aeta_left": _vector_payload(aeta_left, COLLECTIVE_LABELS), "aa_right": _vector_payload(aa_right, EM_LABELS), "etaa_right": _vector_payload(etaa_right, COLLECTIVE_LABELS)}
    block_decomposition = {
        "aa_left": _identity_decomposition_payload(labels=EM_LABELS, identity_name="W_L(K_AA_full) + R_L K_etaA = 0", first_name="W_L(K_AA_full)", first=aa_left_w, second_name="R_L K_etaA", second=aa_left_generator),
        "aeta_left": _identity_decomposition_payload(labels=COLLECTIVE_LABELS, identity_name="W_L(K_Aeta) + R_L K_etaeta = 0", first_name="W_L(K_Aeta)", first=aeta_left_w, second_name="R_L K_etaeta", second=aeta_left_generator),
        "aa_right": _identity_decomposition_payload(labels=EM_LABELS, identity_name="W_R(K_AA_full) + K_Aeta R_R = 0", first_name="W_R(K_AA_full)", first=aa_right_w, second_name="K_Aeta R_R", second=aa_right_generator),
        "etaa_right": _identity_decomposition_payload(labels=COLLECTIVE_LABELS, identity_name="W_R(K_etaA) + K_etaeta R_R = 0", first_name="W_R(K_etaA)", first=etaa_right_w, second_name="K_etaeta R_R", second=etaa_right_generator),
    }
    schur_contributions = {
        "left_from_aa_identity": _vector_payload(left_from_aa_identity, EM_LABELS),
        "left_from_aeta_identity": _vector_payload(left_from_aeta_identity, EM_LABELS),
        "right_from_aa_identity": _vector_payload(right_from_aa_identity, EM_LABELS),
        "right_from_etaa_identity": _vector_payload(right_from_etaa_identity, EM_LABELS),
    }
    ranked = sorted(({"block": name, "norm": float(payload["norm"]), "max_abs": float(payload["max_abs"])} for name, payload in blocks.items()), key=lambda item: item["norm"], reverse=True)
    ranked_schur_contributions = sorted(({"contribution": name, "norm": float(payload["norm"]), "max_abs": float(payload["max_abs"])} for name, payload in schur_contributions.items()), key=lambda item: item["norm"], reverse=True)
    return {
        "identity_version": BLOCK_IDENTITY_VERSION,
        "diagnostic_role": "algebraic_block_identity_localization_not_a_new_criterion",
        "pairing_name": str(pairing_name),
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "omega_eV": float(omega_eV),
        "delta0_eV": float(delta0_eV),
        "generators": {"left": _complex_entries(r_left, COLLECTIVE_LABELS), "right": _complex_entries(r_right, COLLECTIVE_LABELS), "eta2_convention": "eta2 = delta0 * theta", "operator_identity_convention": "D Gamma_A - 2i delta0 Gamma_eta2 = Q_plus tau3 - tau3 Q_minus"},
        "formal_identities": {
            "aa_left": "W_L(K_AA_full) + R_L K_etaA = 0",
            "aeta_left": "W_L(K_Aeta) + R_L K_etaeta = 0",
            "aa_right": "W_R(K_AA_full) + K_Aeta R_R = 0",
            "etaa_right": "W_R(K_etaA) + K_etaeta R_R = 0",
            "schur_reconstruction_left": "W_L(K_GI)=E_AA_L - E_Aeta_L inv(K_etaeta) K_etaA",
            "schur_reconstruction_right": "W_R(K_GI)=E_AA_R - K_Aeta inv(K_etaeta) E_etaA_R",
        },
        "input_shapes": {"k_aa_full": list(aa.shape), "k_aa_bubble": None if aa_bubble is None else list(aa_bubble.shape), "k_aa_direct": None if aa_direct is None else list(aa_direct.shape), "k_aeta": list(aeta.shape), "k_etaa": list(etaa.shape), "k_etaeta": list(etaeta.shape), "schur_response": list(schur.shape)},
        "k_etaeta_matrix": _complex_matrix_entries(etaeta),
        "block_residuals": blocks,
        "block_decomposition": block_decomposition,
        "aa_full_term_decomposition": aa_full_term_decomposition,
        "aa_full_minus_bubble_plus_direct_norm": aa_full_minus_parts_norm,
        "ranked_block_residuals": ranked,
        "dominant_block_residual": ranked[0] if ranked else None,
        "schur_contribution_breakdown": {"contributions": schur_contributions, "ranked_contributions": ranked_schur_contributions, "left_cancellation_fraction": _cancellation_fraction(left_from_aa_identity, left_from_aeta_identity), "right_cancellation_fraction": _cancellation_fraction(right_from_aa_identity, right_from_etaa_identity), "left_cosine_between_contributions": _cosine(left_from_aa_identity, left_from_aeta_identity), "right_cosine_between_contributions": _cosine(right_from_aa_identity, right_from_etaa_identity), "valid_for_casimir_input": False},
        "schur_reconstruction": {"inverse_method": inverse_method, "condition_number": condition_number, "actual_left": _vector_payload(actual_left, EM_LABELS), "actual_right": _vector_payload(actual_right, EM_LABELS), "predicted_left": _vector_payload(predicted_left, EM_LABELS), "predicted_right": _vector_payload(predicted_right, EM_LABELS), "left_difference": _vector_payload(left_difference, EM_LABELS), "right_difference": _vector_payload(right_difference, EM_LABELS), "max_difference_norm": float(max(np.linalg.norm(left_difference), np.linalg.norm(right_difference)))},
        "valid_for_casimir_input": False,
    }
