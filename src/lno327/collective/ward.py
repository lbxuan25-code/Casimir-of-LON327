"""Ward residual contraction helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def physical_ward_residuals(
    response: np.ndarray,
    omega_eV: float,
    q: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(response, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError("response must have shape (3, 3)")
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    left = 1j * omega_eV * matrix[0, :] + qx * matrix[1, :] + qy * matrix[2, :]
    right = 1j * omega_eV * matrix[:, 0] - matrix[:, 1] * qx - matrix[:, 2] * qy
    return left, right


def contact_ward_rhs(
    contact: np.ndarray,
    q: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic-only q-contraction of current-current contact terms."""
    matrix = np.asarray(contact, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError("contact must have shape (3, 3)")
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    left = qx * matrix[1, :] + qy * matrix[2, :]
    right = -matrix[:, 1] * qx - matrix[:, 2] * qy
    return left, right


def physical_ward_residuals_contact_aware(
    response: np.ndarray,
    omega_eV: float,
    q: Sequence[float] | np.ndarray,
    contact: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Diagnostic-only contact-aware Ward residual candidates.

    This helper does not define a production Ward convention. It exposes
    candidate residuals so validation reports can distinguish homogeneous
    bubble checks from physical kernels that include contact terms.
    """
    homogeneous = physical_ward_residuals(response, omega_eV, q)
    contact_residual = physical_ward_residuals(contact, omega_eV, q)
    minus_contact = (
        homogeneous[0] - contact_residual[0],
        homogeneous[1] - contact_residual[1],
    )
    plus_contact = (
        homogeneous[0] + contact_residual[0],
        homogeneous[1] + contact_residual[1],
    )
    return {
        "homogeneous": homogeneous,
        "contact": contact_residual,
        "minus_contact": minus_contact,
        "plus_contact": plus_contact,
        "contact_rhs": contact_ward_rhs(contact, q),
    }


def physical_ward_residuals_corrected(
    response: np.ndarray,
    omega_eV: float,
    q: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return physical_ward_residuals(response, omega_eV, q)


def physical_ward_residuals_legacy(
    response: np.ndarray,
    omega_eV: float,
    q: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(response, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError("response must have shape (3, 3)")
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    left = 1j * omega_eV * matrix[0, :] + qx * matrix[1, :] + qy * matrix[2, :]
    right = 1j * omega_eV * matrix[:, 0] + matrix[:, 1] * qx + matrix[:, 2] * qy
    return left, right


def hamiltonian_vector_ward_residuals(
    response: np.ndarray,
    omega_eV: float,
    q: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(response, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError("response must have shape (3, 3)")
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    left = 1j * omega_eV * matrix[0, :] - qx * matrix[1, :] - qy * matrix[2, :]
    right = 1j * omega_eV * matrix[:, 0] - matrix[:, 1] * qx - matrix[:, 2] * qy
    return left, right


def ward_residuals(response: np.ndarray, omega_eV: float, q: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return physical_ward_residuals(response, omega_eV, q)


def ward_errors(response: np.ndarray, omega_eV: float, q: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    left, right = physical_ward_residuals(response, omega_eV, q)
    scale = max(float(np.linalg.norm(response)), 1e-300)
    left_error = float(np.linalg.norm(left) / scale)
    right_error = float(np.linalg.norm(right) / scale)
    return left_error, right_error, max(left_error, right_error)


def ward_metadata(response: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, float]:
    left, right = physical_ward_residuals(response, omega_eV, q)
    return {
        "left_norm": float(np.linalg.norm(left)),
        "right_norm": float(np.linalg.norm(right)),
        "max_norm": float(max(np.linalg.norm(left), np.linalg.norm(right))),
    }
