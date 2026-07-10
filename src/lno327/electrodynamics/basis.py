"""In-plane basis transformations for sheet electrodynamics.

The conventions in this module are the single source of truth for plate
rotation and longitudinal/transverse projection:

* ``R(theta)`` is an active counter-clockwise rotation from crystal to lab.
* ``q_crystal = R(-theta) @ q_lab``.
* ``T_lab = R(theta) @ T_crystal @ R(theta).T``.
* LT order is ``(L, T)`` with ``T = z_hat x L``.
"""

from __future__ import annotations

import numpy as np

CRYSTAL_XY_BASIS = "crystal_xy"
LAB_XY_BASIS = "lab_xy"
LAB_LT_BASIS = "lab_LT"


def _as_vector_2(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _as_tensor_2(value: np.ndarray, name: str) -> np.ndarray:
    tensor = np.asarray(value, dtype=complex)
    if tensor.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2), got {tensor.shape}")
    if not np.isfinite(tensor.real).all() or not np.isfinite(tensor.imag).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def rotation_matrix(theta_rad: float) -> np.ndarray:
    """Return the active counter-clockwise 2D rotation matrix."""

    theta = float(theta_rad)
    if not np.isfinite(theta):
        raise ValueError("theta_rad must be finite")
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def q_lab_to_crystal(q_lab: np.ndarray, theta_rad: float) -> np.ndarray:
    """Express a fixed lab wavevector in a plate's crystal coordinates."""

    return rotation_matrix(-float(theta_rad)) @ _as_vector_2(q_lab, "q_lab")


def q_crystal_to_lab(q_crystal: np.ndarray, theta_rad: float) -> np.ndarray:
    """Express a crystal-coordinate wavevector in lab coordinates."""

    return rotation_matrix(theta_rad) @ _as_vector_2(q_crystal, "q_crystal")


def tensor_crystal_to_lab(tensor_crystal: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate a rank-two in-plane tensor from crystal to lab coordinates."""

    rotation = rotation_matrix(theta_rad)
    tensor = _as_tensor_2(tensor_crystal, "tensor_crystal")
    return rotation @ tensor @ rotation.T


def tensor_lab_to_crystal(tensor_lab: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate a rank-two in-plane tensor from lab to crystal coordinates."""

    rotation = rotation_matrix(-float(theta_rad))
    tensor = _as_tensor_2(tensor_lab, "tensor_lab")
    return rotation @ tensor @ rotation.T


def xy_to_lt_rotation(q_x: float, q_y: float, *, allow_q_zero: bool = False) -> np.ndarray:
    """Return the lab-xy to lab-LT projection matrix for an in-plane q."""

    qx = float(q_x)
    qy = float(q_y)
    if not np.isfinite(qx) or not np.isfinite(qy):
        raise ValueError("q components must be finite")
    magnitude = float(np.hypot(qx, qy))
    if magnitude == 0.0:
        if allow_q_zero:
            return np.eye(2, dtype=float)
        raise ValueError("Q must be nonzero to define the LT basis")
    qx_hat = qx / magnitude
    qy_hat = qy / magnitude
    return np.array([[qx_hat, qy_hat], [-qy_hat, qx_hat]], dtype=float)


def tensor_xy_to_lt(
    tensor_xy: np.ndarray,
    q_x: float,
    q_y: float,
    *,
    allow_q_zero: bool = False,
) -> np.ndarray:
    """Project a lab-xy rank-two tensor into the common lab ``(L, T)`` basis."""

    projection = xy_to_lt_rotation(q_x, q_y, allow_q_zero=allow_q_zero)
    tensor = _as_tensor_2(tensor_xy, "tensor_xy")
    return projection @ tensor @ projection.T


def tensor_lt_to_xy(
    tensor_lt: np.ndarray,
    q_x: float,
    q_y: float,
    *,
    allow_q_zero: bool = False,
) -> np.ndarray:
    """Transform a lab-LT rank-two tensor back into lab-xy coordinates."""

    projection = xy_to_lt_rotation(q_x, q_y, allow_q_zero=allow_q_zero)
    tensor = _as_tensor_2(tensor_lt, "tensor_lt")
    return projection.T @ tensor @ projection


def tensor_crystal_to_lab_lt(
    tensor_crystal: np.ndarray,
    q_lab: np.ndarray,
    theta_rad: float,
    *,
    allow_q_zero: bool = False,
) -> np.ndarray:
    """Rotate a plate tensor to lab xy and then project onto the lab LT basis."""

    q = _as_vector_2(q_lab, "q_lab")
    tensor_lab = tensor_crystal_to_lab(tensor_crystal, theta_rad)
    return tensor_xy_to_lt(tensor_lab, float(q[0]), float(q[1]), allow_q_zero=allow_q_zero)


__all__ = [
    "CRYSTAL_XY_BASIS",
    "LAB_LT_BASIS",
    "LAB_XY_BASIS",
    "q_crystal_to_lab",
    "q_lab_to_crystal",
    "rotation_matrix",
    "tensor_crystal_to_lab",
    "tensor_crystal_to_lab_lt",
    "tensor_lab_to_crystal",
    "tensor_lt_to_xy",
    "tensor_xy_to_lt",
    "xy_to_lt_rotation",
]
