"""Casimir trace-log integrand prototype helpers.

This module is limited to single-point trace-log integrand prototypes.  It does
not perform a full Matsubara sum, Q integral, or compute energy, force, or
torque.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .lifshitz_readiness import round_trip_factor, trace_log_integrand, trace_log_matrix


def casimir_integrand_single_point(
    R1_te_tm: np.ndarray,
    R2_te_tm: np.ndarray,
    kappa_m_inv: float,
    separation_m: float,
) -> dict[str, Any]:
    """Return the prototype single-point trace-log integrand package."""

    return {
        "round_trip_factor": round_trip_factor(kappa_m_inv, separation_m),
        "trace_log_matrix": trace_log_matrix(R1_te_tm, R2_te_tm, kappa_m_inv, separation_m),
        "logdet_integrand": trace_log_integrand(R1_te_tm, R2_te_tm, kappa_m_inv, separation_m),
    }


def rotation_matrix_2d(theta_rad: float) -> np.ndarray:
    """Return the toy two-dimensional rotation matrix R(theta)."""

    cos_t = float(np.cos(theta_rad))
    sin_t = float(np.sin(theta_rad))
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)


def rotate_2x2_te_tm_toy_matrix(R: np.ndarray, theta_rad: float) -> np.ndarray:
    """Return R(theta) matrix R(theta)^T for toy anisotropic checks only."""

    matrix = np.asarray(R)
    if matrix.shape != (2, 2):
        raise ValueError("R must have shape (2, 2)")
    rotation = rotation_matrix_2d(theta_rad)
    return rotation @ matrix.astype(complex, copy=False) @ rotation.T


def toy_zero_reflection() -> np.ndarray:
    """Return a zero TE/TM toy reflection matrix."""

    return np.zeros((2, 2), dtype=complex)


def toy_isotropic_reflection(r_s: complex, r_p: complex) -> np.ndarray:
    """Return a diagonal TE/TM toy reflection matrix."""

    return np.diag([complex(r_s), complex(r_p)]).astype(complex)


def toy_anisotropic_symmetric_reflection(
    r_s: complex,
    r_p: complex,
    mixing: complex = 0.0,
) -> np.ndarray:
    """Return a symmetric toy TE/TM reflection matrix with offdiag mixing."""

    return np.array([[complex(r_s), complex(mixing)], [complex(mixing), complex(r_p)]], dtype=complex)


def casimir_integrand_prototype_metadata() -> dict[str, Any]:
    """Return frozen conventions and explicit non-production scope."""

    return {
        "basis": "TE_TM_amplitude_basis",
        "ordering": ["s", "p"],
        "rows": "reflected polarization",
        "columns": "incident polarization",
        "matrix_formula": "M = I - exp(-2*kappa*d) * R1 @ R2",
        "integrand_formula": "log(det(M))",
        "round_trip_factor_formula": "exp(-2*kappa*d)",
        "toy_rotation_only_not_physical_material_rotation": True,
        "no_full_matsubara_sum": True,
        "no_full_Q_integral": True,
        "no_casimir_energy": True,
        "no_casimir_force": True,
        "no_casimir_torque": True,
        "not_casimir_ready_claim": True,
    }
