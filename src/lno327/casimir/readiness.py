"""Pre-Lifshitz trace-log readiness helpers.

These helpers define integrand-level matrix conventions for TE/TM reflection
matrices.  They do not perform a full Matsubara sum, Q integral, or compute
energy, force, or torque.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_2x2_complex(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.shape != (2, 2):
        raise ValueError("matrix must have shape (2, 2)")
    return array.astype(complex, copy=False)


def round_trip_factor(kappa_m_inv: float, separation_m: float) -> float:
    """Return exp(-2*kappa*d)."""

    if kappa_m_inv < 0.0:
        raise ValueError("kappa_m_inv must be non-negative")
    if separation_m <= 0.0:
        raise ValueError("separation_m must be positive")
    return float(np.exp(-2.0 * float(kappa_m_inv) * float(separation_m)))


def trace_log_matrix(
    R1_te_tm: np.ndarray,
    R2_te_tm: np.ndarray,
    kappa_m_inv: float,
    separation_m: float,
) -> np.ndarray:
    """Return M = I - exp(-2*kappa*d) * R1 @ R2."""

    r1 = _as_2x2_complex(R1_te_tm)
    r2 = _as_2x2_complex(R2_te_tm)
    u2 = round_trip_factor(kappa_m_inv, separation_m)
    return np.eye(2, dtype=complex) - u2 * (r1 @ r2)


def trace_log_integrand(
    R1_te_tm: np.ndarray,
    R2_te_tm: np.ndarray,
    kappa_m_inv: float,
    separation_m: float,
) -> complex:
    """Return log(det(M)) for a single fixed xi, Q, and separation."""

    matrix = trace_log_matrix(R1_te_tm, R2_te_tm, kappa_m_inv, separation_m)
    return complex(np.log(np.linalg.det(matrix)))


def scalar_sheet_te_tm_reflection(sigma_tilde: float, eta_L: float, eta_T: float) -> np.ndarray:
    """Return diagonal TE/TM reflection matrix for a scalar synthetic sheet."""

    if eta_L <= 0.0 or eta_T <= 0.0:
        raise ValueError("eta_L and eta_T must be positive")
    sigma = float(sigma_tilde)
    r_ss = -sigma / (2.0 * float(eta_T) + sigma)
    r_pp = sigma / (2.0 * float(eta_L) + sigma)
    return np.diag([r_ss, r_pp]).astype(complex)


def rotation_matrix_2d(theta_rad: float) -> np.ndarray:
    """Return the active two-dimensional rotation matrix R(theta)."""

    cos_t = float(np.cos(theta_rad))
    sin_t = float(np.sin(theta_rad))
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)


def lab_q_to_crystal_q(Q_lab: np.ndarray, theta_rad: float) -> np.ndarray:
    """Return Q_crystal = R(-theta) Q_lab."""

    q = np.asarray(Q_lab, dtype=float)
    if q.shape != (2,):
        raise ValueError("Q_lab must have shape (2,)")
    return rotation_matrix_2d(-float(theta_rad)) @ q


def pre_lifshitz_readiness_metadata() -> dict[str, Any]:
    """Return frozen conventions for the pre-Lifshitz readiness audit."""

    return {
        "matrix_ordering": ["s", "p"],
        "rows": "reflected polarization",
        "columns": "incident polarization",
        "R_definition": "E_ref = R E_inc",
        "trace_log_matrix_formula": "M = I - exp(-2*kappa*d) * R1 @ R2",
        "integrand_formula": "log(det(M))",
        "round_trip_factor_formula": "exp(-2*kappa*d)",
        "R1_R2_basis_requirement": "same lab-frame TE/TM basis",
        "plate_convention": "R1 lower plate and R2 upper plate, both viewed from the vacuum gap side",
        "rotation_convention": {
            "theta": "rotation angle of plate 2 crystal axes relative to plate 1/lab axes",
            "Q_crystal": "R(-theta) Q_lab",
            "final_basis": "all final R_TE_TM matrices are represented in the common lab TE/TM basis",
        },
        "no_full_matsubara_sum": True,
        "no_full_Q_integral": True,
        "no_casimir_energy": True,
        "no_casimir_force": True,
        "no_casimir_torque": True,
        "not_casimir_ready_claim": True,
    }
