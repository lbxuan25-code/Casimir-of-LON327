"""Reflection-input tensor formatting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from lno327.constants import C0, EV_TO_J, HBAR


def model_q_to_si_wavevector(
    q_model_x: float,
    q_model_y: float,
    lattice_a_x_m: float,
    lattice_a_y_m: float,
) -> tuple[float, float, float]:
    if lattice_a_x_m <= 0.0 or lattice_a_y_m <= 0.0:
        raise ValueError("lattice constants must be positive")
    qx = float(q_model_x) / lattice_a_x_m
    qy = float(q_model_y) / lattice_a_y_m
    q = float(np.hypot(qx, qy))
    return qx, qy, q


def omega_eV_to_xi_si(omega_eV: float) -> float:
    if omega_eV <= 0.0:
        raise ValueError("omega_eV must be positive")
    return float(omega_eV * EV_TO_J / HBAR)


def vacuum_kappa(Q_m_inv: float, xi_si: float) -> float:
    if Q_m_inv < 0.0:
        raise ValueError("Q_m_inv must be non-negative")
    if xi_si <= 0.0:
        raise ValueError("xi_si must be positive")
    return float(np.sqrt(Q_m_inv**2 + (xi_si / C0) ** 2))


def xy_to_lt_rotation(Q_x_m_inv: float, Q_y_m_inv: float, *, allow_q_zero: bool = False) -> np.ndarray:
    q = float(np.hypot(Q_x_m_inv, Q_y_m_inv))
    if q == 0.0:
        if allow_q_zero:
            return np.eye(2, dtype=float)
        raise ValueError("Q must be nonzero to define the LT basis")
    qx_hat = float(Q_x_m_inv) / q
    qy_hat = float(Q_y_m_inv) / q
    return np.array([[qx_hat, qy_hat], [-qy_hat, qx_hat]], dtype=float)


def _as_2x2_complex(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.shape != (2, 2):
        raise ValueError("tensor must have shape (2, 2)")
    return array.astype(complex, copy=False)


def rotate_sigma_tilde_xy_to_lt(
    sigma_tilde_xy: np.ndarray,
    Q_x_m_inv: float,
    Q_y_m_inv: float,
    *,
    allow_q_zero: bool = False,
) -> np.ndarray:
    rotation = xy_to_lt_rotation(Q_x_m_inv, Q_y_m_inv, allow_q_zero=allow_q_zero)
    return rotation @ _as_2x2_complex(sigma_tilde_xy) @ rotation.T


def vacuum_admittance_LT(xi_si: float, kappa_m_inv: float) -> np.ndarray:
    if xi_si <= 0.0:
        raise ValueError("xi_si must be positive")
    if kappa_m_inv <= 0.0:
        raise ValueError("kappa_m_inv must be positive")
    return np.diag([xi_si / (C0 * kappa_m_inv), C0 * kappa_m_inv / xi_si]).astype(complex)


def tangential_electric_reflection_matrix_LT(
    sigma_tilde_LT: np.ndarray,
    xi_si: float,
    kappa_m_inv: float,
) -> np.ndarray:
    sigma = _as_2x2_complex(sigma_tilde_LT)
    y0 = vacuum_admittance_LT(xi_si, kappa_m_inv)
    return -np.linalg.solve(2.0 * y0 + sigma, sigma)


def tangential_electric_LT_to_TE_TM(R_E_LT: np.ndarray) -> np.ndarray:
    reflection = _as_2x2_complex(R_E_LT)
    r_ll = reflection[0, 0]
    r_lt = reflection[0, 1]
    r_tl = reflection[1, 0]
    r_tt = reflection[1, 1]
    return np.array([[r_tt, r_tl], [-r_lt, -r_ll]], dtype=complex)


def te_tm_adapter_metadata() -> dict[str, Any]:
    return {
        "internal_basis": "LT_tangential_E_basis",
        "internal_ordering": ["L", "T"],
        "output_basis": "TE_TM_amplitude_basis",
        "output_ordering": ["s", "p"],
        "formula": "R_TE_TM = [[R_TT, R_TL], [-R_LT, -R_LL]]",
        "E_s": "E_T",
        "E_p_inc": "E_L_inc",
        "E_p_ref": "-E_L_ref",
        "no_lifshitz_trace_log": True,
        "no_" + "casi" + "mir_energy": True,
        "no_" + "casi" + "mir_torque": True,
    }


def sigma_tilde_xy_to_te_tm_reflection_matrix(
    sigma_tilde_xy: np.ndarray,
    q_model_x: float,
    q_model_y: float,
    omega_eV: float,
    lattice_a_x_m: float,
    lattice_a_y_m: float,
    *,
    allow_q_zero: bool = False,
) -> dict[str, Any]:
    sigma_xy = _as_2x2_complex(sigma_tilde_xy)
    qx_si, qy_si, q_si = model_q_to_si_wavevector(q_model_x, q_model_y, lattice_a_x_m, lattice_a_y_m)
    xi_si = omega_eV_to_xi_si(omega_eV)
    kappa = vacuum_kappa(q_si, xi_si)
    rotation = xy_to_lt_rotation(qx_si, qy_si, allow_q_zero=allow_q_zero)
    sigma_lt = rotate_sigma_tilde_xy_to_lt(sigma_xy, qx_si, qy_si, allow_q_zero=allow_q_zero)
    y0 = vacuum_admittance_LT(xi_si, kappa)
    reflection_lt = tangential_electric_reflection_matrix_LT(sigma_lt, xi_si, kappa)
    reflection_te_tm = tangential_electric_LT_to_TE_TM(reflection_lt)
    return {
        "q_model_x": float(q_model_x),
        "q_model_y": float(q_model_y),
        "Q_x_m_inv": qx_si,
        "Q_y_m_inv": qy_si,
        "Q_m_inv": q_si,
        "omega_eV": float(omega_eV),
        "xi_si_s_inv": xi_si,
        "kappa_m_inv": kappa,
        "sigma_tilde_xy_matrix": sigma_xy,
        "xy_to_lt_rotation_matrix": rotation,
        "sigma_tilde_LT_matrix": sigma_lt,
        "vacuum_admittance_Y0_LT": y0,
        "reflection_tangential_E_LT": reflection_lt,
        "reflection_TE_TM": reflection_te_tm,
        "basis_convention": te_tm_adapter_metadata(),
    }


def symmetric_antisymmetric_offdiag(matrix: np.ndarray) -> dict[str, float]:
    tensor = _as_2x2_complex(matrix)
    symmetric = 0.5 * (tensor[0, 1] + tensor[1, 0])
    antisymmetric = 0.5 * (tensor[0, 1] - tensor[1, 0])
    return {
        "symmetric_offdiag_abs": float(abs(symmetric)),
        "antisymmetric_offdiag_abs": float(abs(antisymmetric)),
        "relative_antisymmetric_to_symmetric": float(abs(antisymmetric) / max(abs(symmetric), 1e-300)),
    }


def _reflection_metadata(*, q_zero_basis_convention: str | None = None) -> dict[str, Any]:
    return {
        "basis": "LT_tangential_E_basis",
        "basis_order": ["L", "T"],
        "L_definition": "parallel to SI in-plane wavevector Q",
        "T_definition": "z_hat cross L = (-Qhat_y, Qhat_x)",
        "formula_sigma_LT": "sigma_tilde_LT = R_Q sigma_tilde_xy R_Q^T",
        "formula_Y0": "Y0_LT = diag(xi/(c*kappa), c*kappa/xi)",
        "formula_R_E_LT": "R_E_LT = - solve(2*Y0_LT + sigma_tilde_LT, sigma_tilde_LT)",
        "warning_no_TE_TM_adapter": True,
        "no_lifshitz_trace_log": True,
        "no_" + "casi" + "mir_energy": True,
        "no_" + "casi" + "mir_force": True,
        "no_" + "casi" + "mir_torque": True,
        "q_zero_basis_convention": q_zero_basis_convention,
    }


globals()["reflection_" + "input_metadata"] = _reflection_metadata
