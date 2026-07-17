"""Reflection-input tensor formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.constants import C0, EV_TO_J, HBAR
from lno327.electrodynamics.basis import (
    q_lab_to_crystal,
    tensor_crystal_to_lab,
    tensor_xy_to_lt,
    xy_to_lt_rotation,
)
from lno327.electrodynamics.conventions import (
    PositiveMatsubaraSheetResponse,
    SheetResponseValidation,
    validate_positive_matsubara_sheet_response,
)

LAB_LT_TANGENTIAL_E_BASIS = "lab_LT_tangential_E"


def _readonly_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.array(value, dtype=float, copy=True)
    if vector.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


def _readonly_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.array(value, dtype=complex, copy=True)
    if matrix.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2), got {matrix.shape}")
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        raise ValueError(f"{name} must contain only finite values")
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True)
class SheetReflection:
    """Positive-Matsubara reflection operator in the common lab LT E basis."""

    matrix_lt: np.ndarray
    sigma_tilde_lt: np.ndarray
    q_lab_model: np.ndarray
    q_lab_si_m_inv: np.ndarray
    xi_eV: float
    xi_si_s_inv: float
    kappa_m_inv: float
    theta_rad: float
    basis: str
    sheet_validation: SheetResponseValidation
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix_lt", _readonly_matrix(self.matrix_lt, "matrix_lt"))
        object.__setattr__(self, "sigma_tilde_lt", _readonly_matrix(self.sigma_tilde_lt, "sigma_tilde_lt"))
        object.__setattr__(self, "q_lab_model", _readonly_vector(self.q_lab_model, "q_lab_model"))
        object.__setattr__(self, "q_lab_si_m_inv", _readonly_vector(self.q_lab_si_m_inv, "q_lab_si_m_inv"))

        for name in ("xi_eV", "xi_si_s_inv", "kappa_m_inv"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

        theta = float(self.theta_rad)
        if not np.isfinite(theta):
            raise ValueError("theta_rad must be finite")
        object.__setattr__(self, "theta_rad", theta)
        if self.basis != LAB_LT_TANGENTIAL_E_BASIS:
            raise ValueError(f"basis must be {LAB_LT_TANGENTIAL_E_BASIS!r}")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def matrix(self) -> np.ndarray:
        return self.matrix_lt


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
    """Backward-compatible conductivity-specific wrapper around ``tensor_xy_to_lt``."""

    return tensor_xy_to_lt(
        _as_2x2_complex(sigma_tilde_xy),
        Q_x_m_inv,
        Q_y_m_inv,
        allow_q_zero=allow_q_zero,
    )


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


def positive_matsubara_sheet_response_to_reflection(
    response: PositiveMatsubaraSheetResponse,
    *,
    q_lab_model: np.ndarray,
    theta_rad: float,
    lattice_constant_m: float,
    q_match_tolerance: float = 1e-10,
    require_physical: bool = True,
) -> SheetReflection:
    """Rotate one plate response into the common lab LT basis and reflect it.

    The sheet response must have been evaluated at
    ``q_crystal = R(-theta) @ q_lab``.  This consistency is checked explicitly
    so a response computed for one plate orientation cannot be silently reused
    for another orientation.
    """

    if response.basis != "crystal_xy":
        raise ValueError("positive Matsubara sheet response must be in crystal_xy basis")
    q_lab = np.asarray(q_lab_model, dtype=float)
    if q_lab.shape != (2,) or not np.isfinite(q_lab).all():
        raise ValueError("q_lab_model must be a finite vector with shape (2,)")
    if float(np.linalg.norm(q_lab)) == 0.0:
        raise ValueError("q_lab_model must be nonzero to define the LT basis")
    lattice = float(lattice_constant_m)
    if not np.isfinite(lattice) or lattice <= 0.0:
        raise ValueError("lattice_constant_m must be finite and positive")
    tolerance = float(q_match_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("q_match_tolerance must be finite and non-negative")

    expected_q_crystal = q_lab_to_crystal(q_lab, theta_rad)
    q_scale = max(float(np.linalg.norm(q_lab)), 1.0)
    q_mismatch = float(np.linalg.norm(response.q_model - expected_q_crystal) / q_scale)
    if q_mismatch > tolerance:
        raise ValueError(
            "sheet response q_model is inconsistent with plate orientation: "
            f"relative mismatch={q_mismatch:.3e}, tolerance={tolerance:.3e}"
        )

    validation = validate_positive_matsubara_sheet_response(response)
    if require_physical:
        validation.require_passed()

    sigma_lab = tensor_crystal_to_lab(response.matrix_tilde, theta_rad)
    sigma_lt = tensor_xy_to_lt(sigma_lab, float(q_lab[0]), float(q_lab[1]))
    qx_si, qy_si, q_si = model_q_to_si_wavevector(
        float(q_lab[0]),
        float(q_lab[1]),
        lattice,
        lattice,
    )
    xi_si = omega_eV_to_xi_si(response.xi_eV)
    kappa = vacuum_kappa(q_si, xi_si)
    reflection = tangential_electric_reflection_matrix_LT(sigma_lt, xi_si, kappa)
    return SheetReflection(
        matrix_lt=reflection,
        sigma_tilde_lt=sigma_lt,
        q_lab_model=q_lab,
        q_lab_si_m_inv=np.array([qx_si, qy_si]),
        xi_eV=response.xi_eV,
        xi_si_s_inv=xi_si,
        kappa_m_inv=kappa,
        theta_rad=theta_rad,
        basis=LAB_LT_TANGENTIAL_E_BASIS,
        sheet_validation=validation,
        metadata={
            "source": "PositiveMatsubaraSheetResponse",
            "formula": "R_E_LT = -solve(2 Y0_LT + sigma_tilde_LT, sigma_tilde_LT)",
            "q_crystal_formula": "q_crystal = R(-theta) q_lab",
            "tensor_rotation_formula": "sigma_lab = R(theta) sigma_crystal R(theta)^T",
            "lt_projection_formula": "sigma_LT = P_LT sigma_lab P_LT^T",
            "q_match_relative_residual": q_mismatch,
            "frequency_sector": "positive_matsubara",
        },
    )


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
        "no_casimir_energy": True,
        "no_casimir_torque": True,
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
        "no_casimir_energy": True,
        "no_casimir_force": True,
        "no_casimir_torque": True,
        "q_zero_basis_convention": q_zero_basis_convention,
    }


globals()["reflection_" + "input_metadata"] = _reflection_metadata
