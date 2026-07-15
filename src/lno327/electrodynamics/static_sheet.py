"""Zero-Matsubara sheet electrodynamics for primitive finite-q BdG kernels.

The static mode is a thermodynamic susceptibility problem, not a conductivity
limit.  This module therefore never divides the microscopic kernel by a
frequency.  It extracts the density susceptibility and transverse magnetic
stiffness directly from ``K_eff(q, xi=0)`` after the primitive crystal-xy Ward
contract has been evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.constants import C0, E2_OVER_HBAR, EV_TO_J, HBAR, SIGMA0
from lno327.electrodynamics.basis import q_lab_to_crystal, xy_to_lt_rotation
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.electrodynamics.reflection import (
    LAB_LT_TANGENTIAL_E_BASIS,
    model_q_to_si_wavevector,
)
from lno327.response.effective_kernel import EffectiveEMKernel
from lno327.response.ward_validation import EffectiveWardValidation

STATIC_LOCAL_BASIS = "crystal_local_A0LT"
ZERO_MATSUBARA_SECTOR = "zero_matsubara"


def _readonly_real_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.array(value, dtype=float, copy=True)
    if vector.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


def _readonly_complex_matrix(
    value: np.ndarray,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    matrix = np.array(value, dtype=complex, copy=True)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        raise ValueError(f"{name} must contain only finite values")
    matrix.setflags(write=False)
    return matrix


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _finite_positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _dimensionless_static_kernel(kernel_lt: np.ndarray, energy_scale_eV: float) -> np.ndarray:
    """Scale the mixed-unit ``(A0,L,T)`` kernel into dimensionless diagnostics.

    ``K_00`` has units ``1/eV``, the density-current blocks are dimensionless,
    and the spatial block has units ``eV`` under the fixed source convention.
    """

    matrix = np.asarray(kernel_lt, dtype=complex)
    scaled = matrix.copy()
    scaled[0, 0] *= energy_scale_eV
    scaled[1:3, 1:3] /= energy_scale_eV
    return scaled


@dataclass(frozen=True)
class StaticSheetValidation:
    """Hard static-sheet checks plus nonblocking longitudinal telemetry.

    The crystal-xy effective Ward identity is the gauge-closure hard gate.  The
    local-LT longitudinal norm is retained at every point as a convergence
    diagnostic, but it does not by itself reject an otherwise finite, real,
    passive static sheet response.
    """

    finite: bool
    ward_passed: bool
    relative_imaginary_norm: float
    relative_longitudinal_gauge_residual: float
    relative_density_transverse_mixing: float
    chi_bar: float
    dbar_t: float
    reality_tolerance: float
    longitudinal_tolerance: float
    mixing_tolerance: float
    passivity_tolerance: float

    def __post_init__(self) -> None:
        for name in (
            "relative_imaginary_norm",
            "relative_longitudinal_gauge_residual",
            "relative_density_transverse_mixing",
            "reality_tolerance",
            "longitudinal_tolerance",
            "mixing_tolerance",
            "passivity_tolerance",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        for name in ("chi_bar", "dbar_t"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    @property
    def longitudinal_within_tolerance(self) -> bool:
        return bool(
            self.relative_longitudinal_gauge_residual <= self.longitudinal_tolerance
        )

    @property
    def longitudinal_warning(self) -> bool:
        return not self.longitudinal_within_tolerance

    @property
    def passed(self) -> bool:
        return bool(
            self.finite
            and self.ward_passed
            and self.relative_imaginary_norm <= self.reality_tolerance
            and self.relative_density_transverse_mixing <= self.mixing_tolerance
            and self.chi_bar >= -self.passivity_tolerance
            and self.dbar_t >= -self.passivity_tolerance
        )

    def require_passed(self) -> None:
        if not self.passed:
            raise ValueError(
                "zero-Matsubara sheet response failed hard validation: "
                f"finite={self.finite}, ward_passed={self.ward_passed}, "
                f"relative_imaginary_norm={self.relative_imaginary_norm:.3e}, "
                "relative_density_transverse_mixing="
                f"{self.relative_density_transverse_mixing:.3e}, "
                f"chi_bar={self.chi_bar:.3e}, dbar_t={self.dbar_t:.3e}; "
                "longitudinal_diagnostic_only="
                f"{self.relative_longitudinal_gauge_residual:.3e}"
            )


@dataclass(frozen=True)
class StaticSheetResponse:
    """Static density and transverse-stiffness channels in local crystal LT."""

    kernel_lt: np.ndarray
    chi_bar: float
    dbar_t: float
    q_model: np.ndarray
    energy_scale_eV: float
    degeneracy: float
    basis: str
    validation: StaticSheetValidation
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kernel_lt",
            _readonly_complex_matrix(self.kernel_lt, (3, 3), "kernel_lt"),
        )
        object.__setattr__(self, "q_model", _readonly_real_vector(self.q_model, "q_model"))
        if float(np.linalg.norm(self.q_model)) == 0.0:
            raise ValueError("zero-Matsubara response requires nonzero q_model")
        for name in ("chi_bar", "dbar_t"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "energy_scale_eV",
            _finite_positive(self.energy_scale_eV, "energy_scale_eV"),
        )
        object.__setattr__(
            self,
            "degeneracy",
            _finite_positive(self.degeneracy, "degeneracy"),
        )
        if self.basis != STATIC_LOCAL_BASIS:
            raise ValueError(f"basis must be {STATIC_LOCAL_BASIS!r}")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def q_norm(self) -> float:
        return float(np.linalg.norm(self.q_model))


@dataclass(frozen=True)
class StaticSheetReflection:
    """Zero-Matsubara reflection in the common lab LT tangential-E basis."""

    matrix_lt: np.ndarray
    q_lab_model: np.ndarray
    q_lab_si_m_inv: np.ndarray
    xi_si_s_inv: float
    kappa_m_inv: float
    theta_rad: float
    lambda_l: float
    lambda_t: float
    basis: str
    sheet_validation: StaticSheetValidation
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matrix_lt",
            _readonly_complex_matrix(self.matrix_lt, (2, 2), "matrix_lt"),
        )
        object.__setattr__(
            self,
            "q_lab_model",
            _readonly_real_vector(self.q_lab_model, "q_lab_model"),
        )
        object.__setattr__(
            self,
            "q_lab_si_m_inv",
            _readonly_real_vector(self.q_lab_si_m_inv, "q_lab_si_m_inv"),
        )
        xi = float(self.xi_si_s_inv)
        if xi != 0.0:
            raise ValueError("zero-Matsubara reflection requires xi_si_s_inv == 0")
        object.__setattr__(self, "xi_si_s_inv", 0.0)
        object.__setattr__(
            self,
            "kappa_m_inv",
            _finite_positive(self.kappa_m_inv, "kappa_m_inv"),
        )
        theta = float(self.theta_rad)
        if not np.isfinite(theta):
            raise ValueError("theta_rad must be finite")
        object.__setattr__(self, "theta_rad", theta)
        for name in ("lambda_l", "lambda_t"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.basis != LAB_LT_TANGENTIAL_E_BASIS:
            raise ValueError(f"basis must be {LAB_LT_TANGENTIAL_E_BASIS!r}")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def matrix(self) -> np.ndarray:
        return self.matrix_lt


def static_matsubara_kernel_to_sheet_response(
    kernel: EffectiveEMKernel,
    ward_validation: EffectiveWardValidation,
    *,
    energy_scale_eV: float = 1.0,
    degeneracy: float = 1.0,
    frequency_tolerance_eV: float = 1e-14,
    reality_tolerance: float = 1e-9,
    longitudinal_tolerance: float = 1e-7,
    mixing_tolerance: float = 1e-7,
    passivity_tolerance: float = 1e-10,
) -> StaticSheetResponse:
    """Extract ``chi_bar`` and ``Dbar_T`` directly from ``K_eff(q, 0)``.

    The local LT projection is only a static channel decomposition.  Ward
    validation itself has already been performed in primitive crystal xy.
    """

    frequency_tolerance = _finite_nonnegative(
        frequency_tolerance_eV, "frequency_tolerance_eV"
    )
    if abs(float(kernel.xi_eV)) > frequency_tolerance:
        raise ValueError("zero-Matsubara conversion requires kernel.xi_eV == 0")
    if abs(float(ward_validation.xi_eV)) > frequency_tolerance:
        raise ValueError("zero-Matsubara conversion requires Ward xi_eV == 0")
    if not np.allclose(kernel.q_model, ward_validation.q_model, rtol=0.0, atol=1e-14):
        raise ValueError("kernel and Ward validation q_model do not match")
    q = np.asarray(kernel.q_model, dtype=float)
    if float(np.linalg.norm(q)) == 0.0:
        raise ValueError("zero-Matsubara conversion requires nonzero q_model")

    energy = _finite_positive(energy_scale_eV, "energy_scale_eV")
    factor = _finite_positive(degeneracy, "degeneracy")
    reality = _finite_nonnegative(reality_tolerance, "reality_tolerance")
    longitudinal = _finite_nonnegative(longitudinal_tolerance, "longitudinal_tolerance")
    mixing = _finite_nonnegative(mixing_tolerance, "mixing_tolerance")
    passivity = _finite_nonnegative(passivity_tolerance, "passivity_tolerance")

    projection = xy_to_lt_rotation(float(q[0]), float(q[1]))
    transform = np.eye(3, dtype=float)
    transform[1:3, 1:3] = projection
    kernel_lt = transform @ np.asarray(kernel.k_eff, dtype=complex) @ transform.T
    scaled = _dimensionless_static_kernel(kernel_lt, energy)
    scale = max(float(np.linalg.norm(scaled.real)), 1.0)
    relative_imaginary = float(np.linalg.norm(scaled.imag) / scale)

    longitudinal_entries = np.asarray(
        [
            scaled[0, 1],
            scaled[1, 0],
            scaled[1, 1],
            scaled[1, 2],
            scaled[2, 1],
        ],
        dtype=complex,
    )
    density_transverse_entries = np.asarray(
        [scaled[0, 2], scaled[2, 0]],
        dtype=complex,
    )
    relative_longitudinal = float(np.linalg.norm(longitudinal_entries) / scale)
    relative_density_transverse = float(
        np.linalg.norm(density_transverse_entries) / scale
    )

    chi_bar_complex = -energy * complex(kernel_lt[0, 0])
    dbar_t_complex = -complex(kernel_lt[2, 2]) / energy
    chi_bar = float(chi_bar_complex.real)
    dbar_t = float(dbar_t_complex.real)
    finite = bool(
        np.isfinite(kernel_lt.real).all()
        and np.isfinite(kernel_lt.imag).all()
        and np.isfinite(chi_bar)
        and np.isfinite(dbar_t)
    )
    validation = StaticSheetValidation(
        finite=finite,
        ward_passed=bool(ward_validation.passed),
        relative_imaginary_norm=relative_imaginary,
        relative_longitudinal_gauge_residual=relative_longitudinal,
        relative_density_transverse_mixing=relative_density_transverse,
        chi_bar=chi_bar,
        dbar_t=dbar_t,
        reality_tolerance=reality,
        longitudinal_tolerance=longitudinal,
        mixing_tolerance=mixing,
        passivity_tolerance=passivity,
    )
    return StaticSheetResponse(
        kernel_lt=kernel_lt,
        chi_bar=chi_bar,
        dbar_t=dbar_t,
        q_model=q,
        energy_scale_eV=energy,
        degeneracy=factor,
        basis=STATIC_LOCAL_BASIS,
        validation=validation,
        metadata={
            "frequency_sector": ZERO_MATSUBARA_SECTOR,
            "source": "EffectiveEMKernel at xi_eV=0",
            "ward_convention": ward_validation.metadata.get("convention"),
            "ward_validated_in_basis": "crystal_A0_xy",
            "lt_role": "static channel decomposition only",
            "chi_formula": "chi_bar = - E0 * K_eff_00(q, 0)",
            "stiffness_formula": "Dbar_T = - K_eff_TT(q, 0) / E0",
            "conductivity_division_forbidden": True,
            "local_projection_matrix": transform.copy(),
            "chi_bar_complex_before_reality_projection": chi_bar_complex,
            "dbar_t_complex_before_reality_projection": dbar_t_complex,
            "longitudinal_hard_gate": False,
            "longitudinal_within_tolerance": validation.longitudinal_within_tolerance,
            "longitudinal_warning": validation.longitudinal_warning,
        },
    )


def static_sheet_response_to_reflection(
    response: StaticSheetResponse,
    *,
    q_lab_model: np.ndarray,
    theta_rad: float,
    lattice_constant_m: float = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
    q_match_tolerance: float = 1e-10,
    require_physical: bool = True,
) -> StaticSheetReflection:
    """Map static density/stiffness channels to a common lab-LT reflection."""

    if response.basis != STATIC_LOCAL_BASIS:
        raise ValueError("static sheet response must use the crystal local A0LT basis")
    q_lab = _readonly_real_vector(q_lab_model, "q_lab_model")
    if float(np.linalg.norm(q_lab)) == 0.0:
        raise ValueError("q_lab_model must be nonzero")
    lattice = _finite_positive(lattice_constant_m, "lattice_constant_m")
    tolerance = _finite_nonnegative(q_match_tolerance, "q_match_tolerance")
    theta = float(theta_rad)
    if not np.isfinite(theta):
        raise ValueError("theta_rad must be finite")

    expected_q_crystal = q_lab_to_crystal(q_lab, theta)
    q_scale = max(float(np.linalg.norm(q_lab)), 1.0)
    q_mismatch = float(
        np.linalg.norm(response.q_model - expected_q_crystal) / q_scale
    )
    if q_mismatch > tolerance:
        raise ValueError(
            "static sheet response q_model is inconsistent with plate orientation: "
            f"relative mismatch={q_mismatch:.3e}, tolerance={tolerance:.3e}"
        )
    if require_physical:
        response.validation.require_passed()

    beta = response.energy_scale_eV * EV_TO_J * lattice / (HBAR * C0)
    gamma = E2_OVER_HBAR / SIGMA0
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("static electrodynamic beta must be finite and positive")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("static electrodynamic gamma must be finite and positive")

    q_norm = float(np.linalg.norm(q_lab))
    lambda_l = response.degeneracy * (gamma / beta) * response.chi_bar / q_norm
    lambda_t = response.degeneracy * (gamma * beta) * response.dbar_t / q_norm
    if 2.0 + lambda_l <= 0.0 or 2.0 + lambda_t <= 0.0:
        raise ValueError("static reflection denominator reached a nonphysical pole")
    reflection = np.diag(
        [
            -lambda_l / (2.0 + lambda_l),
            -lambda_t / (2.0 + lambda_t),
        ]
    ).astype(complex)

    qx_si, qy_si, q_si = model_q_to_si_wavevector(
        float(q_lab[0]),
        float(q_lab[1]),
        lattice,
        lattice,
    )
    return StaticSheetReflection(
        matrix_lt=reflection,
        q_lab_model=q_lab,
        q_lab_si_m_inv=np.asarray([qx_si, qy_si], dtype=float),
        xi_si_s_inv=0.0,
        kappa_m_inv=q_si,
        theta_rad=theta,
        lambda_l=lambda_l,
        lambda_t=lambda_t,
        basis=LAB_LT_TANGENTIAL_E_BASIS,
        sheet_validation=response.validation,
        metadata={
            "frequency_sector": ZERO_MATSUBARA_SECTOR,
            "matsubara_prime_weight": 0.5,
            "source": "StaticSheetResponse",
            "beta": beta,
            "gamma": gamma,
            "lambda_l_formula": "g * (gamma / beta) * chi_bar / q",
            "lambda_t_formula": "g * gamma * beta * Dbar_T / q",
            "reflection_formula": "diag(-lambda_L/(2+lambda_L), -lambda_T/(2+lambda_T))",
            "basis_order": ("L", "T"),
            "q_crystal_formula": "q_crystal = R(-theta) q_lab",
            "q_match_relative_residual": q_mismatch,
            "conductivity_limit_not_used": True,
            "longitudinal_hard_gate": False,
            "longitudinal_within_tolerance": (
                response.validation.longitudinal_within_tolerance
            ),
            "longitudinal_warning": response.validation.longitudinal_warning,
        },
    )


__all__ = [
    "STATIC_LOCAL_BASIS",
    "ZERO_MATSUBARA_SECTOR",
    "StaticSheetReflection",
    "StaticSheetResponse",
    "StaticSheetValidation",
    "static_matsubara_kernel_to_sheet_response",
    "static_sheet_response_to_reflection",
]
