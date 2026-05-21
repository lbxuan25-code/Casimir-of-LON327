"""Casimir-Lifshitz energy and torque building blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conductivity import ConductivityTensor, anisotropy_delta
from .constants import C0, KB, SIGMA0


@dataclass(frozen=True)
class CasimirSetup:
    """Geometry and thermodynamic inputs for two parallel 2D plates."""

    temperature: float
    distance: float
    area: float = 1.0


def matsubara_frequency(n: int, temperature: float) -> float:
    """Return bosonic Matsubara angular frequency xi_n."""

    from .constants import HBAR

    return 2.0 * np.pi * KB * temperature * n / HBAR


def reflection_matrix_weak_2d(
    xi: float,
    k_parallel: float,
    phi: float,
    conductivity: ConductivityTensor,
) -> np.ndarray:
    """Return Dai-Jiang 2D reflection matrix using delta/sigma_t variables.

    This is the imaginary-frequency form of their Eq. (9). Conductivities are
    supplied directly; future Kubo code should feed this function.
    """

    sigma_t = (conductivity.xx + conductivity.yy) / SIGMA0
    delta = anisotropy_delta(conductivity)
    kappa = np.sqrt((xi / C0) ** 2 + k_parallel**2)
    cos2 = np.cos(2.0 * phi)
    sin2 = np.sin(2.0 * phi)
    denom = (
        (xi**2 / C0**2) * sigma_t * (1.0 - delta * cos2)
        + (xi * kappa / (4.0 * C0)) * sigma_t**2 * (1.0 - delta**2)
        + kappa**2 * sigma_t * (1.0 + delta * cos2)
        + 4.0 * xi * kappa / C0
    )
    if np.isclose(denom, 0.0):
        raise ZeroDivisionError("reflection denominator is numerically zero")
    rss = (
        -((xi**2 / C0**2) * sigma_t * (1.0 - delta * cos2))
        - (xi * kappa / (4.0 * C0)) * sigma_t**2 * (1.0 - delta**2)
    ) / denom
    rpp = (
        (xi * kappa / (4.0 * C0)) * sigma_t**2 * (1.0 - delta**2)
        + kappa**2 * sigma_t * (1.0 + delta * cos2)
    ) / denom
    rps = -(xi * kappa / C0) * sigma_t * delta * sin2 / denom
    rsp = (xi * kappa / C0) * sigma_t * delta * sin2 / denom
    return np.array([[rss, rsp], [rps, rpp]], dtype=complex)


def casimir_energy_integrand(
    setup: CasimirSetup,
    xi: float,
    k_parallel: float,
    phi: float,
    theta: float,
    left: ConductivityTensor,
    right: ConductivityTensor,
) -> complex:
    """Return the k-space integrand of the two-plate Casimir energy."""

    kappa = np.sqrt((xi / C0) ** 2 + k_parallel**2)
    r1 = reflection_matrix_weak_2d(xi, k_parallel, phi, left)
    r2 = reflection_matrix_weak_2d(xi, k_parallel, phi + theta, right)
    matrix = np.eye(2, dtype=complex) - r1 @ r2 * np.exp(-2.0 * kappa * setup.distance)
    return k_parallel * np.log(np.linalg.det(matrix))


def casimir_torque_integrand(
    setup: CasimirSetup,
    xi: float,
    k_parallel: float,
    phi: float,
    theta: float,
    left: ConductivityTensor,
    right: ConductivityTensor,
    dtheta: float = 1e-5,
) -> complex:
    """Return a finite-difference torque integrand, -partial_theta E."""

    e_plus = casimir_energy_integrand(setup, xi, k_parallel, phi, theta + dtheta, left, right)
    e_minus = casimir_energy_integrand(setup, xi, k_parallel, phi, theta - dtheta, left, right)
    return -(e_plus - e_minus) / (2.0 * dtheta)
