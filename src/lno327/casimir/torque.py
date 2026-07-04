"""Finite-difference Casimir torque integrand."""

from __future__ import annotations

from lno327.electrodynamics.conductivity import ConductivityTensor

from .lifshitz import casimir_energy_integrand
from .setup import CasimirSetup


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
