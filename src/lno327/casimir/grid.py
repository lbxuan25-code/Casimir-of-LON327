"""Casimir integration grid planning scaffold helpers.

This module only builds grids, simple scaffold weights, and requirement
metadata.  It does not perform a full Matsubara sum, Q integral, or compute
energy, force, or torque.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import C0, EV_TO_J, HBAR, KB


def matsubara_xi_grid(temperature_K: float, n_max: int) -> np.ndarray:
    """Return xi_n = 2*pi*n*k_B*T/hbar for n=0..n_max in s^-1."""

    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if n_max < 0:
        raise ValueError("n_max must be non-negative")
    n = np.arange(int(n_max) + 1, dtype=float)
    return 2.0 * np.pi * n * KB * float(temperature_K) / HBAR


def matsubara_prime_weights(n_max: int) -> np.ndarray:
    """Return Matsubara prime weights with w_0=1/2 and w_n=1 for n>0."""

    if n_max < 0:
        raise ValueError("n_max must be non-negative")
    weights = np.ones(int(n_max) + 1, dtype=float)
    weights[0] = 0.5
    return weights


def xi_si_to_omega_eV(xi_si: np.ndarray | float) -> np.ndarray | float:
    """Convert xi in s^-1 to hbar*xi in eV."""

    return np.asarray(xi_si) * HBAR / EV_TO_J if isinstance(xi_si, np.ndarray) else float(xi_si) * HBAR / EV_TO_J


def omega_eV_to_xi_si(omega_eV: np.ndarray | float) -> np.ndarray | float:
    """Convert hbar*xi in eV to xi in s^-1."""

    return np.asarray(omega_eV) * EV_TO_J / HBAR if isinstance(omega_eV, np.ndarray) else float(omega_eV) * EV_TO_J / HBAR


def q_polar_grid(
    q_max_m_inv: float,
    n_q: int,
    n_phi: int,
    *,
    q_min_m_inv: float = 0.0,
    endpoint_phi: bool = False,
) -> dict[str, np.ndarray]:
    """Return a simple polar Q, phi grid and Cartesian mesh."""

    if q_min_m_inv < 0.0:
        raise ValueError("q_min_m_inv must be non-negative")
    if q_max_m_inv < q_min_m_inv:
        raise ValueError("q_max_m_inv must be >= q_min_m_inv")
    if n_q <= 0:
        raise ValueError("n_q must be positive")
    if n_phi <= 0:
        raise ValueError("n_phi must be positive")
    q = np.linspace(float(q_min_m_inv), float(q_max_m_inv), int(n_q))
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=endpoint_phi)
    q_mesh, phi_mesh = np.meshgrid(q, phi, indexing="ij")
    return {
        "Q_m_inv": q,
        "phi_rad": phi,
        "Qx_m_inv": q_mesh * np.cos(phi_mesh),
        "Qy_m_inv": q_mesh * np.sin(phi_mesh),
    }


def kappa_si(Q_m_inv: np.ndarray | float, xi_si: np.ndarray | float) -> np.ndarray | float:
    """Return kappa = sqrt(Q^2 + xi^2/c^2)."""

    return np.sqrt(np.asarray(Q_m_inv) ** 2 + (np.asarray(xi_si) / C0) ** 2)


def round_trip_factor_from_xi_Q_d(xi_si: float, Q_m_inv: float, separation_m: float) -> float:
    """Return exp(-2*kappa*d) from xi, Q, and separation."""

    if separation_m <= 0.0:
        raise ValueError("separation_m must be positive")
    return float(np.exp(-2.0 * float(kappa_si(Q_m_inv, xi_si)) * float(separation_m)))


def simple_trapezoid_q_weights(Q_m_inv: np.ndarray) -> np.ndarray:
    """Return scaffold weights approximating Q dQ on a one-dimensional Q grid."""

    q = np.asarray(Q_m_inv, dtype=float)
    if q.ndim != 1:
        raise ValueError("Q_m_inv must be one-dimensional")
    if len(q) == 0:
        raise ValueError("Q_m_inv must not be empty")
    if np.any(q < 0.0):
        raise ValueError("Q_m_inv must be non-negative")
    if len(q) == 1:
        return np.zeros_like(q)
    edges = np.empty(len(q) + 1, dtype=float)
    edges[1:-1] = 0.5 * (q[:-1] + q[1:])
    edges[0] = max(0.0, q[0] - 0.5 * (q[1] - q[0]))
    edges[-1] = q[-1] + 0.5 * (q[-1] - q[-2])
    delta_q = np.diff(edges)
    return q * delta_q


def uniform_phi_weights(phi_rad: np.ndarray) -> np.ndarray:
    """Return uniform angular scaffold weights."""

    phi = np.asarray(phi_rad, dtype=float)
    if phi.ndim != 1:
        raise ValueError("phi_rad must be one-dimensional")
    if len(phi) == 0:
        raise ValueError("phi_rad must not be empty")
    return np.full_like(phi, 2.0 * np.pi / len(phi), dtype=float)


def polar_measure_weights(Q_m_inv: np.ndarray, phi_rad: np.ndarray) -> np.ndarray:
    """Return scaffold weights Q dQ dphi/(2*pi)^2."""

    q_weights = simple_trapezoid_q_weights(Q_m_inv)
    phi_weights = uniform_phi_weights(phi_rad)
    return np.outer(q_weights, phi_weights) / (2.0 * np.pi) ** 2


def material_response_grid_requirements() -> dict[str, Any]:
    """Return requirements for a future production material-response grid."""

    return {
        "required_data": "sigma_tilde(i*xi_n,Q,phi) or R_TE_TM(i*xi_n,Q,phi) on a two-dimensional Matsubara/polar grid",
        "existing_validation_cases_insufficient": "Existing 8 validation reflection cases are not a production integration grid.",
        "response_strategy": "Use a validated interpolation strategy or compute response directly at every grid point.",
        "plate_2_rotation": "Q_crystal_2 = R(-theta) Q_lab",
        "final_basis": "All reflection matrices must be represented in the common lab TE/TM basis.",
        "Q_zero_warning": "Q=0 has undefined TE/TM in-plane direction and must be handled by symmetry/limit or excluded from angular-grid production runs.",
        "cutoff_convergence": "High-frequency and large-Q cutoffs require convergence tests.",
        "grid_convergence": "n_max, Q_max, n_Q, and n_phi require convergence audits.",
        "quadrature_warning": "Simple scaffold weights are not production quadrature.",
    }


def casimir_grid_scaffold_metadata() -> dict[str, Any]:
    """Return metadata for the Stage 5.9 grid scaffold."""

    return {
        "formula": "F/A = k_B*T*sum_n' integral d^2Q/(2*pi)^2 logdet[I-exp(-2*kappa*d) R1 R2]",
        "variables": ["xi_n", "Q", "phi", "theta", "d"],
        "measure": "Q dQ dphi / (2*pi)^2",
        "matsubara_prime_weight": "w0=1/2, w_n>0=1",
        "dimensionless_variable_proposal": "y = 2*kappa*d",
        "Q_zero_warning": "Q=0 has undefined TE/TM in-plane direction and must be handled by symmetry/limit or excluded from angular-grid production runs.",
        "no_full_matsubara_sum": True,
        "no_full_Q_integral": True,
        "no_energy_output": True,
        "no_force_output": True,
        "no_torque_output": True,
        "not_casimir_ready_claim": True,
    }
