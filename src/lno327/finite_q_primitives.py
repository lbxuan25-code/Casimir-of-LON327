"""Shared finite-q BdG response primitives.

This module contains generic low-level numerical helpers used by the finite-q
engine and by the legacy compatibility facade. It does not know about pairing
ansatz names or order-parameter vertex preprocessing choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .conductivity import KuboConfig, fermi_function
from .models.lno327_four_orbital.bdg import bdg_hamiltonian
from .tb_fourier import peierls_hamiltonian_contact_vertex, peierls_hamiltonian_vector_vertex
from .ward_response import physical_ward_residuals


@dataclass(frozen=True)
class BdGFiniteQResponseComponents:
    bare_bubble: np.ndarray
    direct: np.ndarray
    bare_total: np.ndarray

    phase_coupling_left: np.ndarray
    phase_coupling_right: np.ndarray
    phase_phase_bubble: complex
    phase_phase_direct: complex
    phase_phase_total: complex

    minus_schur: np.ndarray
    plus_schur: np.ndarray
    collective_bubble: np.ndarray
    collective_counterterm: np.ndarray
    collective_total: np.ndarray
    em_collective_left: np.ndarray
    collective_em_right: np.ndarray
    amplitude_phase_schur: np.ndarray
    gauge_restored: np.ndarray

    metadata: dict[str, Any]

    @property
    def phase_phase(self) -> complex:
        """Backward-compatible alias for the total phase kernel."""

        return self.phase_phase_total


def validate_finite_q_inputs(
    q_model: np.ndarray,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: KuboConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model must have shape (2,)")
    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("k_points must have shape (n, 2) and must not be empty")
    weights = np.asarray(k_weights, dtype=float)
    if weights.shape != (points.shape[0],):
        raise ValueError("k_weights must have shape (n,)")
    if config.omega_eV < 0.0:
        raise ValueError("omega_eV must be non-negative")
    return q, points, weights


def bdg_finite_q_vector_vertex(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
) -> np.ndarray:
    """Return BdG finite-q source vertex built from the normal Peierls vertex."""

    particle_block = peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, direction)
    hole_block = -peierls_hamiltonian_vector_vertex(-kx, -ky, -qx, -qy, direction).T
    zero = np.zeros((4, 4), dtype=complex)
    return np.block([[particle_block, zero], [zero, hole_block]]).astype(complex)


def bdg_finite_q_contact_vertex(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction_i: str,
    direction_j: str,
) -> np.ndarray:
    """Return BdG finite-q contact vertex from the normal Peierls contact."""

    particle_block = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
    hole_block = -peierls_hamiltonian_contact_vertex(-kx, -ky, -qx, -qy, direction_i, direction_j).T
    zero = np.zeros((4, 4), dtype=complex)
    return np.block([[particle_block, zero], [zero, hole_block]]).astype(complex)


def phase_vertex(pairing: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(pairing)
    return np.block([[zero, 1j * pairing], [-1j * pairing.conjugate().T, zero]]).astype(complex)


def phase_phase_direct_vertex(delta_theta: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(delta_theta)
    return np.block([[zero, -delta_theta], [-delta_theta.conjugate().T, zero]]).astype(complex)


def density_vertex() -> np.ndarray:
    eye = np.eye(4, dtype=complex)
    return np.block([[eye, np.zeros((4, 4), dtype=complex)], [np.zeros((4, 4), dtype=complex), -eye]])


def fermi_derivative(energy_eV: float, fermi_level_eV: float, temperature_eV: float, eta_eV: float) -> float:
    shifted = float(energy_eV) - float(fermi_level_eV)
    if temperature_eV <= 0.0:
        width = max(float(eta_eV), 1e-12)
        return -float(width / (np.pi * (shifted**2 + width**2)))
    x = np.clip(shifted / (2.0 * temperature_eV), -350.0, 350.0)
    return -float(1.0 / (4.0 * temperature_eV * np.cosh(x) ** 2))


def kubo_factor(
    em: float,
    en: float,
    fm: float,
    fn: float,
    omega_eV: float,
    *,
    static_limit: bool = False,
    fermi_level_eV: float = 0.0,
    temperature_eV: float | None = None,
    eta_eV: float = 1e-8,
) -> complex:
    delta_e = float(em) - float(en)
    if static_limit and abs(float(omega_eV)) <= eta_eV and abs(delta_e) < eta_eV:
        if temperature_eV is None:
            raise ValueError("temperature_eV is required for static degenerate Kubo factor")
        return fermi_derivative(float(em), fermi_level_eV, temperature_eV, eta_eV)
    return (float(fm) - float(fn)) / (1j * float(omega_eV) + delta_e)


def vertex_band(states_minus: np.ndarray, vertex: np.ndarray, states_plus: np.ndarray) -> np.ndarray:
    return states_minus.conjugate().T @ vertex @ states_plus


def add_bubble(
    accumulator: np.ndarray,
    left_vertices: tuple[np.ndarray, ...],
    right_vertices: tuple[np.ndarray, ...],
    energies_minus: np.ndarray,
    states_minus: np.ndarray,
    occupations_minus: np.ndarray,
    energies_plus: np.ndarray,
    states_plus: np.ndarray,
    occupations_plus: np.ndarray,
    omega_eV: float,
    weight: float,
    config: KuboConfig | None = None,
    static_limit: bool = False,
) -> None:
    left_band = tuple(vertex_band(states_minus, vertex, states_plus) for vertex in left_vertices)
    right_band = tuple(vertex_band(states_minus, vertex, states_plus) for vertex in right_vertices)
    for m, energy_minus in enumerate(energies_minus):
        for n, energy_plus in enumerate(energies_plus):
            occupation_diff = float(occupations_minus[m] - occupations_plus[n])
            if occupation_diff == 0.0 and not static_limit:
                continue
            if config is None:
                raw_factor = occupation_diff / (1j * omega_eV + float(energy_minus - energy_plus))
            else:
                raw_factor = kubo_factor(
                    float(energy_minus),
                    float(energy_plus),
                    float(occupations_minus[m]),
                    float(occupations_plus[n]),
                    omega_eV,
                    static_limit=static_limit,
                    fermi_level_eV=config.fermi_level_eV,
                    temperature_eV=config.temperature_eV,
                    eta_eV=config.eta_eV,
                )
            factor = 0.5 * weight * raw_factor
            for mu, left in enumerate(left_band):
                for nu, right in enumerate(right_band):
                    accumulator[mu, nu] += factor * left[m, n] * np.conjugate(right[m, n])


def thermal_expectation_bdg(
    kx: float,
    ky: float,
    pairing: np.ndarray,
    vertex: np.ndarray,
    config: KuboConfig,
) -> complex:
    energies, states = np.linalg.eigh(bdg_hamiltonian(kx, ky, pairing))
    occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
    vertex_in_band = states.conjugate().T @ vertex @ states
    return complex(0.5 * np.sum(occupations * np.diag(vertex_in_band)))


def ward_metadata(response: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, float]:
    left, right = physical_ward_residuals(response, omega_eV, q)
    return {
        "left_norm": float(np.linalg.norm(left)),
        "right_norm": float(np.linalg.norm(right)),
        "max_norm": float(max(np.linalg.norm(left), np.linalg.norm(right))),
    }
