"""Diagnostic-only normal-state finite-momentum current-current response.

This module does not provide a gauge-closed finite-q conductivity.  At exactly
q=0 the public diagnostic function deliberately falls back to the existing
local normal-state Kubo reference.  At nonzero q it evaluates a shifted-state
positive current-current bubble with a midpoint velocity vertex.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .conductivity import (
    KuboConfig,
    fermi_function,
    kubo_conductivity_imag_axis,
)
from .constants import E2_OVER_HBAR
from .model import normal_state_hamiltonian, normal_state_velocity_operator

HamiltonianBuilder = Callable[[float, float], np.ndarray]
VelocityBuilder = Callable[[float, float, str], np.ndarray]


@dataclass(frozen=True)
class ShiftedNormalEigensystem:
    """Normal-state eigensystems at k-q/2 and k+q/2."""

    energies_minus_eV: np.ndarray
    states_minus: np.ndarray
    occupations_minus: np.ndarray
    energies_plus_eV: np.ndarray
    states_plus: np.ndarray
    occupations_plus: np.ndarray


def shifted_normal_eigensystem(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    config: KuboConfig,
    hamiltonian: HamiltonianBuilder = normal_state_hamiltonian,
) -> ShiftedNormalEigensystem:
    """Return normal-state eigensystems at symmetrically shifted momenta."""

    energies_minus, states_minus = np.linalg.eigh(hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy))
    energies_plus, states_plus = np.linalg.eigh(hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy))
    occupations_minus = fermi_function(
        energies_minus, config.fermi_level_eV, config.temperature_eV
    )
    occupations_plus = fermi_function(
        energies_plus, config.fermi_level_eV, config.temperature_eV
    )
    return ShiftedNormalEigensystem(
        energies_minus,
        states_minus,
        occupations_minus,
        energies_plus,
        states_plus,
        occupations_plus,
    )


def midpoint_velocity_vertex(
    kx: float,
    ky: float,
    direction: str,
    states_minus: np.ndarray,
    states_plus: np.ndarray,
    velocity: VelocityBuilder = normal_state_velocity_operator,
) -> np.ndarray:
    """Return <m,k-q/2|v_direction(k)|n,k+q/2>."""

    vertex = velocity(kx, ky, direction)
    return states_minus.conjugate().T @ vertex @ states_plus


def _validated_points_and_weights(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None,
    config: KuboConfig,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("k_points must have shape (n, 2) and must not be empty")
    if config.omega_eV < 0.0:
        raise ValueError("omega_eV must be non-negative")
    if config.eta_eV <= 0.0:
        raise ValueError("eta_eV must be positive")
    if weights is None:
        normalized_weights = np.full(points.shape[0], 1.0 / points.shape[0])
    else:
        normalized_weights = np.asarray(weights, dtype=float)
        if normalized_weights.shape != (points.shape[0],):
            raise ValueError("k_weights must have shape (n,)")
    return points, normalized_weights


def normal_finite_q_current_current_kernel_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    k_weights: Sequence[float] | np.ndarray | None = None,
    hamiltonian: HamiltonianBuilder = normal_state_hamiltonian,
    velocity: VelocityBuilder = normal_state_velocity_operator,
) -> np.ndarray:
    """Return the diagnostic normal finite-q current-current response matrix.

    Exactly q=0 uses ``kubo_conductivity_imag_axis`` as the required local
    reference fallback.  Nonzero q uses the shifted-state positive bubble with
    midpoint velocity vertices.  Consequently, this function is an interface
    diagnostic and must not be interpreted as a gauge-closed finite-q
    conductivity or a Casimir input.
    """

    points, weights = _validated_points_and_weights(k_points, k_weights, config)
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")
    if np.all(q_vector == 0.0):
        return kubo_conductivity_imag_axis(
            points,
            config,
            weights,
            hamiltonian=hamiltonian,
            velocity=velocity,
        ).matrix()

    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    omega = config.omega_eV + config.eta_eV
    response = np.zeros((2, 2), dtype=complex)
    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        bands = shifted_normal_eigensystem(kx, ky, qx, qy, config, hamiltonian)
        vertices = (
            midpoint_velocity_vertex(kx, ky, "x", bands.states_minus, bands.states_plus, velocity),
            midpoint_velocity_vertex(kx, ky, "y", bands.states_minus, bands.states_plus, velocity),
        )
        for m, energy_minus in enumerate(bands.energies_minus_eV):
            for n, energy_plus in enumerate(bands.energies_plus_eV):
                delta_energy = float(energy_minus - energy_plus)
                delta_occupation = float(
                    bands.occupations_minus[m] - bands.occupations_plus[n]
                )
                denominator = delta_energy**2 + omega**2
                response_factor = -delta_occupation * delta_energy / denominator
                if response_factor == 0.0:
                    continue
                for alpha in range(2):
                    for beta in range(2):
                        response[alpha, beta] += (
                            weight
                            * response_factor
                            * vertices[alpha][m, n]
                            * np.conjugate(vertices[beta][m, n])
                        )

    if config.output_si:
        response *= E2_OVER_HBAR
    return response


def c4_covariance_error(matrix_q: np.ndarray, matrix_rotated_q: np.ndarray) -> float:
    """Return ||K(Rq)-R K(q) R^T|| / max(||K(Rq)||, ||K(q)||, eps)."""

    matrix_q = np.asarray(matrix_q, dtype=complex)
    matrix_rotated_q = np.asarray(matrix_rotated_q, dtype=complex)
    if matrix_q.shape != (2, 2) or matrix_rotated_q.shape != (2, 2):
        raise ValueError("both response matrices must have shape (2, 2)")
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    expected = rotation @ matrix_q @ rotation.T
    scale = max(float(np.linalg.norm(matrix_q)), float(np.linalg.norm(matrix_rotated_q)), 1e-300)
    return float(np.linalg.norm(matrix_rotated_q - expected) / scale)
