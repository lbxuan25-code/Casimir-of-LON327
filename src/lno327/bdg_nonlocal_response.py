"""Diagnostic-only BdG finite-q current-current kernels.

This module exposes a BdG current-current kernel block for contract tests.  It
does not provide a gauge-closed finite-q conductivity, density-current
response, Ward identity check, reflection input, or Casimir input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .bdg_response import bdg_current_vertex
from .conductivity import KuboConfig, fermi_function
from .constants import E2_OVER_HBAR
from .models.lno327_four_orbital.bdg import bdg_hamiltonian
from .models.lno327_four_orbital.pairing import pairing_matrix
from .models.lno327_four_orbital.parameters import PairingAmplitudes, PairingKind


@dataclass(frozen=True)
class ShiftedBdGEigensystem:
    """BdG eigensystems at k-q/2 and k+q/2."""

    energies_minus_eV: np.ndarray
    states_minus: np.ndarray
    occupations_minus: np.ndarray
    energies_plus_eV: np.ndarray
    states_plus: np.ndarray
    occupations_plus: np.ndarray


def shifted_bdg_eigensystem(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None,
    config: KuboConfig,
) -> ShiftedBdGEigensystem:
    """Return BdG eigensystems at symmetrically shifted momenta."""

    kx_minus = kx - 0.5 * qx
    ky_minus = ky - 0.5 * qy
    kx_plus = kx + 0.5 * qx
    ky_plus = ky + 0.5 * qy
    delta_minus = pairing_matrix(pairing_kind, kx_minus, ky_minus, pairing_params)
    delta_plus = pairing_matrix(pairing_kind, kx_plus, ky_plus, pairing_params)
    energies_minus, states_minus = np.linalg.eigh(bdg_hamiltonian(kx_minus, ky_minus, delta_minus))
    energies_plus, states_plus = np.linalg.eigh(bdg_hamiltonian(kx_plus, ky_plus, delta_plus))
    occupations_minus = fermi_function(
        energies_minus,
        config.fermi_level_eV,
        config.temperature_eV,
    )
    occupations_plus = fermi_function(
        energies_plus,
        config.fermi_level_eV,
        config.temperature_eV,
    )
    return ShiftedBdGEigensystem(
        energies_minus,
        states_minus,
        occupations_minus,
        energies_plus,
        states_plus,
        occupations_plus,
    )


def midpoint_bdg_current_vertex(
    kx: float,
    ky: float,
    direction: str,
    states_minus: np.ndarray,
    states_plus: np.ndarray,
) -> np.ndarray:
    """Return <m,k-q/2|J_direction^BdG(k)|n,k+q/2>."""

    vertex = bdg_current_vertex(kx, ky, direction)
    return states_minus.conjugate().T @ vertex @ states_plus


def _validated_points_weights_q(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("k_points must have shape (n, 2) and must not be empty")
    if config.omega_eV < 0.0:
        raise ValueError("omega_eV must be non-negative")
    if config.eta_eV <= 0.0:
        raise ValueError("eta_eV must be positive")
    q_vector = np.asarray(q, dtype=float)
    if q_vector.shape != (2,):
        raise ValueError("q must have shape (2,)")
    if weights is None:
        normalized_weights = np.full(points.shape[0], 1.0 / points.shape[0])
    else:
        normalized_weights = np.asarray(weights, dtype=float)
        if normalized_weights.shape != (points.shape[0],):
            raise ValueError("k_weights must have shape (n,)")
    return points, normalized_weights, q_vector


def bdg_current_current_kernel_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    q: Sequence[float] | np.ndarray,
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None = None,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return the BdG current-current kernel block K(i omega_n, q).

    This is the paramagnetic current-current block only.  It is not K_total,
    not a finite-q conductivity, and not a gauge-closed Pi_mu_nu response.
    Exactly q=0 and q!=0 are evaluated inside this same interface.  The
    degeneracy tolerance ``config.eta_eV`` is used only to skip numerically
    near-degenerate denominators; it is not added to the Matsubara frequency.
    A final 1/2 Nambu prefactor compensates the BdG particle-hole redundancy.
    """

    points, weights, q_vector = _validated_points_weights_q(k_points, config, q, k_weights)
    qx, qy = (float(q_vector[0]), float(q_vector[1]))
    omega = config.omega_eV
    degeneracy_tol_eV = config.eta_eV
    response = np.zeros((2, 2), dtype=complex)

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        bands = shifted_bdg_eigensystem(kx, ky, qx, qy, pairing_kind, pairing_params, config)
        vertices = (
            midpoint_bdg_current_vertex(kx, ky, "x", bands.states_minus, bands.states_plus),
            midpoint_bdg_current_vertex(kx, ky, "y", bands.states_minus, bands.states_plus),
        )

        for m, energy_minus in enumerate(bands.energies_minus_eV):
            for n, energy_plus in enumerate(bands.energies_plus_eV):
                delta_energy = float(energy_minus - energy_plus)
                delta_occupation = float(bands.occupations_minus[m] - bands.occupations_plus[n])
                denominator = delta_energy**2 + omega**2
                if denominator <= degeneracy_tol_eV**2:
                    continue
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

    response *= 0.5
    if config.output_si:
        response *= E2_OVER_HBAR
    return response
