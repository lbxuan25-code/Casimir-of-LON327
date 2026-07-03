"""Model-agnostic local normal-state Kubo response."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lno327.bdg.spectrum import normal_eigensystem_from_model, transform_operator_to_band_basis
from lno327.constants import E2_OVER_HBAR
from lno327.electrodynamics.conductivity import ConductivityTensor
from lno327.response.config import KuboConfig
from lno327.response.occupations import fermi_function, negative_fermi_derivative
from lno327.response.validation import validate_k_points_and_weights


@dataclass(frozen=True)
class NormalConductivityEigensystem:
    energies_eV: np.ndarray
    states: np.ndarray
    occupations: np.ndarray
    negative_fermi_derivative: np.ndarray
    velocity_x_band: np.ndarray
    velocity_y_band: np.ndarray


def normal_conductivity_eigensystem_from_model(
    spec,
    kx: float,
    ky: float,
    config: KuboConfig,
) -> NormalConductivityEigensystem:
    bands = normal_eigensystem_from_model(spec, kx, ky)
    occupations = fermi_function(bands.energies, config.fermi_level_eV, config.temperature_eV)
    minus_df = negative_fermi_derivative(
        bands.energies,
        config.fermi_level_eV,
        config.temperature_eV,
        config.eta_eV,
    )
    vx = transform_operator_to_band_basis(bands.states, spec.velocity_operator(kx, ky, "x"))
    vy = transform_operator_to_band_basis(bands.states, spec.velocity_operator(kx, ky, "y"))
    return NormalConductivityEigensystem(bands.energies, bands.states, occupations, minus_df, vx, vy)


def kubo_conductivity_imag_axis_from_model(
    spec,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> ConductivityTensor:
    points, weights = validate_k_points_and_weights(k_points, config, k_weights)
    omega = config.omega_eV + config.eta_eV
    sigma = np.zeros((2, 2), dtype=complex)

    for weight, (kx, ky) in zip(weights, points, strict=True):
        bands = normal_conductivity_eigensystem_from_model(spec, float(kx), float(ky), config)
        velocity_bands = [bands.velocity_x_band, bands.velocity_y_band]

        for m, energy_m in enumerate(bands.energies_eV):
            for n, energy_n in enumerate(bands.energies_eV):
                if m == n:
                    kernel = bands.negative_fermi_derivative[m] / omega
                else:
                    occupation_diff = bands.occupations[m] - bands.occupations[n]
                    if np.isclose(occupation_diff, 0.0):
                        continue
                    delta = energy_m - energy_n
                    if abs(delta) < config.eta_eV:
                        continue
                    kernel = -occupation_diff * delta / (delta**2 + omega**2)
                for alpha in range(2):
                    for beta in range(2):
                        sigma[alpha, beta] += (
                            weight
                            * kernel
                            * velocity_bands[alpha][m, n]
                            * velocity_bands[beta][n, m]
                        )

    if config.output_si:
        sigma *= E2_OVER_HBAR

    return ConductivityTensor(sigma[0, 0], sigma[1, 1], sigma[0, 1], sigma[1, 0])


def kubo_conductivity_real_axis_from_model(
    spec,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> ConductivityTensor:
    points, weights = validate_k_points_and_weights(k_points, config, k_weights)
    z = config.omega_eV + 1j * config.eta_eV
    drude_denominator = config.eta_eV - 1j * config.omega_eV
    sigma = np.zeros((2, 2), dtype=complex)

    for weight, (kx, ky) in zip(weights, points, strict=True):
        bands = normal_conductivity_eigensystem_from_model(spec, float(kx), float(ky), config)
        velocity_bands = [bands.velocity_x_band, bands.velocity_y_band]

        for m, energy_m in enumerate(bands.energies_eV):
            for n, energy_n in enumerate(bands.energies_eV):
                if m == n:
                    kernel = bands.negative_fermi_derivative[m] / drude_denominator
                else:
                    occupation_diff = bands.occupations[m] - bands.occupations[n]
                    if np.isclose(occupation_diff, 0.0):
                        continue
                    delta = energy_m - energy_n
                    if abs(delta) < config.eta_eV:
                        continue
                    kernel = -occupation_diff * delta / (delta**2 - z**2)
                for alpha in range(2):
                    for beta in range(2):
                        sigma[alpha, beta] += (
                            weight
                            * kernel
                            * velocity_bands[alpha][m, n]
                            * velocity_bands[beta][n, m]
                        )

    if config.output_si:
        sigma *= E2_OVER_HBAR

    return ConductivityTensor(sigma[0, 0], sigma[1, 1], sigma[0, 1], sigma[1, 0])
