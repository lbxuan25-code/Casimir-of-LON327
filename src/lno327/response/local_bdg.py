"""Model-agnostic local BdG response."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lno327.bdg.nambu import charge_current_vertex_from_model, diamagnetic_vertex_from_model
from lno327.bdg.spectrum import bdg_eigensystem_from_model, transform_operator_to_band_basis
from lno327.response.bubble import band_basis_bubble_imag_axis
from lno327.response.config import KuboConfig
from lno327.response.containers import KernelComponents
from lno327.response.local_normal import validate_k_points_and_weights
from lno327.response.occupations import fermi_function, negative_fermi_derivative


@dataclass(frozen=True)
class BdGLocalEigensystem:
    energies_eV: np.ndarray
    states: np.ndarray
    occupations: np.ndarray
    negative_fermi_derivative: np.ndarray
    current_x_band: np.ndarray
    current_y_band: np.ndarray


@dataclass(frozen=True)
class BdGLocalSuperconductingResponse:
    paramagnetic: np.ndarray
    diamagnetic: np.ndarray
    total: np.ndarray
    sigma_like_response: np.ndarray


def bdg_local_eigensystem_from_model(
    spec,
    kx: float,
    ky: float,
    channel: str,
    config: KuboConfig | None = None,
) -> BdGLocalEigensystem:
    bands = bdg_eigensystem_from_model(spec, kx, ky, channel)
    if config is None:
        occupations = np.zeros_like(bands.energies, dtype=float)
        minus_df = np.zeros_like(bands.energies, dtype=float)
    else:
        occupations = fermi_function(bands.energies, config.fermi_level_eV, config.temperature_eV)
        minus_df = negative_fermi_derivative(
            bands.energies,
            config.fermi_level_eV,
            config.temperature_eV,
            config.eta_eV,
        )

    jx = transform_operator_to_band_basis(
        bands.states,
        charge_current_vertex_from_model(spec, kx, ky, "x"),
    )
    jy = transform_operator_to_band_basis(
        bands.states,
        charge_current_vertex_from_model(spec, kx, ky, "y"),
    )
    return BdGLocalEigensystem(bands.energies, bands.states, occupations, minus_df, jx, jy)


def bdg_local_paramagnetic_kernel_imag_axis(
    spec,
    channel: str,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    points, weights = validate_k_points_and_weights(k_points, config, k_weights)
    kernel_matrix = np.zeros((2, 2), dtype=complex)

    for weight, (kx, ky) in zip(weights, points, strict=True):
        bands = bdg_local_eigensystem_from_model(spec, float(kx), float(ky), channel, config)
        kernel_matrix += weight * band_basis_bubble_imag_axis(
            bands.energies_eV,
            bands.occupations,
            bands.negative_fermi_derivative,
            (bands.current_x_band, bands.current_y_band),
            config.omega_eV,
            config.eta_eV,
            prefactor=0.5,
        )

    return kernel_matrix


def bdg_local_diamagnetic_kernel(
    spec,
    channel: str,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    points, weights = validate_k_points_and_weights(k_points, config, k_weights)
    kernel_matrix = np.zeros((2, 2), dtype=complex)
    directions = ("x", "y")

    for weight, (kx, ky) in zip(weights, points, strict=True):
        bands = bdg_local_eigensystem_from_model(spec, float(kx), float(ky), channel, config)
        for alpha, direction_a in enumerate(directions):
            for beta, direction_b in enumerate(directions):
                vertex = diamagnetic_vertex_from_model(
                    spec,
                    float(kx),
                    float(ky),
                    direction_a,
                    direction_b,
                )
                vertex_band = transform_operator_to_band_basis(bands.states, vertex)
                kernel_matrix[alpha, beta] += (
                    0.5 * weight * np.sum(bands.occupations * np.diag(vertex_band))
                )

    return kernel_matrix


def bdg_local_total_kernel_imag_axis(
    spec,
    channel: str,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> KernelComponents:
    para = bdg_local_paramagnetic_kernel_imag_axis(spec, channel, k_points, config, k_weights)
    dia = bdg_local_diamagnetic_kernel(spec, channel, k_points, config, k_weights)
    return KernelComponents(paramagnetic=para, diamagnetic=dia, total=dia - para)


def bdg_local_superconducting_response_imag_axis(
    spec,
    channel: str,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> BdGLocalSuperconductingResponse:
    if config.omega_eV <= 0.0:
        raise ValueError("omega_eV must be positive for Sigma_SC = K_total / omega_eV")

    components = bdg_local_total_kernel_imag_axis(spec, channel, k_points, config, k_weights)
    return BdGLocalSuperconductingResponse(
        paramagnetic=components.paramagnetic,
        diamagnetic=components.diamagnetic,
        total=components.total,
        sigma_like_response=components.total / config.omega_eV,
    )
