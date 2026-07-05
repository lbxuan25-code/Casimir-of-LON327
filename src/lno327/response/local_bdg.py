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
from lno327.response.occupations import fermi_function, negative_fermi_derivative
from lno327.response.validation import validate_k_points_and_weights


@dataclass(frozen=True)
class BdGLocalEigensystem:
    energies_eV: np.ndarray
    states: np.ndarray
    occupations: np.ndarray
    negative_fermi_derivative: np.ndarray
    current_x_band: np.ndarray
    current_y_band: np.ndarray


@dataclass(frozen=True)
class BdGLocalWorkspaceEntry:
    weight: float
    kx: float
    ky: float
    eigensystem: BdGLocalEigensystem
    diamagnetic_vertices_band: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class BdGLocalWorkspace:
    k_points: np.ndarray
    k_weights: np.ndarray
    channel: str
    config: KuboConfig
    entries: tuple[BdGLocalWorkspaceEntry, ...]


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


def precompute_bdg_local_workspace_from_model(
    spec,
    channel: str,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> BdGLocalWorkspace:
    points, weights = validate_k_points_and_weights(k_points, config, k_weights)
    directions = ("x", "y")
    entries = []
    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        bands = bdg_local_eigensystem_from_model(spec, kx, ky, channel, config)
        dia_vertices = tuple(
            tuple(
                transform_operator_to_band_basis(
                    bands.states,
                    diamagnetic_vertex_from_model(spec, kx, ky, direction_a, direction_b),
                )
                for direction_b in directions
            )
            for direction_a in directions
        )
        entries.append(BdGLocalWorkspaceEntry(float(weight), kx, ky, bands, dia_vertices))
    return BdGLocalWorkspace(points, weights, channel, config, tuple(entries))


def bdg_local_paramagnetic_kernel_imag_axis_from_workspace(
    workspace: BdGLocalWorkspace,
    config: KuboConfig | None = None,
) -> np.ndarray:
    eval_config = config or workspace.config
    kernel_matrix = np.zeros((2, 2), dtype=complex)
    for entry in workspace.entries:
        bands = entry.eigensystem
        kernel_matrix += entry.weight * band_basis_bubble_imag_axis(
            bands.energies_eV,
            bands.occupations,
            bands.negative_fermi_derivative,
            (bands.current_x_band, bands.current_y_band),
            eval_config.omega_eV,
            eval_config.eta_eV,
            prefactor=0.5,
        )
    return kernel_matrix


def bdg_local_paramagnetic_kernel_imag_axis(
    spec,
    channel: str,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    workspace = precompute_bdg_local_workspace_from_model(spec, channel, k_points, config, k_weights)
    return bdg_local_paramagnetic_kernel_imag_axis_from_workspace(workspace, config)


def _weighted_bdg_local_eigensystems(
    spec,
    channel: str,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
):
    points, weights = validate_k_points_and_weights(k_points, config, k_weights)
    for weight, (kx, ky) in zip(weights, points, strict=True):
        kx_float = float(kx)
        ky_float = float(ky)
        yield (
            float(weight),
            kx_float,
            ky_float,
            bdg_local_eigensystem_from_model(spec, kx_float, ky_float, channel, config),
        )


def bdg_local_diamagnetic_kernel_from_workspace(
    workspace: BdGLocalWorkspace,
    config: KuboConfig | None = None,
) -> np.ndarray:
    kernel_matrix = np.zeros((2, 2), dtype=complex)
    for entry in workspace.entries:
        bands = entry.eigensystem
        for alpha in range(2):
            for beta in range(2):
                vertex_band = entry.diamagnetic_vertices_band[alpha][beta]
                kernel_matrix[alpha, beta] += (
                    0.5 * entry.weight * np.sum(bands.occupations * np.diag(vertex_band))
                )
    return kernel_matrix


def bdg_local_diamagnetic_kernel(
    spec,
    channel: str,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    workspace = precompute_bdg_local_workspace_from_model(spec, channel, k_points, config, k_weights)
    return bdg_local_diamagnetic_kernel_from_workspace(workspace, config)


def bdg_local_total_kernel_imag_axis_from_workspace(
    workspace: BdGLocalWorkspace,
    config: KuboConfig | None = None,
) -> KernelComponents:
    para = bdg_local_paramagnetic_kernel_imag_axis_from_workspace(workspace, config)
    dia = bdg_local_diamagnetic_kernel_from_workspace(workspace, config)
    return KernelComponents(paramagnetic=para, diamagnetic=dia, total=dia - para)


def bdg_local_total_kernel_imag_axis(
    spec,
    channel: str,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> KernelComponents:
    workspace = precompute_bdg_local_workspace_from_model(spec, channel, k_points, config, k_weights)
    return bdg_local_total_kernel_imag_axis_from_workspace(workspace, config)


def bdg_local_superconducting_response_imag_axis_from_workspace(
    workspace: BdGLocalWorkspace,
    config: KuboConfig | None = None,
) -> BdGLocalSuperconductingResponse:
    eval_config = config or workspace.config
    if eval_config.omega_eV <= 0.0:
        raise ValueError("omega_eV must be positive for Sigma_SC = K_total / omega_eV")

    components = bdg_local_total_kernel_imag_axis_from_workspace(workspace, eval_config)
    return BdGLocalSuperconductingResponse(
        paramagnetic=components.paramagnetic,
        diamagnetic=components.diamagnetic,
        total=components.total,
        sigma_like_response=components.total / eval_config.omega_eV,
    )


def bdg_local_superconducting_response_imag_axis(
    spec,
    channel: str,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> BdGLocalSuperconductingResponse:
    workspace = precompute_bdg_local_workspace_from_model(spec, channel, k_points, config, k_weights)
    return bdg_local_superconducting_response_imag_axis_from_workspace(workspace, config)
