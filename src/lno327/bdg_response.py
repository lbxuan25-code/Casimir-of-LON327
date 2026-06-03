"""Minimal BdG paramagnetic electromagnetic response building blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .conductivity import KuboConfig, fermi_function, negative_fermi_derivative
from .model import normal_state_mass_operator, normal_state_velocity_operator
from .pairing import PairingAmplitudes, PairingKind, bdg_hamiltonian, pairing_matrix


@dataclass(frozen=True)
class BdGEigensystem:
    """BdG eigenvalues, eigenvectors, occupations, and current vertices."""

    energies_eV: np.ndarray
    states: np.ndarray
    occupations: np.ndarray
    negative_fermi_derivative: np.ndarray
    current_x_band: np.ndarray
    current_y_band: np.ndarray


@dataclass(frozen=True)
class BdGKernelComponents:
    """BdG electromagnetic kernel components on the imaginary axis.

    ``paramagnetic`` is the positive current-current bubble. ``diamagnetic`` is
    the mass/contact term. In the current Peierls/free-energy validated
    convention, ``total`` is the stiffness kernel ``K_dia - K_para``.
    """

    paramagnetic: np.ndarray
    diamagnetic: np.ndarray
    total: np.ndarray


@dataclass(frozen=True)
class BdGSuperconductingResponse:
    """BdG imaginary-axis superconducting response diagnostic.

    ``sigma_like_response`` is defined here as K_total / omega_eV for positive
    Matsubara energies, where K_total uses the dia-minus-para stiffness
    convention. It is a sheet response kernel used for comparison with
    normal-state sigma(i omega), not a real-axis optical conductivity.
    """

    paramagnetic: np.ndarray
    diamagnetic: np.ndarray
    total: np.ndarray
    sigma_like_response: np.ndarray


def bdg_current_vertex(kx: float, ky: float, direction: str) -> np.ndarray:
    """Return the 8x8 BdG charge-current vertex.

    This is intentionally not ``dH_BdG/dk``. It contains only the normal-state
    charge-current blocks:

    J_a^BdG(k) = [[d_a H0(k), 0], [0, -d_a H0^T(-k)]].
    """

    if direction not in {"x", "y"}:
        raise ValueError("direction must be 'x' or 'y'")

    particle_block = normal_state_velocity_operator(kx, ky, direction)
    hole_block = -normal_state_velocity_operator(-kx, -ky, direction).T
    zero = np.zeros((4, 4), dtype=complex)
    return np.block(
        [
            [particle_block, zero],
            [zero, hole_block],
        ]
    ).astype(complex)


def bdg_current_vertices(kx: float, ky: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(J_x^BdG, J_y^BdG)`` at one momentum."""

    return bdg_current_vertex(kx, ky, "x"), bdg_current_vertex(kx, ky, "y")


def bdg_diamagnetic_vertex(
    kx: float,
    ky: float,
    direction_a: str,
    direction_b: str,
) -> np.ndarray:
    """Return the 8x8 BdG diamagnetic vertex from normal-state H0 only.

    No derivatives of Delta(k) are included. The hole block mirrors the
    charge-current convention used by ``bdg_current_vertex``.
    """

    if direction_a not in {"x", "y"} or direction_b not in {"x", "y"}:
        raise ValueError("directions must be 'x' or 'y'")

    particle_block = normal_state_mass_operator(kx, ky, direction_a, direction_b)
    hole_block = -normal_state_mass_operator(-kx, -ky, direction_a, direction_b).T
    zero = np.zeros((4, 4), dtype=complex)
    return np.block(
        [
            [particle_block, zero],
            [zero, hole_block],
        ]
    ).astype(complex)


def bdg_eigensystem(
    kx: float,
    ky: float,
    pairing: np.ndarray,
    config: KuboConfig | None = None,
) -> BdGEigensystem:
    """Diagonalize the BdG Hamiltonian and transform current vertices."""

    h_bdg = bdg_hamiltonian(kx, ky, pairing)
    energies, states = np.linalg.eigh(h_bdg)
    if config is None:
        occupations = np.zeros_like(energies, dtype=float)
        minus_df = np.zeros_like(energies, dtype=float)
    else:
        occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
        minus_df = negative_fermi_derivative(
            energies,
            config.fermi_level_eV,
            config.temperature_eV,
            config.eta_eV,
        )

    jx, jy = bdg_current_vertices(kx, ky)
    current_x_band = states.conjugate().T @ jx @ states
    current_y_band = states.conjugate().T @ jy @ states
    return BdGEigensystem(energies, states, occupations, minus_df, current_x_band, current_y_band)


def _validate_kernel_inputs(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    weights: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("k_points must have shape (n, 2)")
    if points.shape[0] == 0:
        raise ValueError("k_points must not be empty")
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


def bdg_paramagnetic_kernel_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None = None,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return the positive 2x2 BdG paramagnetic current-current bubble.

    This is a paramagnetic kernel only. It does not include the diamagnetic
    term and should not be interpreted as a full superconducting conductivity.

    The final 1/2 prefactor compensates the particle-hole redundancy of the BdG
    basis, matching the Nambu prefactor used in the diamagnetic kernel.
    """

    points, weights = _validate_kernel_inputs(k_points, config, k_weights)
    omega = config.omega_eV + config.eta_eV
    kernel_matrix = np.zeros((2, 2), dtype=complex)

    for weight, (kx, ky) in zip(weights, points, strict=True):
        delta = pairing_matrix(pairing_kind, float(kx), float(ky), pairing_params)
        bands = bdg_eigensystem(float(kx), float(ky), delta, config)
        currents = [bands.current_x_band, bands.current_y_band]

        for m, energy_m in enumerate(bands.energies_eV):
            for n, energy_n in enumerate(bands.energies_eV):
                if m == n:
                    response_factor = bands.negative_fermi_derivative[m]
                else:
                    occupation_diff = bands.occupations[m] - bands.occupations[n]
                    if np.isclose(occupation_diff, 0.0):
                        continue
                    energy_diff = energy_m - energy_n
                    if abs(energy_diff) < config.eta_eV:
                        continue
                    response_factor = -occupation_diff * energy_diff / (energy_diff**2 + omega**2)
                for alpha in range(2):
                    for beta in range(2):
                        kernel_matrix[alpha, beta] += (
                            weight
                            * response_factor
                            * currents[alpha][m, n]
                            * currents[beta][n, m]
                        )

    return 0.5 * kernel_matrix


def bdg_diamagnetic_kernel(
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return the 2x2 BdG diamagnetic kernel.

    This is a diamagnetic kernel only. It does not include K_para and should
    not be interpreted as a full superconducting conductivity.

    The trace over Nambu quasiparticles carries an explicit 1/2 prefactor to
    compensate the particle-hole redundancy of the BdG basis used here.
    """

    points, weights = _validate_kernel_inputs(k_points, config, k_weights)
    kernel_matrix = np.zeros((2, 2), dtype=complex)
    directions = ("x", "y")

    for weight, (kx, ky) in zip(weights, points, strict=True):
        delta = pairing_matrix(pairing_kind, float(kx), float(ky), pairing_params)
        bands = bdg_eigensystem(float(kx), float(ky), delta, config)
        for alpha, direction_a in enumerate(directions):
            for beta, direction_b in enumerate(directions):
                vertex = bdg_diamagnetic_vertex(float(kx), float(ky), direction_a, direction_b)
                vertex_band = bands.states.conjugate().T @ vertex @ bands.states
                kernel_matrix[alpha, beta] += (
                    0.5 * weight * np.sum(bands.occupations * np.diag(vertex_band))
                )

    return kernel_matrix


def bdg_total_kernel_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None = None,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> BdGKernelComponents:
    """Return K_para, K_dia, and K_total = K_dia - K_para at i omega.

    K_para is the positive current-current bubble and K_dia is the mass/contact
    term. In this convention, Peierls/free-energy validation identifies the
    electromagnetic stiffness kernel as K_dia - K_para. This is not a Casimir
    input yet and is not labeled as an experimental optical conductivity.
    """

    para = bdg_paramagnetic_kernel_imag_axis(
        k_points,
        config,
        pairing_kind,
        pairing_params,
        k_weights,
    )
    dia = bdg_diamagnetic_kernel(
        pairing_kind,
        pairing_params,
        k_points,
        config,
        k_weights,
    )
    return BdGKernelComponents(paramagnetic=para, diamagnetic=dia, total=dia - para)


def bdg_superconducting_response_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None = None,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> BdGSuperconductingResponse:
    """Return Sigma_SC(i omega) = K_total(i omega) / omega_eV for omega_eV > 0.

    The returned ``sigma_like_response`` is an imaginary-axis superconducting
    sheet response kernel. The K_total used here is the dia-minus-para
    stiffness kernel returned by ``bdg_total_kernel_imag_axis``.
    """

    if config.omega_eV <= 0.0:
        raise ValueError("omega_eV must be positive for Sigma_SC = K_total / omega_eV")

    components = bdg_total_kernel_imag_axis(
        k_points,
        config,
        pairing_kind,
        pairing_params,
        k_weights,
    )
    return BdGSuperconductingResponse(
        paramagnetic=components.paramagnetic,
        diamagnetic=components.diamagnetic,
        total=components.total,
        sigma_like_response=components.total / config.omega_eV,
    )
