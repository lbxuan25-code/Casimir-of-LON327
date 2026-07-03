"""Conductivity helpers for future Kubo and Casimir calculations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .constants import E2_OVER_HBAR, KB_EV_PER_K
from .models.lno327_four_orbital.normal import normal_state_hamiltonian
from .models.lno327_four_orbital.vertices import normal_state_velocity_operator

HamiltonianBuilder = Callable[[float, float], np.ndarray]
VelocityBuilder = Callable[[float, float, str], np.ndarray]


@dataclass(frozen=True)
class ConductivityTensor:
    """2D optical conductivity tensor at a fixed imaginary frequency."""

    xx: complex
    yy: complex
    xy: complex = 0.0
    yx: complex = 0.0

    def matrix(self) -> np.ndarray:
        return np.array([[self.xx, self.xy], [self.yx, self.yy]], dtype=complex)


@dataclass(frozen=True)
class KuboConfig:
    """Inputs for the band-basis Kubo conductivity.

    All energy-like quantities are in eV: Hamiltonian eigenvalues, Fermi level,
    external frequency, broadening, and kBT.
    """

    omega_eV: float
    temperature_eV: float
    fermi_level_eV: float = 0.0
    eta_eV: float = 1e-6
    output_si: bool = True

    @classmethod
    def from_kelvin(
        cls,
        omega_eV: float,
        temperature_K: float,
        fermi_level_eV: float = 0.0,
        eta_eV: float = 1e-6,
        output_si: bool = True,
    ) -> "KuboConfig":
        return cls(
            omega_eV=omega_eV,
            temperature_eV=temperature_K * KB_EV_PER_K,
            fermi_level_eV=fermi_level_eV,
            eta_eV=eta_eV,
            output_si=output_si,
        )


def rotate_conductivity(conductivity: ConductivityTensor, phi: float) -> ConductivityTensor:
    """Rotate a 2D conductivity tensor counter-clockwise by phi."""

    c = np.cos(phi)
    s = np.sin(phi)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    rotated = rotation @ conductivity.matrix() @ rotation.T
    return ConductivityTensor(rotated[0, 0], rotated[1, 1], rotated[0, 1], rotated[1, 0])


def anisotropy_delta(conductivity: ConductivityTensor) -> complex:
    """Return delta = (sigma_xx - sigma_yy)/(sigma_xx + sigma_yy)."""

    denom = conductivity.xx + conductivity.yy
    if np.isclose(denom, 0.0):
        raise ValueError("sigma_xx + sigma_yy must be nonzero")
    return (conductivity.xx - conductivity.yy) / denom


def anisotropy_summary(conductivity: ConductivityTensor) -> dict[str, complex]:
    """Return compact diagnostics for a 2D conductivity tensor."""

    trace = conductivity.xx + conductivity.yy
    delta = anisotropy_delta(conductivity)
    hall_symmetric = 0.5 * (conductivity.xy + conductivity.yx)
    hall_antisymmetric = 0.5 * (conductivity.xy - conductivity.yx)
    return {
        "sigma_trace": trace,
        "delta": delta,
        "sigma_xy": conductivity.xy,
        "sigma_yx": conductivity.yx,
        "offdiag_symmetric": hall_symmetric,
        "offdiag_antisymmetric": hall_antisymmetric,
    }


def conductivity_matrix_diagnostics(conductivity: ConductivityTensor) -> dict[str, np.ndarray | complex | float]:
    """Return matrix diagnostics for a 2D conductivity tensor."""

    sigma_matrix = conductivity.matrix()
    eigenvalues, eigenvectors = np.linalg.eig(sigma_matrix)
    offdiag_norm = float(np.linalg.norm([sigma_matrix[0, 1], sigma_matrix[1, 0]]))
    relative_xx_yy_error = 0.0
    diagonal_scale = 0.5 * (abs(conductivity.xx) + abs(conductivity.yy))
    if not np.isclose(diagonal_scale, 0.0):
        relative_xx_yy_error = float(abs(conductivity.xx - conductivity.yy) / diagonal_scale)

    return {
        "sigma_matrix": sigma_matrix,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "anisotropy_delta": anisotropy_delta(conductivity),
        "offdiag_norm": offdiag_norm,
        "relative_xx_yy_error": relative_xx_yy_error,
    }


def fermi_function(energy_eV: np.ndarray, fermi_level_eV: float, temperature_eV: float) -> np.ndarray:
    """Return Fermi occupation for eV energies."""

    shifted = np.asarray(energy_eV, dtype=float) - fermi_level_eV
    if temperature_eV <= 0.0:
        return (shifted < 0.0).astype(float)

    x = np.clip(shifted / temperature_eV, -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)


def negative_fermi_derivative(
    energy_eV: np.ndarray,
    fermi_level_eV: float,
    temperature_eV: float,
    eta_eV: float,
) -> np.ndarray:
    """Return -df/dE in 1/eV with a finite-width zero-temperature fallback."""

    shifted = np.asarray(energy_eV, dtype=float) - fermi_level_eV
    if temperature_eV <= 0.0:
        width = max(eta_eV, 1e-12)
        return width / (np.pi * (shifted**2 + width**2))

    x = np.clip(shifted / (2.0 * temperature_eV), -350.0, 350.0)
    return 1.0 / (4.0 * temperature_eV * np.cosh(x) ** 2)


def bosonic_matsubara_energy_eV(n: int, temperature_K: float) -> float:
    """Return hbar*xi_n = 2*pi*n*kBT in eV."""

    if n < 0:
        raise ValueError("n must be non-negative")
    if temperature_K < 0.0:
        raise ValueError("temperature_K must be non-negative")
    return 2.0 * np.pi * n * temperature_K * KB_EV_PER_K


def uniform_bz_mesh(nkx: int, nky: int | None = None) -> np.ndarray:
    """Return a midpoint uniform mesh over [-pi, pi) x [-pi, pi)."""

    nky = nkx if nky is None else nky
    if nkx <= 0 or nky <= 0:
        raise ValueError("nkx and nky must be positive")
    kx_values = -np.pi + (np.arange(nkx) + 0.5) * (2.0 * np.pi / nkx)
    ky_values = -np.pi + (np.arange(nky) + 0.5) * (2.0 * np.pi / nky)
    return np.array([(kx, ky) for kx in kx_values for ky in ky_values], dtype=float)


def k_weights(k_points: Sequence[tuple[float, float]] | np.ndarray) -> np.ndarray:
    """Return normalized weights for int_BZ d2k/(2pi)^2 on a supplied mesh."""

    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("k_points must have shape (n, 2) and must not be empty")
    return np.full(points.shape[0], 1.0 / points.shape[0])


@dataclass(frozen=True)
class ConductivityEigensystem:
    """Band energies, eigenvectors, occupations, and velocity vertices at one k."""

    energies_eV: np.ndarray
    states: np.ndarray
    occupations: np.ndarray
    negative_fermi_derivative: np.ndarray
    velocity_x_band: np.ndarray
    velocity_y_band: np.ndarray


def conductivity_eigensystem(
    kx: float,
    ky: float,
    config: KuboConfig,
    hamiltonian: HamiltonianBuilder = normal_state_hamiltonian,
    velocity: VelocityBuilder = normal_state_velocity_operator,
) -> ConductivityEigensystem:
    """Diagonalize H(k) and transform velocity vertices to the band basis."""

    h_k = hamiltonian(kx, ky)
    energies, states = np.linalg.eigh(h_k)
    occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
    minus_df = negative_fermi_derivative(
        energies, config.fermi_level_eV, config.temperature_eV, config.eta_eV
    )
    vx = states.conjugate().T @ velocity(kx, ky, "x") @ states
    vy = states.conjugate().T @ velocity(kx, ky, "y") @ states
    return ConductivityEigensystem(energies, states, occupations, minus_df, vx, vy)


def _validate_kubo_inputs(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
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

    if k_weights is None:
        weights = np.full(points.shape[0], 1.0 / points.shape[0])
    else:
        weights = np.asarray(k_weights, dtype=float)
        if weights.shape != (points.shape[0],):
            raise ValueError("k_weights must have shape (n,)")
    return points, weights


def kubo_conductivity_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
    hamiltonian: HamiltonianBuilder = normal_state_hamiltonian,
    velocity: VelocityBuilder = normal_state_velocity_operator,
) -> ConductivityTensor:
    """Compute sigma(i xi) on the imaginary-frequency axis.

    The Brillouin-zone convention is sum_k w_k with sum w_k = 1, equivalent to
    int_BZ d2k/(2pi)^2 for the square [-pi, pi)^2 zone.
    """

    points, weights = _validate_kubo_inputs(k_points, config, k_weights)
    omega = config.omega_eV + config.eta_eV
    sigma = np.zeros((2, 2), dtype=complex)

    for weight, (kx, ky) in zip(weights, points, strict=True):
        bands = conductivity_eigensystem(float(kx), float(ky), config, hamiltonian, velocity)
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


def kubo_conductivity_real_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
    hamiltonian: HamiltonianBuilder = normal_state_hamiltonian,
    velocity: VelocityBuilder = normal_state_velocity_operator,
) -> ConductivityTensor:
    """Compute sigma(omega) on the real-frequency axis with eta broadening."""

    points, weights = _validate_kubo_inputs(k_points, config, k_weights)
    z = config.omega_eV + 1j * config.eta_eV
    drude_denominator = config.eta_eV - 1j * config.omega_eV
    sigma = np.zeros((2, 2), dtype=complex)

    for weight, (kx, ky) in zip(weights, points, strict=True):
        bands = conductivity_eigensystem(float(kx), float(ky), config, hamiltonian, velocity)
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
