"""Conductivity helpers for future Kubo and Casimir calculations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .constants import E2_OVER_HBAR, KB_EV_PER_K
from .model import ground_state_hamiltonian, ground_state_velocity_operator

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
    external imaginary frequency, broadening, and kBT.
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


def fermi_function(energy_eV: np.ndarray, fermi_level_eV: float, temperature_eV: float) -> np.ndarray:
    """Return Fermi occupation for eV energies."""

    shifted = np.asarray(energy_eV, dtype=float) - fermi_level_eV
    if temperature_eV <= 0.0:
        return (shifted < 0.0).astype(float)

    x = np.clip(shifted / temperature_eV, -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)


def bosonic_matsubara_energy_eV(n: int, temperature_K: float) -> float:
    """Return hbar*xi_n = 2*pi*n*kBT in eV."""

    if n < 0:
        raise ValueError("n must be non-negative")
    if temperature_K < 0.0:
        raise ValueError("temperature_K must be non-negative")
    return 2.0 * np.pi * n * temperature_K * KB_EV_PER_K


def kubo_conductivity(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
    hamiltonian: HamiltonianBuilder = ground_state_hamiltonian,
    velocity: VelocityBuilder = ground_state_velocity_operator,
) -> ConductivityTensor:
    """Compute the 2D sheet conductivity tensor from a band-basis Kubo sum.

    The velocity vertices are dH/dkx and dH/dky in eV for dimensionless crystal
    momenta. The implemented response uses the real imaginary-axis kernel
    -(f_m-f_n)(E_m-E_n)/[(E_m-E_n)^2+Omega^2].
    With ``output_si=True`` the dimensionless response is multiplied by e^2/hbar.
    """

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

    omega = config.omega_eV + config.eta_eV
    sigma = np.zeros((2, 2), dtype=complex)

    for weight, (kx, ky) in zip(weights, points, strict=True):
        h_k = hamiltonian(float(kx), float(ky))
        energies, states = np.linalg.eigh(h_k)
        occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
        velocity_bands = []
        for direction in ("x", "y"):
            vertex = velocity(float(kx), float(ky), direction)
            velocity_bands.append(states.conjugate().T @ vertex @ states)

        for m, energy_m in enumerate(energies):
            for n, energy_n in enumerate(energies):
                if m == n:
                    continue
                occupation_diff = occupations[m] - occupations[n]
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
