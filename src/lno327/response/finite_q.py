"""Model-independent finite-q response algebra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lno327.bdg.spectrum import diagonalize_hermitian, transform_operator_to_band_basis
from lno327.response.config import KuboConfig
from lno327.response.occupations import fermi_function


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


def fermi_derivative(
    energy_eV: float,
    fermi_level_eV: float,
    temperature_eV: float,
    eta_eV: float,
) -> float:
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
    """Return forward finite-q vertex matrix elements in minus-plus storage.

    The matrix vertex is defined by Psi†_{k+q/2} Gamma(k,q) Psi_{k-q/2}.
    For loop indices m in the k-q/2 band and n in the k+q/2 band, the bubble
    needs <n,+|Gamma(k,q)|m,->.  We store those elements as [m, n] so the Kubo
    loops can remain ordered by energies_minus then energies_plus.
    """

    return (states_plus.conjugate().T @ vertex @ states_minus).T


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


def add_band_bubble(
    accumulator: np.ndarray,
    left_band_vertices: tuple[np.ndarray, ...],
    right_band_vertices: tuple[np.ndarray, ...],
    energies_minus: np.ndarray,
    occupations_minus: np.ndarray,
    energies_plus: np.ndarray,
    occupations_plus: np.ndarray,
    omega_eV: float,
    weight: float,
    config: KuboConfig | None = None,
    static_limit: bool = False,
    prefactor: float = 0.5,
) -> None:
    """Accumulate a bubble from forward vertices stored as [minus, plus]."""

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
            factor = float(prefactor) * float(weight) * raw_factor
            for mu, left in enumerate(left_band_vertices):
                for nu, right in enumerate(right_band_vertices):
                    accumulator[mu, nu] += factor * left[m, n] * np.conjugate(right[m, n])


def thermal_expectation_bdg_from_hamiltonian(
    hamiltonian: np.ndarray,
    vertex: np.ndarray,
    config: KuboConfig,
    *,
    prefactor: float = 0.5,
) -> complex:
    bands = diagonalize_hermitian(hamiltonian)
    occupations = fermi_function(bands.energies, config.fermi_level_eV, config.temperature_eV)
    vertex_in_band = transform_operator_to_band_basis(bands.states, vertex)
    return complex(prefactor * np.sum(occupations * np.diag(vertex_in_band)))


def thermal_expectation_bdg_from_model(
    spec,
    kx: float,
    ky: float,
    channel: str,
    vertex: np.ndarray,
    config: KuboConfig,
    *,
    prefactor: float = 0.5,
) -> complex:
    hamiltonian = spec.bdg_hamiltonian(kx, ky, channel)
    return thermal_expectation_bdg_from_hamiltonian(
        hamiltonian,
        vertex,
        config,
        prefactor=prefactor,
    )
