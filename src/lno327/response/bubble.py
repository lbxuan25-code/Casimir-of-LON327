"""Band-basis bubble algebra for imaginary-axis response kernels."""

from __future__ import annotations

import numpy as np


def response_factor_imag_axis(
    energy_m: float,
    energy_n: float,
    occupation_m: float,
    occupation_n: float,
    negative_df_m: float,
    omega_eV: float,
    eta_eV: float,
    *,
    same_state: bool,
) -> float:
    if same_state:
        return float(negative_df_m)

    occupation_diff = occupation_m - occupation_n
    if np.isclose(occupation_diff, 0.0):
        return 0.0
    energy_diff = energy_m - energy_n
    if abs(energy_diff) < eta_eV:
        return 0.0
    omega = omega_eV + eta_eV
    return float(-occupation_diff * energy_diff / (energy_diff**2 + omega**2))


def band_basis_bubble_imag_axis(
    energies: np.ndarray,
    occupations: np.ndarray,
    negative_fermi_derivative: np.ndarray,
    vertices: tuple[np.ndarray, ...],
    omega_eV: float,
    eta_eV: float,
    *,
    prefactor: float = 1.0,
) -> np.ndarray:
    energy_values = np.asarray(energies, dtype=float)
    occupation_values = np.asarray(occupations, dtype=float)
    minus_df_values = np.asarray(negative_fermi_derivative, dtype=float)
    if energy_values.ndim != 1:
        raise ValueError("energies must be a one-dimensional array")
    if occupation_values.shape != energy_values.shape:
        raise ValueError("occupations must have the same shape as energies")
    if minus_df_values.shape != energy_values.shape:
        raise ValueError("negative_fermi_derivative must have the same shape as energies")
    if len(vertices) == 0:
        raise ValueError("vertices must not be empty")

    dim = energy_values.shape[0]
    vertex_matrices = tuple(np.asarray(vertex) for vertex in vertices)
    for vertex in vertex_matrices:
        if vertex.shape != (dim, dim):
            raise ValueError("each vertex must have shape (n_bands, n_bands)")

    kernel = np.zeros((len(vertex_matrices), len(vertex_matrices)), dtype=complex)
    for m, energy_m in enumerate(energy_values):
        for n, energy_n in enumerate(energy_values):
            factor = response_factor_imag_axis(
                float(energy_m),
                float(energy_n),
                float(occupation_values[m]),
                float(occupation_values[n]),
                float(minus_df_values[m]),
                omega_eV,
                eta_eV,
                same_state=m == n,
            )
            if factor == 0.0:
                continue
            for alpha, vertex_alpha in enumerate(vertex_matrices):
                for beta, vertex_beta in enumerate(vertex_matrices):
                    kernel[alpha, beta] += factor * vertex_alpha[m, n] * vertex_beta[n, m]

    return prefactor * kernel


def two_sided_response_factor_imag_axis(
    energy_left: float,
    energy_right: float,
    occupation_left: float,
    occupation_right: float,
    omega_eV: float,
    eta_eV: float,
) -> float:
    delta_energy = energy_left - energy_right
    delta_occupation = occupation_left - occupation_right
    denominator = delta_energy**2 + omega_eV**2
    if denominator <= eta_eV**2:
        return 0.0
    return float(-delta_occupation * delta_energy / denominator)


def two_sided_band_basis_bubble_imag_axis(
    energies_left: np.ndarray,
    energies_right: np.ndarray,
    occupations_left: np.ndarray,
    occupations_right: np.ndarray,
    vertices_left_right: tuple[np.ndarray, ...],
    omega_eV: float,
    eta_eV: float,
    *,
    prefactor: float = 1.0,
) -> np.ndarray:
    left_energies = np.asarray(energies_left, dtype=float)
    right_energies = np.asarray(energies_right, dtype=float)
    left_occupations = np.asarray(occupations_left, dtype=float)
    right_occupations = np.asarray(occupations_right, dtype=float)
    if left_energies.ndim != 1:
        raise ValueError("energies_left must be a one-dimensional array")
    if right_energies.ndim != 1:
        raise ValueError("energies_right must be a one-dimensional array")
    if left_occupations.shape != left_energies.shape:
        raise ValueError("occupations_left must have the same shape as energies_left")
    if right_occupations.shape != right_energies.shape:
        raise ValueError("occupations_right must have the same shape as energies_right")
    if len(vertices_left_right) == 0:
        raise ValueError("vertices_left_right must not be empty")

    n_left = left_energies.shape[0]
    n_right = right_energies.shape[0]
    vertex_matrices = tuple(np.asarray(vertex) for vertex in vertices_left_right)
    for vertex in vertex_matrices:
        if vertex.shape != (n_left, n_right):
            raise ValueError("each vertex must have shape (n_left, n_right)")

    kernel = np.zeros((len(vertex_matrices), len(vertex_matrices)), dtype=complex)
    for m, energy_left in enumerate(left_energies):
        for n, energy_right in enumerate(right_energies):
            factor = two_sided_response_factor_imag_axis(
                float(energy_left),
                float(energy_right),
                float(left_occupations[m]),
                float(right_occupations[n]),
                omega_eV,
                eta_eV,
            )
            if factor == 0.0:
                continue
            for alpha, vertex_alpha in enumerate(vertex_matrices):
                for beta, vertex_beta in enumerate(vertex_matrices):
                    kernel[alpha, beta] += (
                        factor * vertex_alpha[m, n] * np.conjugate(vertex_beta[m, n])
                    )

    return prefactor * kernel
