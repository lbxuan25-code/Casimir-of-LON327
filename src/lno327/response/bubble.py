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
