"""Occupation factors shared by response kernels."""

from __future__ import annotations

import numpy as np


def fermi_function(
    energies: np.ndarray,
    fermi_level_eV: float,
    temperature_eV: float,
) -> np.ndarray:
    shifted = np.asarray(energies, dtype=float) - fermi_level_eV
    if temperature_eV <= 0.0:
        return (shifted < 0.0).astype(float)

    x = np.clip(shifted / temperature_eV, -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)


def negative_fermi_derivative(
    energies: np.ndarray,
    fermi_level_eV: float,
    temperature_eV: float,
    eta_eV: float,
) -> np.ndarray:
    shifted = np.asarray(energies, dtype=float) - fermi_level_eV
    if temperature_eV <= 0.0:
        width = max(eta_eV, 1e-12)
        return width / (np.pi * (shifted**2 + width**2))

    x = np.clip(shifted / (2.0 * temperature_eV), -350.0, 350.0)
    return 1.0 / (4.0 * temperature_eV * np.cosh(x) ** 2)


def occupation_difference(occupations: np.ndarray) -> np.ndarray:
    values = np.asarray(occupations)
    if values.ndim != 1:
        raise ValueError("occupations must be a one-dimensional array")
    return values[:, np.newaxis] - values[np.newaxis, :]
