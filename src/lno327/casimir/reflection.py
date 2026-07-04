"""Weak-coupling 2D reflection matrix used by the Casimir integrand."""

from __future__ import annotations

import numpy as np

from lno327.constants import C0, SIGMA0
from lno327.electrodynamics.conductivity import ConductivityTensor, anisotropy_delta


def reflection_matrix_weak_2d(
    xi: float,
    k_parallel: float,
    phi: float,
    conductivity: ConductivityTensor,
) -> np.ndarray:
    """Return a weak-coupling 2D reflection matrix using delta/sigma_t variables."""

    sigma_t = (conductivity.xx + conductivity.yy) / SIGMA0
    delta = anisotropy_delta(conductivity)
    kappa = np.sqrt((xi / C0) ** 2 + k_parallel**2)
    cos2 = np.cos(2.0 * phi)
    sin2 = np.sin(2.0 * phi)
    denom = (
        (xi**2 / C0**2) * sigma_t * (1.0 - delta * cos2)
        + (xi * kappa / (4.0 * C0)) * sigma_t**2 * (1.0 - delta**2)
        + kappa**2 * sigma_t * (1.0 + delta * cos2)
        + 4.0 * xi * kappa / C0
    )
    if np.isclose(denom, 0.0):
        raise ZeroDivisionError("reflection denominator is numerically zero")
    rss = (
        -((xi**2 / C0**2) * sigma_t * (1.0 - delta * cos2))
        - (xi * kappa / (4.0 * C0)) * sigma_t**2 * (1.0 - delta**2)
    ) / denom
    rpp = (
        (xi * kappa / (4.0 * C0)) * sigma_t**2 * (1.0 - delta**2)
        + kappa**2 * sigma_t * (1.0 + delta * cos2)
    ) / denom
    rps = -(xi * kappa / C0) * sigma_t * delta * sin2 / denom
    rsp = (xi * kappa / C0) * sigma_t * delta * sin2 / denom
    return np.array([[rss, rsp], [rps, rpp]], dtype=complex)
