"""Normal-state Hamiltonian for the symmetry-focused two-band model."""

from __future__ import annotations

import numpy as np

from lno327.models.symmetry_bdg_2band.parameters import TwoBandParameters

TAU0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
TAUX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
TAUZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def normal_coefficients(
    kx: float,
    ky: float,
    params: TwoBandParameters | None = None,
) -> tuple[float, float, float]:
    params = params or TwoBandParameters()
    cx = np.cos(kx)
    cy = np.cos(ky)
    xi0 = -2.0 * params.t * (cx + cy) - 4.0 * params.tp * cx * cy - params.mu
    xix = params.t_perp + 2.0 * params.t_perp_p * (cx + cy)
    xiz = params.m - 2.0 * params.t_z * (cx + cy)
    return float(xi0), float(xix), float(xiz)


def normal_hamiltonian(
    kx: float,
    ky: float,
    params: TwoBandParameters | None = None,
) -> np.ndarray:
    """Return h(k) = xi0 tau0 + xix taux + xiz tauz."""

    xi0, xix, xiz = normal_coefficients(kx, ky, params)
    return xi0 * TAU0 + xix * TAUX + xiz * TAUZ
