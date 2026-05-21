"""Qiu et al. bilayer two-orbital normal-state Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ORBITAL_BASIS = ("dz1", "dx1", "dz2", "dx2")


@dataclass(frozen=True)
class QiuExchangeParameters:
    """Exchange and filling parameters quoted in Qiu et al."""

    chemical_potential: float = 0.05
    j_perp_dz: float = 0.135
    j_parallel_dx: float = 0.084
    j_parallel_xz: float = 0.03
    hund: float = 1.0
    filling_dz: float = 0.8
    filling_dx: float = 0.58


@dataclass(frozen=True)
class QiuTightBindingParameters:
    """Appendix-A tight-binding coefficients in eV for basis dz1, dx1, dz2, dx2.

    The normal-state matrix follows Qiu et al. Appendix A:
    H(k) = [[H_parallel, H_perp], [H_perp, H_parallel]] - mu I.
    """

    tz_1: float = -0.217
    tz_2: float = -0.073
    tz_3: float = -0.021
    tz_4: float = -0.005
    tz_0: float = 0.431
    tx_1: float = -0.922
    tx_2: float = 0.301
    tx_3: float = -0.108
    tx_4: float = -0.025
    tx_0: float = 0.881
    tz_perp_0: float = -0.550
    tz_perp_1: float = 0.041
    tx_perp_0: float = 0.005
    vxz_1: float = 0.429
    vxz_2: float = 0.041
    vxz_perp_1: float = -0.061


def _cos_terms(kx: float, ky: float) -> tuple[float, float, float, float, float, float]:
    cx = np.cos(kx)
    cy = np.cos(ky)
    c2x = np.cos(2.0 * kx)
    c2y = np.cos(2.0 * ky)
    c3x = np.cos(3.0 * kx)
    c3y = np.cos(3.0 * ky)
    return cx, cy, c2x, c2y, c3x, c3y


def qiu_bilayer_hamiltonian(
    kx: float,
    ky: float,
    tb: QiuTightBindingParameters | None = None,
    exchange: QiuExchangeParameters | None = None,
) -> np.ndarray:
    """Return the 4x4 normal-state Hamiltonian H_t(k) in eV.

    The returned matrix is spin independent and omits RMFT renormalization factors.
    """

    tb = tb or QiuTightBindingParameters()
    exchange = exchange or QiuExchangeParameters()
    cx, cy, c2x, c2y, c3x, c3y = _cos_terms(kx, ky)

    tz = (
        tb.tz_1 * (cx + cy)
        + tb.tz_2 * cx * cy
        + tb.tz_3 * (c2x + c2y)
        + tb.tz_4 * (c3x + c3y)
        + tb.tz_0
    )
    tx = (
        tb.tx_1 * (cx + cy)
        + tb.tx_2 * cx * cy
        + tb.tx_3 * (c2x + c2y)
        + tb.tx_4 * (c3x + c3y)
        + tb.tx_0
    )
    tz_perp = tb.tz_perp_0 + tb.tz_perp_1 * (cx + cy)
    tx_perp = tb.tx_perp_0
    vxz = tb.vxz_1 * (cx - cy) + tb.vxz_2 * (c2x - c2y)
    vxz_perp = tb.vxz_perp_1 * (cx - cy)

    h_parallel = np.array([[tz, vxz], [vxz, tx]], dtype=float)
    h_perp = np.array([[tz_perp, vxz_perp], [vxz_perp, tx_perp]], dtype=float)
    h = np.block([[h_parallel, h_perp], [h_perp, h_parallel]])
    return h - exchange.chemical_potential * np.eye(4)
