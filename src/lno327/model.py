"""Project ground-state bilayer two-orbital Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ORBITAL_BASIS = ("dz1", "dx1", "dz2", "dx2")


@dataclass(frozen=True)
class GroundStateExchangeParameters:
    """Exchange and filling parameters for the adopted ground-state model."""

    chemical_potential: float = 0.05
    j_perp_dz: float = 0.135
    j_parallel_dx: float = 0.084
    j_parallel_xz: float = 0.03
    hund: float = 1.0
    filling_dz: float = 0.8
    filling_dx: float = 0.58


@dataclass(frozen=True)
class GroundStateTightBindingParameters:
    """Tight-binding coefficients in eV for basis dz1, dx1, dz2, dx2.

    The adopted normal-state matrix is:
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


def _sin_terms(kx: float, ky: float) -> tuple[float, float, float, float, float, float]:
    sx = np.sin(kx)
    sy = np.sin(ky)
    s2x = np.sin(2.0 * kx)
    s2y = np.sin(2.0 * ky)
    s3x = np.sin(3.0 * kx)
    s3y = np.sin(3.0 * ky)
    return sx, sy, s2x, s2y, s3x, s3y


def ground_state_hamiltonian(
    kx: float,
    ky: float,
    tb: GroundStateTightBindingParameters | None = None,
    exchange: GroundStateExchangeParameters | None = None,
) -> np.ndarray:
    """Return the 4x4 normal-state Hamiltonian H_t(k) in eV.

    The returned matrix is spin independent and omits RMFT renormalization factors.
    """

    tb = tb or GroundStateTightBindingParameters()
    exchange = exchange or GroundStateExchangeParameters()
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


def ground_state_velocity_operator(
    kx: float,
    ky: float,
    direction: str,
    tb: GroundStateTightBindingParameters | None = None,
) -> np.ndarray:
    """Return the Kubo velocity vertex dH/dk_direction in eV.

    The crystal momenta are dimensionless lattice momenta. Until a lattice
    constant is fixed, this derivative is the natural current vertex for the
    dimensionless Brillouin-zone formulation rather than a velocity in m/s.
    """

    if direction not in {"x", "y"}:
        raise ValueError("direction must be 'x' or 'y'")

    tb = tb or GroundStateTightBindingParameters()
    cx, cy, _, _, _, _ = _cos_terms(kx, ky)
    sx, sy, s2x, s2y, s3x, s3y = _sin_terms(kx, ky)

    if direction == "x":
        dtz = (
            -tb.tz_1 * sx
            - tb.tz_2 * sx * cy
            - 2.0 * tb.tz_3 * s2x
            - 3.0 * tb.tz_4 * s3x
        )
        dtx = (
            -tb.tx_1 * sx
            - tb.tx_2 * sx * cy
            - 2.0 * tb.tx_3 * s2x
            - 3.0 * tb.tx_4 * s3x
        )
        dtz_perp = -tb.tz_perp_1 * sx
        dvxz = -tb.vxz_1 * sx - 2.0 * tb.vxz_2 * s2x
        dvxz_perp = -tb.vxz_perp_1 * sx
    else:
        dtz = (
            -tb.tz_1 * sy
            - tb.tz_2 * cx * sy
            - 2.0 * tb.tz_3 * s2y
            - 3.0 * tb.tz_4 * s3y
        )
        dtx = (
            -tb.tx_1 * sy
            - tb.tx_2 * cx * sy
            - 2.0 * tb.tx_3 * s2y
            - 3.0 * tb.tx_4 * s3y
        )
        dtz_perp = -tb.tz_perp_1 * sy
        dvxz = tb.vxz_1 * sy + 2.0 * tb.vxz_2 * s2y
        dvxz_perp = tb.vxz_perp_1 * sy

    dtx_perp = 0.0
    dh_parallel = np.array([[dtz, dvxz], [dvxz, dtx]], dtype=float)
    dh_perp = np.array([[dtz_perp, dvxz_perp], [dvxz_perp, dtx_perp]], dtype=float)
    return np.block([[dh_parallel, dh_perp], [dh_perp, dh_parallel]])


def ground_state_velocity_operators(kx: float, ky: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (dH/dkx, dH/dky), both in eV."""

    return (
        ground_state_velocity_operator(kx, ky, "x"),
        ground_state_velocity_operator(kx, ky, "y"),
    )
