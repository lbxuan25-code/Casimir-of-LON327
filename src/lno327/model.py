"""Normal-state bilayer two-orbital Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ORBITAL_BASIS = ("dz1", "dx1", "dz2", "dx2")


@dataclass(frozen=True)
class NormalStateParameters:
    """Tight-binding coefficients in eV for basis dz1, dx1, dz2, dx2.

    The adopted normal-state matrix is:
    H(k) = [[H_parallel, H_perp], [H_perp, H_parallel]] - mu I.
    """

    chemical_potential: float = 0.05
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


def normal_state_hamiltonian(
    kx: float,
    ky: float,
    params: NormalStateParameters | None = None,
) -> np.ndarray:
    """Return the 4x4 normal-state Hamiltonian H_t(k) in eV.

    The returned matrix is spin independent and contains only the adopted
    tight-binding normal-state model.
    """

    params = params or NormalStateParameters()
    cx, cy, c2x, c2y, c3x, c3y = _cos_terms(kx, ky)

    tz = (
        params.tz_1 * (cx + cy)
        + params.tz_2 * cx * cy
        + params.tz_3 * (c2x + c2y)
        + params.tz_4 * (c3x + c3y)
        + params.tz_0
    )
    tx = (
        params.tx_1 * (cx + cy)
        + params.tx_2 * cx * cy
        + params.tx_3 * (c2x + c2y)
        + params.tx_4 * (c3x + c3y)
        + params.tx_0
    )
    tz_perp = params.tz_perp_0 + params.tz_perp_1 * (cx + cy)
    tx_perp = params.tx_perp_0
    vxz = params.vxz_1 * (cx - cy) + params.vxz_2 * (c2x - c2y)
    vxz_perp = params.vxz_perp_1 * (cx - cy)

    h_parallel = np.array([[tz, vxz], [vxz, tx]], dtype=float)
    h_perp = np.array([[tz_perp, vxz_perp], [vxz_perp, tx_perp]], dtype=float)
    h = np.block([[h_parallel, h_perp], [h_perp, h_parallel]])
    return h - params.chemical_potential * np.eye(4)


def normal_state_velocity_operator(
    kx: float,
    ky: float,
    direction: str,
    params: NormalStateParameters | None = None,
) -> np.ndarray:
    """Return the Kubo velocity vertex dH/dk_direction in eV.

    The crystal momenta are dimensionless lattice momenta. Until a lattice
    constant is fixed, this derivative is the natural current vertex for the
    dimensionless Brillouin-zone formulation rather than a velocity in m/s.
    """

    if direction not in {"x", "y"}:
        raise ValueError("direction must be 'x' or 'y'")

    params = params or NormalStateParameters()
    cx, cy, _, _, _, _ = _cos_terms(kx, ky)
    sx, sy, s2x, s2y, s3x, s3y = _sin_terms(kx, ky)

    if direction == "x":
        dtz = (
            -params.tz_1 * sx
            - params.tz_2 * sx * cy
            - 2.0 * params.tz_3 * s2x
            - 3.0 * params.tz_4 * s3x
        )
        dtx = (
            -params.tx_1 * sx
            - params.tx_2 * sx * cy
            - 2.0 * params.tx_3 * s2x
            - 3.0 * params.tx_4 * s3x
        )
        dtz_perp = -params.tz_perp_1 * sx
        dvxz = -params.vxz_1 * sx - 2.0 * params.vxz_2 * s2x
        dvxz_perp = -params.vxz_perp_1 * sx
    else:
        dtz = (
            -params.tz_1 * sy
            - params.tz_2 * cx * sy
            - 2.0 * params.tz_3 * s2y
            - 3.0 * params.tz_4 * s3y
        )
        dtx = (
            -params.tx_1 * sy
            - params.tx_2 * cx * sy
            - 2.0 * params.tx_3 * s2y
            - 3.0 * params.tx_4 * s3y
        )
        dtz_perp = -params.tz_perp_1 * sy
        dvxz = params.vxz_1 * sy + 2.0 * params.vxz_2 * s2y
        dvxz_perp = params.vxz_perp_1 * sy

    dtx_perp = 0.0
    dh_parallel = np.array([[dtz, dvxz], [dvxz, dtx]], dtype=float)
    dh_perp = np.array([[dtz_perp, dvxz_perp], [dvxz_perp, dtx_perp]], dtype=float)
    return np.block([[dh_parallel, dh_perp], [dh_perp, dh_parallel]])


def normal_state_velocity_operators(kx: float, ky: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (dH/dkx, dH/dky), both in eV."""

    return (
        normal_state_velocity_operator(kx, ky, "x"),
        normal_state_velocity_operator(kx, ky, "y"),
    )
