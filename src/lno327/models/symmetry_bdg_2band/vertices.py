"""Analytic q=0 derivative vertices for the symmetry-focused two-band model."""

from __future__ import annotations

import numpy as np

from lno327.models.symmetry_bdg_2band.normal import TAU0, TAUX, TAUZ
from lno327.models.symmetry_bdg_2band.parameters import TwoBandParameters


def _validate_direction(direction: str) -> None:
    if direction not in {"x", "y"}:
        raise ValueError("direction must be 'x' or 'y'")


def velocity_operator(
    kx: float,
    ky: float,
    direction: str,
    params: TwoBandParameters | None = None,
) -> np.ndarray:
    """Return dh/dk_direction."""

    _validate_direction(direction)
    params = params or TwoBandParameters()
    cx = np.cos(kx)
    cy = np.cos(ky)
    sx = np.sin(kx)
    sy = np.sin(ky)

    if direction == "x":
        dxi0 = 2.0 * params.t * sx + 4.0 * params.tp * sx * cy
        dxix = -2.0 * params.t_perp_p * sx
        dxiz = 2.0 * params.t_z * sx
    else:
        dxi0 = 2.0 * params.t * sy + 4.0 * params.tp * cx * sy
        dxix = -2.0 * params.t_perp_p * sy
        dxiz = 2.0 * params.t_z * sy
    return dxi0 * TAU0 + dxix * TAUX + dxiz * TAUZ


def mass_operator(
    kx: float,
    ky: float,
    i: str,
    j: str,
    params: TwoBandParameters | None = None,
) -> np.ndarray:
    """Return d2h/dk_i dk_j."""

    _validate_direction(i)
    _validate_direction(j)
    params = params or TwoBandParameters()
    cx = np.cos(kx)
    cy = np.cos(ky)
    sx = np.sin(kx)
    sy = np.sin(ky)

    if i != j:
        d2xi0 = -4.0 * params.tp * sx * sy
        d2xix = 0.0
        d2xiz = 0.0
    elif i == "x":
        d2xi0 = 2.0 * params.t * cx + 4.0 * params.tp * cx * cy
        d2xix = -2.0 * params.t_perp_p * cx
        d2xiz = 2.0 * params.t_z * cx
    else:
        d2xi0 = 2.0 * params.t * cy + 4.0 * params.tp * cx * cy
        d2xix = -2.0 * params.t_perp_p * cy
        d2xiz = 2.0 * params.t_z * cy
    return d2xi0 * TAU0 + d2xix * TAUX + d2xiz * TAUZ
