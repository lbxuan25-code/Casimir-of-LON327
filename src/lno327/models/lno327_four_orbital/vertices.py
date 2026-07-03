"""Normal-state derivative vertices."""

from __future__ import annotations

import numpy as np

from lno327.models.lno327_four_orbital.normal import _cos_terms, _sin_terms
from lno327.models.lno327_four_orbital.parameters import NormalStateParameters


def normal_state_velocity_operator(
    kx: float,
    ky: float,
    direction: str,
    params: NormalStateParameters | None = None,
) -> np.ndarray:
    """Return the Kubo velocity vertex dH/dk_direction in eV."""

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


def normal_state_mass_operator(
    kx: float,
    ky: float,
    direction_a: str,
    direction_b: str,
    params: NormalStateParameters | None = None,
) -> np.ndarray:
    """Return d2H/dk_direction_a dk_direction_b in eV."""

    if direction_a not in {"x", "y"} or direction_b not in {"x", "y"}:
        raise ValueError("directions must be 'x' or 'y'")

    params = params or NormalStateParameters()
    cx, cy, c2x, c2y, c3x, c3y = _cos_terms(kx, ky)
    sx, sy, _, _, _, _ = _sin_terms(kx, ky)

    if direction_a != direction_b:
        dtz = params.tz_2 * sx * sy
        dtx = params.tx_2 * sx * sy
        dtz_perp = 0.0
        dtx_perp = 0.0
        dvxz = 0.0
        dvxz_perp = 0.0
    elif direction_a == "x":
        dtz = (
            -params.tz_1 * cx
            - params.tz_2 * cx * cy
            - 4.0 * params.tz_3 * c2x
            - 9.0 * params.tz_4 * c3x
        )
        dtx = (
            -params.tx_1 * cx
            - params.tx_2 * cx * cy
            - 4.0 * params.tx_3 * c2x
            - 9.0 * params.tx_4 * c3x
        )
        dtz_perp = -params.tz_perp_1 * cx
        dtx_perp = 0.0
        dvxz = -params.vxz_1 * cx - 4.0 * params.vxz_2 * c2x
        dvxz_perp = -params.vxz_perp_1 * cx
    else:
        dtz = (
            -params.tz_1 * cy
            - params.tz_2 * cx * cy
            - 4.0 * params.tz_3 * c2y
            - 9.0 * params.tz_4 * c3y
        )
        dtx = (
            -params.tx_1 * cy
            - params.tx_2 * cx * cy
            - 4.0 * params.tx_3 * c2y
            - 9.0 * params.tx_4 * c3y
        )
        dtz_perp = -params.tz_perp_1 * cy
        dtx_perp = 0.0
        dvxz = params.vxz_1 * cy + 4.0 * params.vxz_2 * c2y
        dvxz_perp = params.vxz_perp_1 * cy

    dh_parallel = np.array([[dtz, dvxz], [dvxz, dtx]], dtype=float)
    dh_perp = np.array([[dtz_perp, dvxz_perp], [dvxz_perp, dtx_perp]], dtype=float)
    return np.block([[dh_parallel, dh_perp], [dh_perp, dh_parallel]])


def normal_state_mass_operators(kx: float, ky: float) -> dict[tuple[str, str], np.ndarray]:
    """Return all xx, xy, yx, yy normal-state mass operators."""

    return {
        ("x", "x"): normal_state_mass_operator(kx, ky, "x", "x"),
        ("x", "y"): normal_state_mass_operator(kx, ky, "x", "y"),
        ("y", "x"): normal_state_mass_operator(kx, ky, "y", "x"),
        ("y", "y"): normal_state_mass_operator(kx, ky, "y", "y"),
    }
