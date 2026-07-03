"""Normal-state bilayer two-orbital Hamiltonian."""

from __future__ import annotations

import numpy as np

from lno327.models.lno327_four_orbital.parameters import NormalStateParameters


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
    """Return the 4x4 normal-state Hamiltonian H_t(k) in eV."""

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
