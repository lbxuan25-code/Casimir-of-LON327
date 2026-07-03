"""Pairing matrices for the symmetry-focused two-band BdG model."""

from __future__ import annotations

import numpy as np

from lno327.models.symmetry_bdg_2band.normal import TAU0, TAUX
from lno327.models.symmetry_bdg_2band.parameters import PairingChannel, TwoBandParameters


def d_wave_form_factor(kx: float, ky: float) -> float:
    return float(0.5 * (np.cos(kx) - np.cos(ky)))


def pairing_matrix(
    channel: PairingChannel,
    kx: float,
    ky: float,
    params: TwoBandParameters | None = None,
) -> np.ndarray:
    """Return the 2x2 pairing matrix for the requested channel."""

    params = params or TwoBandParameters()
    if channel == "normal":
        return np.zeros((2, 2), dtype=complex)
    if channel == "spp":
        return params.delta_s * TAU0
    if channel == "spm":
        return params.delta_s * TAUX
    if channel == "dwave":
        return params.delta_d * d_wave_form_factor(kx, ky) * TAU0
    raise ValueError("channel must be 'normal', 'spp', 'spm', or 'dwave'")
