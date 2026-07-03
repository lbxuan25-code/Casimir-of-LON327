"""Matsubara-frequency numerical helpers."""

from __future__ import annotations

import numpy as np

from lno327.constants import KB_EV_PER_K


def bosonic_matsubara_energy_eV(n: int, temperature_K: float) -> float:
    """Return hbar*xi_n = 2*pi*n*kBT in eV."""

    if n < 0:
        raise ValueError("n must be non-negative")
    if temperature_K < 0.0:
        raise ValueError("temperature_K must be non-negative")
    return 2.0 * np.pi * n * temperature_K * KB_EV_PER_K
