"""Shared Matsubara-frequency helpers for validation commands."""
from __future__ import annotations

import numpy as np

from lno327.constants import KB_EV_PER_K


def matsubara_energy_eV(index: int, temperature_K: float) -> float:
    """Return hbar*xi_n in eV, namely 2*pi*n*k_B*T, including n=0."""

    n = int(index)
    temperature = float(temperature_K)
    if n < 0:
        raise ValueError("Matsubara index must be non-negative")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    return float(2.0 * np.pi * n * KB_EV_PER_K * temperature)


__all__ = ["matsubara_energy_eV"]
