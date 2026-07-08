"""Matsubara frequency conventions for the TM/TE sandbox."""

from __future__ import annotations

import numbers

import numpy as np

K_B_EV_PER_K = 8.617_333_262_145e-5


def matsubara_xi_eV(n: int, temperature_K: float) -> float:
    """Return xi_eV = 2*pi*n*k_B*T for a bosonic Matsubara index."""

    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise ValueError("matsubara index n must be an integer")
    index = int(n)
    if index < 0:
        raise ValueError("matsubara index n must be non-negative")
    temperature = float(temperature_K)
    if temperature <= 0.0:
        raise ValueError("temperature_K must be positive")
    if index == 0:
        return 0.0
    return float(2.0 * np.pi * index * K_B_EV_PER_K * temperature)


def frequency_payload(n: int, temperature_K: float) -> dict[str, object]:
    """Return public JSON frequency metadata."""

    xi_eV = matsubara_xi_eV(n, temperature_K)
    return {
        "source": "matsubara_index",
        "matsubara_index": int(n),
        "temperature_K": float(temperature_K),
        "xi_eV": xi_eV,
        "zero_matsubara_mode": int(n) == 0,
        "valid_for_casimir_input": False,
    }
