"""Momentum-space grid helpers."""

from __future__ import annotations

import numpy as np


def uniform_bz_mesh(nkx: int, nky: int | None = None) -> np.ndarray:
    """Return a midpoint uniform mesh over [-pi, pi) x [-pi, pi)."""

    nky = nkx if nky is None else nky
    if nkx <= 0 or nky <= 0:
        raise ValueError("nkx and nky must be positive")
    kx_values = -np.pi + (np.arange(nkx) + 0.5) * (2.0 * np.pi / nkx)
    ky_values = -np.pi + (np.arange(nky) + 0.5) * (2.0 * np.pi / nky)
    return np.array([(kx, ky) for kx in kx_values for ky in ky_values], dtype=float)
