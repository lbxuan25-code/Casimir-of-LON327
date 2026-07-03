"""Numerical quadrature weights."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def k_weights(k_points: Sequence[tuple[float, float]] | np.ndarray) -> np.ndarray:
    """Return normalized weights for int_BZ d2k/(2pi)^2 on a supplied mesh."""

    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("k_points must have shape (n, 2) and must not be empty")
    return np.full(points.shape[0], 1.0 / points.shape[0])
