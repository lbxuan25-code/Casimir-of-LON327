"""Collective-channel adapter for existing pairing ansatz objects."""

from __future__ import annotations

import numpy as np


def collective_vertices(ansatz: object, kx: float, ky: float, qx: float, qy: float, pairing_params: object) -> tuple[np.ndarray, ...]:
    """Return amplitude/phase collective vertices from the existing ansatz."""

    vertices = ansatz.collective_vertices(kx, ky, qx, qy, pairing_params)
    return tuple(np.asarray(vertex, dtype=complex) for vertex in vertices)


def collective_counterterm(ansatz: object, config: object, k_points: np.ndarray, weights: np.ndarray, pairing_params: object) -> np.ndarray:
    """Return the existing Hubbard-Stratonovich collective counterterm matrix."""

    return np.asarray(ansatz.hs_counterterm(config, k_points, weights, pairing_params), dtype=complex)

