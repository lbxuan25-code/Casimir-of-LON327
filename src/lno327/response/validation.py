"""Validation helpers shared by response entry points."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lno327.response.config import KuboConfig


def validate_k_points_and_weights(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("k_points must have shape (n, 2)")
    if points.shape[0] == 0:
        raise ValueError("k_points must not be empty")
    if config.omega_eV < 0.0:
        raise ValueError("omega_eV must be non-negative")
    if config.eta_eV <= 0.0:
        raise ValueError("eta_eV must be positive")

    if k_weights is None:
        weights = np.full(points.shape[0], 1.0 / points.shape[0])
    else:
        weights = np.asarray(k_weights, dtype=float)
        if weights.shape != (points.shape[0],):
            raise ValueError("k_weights must have shape (n,)")
    return points, weights


def validate_finite_q_inputs(
    q_model: np.ndarray,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: KuboConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model must have shape (2,)")
    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("k_points must have shape (n, 2) and must not be empty")
    weights = np.asarray(k_weights, dtype=float)
    if weights.shape != (points.shape[0],):
        raise ValueError("k_weights must have shape (n,)")
    if config.omega_eV < 0.0:
        raise ValueError("omega_eV must be non-negative")
    return q, points, weights
