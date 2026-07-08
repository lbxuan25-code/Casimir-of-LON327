"""Direct target-basis vertex construction."""

from __future__ import annotations

import numpy as np

from .conventions import FiniteQConventions, SOURCE_ORDER_DIAGNOSTIC


def longitudinal_transverse_vertices(
    gamma_x: np.ndarray,
    gamma_y: np.ndarray,
    conventions: FiniteQConventions,
) -> tuple[np.ndarray, np.ndarray]:
    """Return GammaL and GammaT using that=(-qhat_y,qhat_x)."""

    gx = np.asarray(gamma_x, dtype=complex)
    gy = np.asarray(gamma_y, dtype=complex)
    gamma_l = conventions.qhat[0] * gx + conventions.qhat[1] * gy
    gamma_t = -conventions.qhat[1] * gx + conventions.qhat[0] * gy
    return gamma_l, gamma_t


def target_vertices(
    gamma0: np.ndarray,
    gammax: np.ndarray,
    gammay: np.ndarray,
    conventions: FiniteQConventions,
    *,
    source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC,
) -> tuple[np.ndarray, ...]:
    """Build GammaG, GammaTM, GammaTE directly from primitive vertices."""

    g0 = np.asarray(gamma0, dtype=complex)
    gamma_l, gamma_t = longitudinal_transverse_vertices(gammax, gammay, conventions)
    vertices = {
        "G": conventions.g0 * g0 + conventions.gL * gamma_l,
        "TM": -conventions.gL * g0 + conventions.g0 * gamma_l,
        "TE": gamma_t,
    }
    return tuple(vertices[label] for label in source_order)

