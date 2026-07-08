"""Target-basis coefficient helpers."""

from __future__ import annotations

import numpy as np

from .conventions import FiniteQConventions, SOURCE_ORDER_DIAGNOSTIC, SOURCE_ORDER_PHYSICAL, require_diagnostic_source_order


def aligned_source_vectors(conventions: FiniteQConventions) -> dict[str, np.ndarray]:
    """Return source vectors in aligned (A0, AL, AT) coordinates."""

    g0 = conventions.g0
    gL = conventions.gL
    return {
        "G": np.asarray([g0, gL, 0.0], dtype=float),
        "TM": np.asarray([-gL, g0, 0.0], dtype=float),
        "TE": np.asarray([0.0, 0.0, 1.0], dtype=float),
    }


def component_source_vectors(conventions: FiniteQConventions) -> dict[str, np.ndarray]:
    """Return source vectors in primitive [A0, Ax, Ay] coordinates."""

    qx, qy = conventions.qhat
    tx, ty = conventions.that
    g0 = conventions.g0
    gL = conventions.gL
    return {
        "G": np.asarray([g0, gL * qx, gL * qy], dtype=float),
        "TM": np.asarray([-gL, g0 * qx, g0 * qy], dtype=float),
        "TE": np.asarray([0.0, tx, ty], dtype=float),
    }


def physical_indices(source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC) -> tuple[int, int]:
    """Return indices for the physical TM/TE block."""

    require_diagnostic_source_order(source_order)
    return tuple(source_order.index(label) for label in SOURCE_ORDER_PHYSICAL)  # type: ignore[return-value]
