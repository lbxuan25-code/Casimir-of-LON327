"""Target-basis spatial contact projection."""

from __future__ import annotations

import numpy as np

from .conventions import FiniteQConventions, SOURCE_ORDER_DIAGNOSTIC


def spatial_contact_vectors(conventions: FiniteQConventions) -> dict[str, np.ndarray]:
    """Return vG, vTM, vTE vectors for C_ab = v_a^i D_ij v_b^j."""

    return {
        "G": conventions.gL * conventions.qhat,
        "TM": conventions.g0 * conventions.qhat,
        "TE": conventions.that,
    }


def project_spatial_contact(
    spatial_contact: np.ndarray,
    conventions: FiniteQConventions,
    *,
    source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC,
) -> np.ndarray:
    """Project a 2x2 spatial D tensor into the target basis."""

    d = np.asarray(spatial_contact, dtype=complex)
    if d.shape != (2, 2):
        raise ValueError("spatial_contact must have shape (2, 2)")
    vectors = spatial_contact_vectors(conventions)
    out = np.zeros((len(source_order), len(source_order)), dtype=complex)
    for i, left in enumerate(source_order):
        for j, right in enumerate(source_order):
            out[i, j] = vectors[left] @ d @ vectors[right]
    return out

