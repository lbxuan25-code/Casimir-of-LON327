"""Projection helpers for model sanity observables."""

from __future__ import annotations

import numpy as np


def anchor_eigenvector_phases(states: np.ndarray, *, eps: float = 1e-14) -> np.ndarray:
    """Return eigenvectors in a deterministic anchor gauge for sanity plots."""

    anchored = np.array(states, dtype=complex, copy=True)
    if anchored.ndim != 2:
        raise ValueError("states must have shape (dim, n_bands)")
    for band in range(anchored.shape[1]):
        vector = anchored[:, band]
        index = int(np.argmax(np.abs(vector)))
        component = vector[index]
        magnitude = abs(component)
        if magnitude < eps:
            continue
        anchored[:, band] *= np.conjugate(component) / magnitude
    return anchored


def band_project_pairing(
    delta: np.ndarray,
    states_k: np.ndarray,
    states_minus_k: np.ndarray,
    *,
    gauge: str = "anchor",
) -> np.ndarray:
    """Project pairing to normal bands as a deterministic-gauge sanity quantity."""

    if gauge == "anchor":
        left_states = anchor_eigenvector_phases(states_k)
        right_states = anchor_eigenvector_phases(states_minus_k)
    elif gauge == "raw":
        left_states = np.asarray(states_k)
        right_states = np.asarray(states_minus_k)
    else:
        raise ValueError("gauge must be 'anchor' or 'raw'")

    delta = np.asarray(delta)
    if left_states.ndim != 2 or right_states.ndim != 2:
        raise ValueError("states must have shape (dim, n_bands)")
    if left_states.shape != right_states.shape:
        raise ValueError("states_k and states_minus_k must have matching shapes")
    if delta.shape != (left_states.shape[0], left_states.shape[0]):
        raise ValueError("delta shape must match eigenvector dimension")

    return np.asarray(
        [
            left_states[:, band].conjugate().T @ delta @ right_states[:, band].conjugate()
            for band in range(left_states.shape[1])
        ],
        dtype=complex,
    )
