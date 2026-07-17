"""Batched two-band Hamiltonian and Peierls finite-q vertex helpers.

The functions in this module are mathematically identical to the scalar two-band
model helpers, but evaluate an arbitrary leading batch of k points in one NumPy
operation.  They intentionally contain no response-level or quadrature policy.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from lno327.models.hopping import HoppingTerm, sinc_stable
from lno327.models.symmetry_bdg_2band.normal import TAU0, TAUX, TAUZ
from lno327.models.symmetry_bdg_2band.parameters import TwoBandParameters


def _points_array(k_points: np.ndarray) -> np.ndarray:
    points = np.asarray(k_points, dtype=float)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError("k_points must have shape (..., 2)")
    if not np.isfinite(points).all():
        raise ValueError("k_points must be finite")
    return points


def _q_array(q_model: np.ndarray) -> np.ndarray:
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")
    return q


def hopping_arrays(
    hopping_terms: Iterable[HoppingTerm],
) -> tuple[np.ndarray, np.ndarray]:
    terms = tuple(hopping_terms)
    if not terms:
        raise ValueError("hopping_terms must not be empty")
    vectors = np.asarray([term[0] for term in terms], dtype=float)
    matrices = np.stack(
        [np.asarray(term[1], dtype=complex) for term in terms],
        axis=0,
    )
    if vectors.shape != (len(terms), 2):
        raise ValueError("hopping vectors must have shape (n_hoppings, 2)")
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("hopping matrices must have shape (n_hoppings, dim, dim)")
    return vectors, matrices


def normal_hamiltonian_batch(
    k_points: np.ndarray,
    params: TwoBandParameters | None = None,
) -> np.ndarray:
    """Return h(k) for all k points with shape (..., 2, 2)."""

    points = _points_array(k_points)
    parameters = params or TwoBandParameters()
    kx = points[..., 0]
    ky = points[..., 1]
    cx = np.cos(kx)
    cy = np.cos(ky)
    xi0 = (
        -2.0 * parameters.t * (cx + cy)
        - 4.0 * parameters.tp * cx * cy
        - parameters.mu
    )
    xix = parameters.t_perp + 2.0 * parameters.t_perp_p * (cx + cy)
    xiz = parameters.m - 2.0 * parameters.t_z * (cx + cy)
    return (
        xi0[..., None, None] * TAU0
        + xix[..., None, None] * TAUX
        + xiz[..., None, None] * TAUZ
    ).astype(complex, copy=False)


def assemble_bdg_hamiltonian_batch(
    h_k: np.ndarray,
    h_minus_k: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Assemble batched 4x4 BdG Hamiltonians from batched 2x2 blocks."""

    particle = np.asarray(h_k, dtype=complex)
    hole_normal = np.asarray(h_minus_k, dtype=complex)
    pairing = np.asarray(delta, dtype=complex)
    if particle.shape[-2:] != (2, 2):
        raise ValueError("h_k must have shape (..., 2, 2)")
    if hole_normal.shape != particle.shape or pairing.shape != particle.shape:
        raise ValueError("h_k, h_minus_k, and delta must have matching shapes")
    result = np.zeros(particle.shape[:-2] + (4, 4), dtype=complex)
    result[..., :2, :2] = particle
    result[..., :2, 2:] = pairing
    result[..., 2:, :2] = np.swapaxes(pairing.conjugate(), -1, -2)
    result[..., 2:, 2:] = -np.swapaxes(hole_normal, -1, -2)
    return result


def bdg_hamiltonian_from_pairing_batch(
    k_points: np.ndarray,
    pairing: np.ndarray,
    params: TwoBandParameters | None = None,
) -> np.ndarray:
    """Return batched two-band BdG Hamiltonians for explicit pairing matrices."""

    points = _points_array(k_points)
    delta = np.asarray(pairing, dtype=complex)
    expected = points.shape[:-1] + (2, 2)
    if delta.shape != expected:
        raise ValueError(f"pairing must have shape {expected}")
    h_k = normal_hamiltonian_batch(points, params)
    h_minus_k = normal_hamiltonian_batch(-points, params)
    return assemble_bdg_hamiltonian_batch(h_k, h_minus_k, delta)


def _validate_hopping_arrays(
    hopping_vectors: np.ndarray,
    hopping_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(hopping_vectors, dtype=float)
    matrices = np.asarray(hopping_matrices, dtype=complex)
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("hopping_vectors must have shape (n_hoppings, 2)")
    if (
        matrices.ndim != 3
        or matrices.shape[0] != vectors.shape[0]
        or matrices.shape[1] != matrices.shape[2]
    ):
        raise ValueError(
            "hopping_matrices must have shape (n_hoppings, dim, dim)"
        )
    return vectors, matrices


def _hopping_phases(
    k_points: np.ndarray,
    hopping_vectors: np.ndarray,
) -> np.ndarray:
    return np.exp(1j * np.einsum("...d,hd->...h", k_points, hopping_vectors))


def peierls_hamiltonian_vector_vertices_batch(
    k_points: np.ndarray,
    q_model: np.ndarray,
    hopping_vectors: np.ndarray,
    hopping_matrices: np.ndarray,
) -> np.ndarray:
    """Return both normal-state Peierls vector vertices, shape (..., 2, d, d)."""

    points = _points_array(k_points)
    q = _q_array(q_model)
    vectors, matrices = _validate_hopping_arrays(
        hopping_vectors, hopping_matrices
    )
    phases = _hopping_phases(points, vectors)
    q_dot_r = vectors @ q
    sinc = np.asarray(sinc_stable(0.5 * q_dot_r), dtype=float)
    coefficients = 1j * vectors * sinc[:, None]
    return np.einsum(
        "...h,hd,hij->...dij",
        phases,
        coefficients,
        matrices,
        optimize=True,
    )


def peierls_hamiltonian_contact_vertices_batch(
    k_points: np.ndarray,
    q_model: np.ndarray,
    hopping_vectors: np.ndarray,
    hopping_matrices: np.ndarray,
) -> np.ndarray:
    """Return all normal-state contact vertices, shape (..., 2, 2, d, d)."""

    points = _points_array(k_points)
    q = _q_array(q_model)
    vectors, matrices = _validate_hopping_arrays(
        hopping_vectors, hopping_matrices
    )
    phases = _hopping_phases(points, vectors)
    q_dot_r = vectors @ q
    sinc2 = np.asarray(sinc_stable(0.5 * q_dot_r), dtype=float) ** 2
    coefficients = (
        -vectors[:, :, None] * vectors[:, None, :] * sinc2[:, None, None]
    )
    return np.einsum(
        "...h,hde,hij->...deij",
        phases,
        coefficients,
        matrices,
        optimize=True,
    )


def peierls_hamiltonian_vertices_batch(
    k_points: np.ndarray,
    q_model: np.ndarray,
    hopping_vectors: np.ndarray,
    hopping_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return vector and contact vertices while sharing one phase evaluation."""

    points = _points_array(k_points)
    q = _q_array(q_model)
    vectors, matrices = _validate_hopping_arrays(
        hopping_vectors, hopping_matrices
    )
    phases = _hopping_phases(points, vectors)
    q_dot_r = vectors @ q
    sinc = np.asarray(sinc_stable(0.5 * q_dot_r), dtype=float)

    vector_coefficients = 1j * vectors * sinc[:, None]
    vector = np.einsum(
        "...h,hd,hij->...dij",
        phases,
        vector_coefficients,
        matrices,
        optimize=True,
    )

    contact_coefficients = (
        -vectors[:, :, None]
        * vectors[:, None, :]
        * (sinc * sinc)[:, None, None]
    )
    contact = np.einsum(
        "...h,hde,hij->...deij",
        phases,
        contact_coefficients,
        matrices,
        optimize=True,
    )
    return vector, contact


__all__ = [
    "assemble_bdg_hamiltonian_batch",
    "bdg_hamiltonian_from_pairing_batch",
    "hopping_arrays",
    "normal_hamiltonian_batch",
    "peierls_hamiltonian_contact_vertices_batch",
    "peierls_hamiltonian_vector_vertices_batch",
    "peierls_hamiltonian_vertices_batch",
]
