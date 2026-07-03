"""Model-independent BdG finite-q Nambu vertex primitives."""

from __future__ import annotations

import numpy as np

from lno327.bdg.nambu import infer_square_dim, nambu_block


def bdg_block_diagonal_vertex(
    particle_block: np.ndarray,
    hole_block: np.ndarray,
) -> np.ndarray:
    particle = np.asarray(particle_block)
    hole = np.asarray(hole_block)
    dim = infer_square_dim(particle)
    if hole.shape != (dim, dim):
        raise ValueError("particle and hole blocks must have the same square shape")
    return nambu_block(particle, hole)


def bdg_finite_q_vertex_from_normal_blocks(
    particle_block: np.ndarray,
    hole_normal_block_at_minus_k_minus_q: np.ndarray,
) -> np.ndarray:
    hole_block = -np.asarray(hole_normal_block_at_minus_k_minus_q).T
    return bdg_block_diagonal_vertex(particle_block, hole_block)


def phase_vertex(pairing: np.ndarray) -> np.ndarray:
    delta = np.asarray(pairing)
    zero = np.zeros_like(delta)
    return np.block([[zero, 1j * delta], [-1j * delta.conjugate().T, zero]]).astype(complex)


def phase_phase_direct_vertex(delta_theta: np.ndarray) -> np.ndarray:
    vertex = np.asarray(delta_theta)
    zero = np.zeros_like(vertex)
    return np.block([[zero, -vertex], [-vertex.conjugate().T, zero]]).astype(complex)


def density_vertex(orbital_dim: int) -> np.ndarray:
    if not isinstance(orbital_dim, int) or orbital_dim <= 0:
        raise ValueError("orbital_dim must be a positive integer")
    eye = np.eye(orbital_dim, dtype=complex)
    zero = np.zeros((orbital_dim, orbital_dim), dtype=complex)
    return np.block([[eye, zero], [zero, -eye]])
