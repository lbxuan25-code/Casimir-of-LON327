"""Model-independent Nambu block helpers."""

from __future__ import annotations

import numpy as np


def _validate_direction(direction: str) -> None:
    if direction not in {"x", "y"}:
        raise ValueError("direction must be 'x' or 'y'")


def infer_square_dim(matrix: np.ndarray) -> int:
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    return int(array.shape[0])


def nambu_block(
    particle_block: np.ndarray,
    hole_block: np.ndarray,
) -> np.ndarray:
    particle = np.asarray(particle_block)
    hole = np.asarray(hole_block)
    dim = infer_square_dim(particle)
    if hole.shape != (dim, dim):
        raise ValueError("particle and hole blocks must have the same square shape")
    zero = np.zeros((dim, dim), dtype=np.result_type(particle, hole, complex))
    return np.block([[particle, zero], [zero, hole]]).astype(complex)


def charge_current_vertex_from_blocks(
    particle_velocity: np.ndarray,
    hole_velocity_at_minus_k: np.ndarray,
) -> np.ndarray:
    return nambu_block(particle_velocity, -np.asarray(hole_velocity_at_minus_k).T)


def diamagnetic_vertex_from_blocks(
    particle_mass: np.ndarray,
    hole_mass_at_minus_k: np.ndarray,
) -> np.ndarray:
    return nambu_block(particle_mass, -np.asarray(hole_mass_at_minus_k).T)


def charge_current_vertex_from_model(
    spec,
    kx: float,
    ky: float,
    direction: str,
) -> np.ndarray:
    _validate_direction(direction)
    return charge_current_vertex_from_blocks(
        spec.velocity_operator(kx, ky, direction),
        spec.velocity_operator(-kx, -ky, direction),
    )


def diamagnetic_vertex_from_model(
    spec,
    kx: float,
    ky: float,
    direction_a: str,
    direction_b: str,
) -> np.ndarray:
    _validate_direction(direction_a)
    _validate_direction(direction_b)
    return diamagnetic_vertex_from_blocks(
        spec.mass_operator(kx, ky, direction_a, direction_b),
        spec.mass_operator(-kx, -ky, direction_a, direction_b),
    )
