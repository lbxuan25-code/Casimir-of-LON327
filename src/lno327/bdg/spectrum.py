"""Model-independent eigensystem helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Eigensystem:
    energies: np.ndarray
    states: np.ndarray


def _require_square_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    return array


def diagonalize_hermitian(matrix: np.ndarray) -> Eigensystem:
    array = _require_square_matrix(matrix)
    energies, states = np.linalg.eigh(array)
    return Eigensystem(energies=energies, states=states)


def normal_eigensystem_from_model(spec, kx: float, ky: float) -> Eigensystem:
    return diagonalize_hermitian(spec.normal_hamiltonian(kx, ky))


def bdg_eigensystem_from_model(spec, kx: float, ky: float, channel: str) -> Eigensystem:
    return diagonalize_hermitian(spec.bdg_hamiltonian(kx, ky, channel))


def transform_operator_to_band_basis(states: np.ndarray, operator: np.ndarray) -> np.ndarray:
    state_matrix = _require_square_matrix(states)
    op_matrix = _require_square_matrix(operator)
    if state_matrix.shape != op_matrix.shape:
        raise ValueError("states and operator must have matching square shape")
    return state_matrix.conjugate().T @ op_matrix @ state_matrix
