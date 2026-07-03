"""Generic BdG Hamiltonian assembly."""

from __future__ import annotations

import numpy as np


def _as_square_matrix(name: str, matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return array


def bdg_hamiltonian_from_blocks(
    normal_k: np.ndarray,
    normal_minus_k: np.ndarray,
    pairing: np.ndarray,
) -> np.ndarray:
    """Assemble [[h(k), Delta(k)], [Delta^dagger(k), -h^T(-k)]]."""

    h_k = _as_square_matrix("normal_k", normal_k)
    h_minus_k = _as_square_matrix("normal_minus_k", normal_minus_k)
    delta = _as_square_matrix("pairing", pairing)
    if h_minus_k.shape != h_k.shape or delta.shape != h_k.shape:
        raise ValueError("normal_k, normal_minus_k, and pairing must have the same square shape")
    if not np.allclose(h_k, h_k.conjugate().T):
        raise ValueError("normal_k must be Hermitian")
    if not np.allclose(h_minus_k, h_minus_k.conjugate().T):
        raise ValueError("normal_minus_k must be Hermitian")
    return np.block([[h_k, delta], [delta.conjugate().T, -h_minus_k.T]]).astype(complex)


def bdg_hamiltonian_from_model_pairing(
    spec,
    kx: float,
    ky: float,
    pairing: np.ndarray,
) -> np.ndarray:
    return bdg_hamiltonian_from_blocks(
        spec.normal_hamiltonian(kx, ky),
        spec.normal_hamiltonian(-kx, -ky),
        pairing,
    )
