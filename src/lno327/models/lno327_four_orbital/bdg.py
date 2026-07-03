"""BdG Hamiltonian assembly for the LNO327 four-orbital model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lno327.models.lno327_four_orbital.normal import normal_state_hamiltonian

NormalStateBuilder = Callable[[float, float], np.ndarray]


def bdg_hamiltonian(
    kx: float,
    ky: float,
    pairing: np.ndarray,
    normal_state: NormalStateBuilder = normal_state_hamiltonian,
) -> np.ndarray:
    """Build an 8x8 BdG Hamiltonian from a 4x4 normal state and pairing matrix."""

    h_k = normal_state(kx, ky)
    h_minus_k = normal_state(-kx, -ky)
    pairing = np.asarray(pairing)

    if h_k.shape != (4, 4) or h_minus_k.shape != (4, 4):
        raise ValueError("normal_state must return a 4x4 matrix")
    if pairing.shape != (4, 4):
        raise ValueError("pairing must be a 4x4 matrix")
    if not np.allclose(h_k, h_k.conjugate().T):
        raise ValueError("normal_state(k) must be Hermitian")
    if not np.allclose(h_minus_k, h_minus_k.conjugate().T):
        raise ValueError("normal_state(-k) must be Hermitian")

    return np.block(
        [
            [h_k, pairing],
            [pairing.conjugate().T, -h_minus_k.T],
        ]
    )


assemble_bdg_hamiltonian = bdg_hamiltonian
