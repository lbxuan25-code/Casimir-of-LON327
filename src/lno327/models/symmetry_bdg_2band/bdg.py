"""BdG Hamiltonian assembly for the symmetry-focused two-band model."""

from __future__ import annotations

import numpy as np

from lno327.models.symmetry_bdg_2band.normal import normal_hamiltonian
from lno327.models.symmetry_bdg_2band.pairing import pairing_matrix
from lno327.models.symmetry_bdg_2band.parameters import PairingChannel, TwoBandParameters


def assemble_bdg_hamiltonian(h_k: np.ndarray, h_minus_k: np.ndarray, delta: np.ndarray) -> np.ndarray:
    h_k = np.asarray(h_k)
    h_minus_k = np.asarray(h_minus_k)
    delta = np.asarray(delta)
    if h_k.shape != (2, 2) or h_minus_k.shape != (2, 2):
        raise ValueError("normal-state blocks must have shape (2, 2)")
    if delta.shape != (2, 2):
        raise ValueError("pairing block must have shape (2, 2)")
    return np.block(
        [
            [h_k, delta],
            [delta.conjugate().T, -h_minus_k.T],
        ]
    )


def bdg_hamiltonian(
    kx: float,
    ky: float,
    channel: PairingChannel,
    params: TwoBandParameters | None = None,
) -> np.ndarray:
    params = params or TwoBandParameters()
    return assemble_bdg_hamiltonian(
        normal_hamiltonian(kx, ky, params),
        normal_hamiltonian(-kx, -ky, params),
        pairing_matrix(channel, kx, ky, params),
    )
