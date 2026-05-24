"""s_pm and d-wave superconducting pairing ansaetze in eV."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .model import normal_state_hamiltonian

NormalStateBuilder = Callable[[float, float], np.ndarray]
PairingKind = Literal["spm", "dwave"]


@dataclass(frozen=True)
class PairingAmplitudes:
    """Pairing seed amplitude in eV.

    The default is a small algebraic seed, not a fitted superconducting gap.
    """

    delta0: float = 0.04


def spm_pairing_matrix(kx: float, ky: float, amp: PairingAmplitudes | None = None) -> np.ndarray:
    """Return the 4x4 bilayer dz2 interlayer s_pm pairing matrix in eV.

    The basis is ``(dz1, dx1, dz2, dx2)``. Only the two dz2-like orbitals
    pair across layers, representing sign-changing bonding/antibonding
    bilayer s_pm pairing.
    """

    amp = amp or PairingAmplitudes()
    return np.array(
        [
            [0.0, 0.0, amp.delta0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [amp.delta0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def dwave_pairing_matrix(kx: float, ky: float, amp: PairingAmplitudes | None = None) -> np.ndarray:
    """Return the 4x4 same-layer dz2-dx2_y2 B1g pairing matrix in eV.

    The explicit momentum factor is A1g, ``cos(kx) + cos(ky)``. Combined with
    the B1g symmetry of the dx2_y2 orbital, the total pairing transforms as
    the d-wave/B1g channel.
    """

    amp = amp or PairingAmplitudes()
    form = np.cos(kx) + np.cos(ky)
    orbital_structure = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    return amp.delta0 * form * orbital_structure


def pairing_matrix(
    kind: PairingKind,
    kx: float,
    ky: float,
    amp: PairingAmplitudes | None = None,
) -> np.ndarray:
    """Return the requested 4x4 pairing matrix in eV."""

    if kind == "spm":
        return spm_pairing_matrix(kx, ky, amp)
    if kind == "dwave":
        return dwave_pairing_matrix(kx, ky, amp)
    raise ValueError("pairing kind must be 'spm' or 'dwave'")


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
