"""Simple s_pm and d-wave superconducting pairing ansaetze."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import qiu_bilayer_hamiltonian


@dataclass(frozen=True)
class PairingAmplitudes:
    """Pairing-bond amplitudes in Qiu Appendix-A notation.

    Defaults are dimensionless tiny seed values for algebraic tests, not fitted
    RMFT solutions.
    """

    dz_parallel: float = 0.02
    dx_parallel: float = 0.02
    dxz_parallel: float = 0.005
    dz_perp: float = 0.04
    dx_perp: float = 0.01


def spm_pairing_matrix(kx: float, ky: float, amp: PairingAmplitudes | None = None) -> np.ndarray:
    """Return the 4x4 Qiu A1g/s_pm pairing matrix.

    Implements Eqs. (A6)-(A7): diagonal in-plane s_pm form factors plus
    inter-orbital d-wave form factor inside the same A1g channel.
    """

    amp = amp or PairingAmplitudes()
    s_form = np.cos(kx) + np.cos(ky)
    d_form = np.cos(kx) - np.cos(ky)
    delta_parallel = 2.0 * np.array(
        [
            [-amp.dz_parallel * s_form, amp.dxz_parallel * d_form],
            [amp.dxz_parallel * d_form, amp.dx_parallel * s_form],
        ],
        dtype=float,
    )
    delta_perp = np.diag([amp.dz_perp, amp.dx_perp])
    return np.block([[delta_parallel, delta_perp], [delta_perp, delta_parallel]])


def dwave_pairing_matrix(kx: float, ky: float, amp: PairingAmplitudes | None = None) -> np.ndarray:
    """Return a simple B1g d-wave pairing matrix for theory comparisons."""

    amp = amp or PairingAmplitudes()
    form = np.cos(kx) - np.cos(ky)
    layer_delta = form * np.diag([amp.dz_parallel, amp.dx_parallel])
    zero = np.zeros((2, 2), dtype=float)
    return np.block([[layer_delta, zero], [zero, layer_delta]])


def bdg_hamiltonian(
    kx: float,
    ky: float,
    pairing: np.ndarray,
    normal_hamiltonian: np.ndarray | None = None,
) -> np.ndarray:
    """Build an 8x8 BdG Hamiltonian from a 4x4 normal state and pairing matrix."""

    h_k = normal_hamiltonian if normal_hamiltonian is not None else qiu_bilayer_hamiltonian(kx, ky)
    h_minus_k = qiu_bilayer_hamiltonian(-kx, -ky) if normal_hamiltonian is None else h_k.conjugate()
    return np.block(
        [
            [h_k, pairing],
            [pairing.conjugate().T, -h_minus_k.T],
        ]
    )
