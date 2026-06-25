"""Real-space pairing-bond diagnostics for finite-q collective vertices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .pairing import PairingAmplitudes

PairingBondStatus = Literal["PASSED", "FAILED", "PAIRING_BOND_REPRESENTATION_UNAVAILABLE"]


@dataclass(frozen=True)
class PairingBond:
    channel: str
    left_orbital: int
    right_orbital: int
    left_offset: tuple[float, float]
    right_offset: tuple[float, float]
    coefficient: complex
    matrix_block: str | None = None

    @property
    def displacement(self) -> np.ndarray:
        return np.asarray(self.right_offset, dtype=float) - np.asarray(self.left_offset, dtype=float)

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (np.asarray(self.right_offset, dtype=float) + np.asarray(self.left_offset, dtype=float))


def pairing_bond_list(pairing: str, amp: PairingAmplitudes | None = None) -> list[PairingBond]:
    """Return the exact finite-Fourier pairing bonds used by the current code."""

    amplitude = amp or PairingAmplitudes()
    delta0 = complex(amplitude.delta0_eV)
    if pairing == "onsite_s":
        return [
            PairingBond("onsite_s", orbital, orbital, (0.0, 0.0), (0.0, 0.0), delta0, "onsite_identity")
            for orbital in range(4)
        ]
    if pairing == "spm":
        return [
            PairingBond("spm", 0, 2, (0.0, 0.0), (0.0, 0.0), delta0, "interlayer_dz2"),
            PairingBond("spm", 2, 0, (0.0, 0.0), (0.0, 0.0), delta0, "interlayer_dz2"),
        ]
    if pairing == "dwave":
        bonds: list[PairingBond] = []
        for left, right in ((0, 1), (1, 0), (2, 3), (3, 2)):
            for offset in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                half = (0.5 * offset[0], 0.5 * offset[1])
                bonds.append(
                    PairingBond(
                        "dwave",
                        left,
                        right,
                        (-half[0], -half[1]),
                        half,
                        0.5 * delta0,
                        "same_layer_interorbital_A1g_cos_sum",
                    )
                )
        return bonds
    return []


def pairing_bond_status(pairing: str) -> PairingBondStatus:
    return "PAIRING_BOND_REPRESENTATION_UNAVAILABLE" if not pairing_bond_list(pairing) else "PASSED"


def pairing_from_bonds(
    pairing: str,
    kx: float,
    ky: float,
    amp: PairingAmplitudes | None = None,
) -> np.ndarray:
    """Reconstruct Delta(k) = sum_b c_b exp(-i k.d_b) P_b from pairing bonds."""

    output = np.zeros((4, 4), dtype=complex)
    k = np.asarray([kx, ky], dtype=float)
    for bond in pairing_bond_list(pairing, amp):
        output[bond.left_orbital, bond.right_orbital] += bond.coefficient * np.exp(
            -1j * float(np.dot(k, bond.displacement))
        )
    return output


def bond_center_form_factor(
    pairing: str,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    amp: PairingAmplitudes,
) -> np.ndarray:
    """Return the bond-center amplitude form factor Delta_bond(k,q)/delta0."""

    delta0 = float(amp.delta0_eV)
    if delta0 == 0.0:
        raise ValueError("pairing bond form factor is undefined for delta0=0")
    output = np.zeros((4, 4), dtype=complex)
    k = np.asarray([kx, ky], dtype=float)
    q = np.asarray([qx, qy], dtype=float)
    for bond in pairing_bond_list(pairing, amp):
        output[bond.left_orbital, bond.right_orbital] += (
            bond.coefficient
            * np.exp(-1j * float(np.dot(k, bond.displacement)))
            * np.exp(1j * float(np.dot(q, bond.center)))
            / delta0
        )
    return output


def bond_endpoint_gauge_form_factor(
    pairing: str,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    amp: PairingAmplitudes,
) -> np.ndarray:
    """Return the endpoint-average gauge-phase form factor from pairing bonds."""

    delta0 = float(amp.delta0_eV)
    if delta0 == 0.0:
        raise ValueError("pairing bond form factor is undefined for delta0=0")
    output = np.zeros((4, 4), dtype=complex)
    k = np.asarray([kx, ky], dtype=float)
    q = np.asarray([qx, qy], dtype=float)
    for bond in pairing_bond_list(pairing, amp):
        displacement = bond.displacement
        output[bond.left_orbital, bond.right_orbital] += (
            bond.coefficient
            * np.exp(-1j * float(np.dot(k, displacement)))
            * np.exp(1j * float(np.dot(q, bond.center)))
            * np.cos(0.5 * float(np.dot(q, displacement)))
            / delta0
        )
    return output
