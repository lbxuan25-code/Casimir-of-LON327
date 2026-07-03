"""Parameters and names for the LNO327 four-orbital model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ORBITAL_BASIS = ("dz1", "dx1", "dz2", "dx2")

PairingKind = Literal["spm", "dwave"]
PairingAnsatzName = Literal["onsite_s", "spm", "dwave"]
PhaseVertexName = Literal["midpoint", "symmetric_kpm", "bond_endpoint_gauge"]
PairingBondStatus = Literal["PASSED", "FAILED", "PAIRING_BOND_REPRESENTATION_UNAVAILABLE"]


@dataclass(frozen=True)
class NormalStateParameters:
    """Tight-binding coefficients in eV for basis dz1, dx1, dz2, dx2."""

    chemical_potential: float = 0.05
    tz_1: float = -0.217
    tz_2: float = -0.073
    tz_3: float = -0.021
    tz_4: float = -0.005
    tz_0: float = 0.431
    tx_1: float = -0.922
    tx_2: float = 0.301
    tx_3: float = -0.108
    tx_4: float = -0.025
    tx_0: float = 0.881
    tz_perp_0: float = -0.550
    tz_perp_1: float = 0.041
    tx_perp_0: float = 0.005
    vxz_1: float = 0.429
    vxz_2: float = 0.041
    vxz_perp_1: float = -0.061


@dataclass(frozen=True)
class PairingAmplitudes:
    """Pairing seed amplitude in eV."""

    delta0_eV: float = 0.04

    @property
    def delta0(self) -> float:
        """Backward-compatible alias for the eV pairing amplitude."""

        return self.delta0_eV
