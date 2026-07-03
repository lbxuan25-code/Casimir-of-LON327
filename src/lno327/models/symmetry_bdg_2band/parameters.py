"""Parameters for the symmetry-focused two-band BdG model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BASIS = ("a", "b")

PairingChannel = Literal["normal", "spp", "spm", "dwave"]


@dataclass(frozen=True)
class TwoBandParameters:
    t: float = 1.0
    tp: float = -0.25
    mu: float = -1.0
    t_perp: float = 0.3
    t_perp_p: float = 0.1
    m: float = 0.2
    t_z: float = 0.05
    delta_s: float = 0.1
    delta_d: float = 0.1
