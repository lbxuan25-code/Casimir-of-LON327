"""Geometry and Matsubara-frequency inputs for Casimir integrands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lno327.constants import HBAR, KB


@dataclass(frozen=True)
class CasimirSetup:
    """Geometry and thermodynamic inputs for two parallel 2D plates."""

    temperature: float
    distance: float
    area: float = 1.0


def matsubara_frequency(n: int, temperature: float) -> float:
    """Return bosonic Matsubara angular frequency xi_n."""

    return 2.0 * np.pi * KB * temperature * n / HBAR
