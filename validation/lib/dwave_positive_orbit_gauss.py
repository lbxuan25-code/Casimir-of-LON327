"""Compatibility wrapper for the common positive-orbit fixed Gauss backend."""
from __future__ import annotations

from typing import Any

from validation.lib.positive_orbit_gauss import (
    PositiveOrbitGaussResult,
    integrate_positive_orbit_gauss,
)

DWavePositiveOrbitGaussResult = PositiveOrbitGaussResult


def integrate_dwave_positive_orbit_gauss(**kwargs: Any) -> PositiveOrbitGaussResult:
    ansatz = kwargs.get("ansatz")
    if getattr(ansatz, "name", None) != "dwave":
        raise ValueError("d-wave compatibility wrapper requires ansatz.name='dwave'")
    return integrate_positive_orbit_gauss(**kwargs)


__all__ = [
    "DWavePositiveOrbitGaussResult",
    "integrate_dwave_positive_orbit_gauss",
]
