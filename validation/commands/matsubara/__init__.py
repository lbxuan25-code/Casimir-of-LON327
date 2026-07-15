"""Matsubara validation commands and private compatibility helpers."""
from __future__ import annotations

import sys
from types import ModuleType

from validation.lib.matsubara import matsubara_energy_eV


# Old modules imported the frequency helper from the removed positive-point command.
# Keep that import path private and non-runnable until all internal imports migrate.
_legacy_frequency_module = ModuleType(f"{__name__}.positive_point")
_legacy_frequency_module.matsubara_energy_eV = matsubara_energy_eV
_legacy_frequency_module.__all__ = ["matsubara_energy_eV"]
sys.modules.setdefault(_legacy_frequency_module.__name__, _legacy_frequency_module)


__all__ = ["matsubara_energy_eV"]
