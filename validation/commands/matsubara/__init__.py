"""Matsubara validation commands and retained private command aliases."""
from __future__ import annotations

import sys
from types import ModuleType

from lno327.casimir.matsubara import matsubara_energy_eV


# Older retained Matsubara commands import the frequency helper from the retired
# positive-point command. Keep that private non-runnable import path until those
# independent command modules are migrated; the implementation is production-owned.
_legacy_frequency_module = ModuleType(f"{__name__}.positive_point")
_legacy_frequency_module.matsubara_energy_eV = matsubara_energy_eV
_legacy_frequency_module.__all__ = ["matsubara_energy_eV"]
sys.modules.setdefault(_legacy_frequency_module.__name__, _legacy_frequency_module)


__all__ = ["matsubara_energy_eV"]
