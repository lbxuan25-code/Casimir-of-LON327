"""Exact-static validation commands and private compatibility helpers.

Command modules select their numerical integration strategy explicitly. The retired
``nk_scan`` command remains absent; a private non-runnable import alias exposes only
shared decomposition helpers until specialized quadrature comparisons migrate.
"""
from __future__ import annotations

import sys
from types import ModuleType

from validation.lib.static_point_diagnostics import (
    collective_channel_diagnostics,
    kll_decomposition_diagnostics,
    longitudinal_component_diagnostics,
    phase_channel_factor_diagnostics,
    ward_side_diagnostics,
)


_legacy_diagnostics_module = ModuleType(f"{__name__}.nk_scan")
_legacy_diagnostics_module._collective_channel_diagnostics = (
    collective_channel_diagnostics
)
_legacy_diagnostics_module._kll_decomposition_diagnostics = (
    kll_decomposition_diagnostics
)
_legacy_diagnostics_module._longitudinal_component_diagnostics = (
    longitudinal_component_diagnostics
)
_legacy_diagnostics_module._phase_channel_factor_diagnostics = (
    phase_channel_factor_diagnostics
)
_legacy_diagnostics_module._ward_side_diagnostics = ward_side_diagnostics
sys.modules.setdefault(_legacy_diagnostics_module.__name__, _legacy_diagnostics_module)


__all__ = []
