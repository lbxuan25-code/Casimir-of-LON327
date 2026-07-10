"""Casimir-Lifshitz building blocks."""

from __future__ import annotations

from .lifshitz import casimir_energy_integrand
from .lifshitz_integrand import LifshitzPoint, passive_sheet_logdet
from .reflection import reflection_matrix_weak_2d
from .setup import CasimirSetup, matsubara_frequency
from .torque import casimir_torque_integrand


def casimir_layer_metadata() -> dict[str, object]:
    return {
        "layer": "casimir_lifshitz_building_blocks",
        "valid_for_casimir_input": False,
        "requires_gauge_closed_response": True,
        "ward_identity_closed_by_this_module": False,
        "notes": (
            "This package contains mathematical integrand building blocks only.",
            "It does not validate finite-q Ward/gauge closure.",
            "It does not make BdG response outputs Casimir-ready.",
        ),
    }


__all__ = [
    "CasimirSetup",
    "LifshitzPoint",
    "casimir_energy_integrand",
    "casimir_layer_metadata",
    "casimir_torque_integrand",
    "matsubara_frequency",
    "passive_sheet_logdet",
    "reflection_matrix_weak_2d",
]
