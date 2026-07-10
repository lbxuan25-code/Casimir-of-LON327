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
        "positive_matsubara_signed_logdet_supported": True,
        "zero_matsubara_signed_logdet_supported": True,
        "zero_matsubara_uses_static_susceptibility_not_conductivity": True,
        "zero_matsubara_prime_weight_applied_by_quadrature": True,
        "notes": (
            "This package contains mathematical integrand building blocks only.",
            "It does not validate finite-q Ward/gauge closure.",
            "The n=0 half weight belongs to the future Matsubara quadrature layer.",
            "It does not make BdG response outputs Casimir-ready by itself.",
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
