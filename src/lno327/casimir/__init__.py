"""Casimir-Lifshitz building blocks."""

from __future__ import annotations

from .fixed_outer_q import (
    OuterQGridPlan,
    OuterQGridSpec,
    OuterQNodeManifest,
    absolute_then_relative,
    aggregate_certified_outer_q,
    build_staged_grid_plan,
    build_union_node_manifest,
    compare_ladders,
)
from .lifshitz import casimir_energy_integrand
from .lifshitz_integrand import LifshitzPoint, passive_sheet_logdet
from .matsubara import matsubara_energy_eV
from .microscopic_model import (
    FiniteQMicroscopicModel,
    available_finite_q_microscopic_models,
    get_finite_q_microscopic_model,
)
from .outer_quadrature import (
    MatsubaraFreeEnergyPerArea,
    OuterQPolarGrid,
    build_outer_q_polar_grid,
    free_energy_per_area_from_logdet,
    integrate_outer_q,
    matsubara_prime_weights,
)
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
        "matsubara_energy_helper_owned_by_production": True,
        "finite_q_model_adapter_owned_by_production": True,
        "outer_q_measure_contract_present": True,
        "outer_q_fixed_nested_planning_present": True,
        "outer_q_radial_variable": "u = 2 Q d",
        "outer_q_full_angular_domain": True,
        "notes": (
            "This package contains mathematical integrand and fixed quadrature building blocks.",
            "Validation consumes Matsubara, model-adapter, planning, and reduction helpers from this package.",
            "It does not validate finite-q Ward/gauge closure.",
            "The outer-q layer applies the n=0 half weight but does not estimate the Matsubara tail.",
            "It does not make BdG response outputs Casimir-ready by itself.",
        ),
    }


__all__ = [
    "CasimirSetup",
    "FiniteQMicroscopicModel",
    "LifshitzPoint",
    "MatsubaraFreeEnergyPerArea",
    "OuterQGridPlan",
    "OuterQGridSpec",
    "OuterQNodeManifest",
    "OuterQPolarGrid",
    "absolute_then_relative",
    "aggregate_certified_outer_q",
    "available_finite_q_microscopic_models",
    "build_outer_q_polar_grid",
    "build_staged_grid_plan",
    "build_union_node_manifest",
    "casimir_energy_integrand",
    "casimir_layer_metadata",
    "casimir_torque_integrand",
    "compare_ladders",
    "free_energy_per_area_from_logdet",
    "get_finite_q_microscopic_model",
    "integrate_outer_q",
    "matsubara_energy_eV",
    "matsubara_frequency",
    "matsubara_prime_weights",
    "passive_sheet_logdet",
    "reflection_matrix_weak_2d",
]
