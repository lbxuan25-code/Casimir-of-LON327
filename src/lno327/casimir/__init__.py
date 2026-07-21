"""LNO327 Casimir numerical library.

Operational calculations are created and executed only through
``python -m scripts.full_casimir plan/run``.  The objects exported here are library
components used by that dispatcher and by tests; they are not independent command
surfaces.
"""
from __future__ import annotations

from .production import (
    FullCasimirConfig,
    FullCasimirResult,
    build_full_casimir_config,
    run_full_casimir,
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
        "layer": "casimir_lifshitz_full_adaptive",
        "canonical_operational_entrypoint": "python -m scripts.full_casimir",
        "canonical_plan_command": "python -m scripts.full_casimir plan",
        "canonical_run_command": "python -m scripts.full_casimir run",
        "package_command_present": False,
        "installed_console_command_present": False,
        "legacy_calculation_scripts_present": False,
        "library_engine": "lno327.casimir.production.run_full_casimir",
        "library_config_builder": "lno327.casimir.production.build_full_casimir_config",
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
        "requires_gauge_closed_response": True,
        "ward_identity_closed_by_this_module": False,
        "positive_matsubara_signed_logdet_supported": True,
        "zero_matsubara_signed_logdet_supported": True,
        "zero_matsubara_uses_static_susceptibility_not_conductivity": True,
        "zero_matsubara_prime_weight_applied_by_quadrature": True,
        "frequency_extendable_certified_point_provider_present": True,
        "outer_q_adaptive_radial_present": True,
        "outer_q_adaptive_angular_present": True,
        "outer_q_joint_radial_angular_budget_present": True,
        "outer_q_joint_direction_selection_present": True,
        "outer_q_adaptive_cutoff_present": True,
        "outer_q_geometric_tail_envelope_present": True,
        "outer_q_angular_offset_audit_present": True,
        "matsubara_adaptive_cutoff_present": True,
        "matsubara_geometric_tail_envelope_present": True,
        "adaptive_outer_integration_architecture_complete": True,
        "real_model_end_to_end_qualified": False,
        "notes": (
            "The package exports numerical library components, not an operational CLI.",
            "Formal calculations require an immutable top-level plan and SHA confirmation.",
            "Physical production authorization remains false until real-model qualification.",
        ),
    }


__all__ = [
    "CasimirSetup",
    "FiniteQMicroscopicModel",
    "FullCasimirConfig",
    "FullCasimirResult",
    "LifshitzPoint",
    "MatsubaraFreeEnergyPerArea",
    "OuterQPolarGrid",
    "available_finite_q_microscopic_models",
    "build_full_casimir_config",
    "build_outer_q_polar_grid",
    "casimir_energy_integrand",
    "casimir_layer_metadata",
    "casimir_torque_integrand",
    "free_energy_per_area_from_logdet",
    "get_finite_q_microscopic_model",
    "integrate_outer_q",
    "matsubara_energy_eV",
    "matsubara_frequency",
    "matsubara_prime_weights",
    "passive_sheet_logdet",
    "reflection_matrix_weak_2d",
    "run_full_casimir",
]
