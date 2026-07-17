"""Casimir-Lifshitz building blocks and adaptive diagnostic controllers."""

from __future__ import annotations

from .fixed_chain import (
    FixedCasimirConfig,
    FixedCasimirExecutionError,
    FixedCasimirResult,
    run_casimir,
)
from .certified_point_provider import (
    CertifiedOuterQProvider,
    CertifiedPointBatch,
    CertifiedPointCacheError,
    FrequencyExtendableCertifiedOuterQProvider,
    certified_primary_logdet,
)
from .adaptive_angular_q import (
    AdaptiveAngularCasimirConfig,
    AdaptiveAngularCasimirResult,
    run_adaptive_angular_casimir,
)
from .adaptive_joint_q import (
    AdaptiveJointCasimirConfig,
    AdaptiveJointCasimirResult,
    run_adaptive_joint_casimir,
)
from .adaptive_matsubara_tail import (
    AdaptiveMatsubaraCasimirConfig,
    AdaptiveMatsubaraCasimirResult,
    run_adaptive_matsubara_casimir,
)
from .adaptive_outer_q import (
    AdaptiveOuterQPanelGrid,
    AdaptiveRadialCasimirConfig,
    AdaptiveRadialCasimirResult,
    AdaptiveRadialPanel,
    build_adaptive_outer_q_panel_grid,
    run_adaptive_radial_casimir,
)
from .adaptive_outer_tail import (
    AdaptiveOuterTailCasimirConfig,
    AdaptiveOuterTailCasimirResult,
    run_adaptive_outer_tail_casimir,
)
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
        "layer": "casimir_lifshitz_fixed_chain",
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
        "requires_gauge_closed_response": True,
        "ward_identity_closed_by_this_module": False,
        "positive_matsubara_signed_logdet_supported": True,
        "zero_matsubara_signed_logdet_supported": True,
        "zero_matsubara_uses_static_susceptibility_not_conductivity": True,
        "zero_matsubara_prime_weight_applied_by_quadrature": True,
        "matsubara_energy_helper_owned_by_production": True,
        "finite_q_model_adapter_owned_by_production": True,
        "fixed_transverse_certifier_owned_by_production": True,
        "fixed_casimir_controller_owned_by_production": True,
        "incremental_certified_point_provider_present": True,
        "frequency_extendable_certified_point_provider_present": True,
        "outer_q_measure_contract_present": True,
        "outer_q_fixed_nested_planning_present": True,
        "outer_q_adaptive_radial_present": True,
        "outer_q_adaptive_angular_present": True,
        "outer_q_joint_radial_angular_budget_present": True,
        "outer_q_joint_direction_selection_present": True,
        "outer_q_adaptive_cutoff_present": True,
        "outer_q_geometric_tail_envelope_present": True,
        "outer_q_angular_order_doubling_present": True,
        "outer_q_angular_offset_audit_present": True,
        "outer_q_adaptive_cutoff_fixed": False,
        "outer_q_adaptive_tail_estimated": True,
        "outer_q_radial_variable": "u = 2 Q d",
        "outer_q_full_angular_domain": True,
        "finite_matsubara_partial_result_supported": True,
        "matsubara_adaptive_cutoff_present": True,
        "matsubara_geometric_tail_envelope_present": True,
        "matsubara_tail_estimated": True,
        "notes": (
            "The unique fixed controller is lno327.casimir.run_casimir.",
            "Adaptive controllers remain independent diagnostic surfaces.",
            "The frequency-extendable provider appends only newly requested Matsubara indices.",
            "The Matsubara controller requires outer-Q tail closure for every included term.",
            "Both outer-Q and Matsubara tails are bounded channelwise without sign cancellation.",
            "Production authorization remains false pending physical qualification of the adaptive tail model.",
        ),
    }


__all__ = [
    "AdaptiveAngularCasimirConfig",
    "AdaptiveAngularCasimirResult",
    "AdaptiveJointCasimirConfig",
    "AdaptiveJointCasimirResult",
    "AdaptiveMatsubaraCasimirConfig",
    "AdaptiveMatsubaraCasimirResult",
    "AdaptiveOuterQPanelGrid",
    "AdaptiveOuterTailCasimirConfig",
    "AdaptiveOuterTailCasimirResult",
    "AdaptiveRadialCasimirConfig",
    "AdaptiveRadialCasimirResult",
    "AdaptiveRadialPanel",
    "CasimirSetup",
    "CertifiedOuterQProvider",
    "CertifiedPointBatch",
    "CertifiedPointCacheError",
    "FiniteQMicroscopicModel",
    "FixedCasimirConfig",
    "FixedCasimirExecutionError",
    "FixedCasimirResult",
    "FrequencyExtendableCertifiedOuterQProvider",
    "LifshitzPoint",
    "MatsubaraFreeEnergyPerArea",
    "OuterQGridPlan",
    "OuterQGridSpec",
    "OuterQNodeManifest",
    "OuterQPolarGrid",
    "absolute_then_relative",
    "aggregate_certified_outer_q",
    "available_finite_q_microscopic_models",
    "build_adaptive_outer_q_panel_grid",
    "build_outer_q_polar_grid",
    "build_staged_grid_plan",
    "build_union_node_manifest",
    "casimir_energy_integrand",
    "casimir_layer_metadata",
    "casimir_torque_integrand",
    "certified_primary_logdet",
    "compare_ladders",
    "free_energy_per_area_from_logdet",
    "get_finite_q_microscopic_model",
    "integrate_outer_q",
    "matsubara_energy_eV",
    "matsubara_frequency",
    "matsubara_prime_weights",
    "passive_sheet_logdet",
    "reflection_matrix_weak_2d",
    "run_adaptive_angular_casimir",
    "run_adaptive_joint_casimir",
    "run_adaptive_matsubara_casimir",
    "run_adaptive_outer_tail_casimir",
    "run_adaptive_radial_casimir",
    "run_casimir",
]
