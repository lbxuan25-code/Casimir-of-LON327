"""Data containers for finite-q response diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .response_interface import ResponseKind

GaugeStatus = Literal["prototype_not_ward_verified"]
DiagnosticStatus = Literal["pass_local_limit", "fail_local_limit", "finite_q_diagnostic"]
DenominatorMode = Literal["raw", "stable"]
SmallQLimitStatus = Literal[
    "not_tested",
    "good_continuity_candidate",
    "prototype_continuity_candidate",
    "not_continuous_enough",
]


@dataclass(frozen=True)
class FiniteQResponseResult:
    """Tagged finite-q response diagnostic result.

    ``finite_q_resolved=True`` only means this prototype evaluated a finite-q
    response diagnostic. It is not a final gauge-invariant finite-q Casimir
    input.
    """

    kind: ResponseKind
    matsubara_index: int
    temperature_K: float
    q_magnitude: float
    q_phi: float
    q_vector: tuple[float, float]
    nk: int
    delta0: float
    eta: float
    response_tensor_model: np.ndarray
    sheet_conductivity_SI: np.ndarray
    reflection_dimensionless: np.ndarray
    finite_q_resolved: bool
    finite_q_response_diagnostic: bool
    local_limit_reference: np.ndarray
    local_reference_hook_passed: bool
    local_limit_abs_error: float
    local_limit_relative_error: float
    small_q_limit_abs_error: float
    small_q_limit_relative_error: float
    small_q_limit_status: SmallQLimitStatus
    q_to_0_continuity_tested: bool
    q_to_0_continuity_passed: bool
    angular_anisotropy_A4_xx: float
    angular_anisotropy_A4_trace: float
    symmetry_diagnostics: dict[str, complex | float | bool]
    gauge_status: GaugeStatus
    diagnostic_status: DiagnosticStatus
    final_casimir_input: bool
    not_final_Casimir_conclusion: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FiniteQLocalLimitDecomposition:
    """Diagnostic-only comparison of finite-q bubble to local response pieces."""

    kind: ResponseKind
    matsubara_index: int
    omega_eV: float
    temperature_K: float
    nk: int
    small_q: float
    q_phi: float
    delta0: float
    eta: float
    finite_q: np.ndarray
    local_sigma: np.ndarray
    local_K_para: np.ndarray
    local_K_dia: np.ndarray
    local_K_total: np.ndarray
    local_K_total_over_omega: np.ndarray
    normal_kubo_sigma: np.ndarray
    error_to_local_sigma: float
    error_to_K_para: float
    error_to_K_total: float
    error_to_K_total_over_omega: float
    error_to_normal_kubo_sigma: float
    best_match_component: str
    best_match_relative_error: float
    diagnostic_status: str
    gauge_status: GaugeStatus
    final_casimir_input: bool
    not_final_Casimir_conclusion: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FiniteQFormulaConsistencyDiagnostic:
    """Low-level formula/vertex diagnostics for the finite-q prototype."""

    kind: ResponseKind
    matsubara_index: int
    temperature_K: float
    q_magnitude: float
    q_phi: float
    nk: int
    delta0: float
    eta: float
    vertex_local_limit_abs_error: float
    vertex_local_limit_relative_error: float
    overlap_unitarity_error: float
    overlap_diagonal_error: float
    overlap_offdiag_norm: float
    wrapped_fraction: float
    possible_bz_wrapping_discontinuity: bool
    min_abs_energy_diff: float
    max_denominator_weight: float
    near_degenerate_count: int
    possible_denominator_instability: bool
    component_errors: dict[str, float]
    best_match_component: str
    small_q_relative_error: float
    diagnostic_status: str
    gauge_status: GaugeStatus
    final_casimir_input: bool
    not_final_Casimir_conclusion: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FiniteQSubspaceConsistencyDiagnostic:
    """Near-degenerate subspace and denominator stability diagnostic."""

    kind: ResponseKind
    matsubara_index: int
    temperature_K: float
    q_magnitude: float
    q_phi: float
    nk: int
    delta0: float
    eta: float
    deg_tol: float
    denominator_mode: DenominatorMode
    num_subspaces_minus: int
    num_subspaces_plus: int
    max_subspace_dimension_minus: int
    max_subspace_dimension_plus: int
    near_degenerate_group_count: int
    eigenstate_overlap_offdiag_norm: float
    eigenstate_overlap_diagonal_error: float
    projector_overlap_error: float
    projector_trace_defect: float
    subspace_mixing_norm: float
    unmatched_subspace_weight: float
    possible_band_phase_or_order_issue: bool
    possible_true_subspace_mixing: bool
    near_degenerate_count: int
    min_abs_energy_diff: float
    near_degenerate_weight_raw: float
    near_degenerate_weight_stable: float
    denominator_regularization_delta: float
    stable_denominator_changed_response_norm: float
    stable_denominator_improves_continuity: bool
    component_errors: dict[str, float]
    best_match_component: str
    small_q_relative_error: float
    diagnostic_status: str
    gauge_status: GaugeStatus
    final_casimir_input: bool
    not_final_Casimir_conclusion: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FiniteQRawQ0Consistency:
    """Diagnostic comparison between raw q=0 bubble and local response layers."""

    kind: ResponseKind
    matsubara_index: int
    temperature_K: float
    nk: int
    delta0: float
    eta: float
    denominator_mode: DenominatorMode
    deg_tol: float
    raw_q0_bubble: np.ndarray
    local_sigma: np.ndarray
    local_K_para: np.ndarray
    local_K_dia: np.ndarray
    local_K_total: np.ndarray
    local_K_total_over_omega: np.ndarray
    normal_kubo_sigma: np.ndarray
    hook_q0_response: np.ndarray
    error_raw_to_local_sigma: float
    error_raw_to_local_K_para: float
    error_raw_to_local_K_total: float
    error_raw_to_local_K_total_over_omega: float
    error_raw_to_normal_kubo_sigma: float
    error_hook_to_local_sigma: float
    best_raw_q0_match_component: str
    best_raw_q0_relative_error: float
    raw_q0_matches_local_sigma: bool
    raw_q0_matches_K_para: bool
    raw_q0_matches_K_total_over_omega: bool
    formula_layer_diagnosis: str
    diagnostic_status: str
    gauge_status: GaugeStatus
    final_casimir_input: bool
    not_final_Casimir_conclusion: bool
    notes: tuple[str, ...]
