"""Finite-q response diagnostics for angular anisotropy prototypes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bdg_response import KuboConfig, bdg_current_vertex, bdg_total_kernel_imag_axis, fermi_function
from .constants import KB_EV_PER_K
from .model import normal_state_hamiltonian, normal_state_velocity_operator
from .pairing import PairingAmplitudes, bdg_hamiltonian, pairing_matrix
from .response_interface import ResponseKind, local_response_imag_axis, matrix_symmetry_diagnostics
from .response_units import (
    model_response_to_reflection_dimensionless,
    model_response_to_sheet_conductivity,
)

GaugeStatus = Literal["prototype_not_ward_verified"]
DiagnosticStatus = Literal["pass_local_limit", "fail_local_limit", "finite_q_diagnostic"]
DenominatorMode = Literal["raw", "stable"]
SmallQLimitStatus = Literal[
    "not_tested",
    "good_continuity_candidate",
    "prototype_continuity_candidate",
    "not_continuous_enough",
]

RATIO_EPS = 1e-300
LOCAL_LIMIT_RELATIVE_TOLERANCE = 1e-8


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


def _wrap_bz(points: np.ndarray) -> np.ndarray:
    wrapped = np.asarray(points, dtype=float).copy()
    wrapped[..., 0] = ((wrapped[..., 0] + np.pi) % (2.0 * np.pi)) - np.pi
    wrapped[..., 1] = ((wrapped[..., 1] + np.pi) % (2.0 * np.pi)) - np.pi
    return wrapped


def _uniform_bz_mesh(nk: int) -> tuple[np.ndarray, np.ndarray]:
    if nk <= 0:
        raise ValueError("nk must be positive")
    step = 2.0 * np.pi / nk
    values = -np.pi + (np.arange(nk) + 0.5) * step
    mesh = np.array([(float(kx), float(ky)) for kx in values for ky in values], dtype=float)
    weights = np.full(mesh.shape[0], 1.0 / mesh.shape[0], dtype=float)
    return mesh, weights


def _bosonic_matsubara_energy_eV(matsubara_index: int, temperature_K: float) -> float:
    if matsubara_index < 1:
        raise ValueError("matsubara_index must be >= 1 for this diagnostic")
    return float(2.0 * np.pi * matsubara_index * KB_EV_PER_K * temperature_K)


def _normal_current_vertex(kx: float, ky: float, direction: str) -> np.ndarray:
    return normal_state_velocity_operator(kx, ky, direction)


def _matrix_for_kind(
    kind: ResponseKind,
    kx: float,
    ky: float,
    delta0_eV: float,
) -> np.ndarray:
    if kind == "normal":
        return normal_state_hamiltonian(kx, ky)
    params = PairingAmplitudes(delta0_eV=delta0_eV)
    delta = pairing_matrix(kind, kx, ky, params)
    return bdg_hamiltonian(kx, ky, delta)


def _vertices_for_kind(kind: ResponseKind, kx: float, ky: float) -> list[np.ndarray]:
    if kind == "normal":
        return [_normal_current_vertex(kx, ky, "x"), _normal_current_vertex(kx, ky, "y")]
    return [bdg_current_vertex(kx, ky, "x"), bdg_current_vertex(kx, ky, "y")]


def _fermi_derivative(energy: float, fermi_level: float, temperature_eV: float) -> float:
    if temperature_eV <= 0.0:
        return 0.0
    argument = np.clip((energy - fermi_level) / (2.0 * temperature_eV), -350.0, 350.0)
    cosh_value = np.cosh(argument)
    return float(-1.0 / (4.0 * temperature_eV * cosh_value * cosh_value))


def _stable_occupation_difference(
    energy_m: float,
    energy_n: float,
    occ_m: float,
    occ_n: float,
    fermi_level: float,
    temperature_eV: float,
    deg_tol: float,
) -> float:
    energy_diff = float(energy_m - energy_n)
    if abs(energy_diff) >= deg_tol:
        return float(occ_m - occ_n)
    energy_mid = 0.5 * (float(energy_m) + float(energy_n))
    return _fermi_derivative(energy_mid, fermi_level, temperature_eV) * energy_diff


def _response_factor(
    energy_m: float,
    energy_n: float,
    occ_m: float,
    occ_n: float,
    config: KuboConfig,
    denominator_mode: DenominatorMode,
    deg_tol: float,
) -> float:
    energy_diff = float(energy_m - energy_n)
    if denominator_mode == "stable":
        occupation_diff = _stable_occupation_difference(
            energy_m,
            energy_n,
            occ_m,
            occ_n,
            config.fermi_level_eV,
            config.temperature_eV,
            deg_tol,
        )
    elif denominator_mode == "raw":
        occupation_diff = float(occ_m - occ_n)
        if abs(energy_diff) < config.eta_eV:
            return 0.0
    else:
        raise ValueError("denominator_mode must be raw or stable")
    if np.isclose(occupation_diff, 0.0):
        return 0.0
    omega = config.omega_eV + config.eta_eV
    return float(-occupation_diff * energy_diff / (energy_diff**2 + omega**2))


def _finite_q_bubble(
    *,
    kind: ResponseKind,
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
    q_vector: np.ndarray,
    nk: int,
    delta0_eV: float,
    denominator_mode: DenominatorMode = "raw",
    deg_tol: float = 1e-7,
) -> np.ndarray:
    mesh, weights = _uniform_bz_mesh(nk)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    matrix = np.zeros((2, 2), dtype=complex)
    directions = ("x", "y")

    for weight, k_center in zip(weights, mesh, strict=True):
        k_minus = _wrap_bz(k_center - 0.5 * q_vector)
        k_plus = _wrap_bz(k_center + 0.5 * q_vector)
        vertex_k = _wrap_bz(k_center)

        if kind == "normal":
            h_minus = normal_state_hamiltonian(float(k_minus[0]), float(k_minus[1]))
            h_plus = normal_state_hamiltonian(float(k_plus[0]), float(k_plus[1]))
            energies_minus, states_minus = np.linalg.eigh(h_minus)
            energies_plus, states_plus = np.linalg.eigh(h_plus)
            occ_minus = fermi_function(energies_minus, config.fermi_level_eV, config.temperature_eV)
            occ_plus = fermi_function(energies_plus, config.fermi_level_eV, config.temperature_eV)
            vertices = [
                _normal_current_vertex(float(vertex_k[0]), float(vertex_k[1]), direction) for direction in directions
            ]
            left_states = states_minus
            right_states = states_plus
        else:
            params = PairingAmplitudes(delta0_eV=delta0_eV)
            delta_minus = pairing_matrix(kind, float(k_minus[0]), float(k_minus[1]), params)
            delta_plus = pairing_matrix(kind, float(k_plus[0]), float(k_plus[1]), params)
            h_minus = bdg_hamiltonian(float(k_minus[0]), float(k_minus[1]), delta_minus)
            h_plus = bdg_hamiltonian(float(k_plus[0]), float(k_plus[1]), delta_plus)
            energies_minus, states_minus = np.linalg.eigh(h_minus)
            energies_plus, states_plus = np.linalg.eigh(h_plus)
            occ_minus = fermi_function(energies_minus, config.fermi_level_eV, config.temperature_eV)
            occ_plus = fermi_function(energies_plus, config.fermi_level_eV, config.temperature_eV)
            vertices = [
                bdg_current_vertex(float(vertex_k[0]), float(vertex_k[1]), direction) for direction in directions
            ]
            left_states = states_minus
            right_states = states_plus

        vertex_band = [left_states.conjugate().T @ vertex @ right_states for vertex in vertices]
        reverse_vertex_band = [right_states.conjugate().T @ vertex @ left_states for vertex in vertices]

        for m, energy_m in enumerate(energies_minus):
            for n, energy_n in enumerate(energies_plus):
                response_factor = _response_factor(
                    float(energy_m),
                    float(energy_n),
                    float(occ_minus[m]),
                    float(occ_plus[n]),
                    config,
                    denominator_mode,
                    deg_tol,
                )
                if np.isclose(response_factor, 0.0):
                    continue
                for alpha in range(2):
                    for beta in range(2):
                        matrix[alpha, beta] += (
                            weight
                            * response_factor
                            * vertex_band[alpha][m, n]
                            * reverse_vertex_band[beta][n, m]
                        )

    if kind in {"spm", "dwave"}:
        matrix = matrix / omega_eV
    return matrix


def _formula_consistency_stats(
    kind: ResponseKind,
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
    q_vector: np.ndarray,
    nk: int,
    delta0_eV: float,
) -> dict[str, float | int | bool]:
    mesh, weights = _uniform_bz_mesh(nk)
    _ = weights
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    omega = config.omega_eV + config.eta_eV
    vertex_abs_errors = []
    vertex_rel_errors = []
    overlap_unitarity = []
    overlap_diagonal = []
    overlap_offdiag = []
    wrapped_count = 0
    min_abs_energy_diff = np.inf
    max_denominator_weight = 0.0
    near_degenerate_count = 0

    for k_center in mesh:
        raw_minus = k_center - 0.5 * q_vector
        raw_plus = k_center + 0.5 * q_vector
        k_minus = _wrap_bz(raw_minus)
        k_plus = _wrap_bz(raw_plus)
        vertex_k = _wrap_bz(k_center)
        if not (np.allclose(raw_minus, k_minus) and np.allclose(raw_plus, k_plus)):
            wrapped_count += 1

        h_minus = _matrix_for_kind(kind, float(k_minus[0]), float(k_minus[1]), delta0_eV)
        h_plus = _matrix_for_kind(kind, float(k_plus[0]), float(k_plus[1]), delta0_eV)
        energies_minus, states_minus = np.linalg.eigh(h_minus)
        energies_plus, states_plus = np.linalg.eigh(h_plus)
        occ_minus = fermi_function(energies_minus, config.fermi_level_eV, config.temperature_eV)
        occ_plus = fermi_function(energies_plus, config.fermi_level_eV, config.temperature_eV)

        local_vertices = _vertices_for_kind(kind, float(vertex_k[0]), float(vertex_k[1]))
        finite_vertices = _vertices_for_kind(kind, float(vertex_k[0]), float(vertex_k[1]))
        for finite_vertex, local_vertex in zip(finite_vertices, local_vertices, strict=True):
            abs_error = float(np.linalg.norm(finite_vertex - local_vertex))
            scale = float(np.linalg.norm(local_vertex))
            vertex_abs_errors.append(abs_error)
            vertex_rel_errors.append(abs_error / (scale + RATIO_EPS))

        overlap = states_plus.conjugate().T @ states_minus
        identity = np.eye(overlap.shape[0], dtype=complex)
        overlap_unitarity.append(float(np.linalg.norm(overlap.conjugate().T @ overlap - identity)))
        overlap_diagonal.append(float(np.max(np.abs(np.abs(np.diag(overlap)) - 1.0))))
        offdiag = overlap - np.diag(np.diag(overlap))
        overlap_offdiag.append(float(np.linalg.norm(offdiag)))

        for m, energy_m in enumerate(energies_minus):
            for n, energy_n in enumerate(energies_plus):
                occupation_diff = occ_minus[m] - occ_plus[n]
                if np.isclose(occupation_diff, 0.0):
                    continue
                energy_diff = float(energy_m - energy_n)
                abs_diff = abs(energy_diff)
                min_abs_energy_diff = min(min_abs_energy_diff, abs_diff)
                if abs_diff < eta_eV:
                    near_degenerate_count += 1
                weight = abs(float(occupation_diff) * energy_diff / (energy_diff**2 + omega**2))
                max_denominator_weight = max(max_denominator_weight, weight)

    wrapped_fraction = float(wrapped_count / max(mesh.shape[0], 1))
    min_diff = float(min_abs_energy_diff) if np.isfinite(min_abs_energy_diff) else np.nan
    return {
        "vertex_local_limit_abs_error": float(max(vertex_abs_errors, default=np.nan)),
        "vertex_local_limit_relative_error": float(max(vertex_rel_errors, default=np.nan)),
        "overlap_unitarity_error": float(max(overlap_unitarity, default=np.nan)),
        "overlap_diagonal_error": float(max(overlap_diagonal, default=np.nan)),
        "overlap_offdiag_norm": float(max(overlap_offdiag, default=np.nan)),
        "wrapped_fraction": wrapped_fraction,
        "possible_bz_wrapping_discontinuity": bool(wrapped_fraction > 0.0),
        "min_abs_energy_diff": min_diff,
        "max_denominator_weight": float(max_denominator_weight),
        "near_degenerate_count": int(near_degenerate_count),
        "possible_denominator_instability": bool(near_degenerate_count > 0),
    }


def group_near_degenerate_levels(energies: np.ndarray, deg_tol: float = 1e-7) -> list[np.ndarray]:
    """Group sorted eigenvalue indices whose adjacent gaps are below ``deg_tol``."""

    if deg_tol <= 0.0:
        raise ValueError("deg_tol must be positive")
    values = np.asarray(energies, dtype=float)
    if values.ndim != 1:
        raise ValueError("energies must be one-dimensional")
    if values.size == 0:
        return []
    groups: list[list[int]] = [[0]]
    for index in range(1, values.size):
        if abs(values[index] - values[index - 1]) <= deg_tol:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [np.asarray(group, dtype=int) for group in groups]


def _projector(states: np.ndarray, group: np.ndarray) -> np.ndarray:
    vectors = states[:, group]
    return vectors @ vectors.conjugate().T


def compute_subspace_projector_overlap(
    states_minus: np.ndarray,
    states_plus: np.ndarray,
    groups_minus: Sequence[np.ndarray],
    groups_plus: Sequence[np.ndarray],
) -> dict[str, float]:
    """Compare near-degenerate projectors between k-q/2 and k+q/2."""

    used_plus: set[int] = set()
    projector_errors = []
    trace_defects = []
    mixing_norms = []
    unmatched_weight = 0.0
    for minus_group in groups_minus:
        dim_minus = int(minus_group.size)
        projector_minus = _projector(states_minus, minus_group)
        best_index = -1
        best_overlap = -np.inf
        best_projector = None
        for plus_index, plus_group in enumerate(groups_plus):
            if plus_index in used_plus:
                continue
            projector_plus = _projector(states_plus, plus_group)
            overlap = float(np.real(np.trace(projector_minus @ projector_plus)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = plus_index
                best_projector = projector_plus
        if best_projector is None:
            unmatched_weight += float(dim_minus)
            continue
        used_plus.add(best_index)
        dim_reference = max(float(dim_minus), 1.0)
        projector_errors.append(abs(dim_reference - best_overlap) / dim_reference)
        trace_defects.append(abs(float(np.real(np.trace(projector_minus))) - dim_minus))
        complement = np.eye(projector_minus.shape[0], dtype=complex) - best_projector
        mixing_norms.append(float(np.linalg.norm(projector_minus @ complement) / dim_reference))
    for plus_index, plus_group in enumerate(groups_plus):
        if plus_index not in used_plus:
            unmatched_weight += float(plus_group.size)
    total_dim = max(float(states_minus.shape[0]), 1.0)
    return {
        "projector_overlap_error": float(max(projector_errors, default=0.0)),
        "projector_trace_defect": float(max(trace_defects, default=0.0)),
        "subspace_mixing_norm": float(max(mixing_norms, default=0.0)),
        "unmatched_subspace_weight": float(unmatched_weight / total_dim),
    }


def _subspace_consistency_stats(
    kind: ResponseKind,
    q_vector: np.ndarray,
    nk: int,
    delta0_eV: float,
    deg_tol: float,
) -> dict[str, float | int | bool]:
    mesh, _ = _uniform_bz_mesh(nk)
    num_minus = []
    num_plus = []
    max_dim_minus = []
    max_dim_plus = []
    near_group_count = 0
    eigen_diag_errors = []
    eigen_offdiag_norms = []
    projector_errors = []
    trace_defects = []
    mixing_norms = []
    unmatched_weights = []

    for k_center in mesh:
        k_minus = _wrap_bz(k_center - 0.5 * q_vector)
        k_plus = _wrap_bz(k_center + 0.5 * q_vector)
        h_minus = _matrix_for_kind(kind, float(k_minus[0]), float(k_minus[1]), delta0_eV)
        h_plus = _matrix_for_kind(kind, float(k_plus[0]), float(k_plus[1]), delta0_eV)
        energies_minus, states_minus = np.linalg.eigh(h_minus)
        energies_plus, states_plus = np.linalg.eigh(h_plus)
        groups_minus = group_near_degenerate_levels(energies_minus, deg_tol)
        groups_plus = group_near_degenerate_levels(energies_plus, deg_tol)

        num_minus.append(len(groups_minus))
        num_plus.append(len(groups_plus))
        max_dim_minus.append(max((int(group.size) for group in groups_minus), default=0))
        max_dim_plus.append(max((int(group.size) for group in groups_plus), default=0))
        near_group_count += sum(int(group.size > 1) for group in groups_minus)
        near_group_count += sum(int(group.size > 1) for group in groups_plus)

        overlap = states_plus.conjugate().T @ states_minus
        eigen_diag_errors.append(float(np.max(np.abs(np.abs(np.diag(overlap)) - 1.0))))
        offdiag = overlap - np.diag(np.diag(overlap))
        eigen_offdiag_norms.append(float(np.linalg.norm(offdiag)))

        projector_stats = compute_subspace_projector_overlap(states_minus, states_plus, groups_minus, groups_plus)
        projector_errors.append(projector_stats["projector_overlap_error"])
        trace_defects.append(projector_stats["projector_trace_defect"])
        mixing_norms.append(projector_stats["subspace_mixing_norm"])
        unmatched_weights.append(projector_stats["unmatched_subspace_weight"])

    eigen_offdiag = float(max(eigen_offdiag_norms, default=np.nan))
    projector_error = float(max(projector_errors, default=np.nan))
    return {
        "num_subspaces_minus": int(max(num_minus, default=0)),
        "num_subspaces_plus": int(max(num_plus, default=0)),
        "max_subspace_dimension_minus": int(max(max_dim_minus, default=0)),
        "max_subspace_dimension_plus": int(max(max_dim_plus, default=0)),
        "near_degenerate_group_count": int(near_group_count),
        "eigenstate_overlap_offdiag_norm": eigen_offdiag,
        "eigenstate_overlap_diagonal_error": float(max(eigen_diag_errors, default=np.nan)),
        "projector_overlap_error": projector_error,
        "projector_trace_defect": float(max(trace_defects, default=np.nan)),
        "subspace_mixing_norm": float(max(mixing_norms, default=np.nan)),
        "unmatched_subspace_weight": float(max(unmatched_weights, default=np.nan)),
        "possible_band_phase_or_order_issue": bool(eigen_offdiag > 1e-2 and projector_error < 0.1 * max(eigen_offdiag, RATIO_EPS)),
        "possible_true_subspace_mixing": bool(projector_error > 1e-2),
    }


def _denominator_stability_stats(
    kind: ResponseKind,
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
    q_vector: np.ndarray,
    nk: int,
    delta0_eV: float,
    deg_tol: float,
) -> dict[str, float | int]:
    mesh, weights = _uniform_bz_mesh(nk)
    _ = weights
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    min_abs_energy_diff = np.inf
    near_degenerate_count = 0
    near_weight_raw = 0.0
    near_weight_stable = 0.0
    regularization_delta = 0.0

    for k_center in mesh:
        k_minus = _wrap_bz(k_center - 0.5 * q_vector)
        k_plus = _wrap_bz(k_center + 0.5 * q_vector)
        h_minus = _matrix_for_kind(kind, float(k_minus[0]), float(k_minus[1]), delta0_eV)
        h_plus = _matrix_for_kind(kind, float(k_plus[0]), float(k_plus[1]), delta0_eV)
        energies_minus, _ = np.linalg.eigh(h_minus)
        energies_plus, _ = np.linalg.eigh(h_plus)
        occ_minus = fermi_function(energies_minus, config.fermi_level_eV, config.temperature_eV)
        occ_plus = fermi_function(energies_plus, config.fermi_level_eV, config.temperature_eV)

        for m, energy_m in enumerate(energies_minus):
            for n, energy_n in enumerate(energies_plus):
                energy_diff = float(energy_m - energy_n)
                abs_diff = abs(energy_diff)
                min_abs_energy_diff = min(min_abs_energy_diff, abs_diff)
                if abs_diff >= deg_tol:
                    continue
                raw_occ = float(occ_minus[m] - occ_plus[n])
                stable_occ = _stable_occupation_difference(
                    float(energy_m),
                    float(energy_n),
                    float(occ_minus[m]),
                    float(occ_plus[n]),
                    config.fermi_level_eV,
                    config.temperature_eV,
                    deg_tol,
                )
                raw_factor = abs(_response_factor(float(energy_m), float(energy_n), float(occ_minus[m]), float(occ_plus[n]), config, "raw", deg_tol))
                stable_factor = abs(
                    _response_factor(float(energy_m), float(energy_n), float(occ_minus[m]), float(occ_plus[n]), config, "stable", deg_tol)
                )
                near_degenerate_count += 1
                near_weight_raw += raw_factor
                near_weight_stable += stable_factor
                regularization_delta += abs(raw_occ - stable_occ)

    min_diff = float(min_abs_energy_diff) if np.isfinite(min_abs_energy_diff) else np.nan
    return {
        "near_degenerate_count": int(near_degenerate_count),
        "min_abs_energy_diff": min_diff,
        "near_degenerate_weight_raw": float(near_weight_raw),
        "near_degenerate_weight_stable": float(near_weight_stable),
        "denominator_regularization_delta": float(regularization_delta),
    }


def _local_reference(
    kind: ResponseKind,
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
    nk: int,
    delta0_eV: float,
) -> np.ndarray:
    mesh, weights = _uniform_bz_mesh(nk)
    response = local_response_imag_axis(
        kind,
        omega_eV,
        mesh,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        pairing_params=PairingAmplitudes(delta0_eV=delta0_eV),
        k_weights=weights,
    )
    return response.matrix


def _relative_error(matrix: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(matrix, dtype=complex) - np.asarray(reference, dtype=complex)
    abs_error = float(np.linalg.norm(diff))
    scale = float(np.linalg.norm(reference))
    return abs_error, float(abs_error / (scale + RATIO_EPS))


def _nan_matrix() -> np.ndarray:
    return np.full((2, 2), np.nan + 0.0j, dtype=complex)


def _relative_error_or_nan(matrix: np.ndarray, reference: np.ndarray) -> float:
    if not np.all(np.isfinite(reference)):
        return np.nan
    return _relative_error(matrix, reference)[1]


def _best_component(errors: dict[str, float]) -> tuple[str, float]:
    finite_items = [(name, value) for name, value in errors.items() if np.isfinite(value)]
    if not finite_items:
        return "none", np.nan
    return min(finite_items, key=lambda item: item[1])


def _component_status(best_component: str, errors: dict[str, float]) -> str:
    best_error = errors.get(best_component, np.nan)
    if not np.isfinite(best_error):
        return "no_clear_local_limit_match"
    local_sigma = errors.get("local_sigma", np.nan)
    total_over_omega = errors.get("local_K_total_over_omega", np.nan)
    para = errors.get("local_K_para", np.nan)
    if best_component == "local_K_para":
        reference = np.nanmin([local_sigma, total_over_omega])
        if np.isfinite(reference) and para < 0.1 * reference:
            return "likely_matches_paramagnetic_kernel;likely_missing_contact_or_diamagnetic_completion"
        return "likely_matches_paramagnetic_kernel"
    if best_error >= 1.0:
        return "no_clear_local_limit_match"
    return f"best_matches_{best_component}"


def _local_component_matrices(
    kind: ResponseKind,
    omega_eV: float,
    temperature_K: float,
    eta: float,
    nk: int,
    delta0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mesh, weights = _uniform_bz_mesh(nk)
    local_sigma = _local_reference(kind, omega_eV, temperature_K, eta, nk, delta0)
    nan = _nan_matrix()
    if kind == "normal":
        return local_sigma, nan, nan, nan, nan, local_sigma

    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta,
        output_si=False,
    )
    components = bdg_total_kernel_imag_axis(
        mesh,
        config,
        kind,  # type: ignore[arg-type]
        PairingAmplitudes(delta0_eV=delta0),
        weights,
    )
    return (
        local_sigma,
        components.paramagnetic,
        components.diamagnetic,
        components.total,
        components.total / omega_eV,
        nan,
    )


def _a4_from_matrix_pair(matrix_phi0: np.ndarray, matrix_phi45: np.ndarray) -> tuple[float, float]:
    xx0 = complex(matrix_phi0[0, 0])
    xx45 = complex(matrix_phi45[0, 0])
    trace0 = complex(np.trace(matrix_phi0))
    trace45 = complex(np.trace(matrix_phi45))
    a4_xx = abs(xx0 - xx45) / (abs(xx0) + abs(xx45) + RATIO_EPS)
    a4_trace = abs(trace0 - trace45) / (abs(trace0) + abs(trace45) + RATIO_EPS)
    return float(a4_xx), float(a4_trace)


def bdg_finite_q_response_imag_axis(
    kind: ResponseKind,
    matsubara_index: int,
    temperature_K: float,
    q_magnitude: float,
    q_phi: float,
    nk: int,
    delta0: float,
    eta: float,
    local_limit_tolerance: float = LOCAL_LIMIT_RELATIVE_TOLERANCE,
) -> FiniteQResponseResult:
    """Return a finite-q response diagnostic using symmetric k +/- q/2 sampling.

    The finite-q vertex is a gradient/Peierls prototype. It is not Ward-identity
    verified and does not close the finite-q diamagnetic kernel.
    """

    if kind not in {"normal", "spm", "dwave"}:
        raise ValueError("kind must be one of: normal, spm, dwave")
    if q_magnitude < 0.0:
        raise ValueError("q_magnitude must be non-negative")
    if eta <= 0.0:
        raise ValueError("eta must be positive")

    omega_eV = _bosonic_matsubara_energy_eV(matsubara_index, temperature_K)
    q_vector = np.array([q_magnitude * np.cos(q_phi), q_magnitude * np.sin(q_phi)], dtype=float)
    local_reference = _local_reference(kind, omega_eV, temperature_K, eta, nk, delta0)
    if np.isclose(q_magnitude, 0.0):
        response_model = np.array(local_reference, dtype=complex, copy=True)
    else:
        response_model = _finite_q_bubble(
            kind=kind,
            omega_eV=omega_eV,
            temperature_K=temperature_K,
            eta_eV=eta,
            q_vector=q_vector,
            nk=nk,
            delta0_eV=delta0,
        )

    abs_error, rel_error = _relative_error(response_model, local_reference)
    sheet = model_response_to_sheet_conductivity(response_model)
    reflection = model_response_to_reflection_dimensionless(response_model)
    diagnostic_status: DiagnosticStatus
    if q_magnitude <= local_limit_tolerance:
        diagnostic_status = "pass_local_limit" if rel_error <= local_limit_tolerance else "fail_local_limit"
    else:
        diagnostic_status = "finite_q_diagnostic"

    symmetry = matrix_symmetry_diagnostics(response_model)
    notes = (
        "finite-q response diagnostic prototype",
        "q_magnitude is dimensionless BZ momentum",
        "symmetric k +/- q/2 sampling",
        "gradient/Peierls finite-q current vertex prototype",
        "finite-q diamagnetic and Ward identity are not closed",
        "not a final gauge-invariant finite-q Casimir input",
        "not a final Casimir torque conclusion",
    )
    return FiniteQResponseResult(
        kind=kind,
        matsubara_index=matsubara_index,
        temperature_K=temperature_K,
        q_magnitude=q_magnitude,
        q_phi=q_phi,
        q_vector=(float(q_vector[0]), float(q_vector[1])),
        nk=nk,
        delta0=delta0,
        eta=eta,
        response_tensor_model=response_model,
        sheet_conductivity_SI=sheet.tensor.matrix(),
        reflection_dimensionless=reflection.tensor.matrix(),
        finite_q_resolved=not np.isclose(q_magnitude, 0.0),
        finite_q_response_diagnostic=True,
        local_limit_reference=local_reference,
        local_reference_hook_passed=bool(np.isclose(q_magnitude, 0.0) and rel_error <= local_limit_tolerance),
        local_limit_abs_error=abs_error,
        local_limit_relative_error=rel_error,
        small_q_limit_abs_error=np.nan,
        small_q_limit_relative_error=np.nan,
        small_q_limit_status="not_tested",
        q_to_0_continuity_tested=False,
        q_to_0_continuity_passed=False,
        angular_anisotropy_A4_xx=np.nan,
        angular_anisotropy_A4_trace=np.nan,
        symmetry_diagnostics=symmetry,
        gauge_status="prototype_not_ward_verified",
        diagnostic_status=diagnostic_status,
        final_casimir_input=False,
        not_final_Casimir_conclusion=True,
        notes=notes,
    )


def finite_q_response_phi_scan(
    kind: ResponseKind,
    matsubara_index: int,
    temperature_K: float,
    q_magnitude: float,
    q_phi_list: Sequence[float],
    nk: int,
    delta0: float,
    eta: float,
) -> list[FiniteQResponseResult]:
    """Evaluate finite-q response over q angle and fill simple A4 diagnostics."""

    results = [
        bdg_finite_q_response_imag_axis(
            kind,
            matsubara_index,
            temperature_K,
            q_magnitude,
            float(q_phi),
            nk,
            delta0,
            eta,
        )
        for q_phi in q_phi_list
    ]
    phi_values = np.asarray(q_phi_list, dtype=float)
    phi0_index = int(np.argmin(np.abs(phi_values - 0.0)))
    phi45_index = int(np.argmin(np.abs(phi_values - np.pi / 4.0)))
    a4_xx, a4_trace = _a4_from_matrix_pair(
        results[phi0_index].response_tensor_model,
        results[phi45_index].response_tensor_model,
    )
    return [
        FiniteQResponseResult(
            **{
                **result.__dict__,
                "angular_anisotropy_A4_xx": a4_xx,
                "angular_anisotropy_A4_trace": a4_trace,
            }
        )
        for result in results
    ]


def finite_q_local_limit_decomposition(
    kind: ResponseKind,
    matsubara_index: int,
    temperature_K: float,
    small_q: float,
    q_phi: float,
    nk: int,
    delta0: float,
    eta: float,
) -> FiniteQLocalLimitDecomposition:
    """Compare finite-q bubble at small q against local response components.

    This helper is diagnostic-only. It does not repair, rescale, or reinterpret
    the finite-q bubble as a final gauge-invariant nonlocal response.
    """

    if small_q <= 0.0:
        raise ValueError("small_q must be positive")
    omega_eV = _bosonic_matsubara_energy_eV(matsubara_index, temperature_K)
    q_vector = np.array([small_q * np.cos(q_phi), small_q * np.sin(q_phi)], dtype=float)
    finite_q = _finite_q_bubble(
        kind=kind,
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta,
        q_vector=q_vector,
        nk=nk,
        delta0_eV=delta0,
    )
    local_sigma, k_para, k_dia, k_total, k_total_over_omega, normal_kubo = _local_component_matrices(
        kind,
        omega_eV,
        temperature_K,
        eta,
        nk,
        delta0,
    )
    errors = {
        "local_sigma": _relative_error_or_nan(finite_q, local_sigma),
        "local_K_para": _relative_error_or_nan(finite_q, k_para),
        "local_K_total": _relative_error_or_nan(finite_q, k_total),
        "local_K_total_over_omega": _relative_error_or_nan(finite_q, k_total_over_omega),
        "normal_kubo_sigma": _relative_error_or_nan(finite_q, normal_kubo),
    }
    best_component, best_error = _best_component(errors)
    notes = [
        "finite-q local-limit decomposition diagnostic",
        "q uses dimensionless BZ momentum",
        "no rescaling or formula repair applied",
        "prototype_not_ward_verified",
        "not a final Casimir torque conclusion",
    ]
    if kind == "normal":
        notes.append("BdG K components are not applicable to normal")
    else:
        notes.append("normal_kubo_sigma is not applicable to BdG kinds")
    return FiniteQLocalLimitDecomposition(
        kind=kind,
        matsubara_index=matsubara_index,
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        nk=nk,
        small_q=small_q,
        q_phi=q_phi,
        delta0=delta0,
        eta=eta,
        finite_q=finite_q,
        local_sigma=local_sigma,
        local_K_para=k_para,
        local_K_dia=k_dia,
        local_K_total=k_total,
        local_K_total_over_omega=k_total_over_omega,
        normal_kubo_sigma=normal_kubo,
        error_to_local_sigma=errors["local_sigma"],
        error_to_K_para=errors["local_K_para"],
        error_to_K_total=errors["local_K_total"],
        error_to_K_total_over_omega=errors["local_K_total_over_omega"],
        error_to_normal_kubo_sigma=errors["normal_kubo_sigma"],
        best_match_component=best_component,
        best_match_relative_error=best_error,
        diagnostic_status=_component_status(best_component, errors),
        gauge_status="prototype_not_ward_verified",
        final_casimir_input=False,
        not_final_Casimir_conclusion=True,
        notes=tuple(notes),
    )


def compare_finite_q_to_local_components(
    kinds: Sequence[ResponseKind],
    matsubara_list: Sequence[int],
    small_q_list: Sequence[float],
    q_phi_list: Sequence[float],
    nk_list: Sequence[int],
    temperature_K: float,
    delta0: float,
    eta: float,
) -> list[FiniteQLocalLimitDecomposition]:
    """Run local-limit decomposition over a compact diagnostic grid."""

    rows: list[FiniteQLocalLimitDecomposition] = []
    for kind in kinds:
        for matsubara_index in matsubara_list:
            for nk in nk_list:
                for small_q in small_q_list:
                    for q_phi in q_phi_list:
                        rows.append(
                            finite_q_local_limit_decomposition(
                                kind,
                                int(matsubara_index),
                                temperature_K,
                                float(small_q),
                                float(q_phi),
                                int(nk),
                                delta0,
                                eta,
                            )
                        )
    return rows


def finite_q_formula_consistency_diagnostic(
    kind: ResponseKind,
    matsubara_index: int,
    temperature_K: float,
    q_magnitude: float,
    q_phi: float,
    nk: int,
    delta0: float,
    eta: float,
) -> FiniteQFormulaConsistencyDiagnostic:
    """Return low-level diagnostics for finite-q formula consistency."""

    if q_magnitude <= 0.0:
        raise ValueError("q_magnitude must be positive")
    omega_eV = _bosonic_matsubara_energy_eV(matsubara_index, temperature_K)
    q_vector = np.array([q_magnitude * np.cos(q_phi), q_magnitude * np.sin(q_phi)], dtype=float)
    stats = _formula_consistency_stats(
        kind,
        omega_eV,
        temperature_K,
        eta,
        q_vector,
        nk,
        delta0,
    )
    decomposition = finite_q_local_limit_decomposition(
        kind,
        matsubara_index,
        temperature_K,
        q_magnitude,
        q_phi,
        nk,
        delta0,
        eta,
    )
    component_errors = {
        "local_sigma": decomposition.error_to_local_sigma,
        "local_K_para": decomposition.error_to_K_para,
        "local_K_total": decomposition.error_to_K_total,
        "local_K_total_over_omega": decomposition.error_to_K_total_over_omega,
        "normal_kubo_sigma": decomposition.error_to_normal_kubo_sigma,
    }
    status_parts = []
    if stats["vertex_local_limit_relative_error"] > 1e-10:
        status_parts.append("possible_vertex_mismatch")
    else:
        status_parts.append("vertex_matches_local_convention")
    if stats["overlap_diagonal_error"] > 1e-2 or stats["overlap_offdiag_norm"] > 1e-1:
        status_parts.append("possible_overlap_band_order_or_phase_issue")
    if stats["possible_bz_wrapping_discontinuity"]:
        status_parts.append("possible_bz_wrapping_discontinuity")
    if stats["possible_denominator_instability"]:
        status_parts.append("possible_denominator_instability")
    if decomposition.best_match_relative_error < 1e-2:
        status_parts.append("small_q_continuity_candidate")
    else:
        status_parts.append("small_q_continuity_not_repaired")
    notes = (
        "finite-q formula consistency diagnostic",
        "no ad hoc rescaling or smoothing applied",
        "finite-q vertex remains prototype_not_ward_verified",
        "final_casimir_input=False",
        "not a final Casimir torque conclusion",
    )
    return FiniteQFormulaConsistencyDiagnostic(
        kind=kind,
        matsubara_index=matsubara_index,
        temperature_K=temperature_K,
        q_magnitude=q_magnitude,
        q_phi=q_phi,
        nk=nk,
        delta0=delta0,
        eta=eta,
        vertex_local_limit_abs_error=float(stats["vertex_local_limit_abs_error"]),
        vertex_local_limit_relative_error=float(stats["vertex_local_limit_relative_error"]),
        overlap_unitarity_error=float(stats["overlap_unitarity_error"]),
        overlap_diagonal_error=float(stats["overlap_diagonal_error"]),
        overlap_offdiag_norm=float(stats["overlap_offdiag_norm"]),
        wrapped_fraction=float(stats["wrapped_fraction"]),
        possible_bz_wrapping_discontinuity=bool(stats["possible_bz_wrapping_discontinuity"]),
        min_abs_energy_diff=float(stats["min_abs_energy_diff"]),
        max_denominator_weight=float(stats["max_denominator_weight"]),
        near_degenerate_count=int(stats["near_degenerate_count"]),
        possible_denominator_instability=bool(stats["possible_denominator_instability"]),
        component_errors=component_errors,
        best_match_component=decomposition.best_match_component,
        small_q_relative_error=decomposition.best_match_relative_error,
        diagnostic_status=";".join(status_parts),
        gauge_status="prototype_not_ward_verified",
        final_casimir_input=False,
        not_final_Casimir_conclusion=True,
        notes=notes,
    )


def compare_finite_q_formula_consistency(
    kinds: Sequence[ResponseKind],
    matsubara_list: Sequence[int],
    q_list: Sequence[float],
    q_phi_list: Sequence[float],
    nk_list: Sequence[int],
    temperature_K: float,
    delta0: float,
    eta: float,
) -> list[FiniteQFormulaConsistencyDiagnostic]:
    """Evaluate finite-q formula diagnostics over a compact quick grid."""

    rows: list[FiniteQFormulaConsistencyDiagnostic] = []
    for kind in kinds:
        for matsubara_index in matsubara_list:
            for nk in nk_list:
                for q_magnitude in q_list:
                    for q_phi in q_phi_list:
                        rows.append(
                            finite_q_formula_consistency_diagnostic(
                                kind,
                                int(matsubara_index),
                                temperature_K,
                                float(q_magnitude),
                                float(q_phi),
                                int(nk),
                                delta0,
                                eta,
                            )
                        )
    return rows


def _component_errors_for_matrix(
    finite_q: np.ndarray,
    kind: ResponseKind,
    omega_eV: float,
    temperature_K: float,
    eta: float,
    nk: int,
    delta0: float,
) -> tuple[dict[str, float], str, float]:
    local_sigma, k_para, _k_dia, k_total, k_total_over_omega, normal_kubo = _local_component_matrices(
        kind,
        omega_eV,
        temperature_K,
        eta,
        nk,
        delta0,
    )
    errors = {
        "local_sigma": _relative_error_or_nan(finite_q, local_sigma),
        "local_K_para": _relative_error_or_nan(finite_q, k_para),
        "local_K_total": _relative_error_or_nan(finite_q, k_total),
        "local_K_total_over_omega": _relative_error_or_nan(finite_q, k_total_over_omega),
        "normal_kubo_sigma": _relative_error_or_nan(finite_q, normal_kubo),
    }
    best_component, best_error = _best_component(errors)
    return errors, best_component, best_error


def finite_q_subspace_consistency_diagnostic(
    kind: ResponseKind,
    matsubara_index: int,
    temperature_K: float,
    q_magnitude: float,
    q_phi: float,
    nk: int,
    delta0: float,
    eta: float,
    deg_tol: float = 1e-7,
    denominator_mode: DenominatorMode = "raw",
) -> FiniteQSubspaceConsistencyDiagnostic:
    """Diagnose subspace/projector smoothness and stable denominator behavior."""

    if q_magnitude <= 0.0:
        raise ValueError("q_magnitude must be positive")
    if denominator_mode not in {"raw", "stable"}:
        raise ValueError("denominator_mode must be raw or stable")
    omega_eV = _bosonic_matsubara_energy_eV(matsubara_index, temperature_K)
    q_vector = np.array([q_magnitude * np.cos(q_phi), q_magnitude * np.sin(q_phi)], dtype=float)
    subspace_stats = _subspace_consistency_stats(kind, q_vector, nk, delta0, deg_tol)
    denominator_stats = _denominator_stability_stats(
        kind,
        omega_eV,
        temperature_K,
        eta,
        q_vector,
        nk,
        delta0,
        deg_tol,
    )
    raw_response = _finite_q_bubble(
        kind=kind,
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta,
        q_vector=q_vector,
        nk=nk,
        delta0_eV=delta0,
        denominator_mode="raw",
        deg_tol=deg_tol,
    )
    stable_response = _finite_q_bubble(
        kind=kind,
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta,
        q_vector=q_vector,
        nk=nk,
        delta0_eV=delta0,
        denominator_mode="stable",
        deg_tol=deg_tol,
    )
    selected_response = stable_response if denominator_mode == "stable" else raw_response
    component_errors, best_component, best_error = _component_errors_for_matrix(
        selected_response,
        kind,
        omega_eV,
        temperature_K,
        eta,
        nk,
        delta0,
    )
    raw_errors, _raw_best_component, raw_best_error = _component_errors_for_matrix(
        raw_response,
        kind,
        omega_eV,
        temperature_K,
        eta,
        nk,
        delta0,
    )
    stable_errors, _stable_best_component, stable_best_error = _component_errors_for_matrix(
        stable_response,
        kind,
        omega_eV,
        temperature_K,
        eta,
        nk,
        delta0,
    )
    _ = raw_errors, stable_errors
    stable_delta = float(np.linalg.norm(stable_response - raw_response) / (np.linalg.norm(raw_response) + RATIO_EPS))
    stable_improves = bool(np.isfinite(stable_best_error) and np.isfinite(raw_best_error) and stable_best_error < 0.8 * raw_best_error)

    status_parts = []
    if subspace_stats["possible_band_phase_or_order_issue"]:
        status_parts.append("likely_gauge_or_band_order_rotation")
    if subspace_stats["possible_true_subspace_mixing"]:
        status_parts.append("possible_true_subspace_mixing")
    if denominator_stats["near_degenerate_count"] > 0:
        status_parts.append("near_degenerate_denominator_present")
    if stable_improves:
        status_parts.append("stable_denominator_improves_continuity")
    if best_error < 1e-2:
        status_parts.append("small_q_continuity_candidate")
    else:
        status_parts.append("small_q_continuity_not_repaired")
    notes = (
        "finite-q subspace / denominator repair diagnostic",
        "projector diagnostics are not used to overwrite the response",
        "stable denominator uses Fermi derivative only for near-degenerate occupation differences",
        "no near-degenerate terms are skipped in stable mode",
        "final_casimir_input=False",
        "not a final Casimir torque conclusion",
    )
    return FiniteQSubspaceConsistencyDiagnostic(
        kind=kind,
        matsubara_index=matsubara_index,
        temperature_K=temperature_K,
        q_magnitude=q_magnitude,
        q_phi=q_phi,
        nk=nk,
        delta0=delta0,
        eta=eta,
        deg_tol=deg_tol,
        denominator_mode=denominator_mode,
        num_subspaces_minus=int(subspace_stats["num_subspaces_minus"]),
        num_subspaces_plus=int(subspace_stats["num_subspaces_plus"]),
        max_subspace_dimension_minus=int(subspace_stats["max_subspace_dimension_minus"]),
        max_subspace_dimension_plus=int(subspace_stats["max_subspace_dimension_plus"]),
        near_degenerate_group_count=int(subspace_stats["near_degenerate_group_count"]),
        eigenstate_overlap_offdiag_norm=float(subspace_stats["eigenstate_overlap_offdiag_norm"]),
        eigenstate_overlap_diagonal_error=float(subspace_stats["eigenstate_overlap_diagonal_error"]),
        projector_overlap_error=float(subspace_stats["projector_overlap_error"]),
        projector_trace_defect=float(subspace_stats["projector_trace_defect"]),
        subspace_mixing_norm=float(subspace_stats["subspace_mixing_norm"]),
        unmatched_subspace_weight=float(subspace_stats["unmatched_subspace_weight"]),
        possible_band_phase_or_order_issue=bool(subspace_stats["possible_band_phase_or_order_issue"]),
        possible_true_subspace_mixing=bool(subspace_stats["possible_true_subspace_mixing"]),
        near_degenerate_count=int(denominator_stats["near_degenerate_count"]),
        min_abs_energy_diff=float(denominator_stats["min_abs_energy_diff"]),
        near_degenerate_weight_raw=float(denominator_stats["near_degenerate_weight_raw"]),
        near_degenerate_weight_stable=float(denominator_stats["near_degenerate_weight_stable"]),
        denominator_regularization_delta=float(denominator_stats["denominator_regularization_delta"]),
        stable_denominator_changed_response_norm=stable_delta,
        stable_denominator_improves_continuity=stable_improves,
        component_errors=component_errors,
        best_match_component=best_component,
        small_q_relative_error=best_error,
        diagnostic_status=";".join(status_parts),
        gauge_status="prototype_not_ward_verified",
        final_casimir_input=False,
        not_final_Casimir_conclusion=True,
        notes=notes,
    )


def compare_subspace_and_eigenstate_overlap(
    kinds: Sequence[ResponseKind],
    matsubara_list: Sequence[int],
    q_list: Sequence[float],
    q_phi_list: Sequence[float],
    nk_list: Sequence[int],
    deg_tol_list: Sequence[float],
    denominator_mode_list: Sequence[DenominatorMode],
    temperature_K: float,
    delta0: float,
    eta: float,
) -> list[FiniteQSubspaceConsistencyDiagnostic]:
    """Run subspace and denominator diagnostics over a compact quick grid."""

    rows: list[FiniteQSubspaceConsistencyDiagnostic] = []
    for kind in kinds:
        for matsubara_index in matsubara_list:
            for nk in nk_list:
                for q_magnitude in q_list:
                    for q_phi in q_phi_list:
                        for deg_tol in deg_tol_list:
                            for denominator_mode in denominator_mode_list:
                                rows.append(
                                    finite_q_subspace_consistency_diagnostic(
                                        kind,
                                        int(matsubara_index),
                                        temperature_K,
                                        float(q_magnitude),
                                        float(q_phi),
                                        int(nk),
                                        delta0,
                                        eta,
                                        float(deg_tol),
                                        denominator_mode,
                                    )
                                )
    return rows
