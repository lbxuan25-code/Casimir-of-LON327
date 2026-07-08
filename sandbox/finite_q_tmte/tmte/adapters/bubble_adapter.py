"""Direct target-basis finite-q BdG bubble adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lno327.bdg.hamiltonian import bdg_hamiltonian_from_model_pairing
from lno327.response.finite_q import add_bubble, thermal_expectation_bdg_from_hamiltonian
from lno327.response.finite_q_bdg import bdg_eigensystem_from_model_pairing
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_finite_q_inputs
from lno327.workflows.finite_q_engine import FiniteQEngineOptions

from ..theory.contacts import project_spatial_contact
from ..theory.conventions import (
    FiniteQConventions,
    SOURCE_ORDER_DIAGNOSTIC,
    finite_q_conventions,
    require_diagnostic_source_order,
    require_xi_matches_omega,
)
from ..theory.vertices import target_vertices
from .collective_adapter import collective_counterterm, collective_vertices
from .primitive_vertices_adapter import primitive_observable_vertices, primitive_source_vertices, primitive_spatial_contact_vertices


@dataclass(frozen=True)
class TargetBareBlocks:
    source_order: tuple[str, ...]
    conventions: FiniteQConventions
    k_ss_bubble: np.ndarray
    k_ss_contact: np.ndarray
    k_ss: np.ndarray
    k_seta: np.ndarray
    k_etas: np.ndarray
    k_etaeta_bubble: np.ndarray
    k_etaeta_counterterm: np.ndarray
    k_etaeta: np.ndarray
    metadata: dict[str, Any]


def _bands_for_point(spec: object, ansatz: object, amp: object, kx: float, ky: float, qx: float, qy: float, shared: bool):
    if shared:
        delta = ansatz.mean_pairing(kx, ky, amp)
        bands = bdg_eigensystem_from_model_pairing(spec, kx, ky, delta)
        return bands, bands, delta
    delta_minus = ansatz.mean_pairing(kx - 0.5 * qx, ky - 0.5 * qy, amp)
    delta_plus = ansatz.mean_pairing(kx + 0.5 * qx, ky + 0.5 * qy, amp)
    bands_minus = bdg_eigensystem_from_model_pairing(spec, kx - 0.5 * qx, ky - 0.5 * qy, delta_minus)
    bands_plus = bdg_eigensystem_from_model_pairing(spec, kx + 0.5 * qx, ky + 0.5 * qy, delta_plus)
    return bands_minus, bands_plus, ansatz.mean_pairing(kx, ky, amp)


def compute_target_bare_blocks(
    *,
    spec: object,
    ansatz: object,
    q_model: np.ndarray,
    xi: float,
    k_points: np.ndarray,
    weights: np.ndarray,
    config: object,
    pairing_params: object,
    options: FiniteQEngineOptions | None = None,
    source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC,
) -> TargetBareBlocks:
    """Compute G/TM/TE target-basis bare blocks without component 3x3 rotation."""

    opts = options or FiniteQEngineOptions(include_phase_correction=False, collective_mode="amplitude_phase", collective_counterterm="goldstone_gap_equation")
    require_diagnostic_source_order(source_order)
    require_xi_matches_omega(xi, config.omega_eV)
    q, points, mesh_weights = validate_finite_q_inputs(q_model, k_points, weights, config)
    conventions = finite_q_conventions(q, xi)
    qx, qy = float(q[0]), float(q[1])
    shared_tol = 1e-14
    shared = bool(np.linalg.norm(q) <= shared_tol)
    if shared:
        raise ValueError("direct TM/TE finite-q path requires q > q_tol")

    n_source = len(source_order)
    k_ss_bubble = np.zeros((n_source, n_source), dtype=complex)
    k_ss_contact = np.zeros((n_source, n_source), dtype=complex)
    k_seta = np.zeros((n_source, 2), dtype=complex)
    k_etas = np.zeros((2, n_source), dtype=complex)
    k_etaeta_bubble = np.zeros((2, 2), dtype=complex)

    for weight, (kx_value, ky_value) in zip(mesh_weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        bands_minus, bands_plus, delta_mid = _bands_for_point(spec, ansatz, pairing_params, kx, ky, qx, qy, shared)
        occ_minus = fermi_function(bands_minus.energies, config.fermi_level_eV, config.temperature_eV)
        occ_plus = fermi_function(bands_plus.energies, config.fermi_level_eV, config.temperature_eV)

        left_primitive = primitive_observable_vertices(spec, kx, ky, qx, qy, current_vertex=opts.current_vertex)
        right_primitive = primitive_source_vertices(spec, kx, ky, qx, qy, current_vertex=opts.current_vertex)
        left_targets = target_vertices(*left_primitive, conventions, source_order=source_order)
        right_targets = target_vertices(*right_primitive, conventions, source_order=source_order)
        add_bubble(
            k_ss_bubble,
            left_targets,
            right_targets,
            bands_minus.energies,
            bands_minus.states,
            occ_minus,
            bands_plus.energies,
            bands_plus.states,
            occ_plus,
            config.omega_eV,
            float(weight),
            config=config,
            static_limit=shared,
        )

        coll = collective_vertices(ansatz, kx, ky, qx, qy, pairing_params)
        add_bubble(k_seta, left_targets, coll, bands_minus.energies, bands_minus.states, occ_minus, bands_plus.energies, bands_plus.states, occ_plus, config.omega_eV, float(weight), config=config, static_limit=shared)
        add_bubble(k_etas, coll, right_targets, bands_minus.energies, bands_minus.states, occ_minus, bands_plus.energies, bands_plus.states, occ_plus, config.omega_eV, float(weight), config=config, static_limit=shared)
        add_bubble(k_etaeta_bubble, coll, coll, bands_minus.energies, bands_minus.states, occ_minus, bands_plus.energies, bands_plus.states, occ_plus, config.omega_eV, float(weight), config=config, static_limit=shared)

        h_mid = bdg_hamiltonian_from_model_pairing(spec, kx, ky, delta_mid)
        spatial = np.zeros((2, 2), dtype=complex)
        contact_vertices = primitive_spatial_contact_vertices(spec, kx, ky, qx, qy, current_vertex=opts.current_vertex)
        for i, di in enumerate(("x", "y")):
            for j, dj in enumerate(("x", "y")):
                spatial[i, j] += -float(weight) * thermal_expectation_bdg_from_hamiltonian(h_mid, contact_vertices[(di, dj)], config)
        k_ss_contact += project_spatial_contact(spatial, conventions, source_order=source_order)

    counterterm = np.zeros((2, 2), dtype=complex)
    if opts.collective_counterterm == "goldstone_gap_equation":
        counterterm = collective_counterterm(ansatz, config, points, mesh_weights, pairing_params)
    k_ss = k_ss_bubble + k_ss_contact
    k_etaeta = k_etaeta_bubble + counterterm
    return TargetBareBlocks(
        source_order=source_order,
        conventions=conventions,
        k_ss_bubble=k_ss_bubble,
        k_ss_contact=k_ss_contact,
        k_ss=k_ss,
        k_seta=k_seta,
        k_etas=k_etas,
        k_etaeta_bubble=k_etaeta_bubble,
        k_etaeta_counterterm=counterterm,
        k_etaeta=k_etaeta,
        metadata={
            "adapter": "direct_target_basis_bubble_adapter_v1",
            "uses_existing_kubo_routine": "lno327.response.finite_q.add_bubble",
            "uses_existing_vector_vertices": "lno327.response.finite_q_bdg.bdg_vector_vertex_from_spec",
            "uses_existing_contact_vertices": "lno327.response.finite_q_bdg.bdg_contact_vertex_from_spec",
            "uses_existing_collective_vertices": "ansatz.collective_vertices",
            "current_vertex": opts.current_vertex,
            "collective_counterterm": opts.collective_counterterm,
            "valid_for_casimir_input": False,
        },
    )


def compute_component_reference_effective(
    *,
    spec: object,
    ansatz: object,
    q_model: np.ndarray,
    xi: float,
    k_points: np.ndarray,
    weights: np.ndarray,
    config: object,
    pairing_params: object,
) -> np.ndarray:
    """Debug-only full component response contraction reference."""

    from lno327.response.finite_q_bdg import precompute_finite_q_bdg_workspace_from_model_ansatz
    from lno327.workflows.finite_q_engine import bdg_finite_q_response_imag_axis_from_workspace

    from ..theory.basis import component_source_vectors

    require_xi_matches_omega(xi, config.omega_eV)
    options = FiniteQEngineOptions(include_phase_correction=False, current_vertex="peierls", collective_mode="amplitude_phase", collective_counterterm="goldstone_gap_equation")
    workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(spec, ansatz, q_model, k_points, weights, config, pairing_params, options)
    component = bdg_finite_q_response_imag_axis_from_workspace(workspace, config=config)
    coeffs = component_source_vectors(finite_q_conventions(q_model, xi))
    order = SOURCE_ORDER_DIAGNOSTIC
    transform = np.vstack([coeffs[label] for label in order])
    return transform @ component.amplitude_phase_schur @ transform.T
