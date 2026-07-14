"""Batched finite-q q workspace with integrated Peierls operator diagnostics.

This is the arbitrary-q production wrapper around the established batched workspace
algebra.  It performs the operator identity audit from Hamiltonians and vertices
already constructed for the q workspace, avoiding a second normal-Hamiltonian and
Peierls-vertex pass.
"""
from __future__ import annotations

import numpy as np

from lno327.bdg.finite_q import density_vertex
from lno327.response.finite_q_optimized import (
    FiniteQMaterialWorkspace,
    FiniteQQWorkspace,
)
from lno327.response.finite_q_q_workspace_batched import (
    _band_vertices_batch,
    _bdg_finite_q_vertices_batch,
    _finite_positive_scalar,
    _phase_phase_direct_vertex_batch,
    _phase_vertex_batch,
    _thermal_expectation_batch,
    supports_batched_finite_q_q_workspace,
)
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_finite_q_inputs

_EM_OBSERVABLE_SIGNS = np.asarray([1.0, -1.0, -1.0], dtype=float)
_COLLECTIVE_CHANNELS = 2
_UNIFIED_CHANNELS = 5


def precompute_finite_q_q_workspace_batched_operator(
    material: FiniteQMaterialWorkspace,
    q_model: np.ndarray,
) -> FiniteQQWorkspace:
    """Build one q workspace and reuse its intermediates for operator diagnostics."""

    if not supports_batched_finite_q_q_workspace(material):
        raise ValueError(
            "material model/ansatz does not support the batched q workspace"
        )

    q, _, _ = validate_finite_q_inputs(
        q_model,
        material.k_points,
        material.k_weights,
        material.config,
    )
    shared = bool(np.linalg.norm(q) <= 1e-14)
    spec = material.spec
    ansatz = material.ansatz
    amp = material.pairing_params
    delta0 = _finite_positive_scalar(
        getattr(amp, "delta0_eV", 0.0),
        "delta0_eV",
    )

    points = np.asarray(material.k_points, dtype=float)
    weights = np.asarray(material.k_weights, dtype=float)
    nk = int(points.shape[0])
    q_half = 0.5 * q
    h_minus = None
    h_plus = None

    if shared:
        energies_minus = np.asarray(material.midpoint_energies, dtype=float)
        energies_plus = energies_minus
        states_minus = np.asarray(material.midpoint_states, dtype=complex)
        states_plus = states_minus
        occupations_minus = np.asarray(
            material.midpoint_occupations,
            dtype=float,
        )
        occupations_plus = occupations_minus
    else:
        points_minus = points - q_half
        points_plus = points + q_half
        pairing_minus = ansatz.mean_pairing_batch(points_minus, amp)
        pairing_plus = ansatz.mean_pairing_batch(points_plus, amp)
        h_minus = np.asarray(
            spec.bdg_hamiltonian_from_pairing_batch(
                points_minus,
                pairing_minus,
            ),
            dtype=complex,
        )
        h_plus = np.asarray(
            spec.bdg_hamiltonian_from_pairing_batch(
                points_plus,
                pairing_plus,
            ),
            dtype=complex,
        )
        energies_minus, states_minus = np.linalg.eigh(h_minus)
        energies_plus, states_plus = np.linalg.eigh(h_plus)
        occupations_minus = fermi_function(
            energies_minus,
            material.config.fermi_level_eV,
            material.config.temperature_eV,
        )
        occupations_plus = fermi_function(
            energies_plus,
            material.config.fermi_level_eV,
            material.config.temperature_eV,
        )

    particle_vector, particle_contact = (
        spec.peierls_hamiltonian_vertices_batch(points, q)
    )
    hole_vector, hole_contact = spec.peierls_hamiltonian_vertices_batch(
        -points,
        -q,
    )
    particle_vector = np.asarray(particle_vector, dtype=complex)
    particle_contact = np.asarray(particle_contact, dtype=complex)
    hole_vector = np.asarray(hole_vector, dtype=complex)
    hole_contact = np.asarray(hole_contact, dtype=complex)
    current_vertices = _bdg_finite_q_vertices_batch(
        particle_vector,
        hole_vector,
    )
    contact_vertices = _bdg_finite_q_vertices_batch(
        particle_contact,
        hole_contact,
    )

    orbital_dim = int(particle_vector.shape[-1])
    if shared:
        operator_lhs = np.zeros((nk, orbital_dim, orbital_dim), dtype=complex)
        operator_rhs = np.zeros_like(operator_lhs)
    else:
        if h_plus is None or h_minus is None:
            raise RuntimeError("nonzero-q workspace lost shifted Hamiltonians")
        operator_lhs = np.einsum(
            "i,kiab->kab",
            q,
            particle_vector,
            optimize=True,
        )
        operator_rhs = (
            h_plus[:, :orbital_dim, :orbital_dim]
            - h_minus[:, :orbital_dim, :orbital_dim]
        )
    operator_delta_norm = np.linalg.norm(
        operator_lhs - operator_rhs,
        axis=(-2, -1),
    )
    operator_scale = np.maximum(
        np.linalg.norm(operator_lhs, axis=(-2, -1)),
        np.linalg.norm(operator_rhs, axis=(-2, -1)),
    )
    operator_delta_norm = np.array(operator_delta_norm, dtype=float, copy=True)
    operator_scale = np.array(operator_scale, dtype=float, copy=True)
    operator_delta_norm.setflags(write=False)
    operator_scale.setflags(write=False)

    rho = density_vertex(orbital_dim)
    rho_batch = np.broadcast_to(rho, (nk, 1) + rho.shape)
    source_vertices = np.concatenate(
        (rho_batch, current_vertices),
        axis=1,
    )
    source_band = _band_vertices_batch(
        states_minus,
        source_vertices,
        states_plus,
    )
    observable_band = (
        _EM_OBSERVABLE_SIGNS[None, :, None, None] * source_band
    )

    collective_vertices = np.asarray(
        ansatz.collective_vertices_batch(points, q, amp),
        dtype=complex,
    )
    if collective_vertices.shape != (
        nk,
        _COLLECTIVE_CHANNELS,
        states_minus.shape[1],
        states_minus.shape[1],
    ):
        raise ValueError(
            "batched collective vertices must have shape (nk, 2, nb, nb)"
        )
    collective_band = _band_vertices_batch(
        states_minus,
        collective_vertices,
        states_plus,
    )

    phase_pairing = np.asarray(
        ansatz.phase_pairing_matrix_batch(points, q, amp),
        dtype=complex,
    )
    theta = _phase_vertex_batch(phase_pairing)
    if not np.allclose(
        theta,
        delta0 * collective_vertices[:, 1],
        rtol=1e-11,
        atol=1e-13,
    ):
        raise ValueError(
            "phase vertex is not delta0 times the eta2 collective vertex"
        )

    left_vertices_band = np.concatenate(
        (observable_band, collective_band),
        axis=1,
    )
    right_vertices_band = np.concatenate(
        (source_band, collective_band),
        axis=1,
    )

    occupation_difference = (
        occupations_minus[:, :, None] - occupations_plus[:, None, :]
    )
    rho_band = source_band[:, 0]
    equal_forward = 0.5 * np.einsum(
        "k,kmn,kmn,kjmn->j",
        weights,
        occupation_difference,
        rho_band,
        np.conjugate(source_band),
        optimize=True,
    )

    contact_expectation = _thermal_expectation_batch(
        material.midpoint_states,
        material.midpoint_occupations,
        contact_vertices,
    )
    direct_spatial = -np.einsum(
        "k,kij->ij",
        weights,
        contact_expectation,
        optimize=True,
    )
    direct_total = np.zeros((3, 3), dtype=complex)
    direct_total[1:, 1:] = direct_spatial

    q_contact_mid = np.zeros(3, dtype=complex)
    q_contact_mid[1:] = np.einsum(
        "i,ij->j",
        q,
        direct_spatial,
        optimize=True,
    )

    particle_plus = spec.peierls_hamiltonian_vector_vertices_batch(
        points + q_half,
        q,
    )
    hole_plus = spec.peierls_hamiltonian_vector_vertices_batch(
        -(points + q_half),
        -q,
    )
    particle_minus = spec.peierls_hamiltonian_vector_vertices_batch(
        points - q_half,
        q,
    )
    hole_minus = spec.peierls_hamiltonian_vector_vertices_batch(
        -(points - q_half),
        -q,
    )
    vector_plus = _bdg_finite_q_vertices_batch(
        particle_plus,
        hole_plus,
    )
    vector_minus = _bdg_finite_q_vertices_batch(
        particle_minus,
        hole_minus,
    )
    delta_v_expectation = _thermal_expectation_batch(
        material.midpoint_states,
        material.midpoint_occupations,
        vector_plus - vector_minus,
    )
    delta_v_mid = np.zeros(3, dtype=complex)
    delta_v_mid[1:] = np.einsum(
        "k,kj->j",
        weights,
        delta_v_expectation,
        optimize=True,
    )

    phase_phase_vertex = _phase_phase_direct_vertex_batch(phase_pairing)
    phase_expectation = _thermal_expectation_batch(
        material.midpoint_states,
        material.midpoint_occupations,
        phase_phase_vertex,
    )
    phase_direct_plus = complex(
        np.einsum(
            "k,k->",
            weights,
            phase_expectation,
            optimize=True,
        )
    )

    ward_rhs = equal_forward - delta_v_mid + q_contact_mid
    shifted_count = 0 if shared else 2 * material.nk
    shifted_eigh_calls = 0 if shared else 2
    return FiniteQQWorkspace(
        material=material,
        q_model=q,
        shared_eigenbasis_q0=shared,
        energies_minus=energies_minus,
        energies_plus=energies_plus,
        occupations_minus=occupations_minus,
        occupations_plus=occupations_plus,
        left_vertices_band=left_vertices_band,
        right_vertices_band=right_vertices_band,
        direct_contact_contribution=direct_total,
        phase_phase_direct_plus=phase_direct_plus,
        phase_phase_direct_minus=-phase_direct_plus,
        ward_rhs_vector=ward_rhs,
        metadata={
            "workspace_kind": "finite_q_q_vectorized_operator_integrated",
            "q_workspace_implementation": "batched_model_capability_operator_integrated",
            "q_dependent": True,
            "shifted_eigensystem_count": shifted_count,
            "shifted_eigh_call_count": shifted_eigh_calls,
            "midpoint_eigensystems_reused": material.nk,
            "ward_rhs_cached": True,
            "ward_rhs_formula": "equal_forward - delta_v_mid + qM_mid",
            "ward_equal_forward": equal_forward.copy(),
            "ward_delta_v_mid": delta_v_mid.copy(),
            "ward_qM_mid": q_contact_mid.copy(),
            "operator_identity_source": "q_workspace_existing_hamiltonians_and_vertices",
            "operator_identity_delta_norms": operator_delta_norm,
            "operator_identity_scales": operator_scale,
            "unified_channel_count": _UNIFIED_CHANNELS,
            "phase_only_derived_from_eta2": True,
        },
    )


__all__ = ["precompute_finite_q_q_workspace_batched_operator"]
