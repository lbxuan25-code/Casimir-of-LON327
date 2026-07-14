"""Batched construction of the two-band finite-q q workspace.

This is the single microscopic q-workspace implementation used by both the
retained complete-orbit evaluator and the arbitrary-q periodic-BZ backend.
Operator diagnostics are optional and consume Hamiltonians and Peierls vertices
already built by the workspace.  Linear per-k ingredients are retained so a
runtime batch can be reduced through smaller canonical floating-point blocks
without rebuilding shifted eigensystems.
"""
from __future__ import annotations

import numpy as np

from lno327.bdg.finite_q import density_vertex
from lno327.response.finite_q_optimized import (
    FiniteQMaterialWorkspace,
    FiniteQQWorkspace,
)
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_finite_q_inputs

_EM_OBSERVABLE_SIGNS = np.asarray([1.0, -1.0, -1.0], dtype=float)
_COLLECTIVE_CHANNELS = 2
_UNIFIED_CHANNELS = 5


def supports_batched_finite_q_q_workspace(
    material: FiniteQMaterialWorkspace,
) -> bool:
    spec = material.spec
    ansatz = material.ansatz
    opts = material.options
    required_spec = (
        "bdg_hamiltonian_from_pairing_batch",
        "peierls_hamiltonian_vector_vertices_batch",
        "peierls_hamiltonian_vertices_batch",
    )
    required_ansatz = (
        "mean_pairing_batch",
        "collective_vertices_batch",
        "phase_pairing_matrix_batch",
    )
    return bool(
        getattr(opts, "current_vertex", None) == "peierls"
        and all(callable(getattr(spec, name, None)) for name in required_spec)
        and all(callable(getattr(ansatz, name, None)) for name in required_ansatz)
    )


def _finite_positive_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _readonly(value: np.ndarray, *, dtype=None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _bdg_finite_q_vertices_batch(
    particle: np.ndarray,
    hole_normal: np.ndarray,
) -> np.ndarray:
    particle_array = np.asarray(particle, dtype=complex)
    hole_array = np.asarray(hole_normal, dtype=complex)
    if particle_array.shape != hole_array.shape:
        raise ValueError("particle and hole vertex batches must have equal shapes")
    if particle_array.ndim < 2 or particle_array.shape[-2] != particle_array.shape[-1]:
        raise ValueError("normal vertex batches must have square trailing axes")
    dim = int(particle_array.shape[-1])
    result = np.zeros(
        particle_array.shape[:-2] + (2 * dim, 2 * dim), dtype=complex
    )
    result[..., :dim, :dim] = particle_array
    result[..., dim:, dim:] = -np.swapaxes(hole_array, -1, -2)
    return result


def _phase_vertex_batch(pairing: np.ndarray) -> np.ndarray:
    delta = np.asarray(pairing, dtype=complex)
    if delta.shape[-2:] != (2, 2):
        raise ValueError("phase pairing matrices must have shape (..., 2, 2)")
    result = np.zeros(delta.shape[:-2] + (4, 4), dtype=complex)
    result[..., :2, 2:] = 1j * delta
    result[..., 2:, :2] = -1j * np.swapaxes(delta.conjugate(), -1, -2)
    return result


def _phase_phase_direct_vertex_batch(pairing: np.ndarray) -> np.ndarray:
    delta = np.asarray(pairing, dtype=complex)
    if delta.shape[-2:] != (2, 2):
        raise ValueError("phase pairing matrices must have shape (..., 2, 2)")
    result = np.zeros(delta.shape[:-2] + (4, 4), dtype=complex)
    result[..., :2, 2:] = -delta
    result[..., 2:, :2] = -np.swapaxes(delta.conjugate(), -1, -2)
    return result


def _band_vertices_batch(
    states_minus: np.ndarray,
    vertices: np.ndarray,
    states_plus: np.ndarray,
) -> np.ndarray:
    minus = np.asarray(states_minus, dtype=complex)
    plus = np.asarray(states_plus, dtype=complex)
    operators = np.asarray(vertices, dtype=complex)
    if minus.shape != plus.shape or minus.ndim != 3:
        raise ValueError("state batches must have matching shape (nk, nb, nb)")
    nk, nb, nb_right = minus.shape
    if nb != nb_right:
        raise ValueError("state matrices must be square")
    if operators.ndim != 4 or operators.shape[0] != nk or operators.shape[-2:] != (nb, nb):
        raise ValueError("vertex batch must have shape (nk, channels, nb, nb)")
    return np.einsum(
        "kpn,kapq,kqm->kamn",
        np.conjugate(plus),
        operators,
        minus,
        optimize=True,
    )


def _thermal_expectation_batch(
    states: np.ndarray,
    occupations: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    state_array = np.asarray(states, dtype=complex)
    occupation_array = np.asarray(occupations, dtype=float)
    vertex_array = np.asarray(vertices, dtype=complex)
    if state_array.ndim != 3 or state_array.shape[1] != state_array.shape[2]:
        raise ValueError("states must have shape (nk, nb, nb)")
    nk, nb, _ = state_array.shape
    if occupation_array.shape != (nk, nb):
        raise ValueError("occupations must have shape (nk, nb)")
    if vertex_array.shape[0] != nk or vertex_array.shape[-2:] != (nb, nb):
        raise ValueError("vertices must have shape (nk, ..., nb, nb)")
    channel_shape = vertex_array.shape[1:-2]
    flat = vertex_array.reshape(nk, -1, nb, nb)
    diagonal = np.einsum(
        "kpm,kcpq,kqm->kcm",
        np.conjugate(state_array),
        flat,
        state_array,
        optimize=True,
    )
    expectation = 0.5 * np.einsum(
        "km,kcm->kc", occupation_array, diagonal, optimize=True
    )
    return expectation.reshape((nk,) + channel_shape)


def _integrated_linear_terms_from_workspace_slice(
    workspace: FiniteQQWorkspace,
    start: int,
    stop: int,
) -> tuple[np.ndarray, complex, np.ndarray]:
    """Integrate header primitives on one canonical subblock.

    The q workspace is built once for a larger runtime chunk.  This helper uses
    retained linear ingredients and the original einsum formulas on a slice, so
    canonical block boundaries alone define floating-point accumulation order.
    """

    first, last = int(start), int(stop)
    nk = int(workspace.material.k_points.shape[0])
    if first < 0 or last > nk or first >= last:
        raise ValueError("invalid q-workspace canonical slice")
    metadata = workspace.metadata
    weights = np.asarray(workspace.material.k_weights, dtype=float)[first:last]
    occupation_difference = np.asarray(
        metadata["linear_occupation_difference"], dtype=float
    )[first:last]
    rho_band = np.asarray(metadata["linear_rho_band"], dtype=complex)[first:last]
    source_band = np.asarray(metadata["linear_source_band"], dtype=complex)[first:last]
    equal_forward = 0.5 * np.einsum(
        "k,kmn,kmn,kjmn->j",
        weights,
        occupation_difference,
        rho_band,
        np.conjugate(source_band),
        optimize=True,
    )

    contact_expectation = np.asarray(
        metadata["linear_contact_expectation"], dtype=complex
    )[first:last]
    direct_spatial = -np.einsum(
        "k,kij->ij", weights, contact_expectation, optimize=True
    )
    direct_total = np.zeros((3, 3), dtype=complex)
    direct_total[1:, 1:] = direct_spatial
    q_contact_mid = np.zeros(3, dtype=complex)
    q_contact_mid[1:] = np.einsum(
        "i,ij->j", np.asarray(workspace.q_model, dtype=float), direct_spatial, optimize=True
    )

    delta_v_expectation = np.asarray(
        metadata["linear_delta_v_expectation"], dtype=complex
    )[first:last]
    delta_v_mid = np.zeros(3, dtype=complex)
    delta_v_mid[1:] = np.einsum(
        "k,kj->j", weights, delta_v_expectation, optimize=True
    )

    phase_expectation = np.asarray(
        metadata["linear_phase_expectation"], dtype=complex
    )[first:last]
    phase_direct_plus = complex(
        np.einsum("k,k->", weights, phase_expectation, optimize=True)
    )
    ward_rhs = equal_forward - delta_v_mid + q_contact_mid
    return direct_total, phase_direct_plus, ward_rhs


def precompute_finite_q_q_workspace_batched(
    material: FiniteQMaterialWorkspace,
    q_model: np.ndarray,
    *,
    operator_diagnostics: bool = False,
) -> FiniteQQWorkspace:
    """Build one q workspace by batching all microscopic k operations."""

    if not supports_batched_finite_q_q_workspace(material):
        raise ValueError("material model/ansatz does not support the batched q workspace")
    q, _, _ = validate_finite_q_inputs(
        q_model, material.k_points, material.k_weights, material.config
    )
    shared = bool(np.linalg.norm(q) <= 1e-14)
    spec = material.spec
    ansatz = material.ansatz
    amp = material.pairing_params
    delta0 = _finite_positive_scalar(getattr(amp, "delta0_eV", 0.0), "delta0_eV")
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
        occupations_minus = np.asarray(material.midpoint_occupations, dtype=float)
        occupations_plus = occupations_minus
    else:
        points_minus = points - q_half
        points_plus = points + q_half
        pairing_minus = ansatz.mean_pairing_batch(points_minus, amp)
        pairing_plus = ansatz.mean_pairing_batch(points_plus, amp)
        h_minus = np.asarray(
            spec.bdg_hamiltonian_from_pairing_batch(points_minus, pairing_minus),
            dtype=complex,
        )
        h_plus = np.asarray(
            spec.bdg_hamiltonian_from_pairing_batch(points_plus, pairing_plus),
            dtype=complex,
        )
        energies_minus, states_minus = np.linalg.eigh(h_minus)
        energies_plus, states_plus = np.linalg.eigh(h_plus)
        occupations_minus = fermi_function(
            energies_minus, material.config.fermi_level_eV, material.config.temperature_eV
        )
        occupations_plus = fermi_function(
            energies_plus, material.config.fermi_level_eV, material.config.temperature_eV
        )

    particle_vector, particle_contact = spec.peierls_hamiltonian_vertices_batch(points, q)
    hole_vector, hole_contact = spec.peierls_hamiltonian_vertices_batch(-points, -q)
    particle_vector = np.asarray(particle_vector, dtype=complex)
    particle_contact = np.asarray(particle_contact, dtype=complex)
    hole_vector = np.asarray(hole_vector, dtype=complex)
    hole_contact = np.asarray(hole_contact, dtype=complex)
    current_vertices = _bdg_finite_q_vertices_batch(particle_vector, hole_vector)
    contact_vertices = _bdg_finite_q_vertices_batch(particle_contact, hole_contact)

    orbital_dim = int(particle_vector.shape[-1])
    operator_delta_norm = np.zeros(nk, dtype=float)
    operator_scale = np.zeros(nk, dtype=float)
    if operator_diagnostics and not shared:
        if h_plus is None or h_minus is None:
            raise RuntimeError("nonzero-q workspace lost shifted Hamiltonians")
        operator_lhs = np.einsum("i,kiab->kab", q, particle_vector, optimize=True)
        operator_rhs = h_plus[:, :orbital_dim, :orbital_dim] - h_minus[:, :orbital_dim, :orbital_dim]
        operator_delta_norm = np.linalg.norm(operator_lhs - operator_rhs, axis=(-2, -1))
        operator_scale = np.maximum(
            np.linalg.norm(operator_lhs, axis=(-2, -1)),
            np.linalg.norm(operator_rhs, axis=(-2, -1)),
        )

    rho = density_vertex(orbital_dim)
    rho_batch = np.broadcast_to(rho, (nk, 1) + rho.shape)
    source_vertices = np.concatenate((rho_batch, current_vertices), axis=1)
    source_band = _band_vertices_batch(states_minus, source_vertices, states_plus)
    observable_band = _EM_OBSERVABLE_SIGNS[None, :, None, None] * source_band

    collective_vertices = np.asarray(
        ansatz.collective_vertices_batch(points, q, amp), dtype=complex
    )
    if collective_vertices.shape != (
        nk,
        _COLLECTIVE_CHANNELS,
        states_minus.shape[1],
        states_minus.shape[1],
    ):
        raise ValueError("batched collective vertices must have shape (nk, 2, nb, nb)")
    collective_band = _band_vertices_batch(states_minus, collective_vertices, states_plus)

    phase_pairing = np.asarray(ansatz.phase_pairing_matrix_batch(points, q, amp), dtype=complex)
    theta = _phase_vertex_batch(phase_pairing)
    if not np.allclose(theta, delta0 * collective_vertices[:, 1], rtol=1e-11, atol=1e-13):
        raise ValueError("phase vertex is not delta0 times the eta2 collective vertex")

    left_vertices_band = np.concatenate((observable_band, collective_band), axis=1)
    right_vertices_band = np.concatenate((source_band, collective_band), axis=1)
    occupation_difference = occupations_minus[:, :, None] - occupations_plus[:, None, :]
    rho_band = source_band[:, 0]

    contact_expectation = _thermal_expectation_batch(
        material.midpoint_states, material.midpoint_occupations, contact_vertices
    )
    particle_plus = spec.peierls_hamiltonian_vector_vertices_batch(points + q_half, q)
    hole_plus = spec.peierls_hamiltonian_vector_vertices_batch(-(points + q_half), -q)
    particle_minus = spec.peierls_hamiltonian_vector_vertices_batch(points - q_half, q)
    hole_minus = spec.peierls_hamiltonian_vector_vertices_batch(-(points - q_half), -q)
    vector_plus = _bdg_finite_q_vertices_batch(particle_plus, hole_plus)
    vector_minus = _bdg_finite_q_vertices_batch(particle_minus, hole_minus)
    delta_v_expectation = _thermal_expectation_batch(
        material.midpoint_states,
        material.midpoint_occupations,
        vector_plus - vector_minus,
    )
    phase_phase_vertex = _phase_phase_direct_vertex_batch(phase_pairing)
    phase_expectation = _thermal_expectation_batch(
        material.midpoint_states, material.midpoint_occupations, phase_phase_vertex
    )

    # Compute the full-runtime integrated values through the same helper used by
    # canonical subblocks, guaranteeing one algebraic implementation.
    provisional = FiniteQQWorkspace(
        material=material,
        q_model=q,
        shared_eigenbasis_q0=shared,
        energies_minus=energies_minus,
        energies_plus=energies_plus,
        occupations_minus=occupations_minus,
        occupations_plus=occupations_plus,
        left_vertices_band=left_vertices_band,
        right_vertices_band=right_vertices_band,
        direct_contact_contribution=np.zeros((3, 3), dtype=complex),
        phase_phase_direct_plus=0.0j,
        phase_phase_direct_minus=0.0j,
        ward_rhs_vector=np.zeros(3, dtype=complex),
        metadata={
            "linear_occupation_difference": _readonly(occupation_difference, dtype=float),
            "linear_rho_band": _readonly(rho_band, dtype=complex),
            "linear_source_band": _readonly(source_band, dtype=complex),
            "linear_contact_expectation": _readonly(contact_expectation, dtype=complex),
            "linear_delta_v_expectation": _readonly(delta_v_expectation, dtype=complex),
            "linear_phase_expectation": _readonly(phase_expectation, dtype=complex),
        },
    )
    direct_total, phase_direct_plus, ward_rhs = _integrated_linear_terms_from_workspace_slice(
        provisional, 0, nk
    )
    shifted_count = 0 if shared else 2 * material.nk
    shifted_eigh_calls = 0 if shared else 2
    metadata = {
        **dict(provisional.metadata),
        "workspace_kind": "finite_q_q_vectorized",
        "q_workspace_implementation": "batched_model_capability",
        "operator_diagnostics_enabled": bool(operator_diagnostics),
        "operator_identity_source": "shared_q_workspace_hamiltonians_and_vertices",
        "operator_identity_delta_norms": _readonly(operator_delta_norm, dtype=float),
        "operator_identity_scales": _readonly(operator_scale, dtype=float),
        "q_dependent": True,
        "q_workspace_build_count": 1,
        "shifted_eigensystem_count": shifted_count,
        "shifted_eigh_call_count": shifted_eigh_calls,
        "midpoint_eigensystems_reused": material.nk,
        "ward_rhs_cached": True,
        "ward_rhs_formula": "equal_forward - delta_v_mid + qM_mid",
        "unified_channel_count": _UNIFIED_CHANNELS,
        "phase_only_derived_from_eta2": True,
    }
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
        metadata=metadata,
    )


__all__ = [
    "_integrated_linear_terms_from_workspace_slice",
    "precompute_finite_q_q_workspace_batched",
    "supports_batched_finite_q_q_workspace",
]
