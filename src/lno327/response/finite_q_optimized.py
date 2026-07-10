"""Vectorized two-level workspaces for repeated finite-q BdG response scans.

This module preserves the existing primitive response conventions while separating
q-independent material data from q-dependent shifted-band data. A q workspace
stores one unified electromagnetic-plus-collective band block, static direct
terms, and the analytic Ward RHS. Repeated Matsubara evaluations then require
only a vectorized Kubo contraction; no eigensystems or vertices are rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.bdg.finite_q import density_vertex, phase_phase_direct_vertex, phase_vertex
from lno327.response.config import KuboConfig
from lno327.response.finite_q import BdGFiniteQResponseComponents, vertex_band
from lno327.response.finite_q_bdg import (
    _DefaultFiniteQOptions,
    _check_options,
    _finalize_components,
    _pairing_params_from_inputs,
    bdg_contact_vertex_from_spec,
    bdg_eigensystem_from_model_pairing,
    bdg_vector_vertex_from_spec,
    require_peierls_finite_q_support,
)
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_finite_q_inputs
from lno327.response.ward_validation import PrimitiveWardRHS

_EM_CHANNELS = 3
_COLLECTIVE_CHANNELS = 2
_UNIFIED_CHANNELS = _EM_CHANNELS + _COLLECTIVE_CHANNELS
_EM_OBSERVABLE_SIGNS = np.asarray([1.0, -1.0, -1.0], dtype=float)


def _readonly_array(value: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _finite_scalar(value: float, name: str, *, positive: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or (positive and scalar <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return scalar


def _compatible_config(reference: KuboConfig, candidate: KuboConfig) -> None:
    for name in ("temperature_eV", "fermi_level_eV", "eta_eV", "output_si"):
        if getattr(reference, name) != getattr(candidate, name):
            raise ValueError(f"q workspace config field {name} changed; rebuild the workspace")


def _thermal_expectation_from_bands(
    states: np.ndarray,
    occupations: np.ndarray,
    vertex: np.ndarray,
) -> complex:
    vertex_in_band = states.conjugate().T @ np.asarray(vertex, dtype=complex) @ states
    return complex(0.5 * np.sum(occupations * np.diag(vertex_in_band)))


@dataclass(frozen=True)
class FiniteQMaterialWorkspace:
    """q-independent midpoint data shared by all q workspaces for one material state."""

    spec: object
    ansatz: object
    k_points: np.ndarray
    k_weights: np.ndarray
    config: KuboConfig
    pairing_params: object
    options: object
    collective_mode: str
    collective_mode_disabled_reason: str | None
    midpoint_energies: np.ndarray
    midpoint_states: np.ndarray
    midpoint_occupations: np.ndarray
    collective_counterterm_matrix: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_points", _readonly_array(self.k_points, dtype=float))
        object.__setattr__(self, "k_weights", _readonly_array(self.k_weights, dtype=float))
        object.__setattr__(self, "midpoint_energies", _readonly_array(self.midpoint_energies, dtype=float))
        object.__setattr__(self, "midpoint_states", _readonly_array(self.midpoint_states, dtype=complex))
        object.__setattr__(self, "midpoint_occupations", _readonly_array(self.midpoint_occupations, dtype=float))
        object.__setattr__(
            self,
            "collective_counterterm_matrix",
            _readonly_array(self.collective_counterterm_matrix, dtype=complex),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

        nk = self.k_points.shape[0]
        if self.k_points.shape != (nk, 2) or self.k_weights.shape != (nk,):
            raise ValueError("material workspace k arrays have inconsistent shapes")
        if self.midpoint_energies.ndim != 2 or self.midpoint_energies.shape[0] != nk:
            raise ValueError("midpoint_energies must have shape (nk, nb)")
        nb = self.midpoint_energies.shape[1]
        if self.midpoint_states.shape != (nk, nb, nb):
            raise ValueError("midpoint_states must have shape (nk, nb, nb)")
        if self.midpoint_occupations.shape != (nk, nb):
            raise ValueError("midpoint_occupations must have shape (nk, nb)")
        if self.collective_counterterm_matrix.shape != (_COLLECTIVE_CHANNELS, _COLLECTIVE_CHANNELS):
            raise ValueError("collective_counterterm_matrix must have shape (2, 2)")

    @property
    def nk(self) -> int:
        return int(self.k_points.shape[0])

    @property
    def nb(self) -> int:
        return int(self.midpoint_energies.shape[1])


@dataclass(frozen=True)
class FiniteQQWorkspace:
    """One-q shifted-band cache for vectorized repeated Matsubara evaluations."""

    material: FiniteQMaterialWorkspace
    q_model: np.ndarray
    shared_eigenbasis_q0: bool
    energies_minus: np.ndarray
    energies_plus: np.ndarray
    occupations_minus: np.ndarray
    occupations_plus: np.ndarray
    left_vertices_band: np.ndarray
    right_vertices_band: np.ndarray
    direct_contact_contribution: np.ndarray
    phase_phase_direct_plus: complex
    phase_phase_direct_minus: complex
    ward_rhs_vector: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_model", _readonly_array(self.q_model, dtype=float))
        object.__setattr__(self, "energies_minus", _readonly_array(self.energies_minus, dtype=float))
        object.__setattr__(self, "energies_plus", _readonly_array(self.energies_plus, dtype=float))
        object.__setattr__(self, "occupations_minus", _readonly_array(self.occupations_minus, dtype=float))
        object.__setattr__(self, "occupations_plus", _readonly_array(self.occupations_plus, dtype=float))
        object.__setattr__(self, "left_vertices_band", _readonly_array(self.left_vertices_band, dtype=complex))
        object.__setattr__(self, "right_vertices_band", _readonly_array(self.right_vertices_band, dtype=complex))
        object.__setattr__(
            self,
            "direct_contact_contribution",
            _readonly_array(self.direct_contact_contribution, dtype=complex),
        )
        object.__setattr__(self, "ward_rhs_vector", _readonly_array(self.ward_rhs_vector, dtype=complex))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

        if self.q_model.shape != (2,) or not np.isfinite(self.q_model).all():
            raise ValueError("q_model must be a finite vector with shape (2,)")
        nk, nb = self.material.nk, self.material.nb
        for name in ("energies_minus", "energies_plus", "occupations_minus", "occupations_plus"):
            if getattr(self, name).shape != (nk, nb):
                raise ValueError(f"{name} must have shape (nk, nb)")
        expected_vertices = (nk, _UNIFIED_CHANNELS, nb, nb)
        if self.left_vertices_band.shape != expected_vertices or self.right_vertices_band.shape != expected_vertices:
            raise ValueError("unified band vertices must have shape (nk, 5, nb, nb)")
        if self.direct_contact_contribution.shape != (3, 3):
            raise ValueError("direct_contact_contribution must have shape (3, 3)")
        if self.ward_rhs_vector.shape != (3,):
            raise ValueError("ward_rhs_vector must have shape (3,)")

    @property
    def nk(self) -> int:
        return self.material.nk

    @property
    def nb(self) -> int:
        return self.material.nb


def precompute_finite_q_material_workspace_from_model_ansatz(
    spec: object,
    ansatz: object,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: KuboConfig,
    pairing_params: object | None = None,
    options: object | None = None,
) -> FiniteQMaterialWorkspace:
    """Build q-independent midpoint eigensystems and one cached counterterm."""

    opts = options or _DefaultFiniteQOptions()
    _check_options(opts)
    _, points, weights = validate_finite_q_inputs(
        np.zeros(2, dtype=float), k_points, k_weights, config
    )
    if opts.current_vertex == "peierls":
        require_peierls_finite_q_support(spec)
    amp = _pairing_params_from_inputs(spec, pairing_params)
    collective_mode = opts.collective_mode
    disabled = None
    if float(getattr(amp, "delta0_eV", 0.0)) == 0.0 and collective_mode == "amplitude_phase":
        collective_mode = "none"
        disabled = "delta0=0 normal limit"
    if collective_mode != "amplitude_phase":
        raise ValueError(
            "optimized production workspace requires collective_mode='amplitude_phase' "
            "with nonzero delta0"
        )

    energies_rows: list[np.ndarray] = []
    states_rows: list[np.ndarray] = []
    occupation_rows: list[np.ndarray] = []
    for kx_value, ky_value in points:
        kx, ky = float(kx_value), float(ky_value)
        pairing = ansatz.mean_pairing(kx, ky, amp)
        bands = bdg_eigensystem_from_model_pairing(spec, kx, ky, pairing)
        occupations = fermi_function(
            bands.energies, config.fermi_level_eV, config.temperature_eV
        )
        energies_rows.append(np.asarray(bands.energies, dtype=float))
        states_rows.append(np.asarray(bands.states, dtype=complex))
        occupation_rows.append(np.asarray(occupations, dtype=float))

    counterterm = np.zeros((_COLLECTIVE_CHANNELS, _COLLECTIVE_CHANNELS), dtype=complex)
    if opts.collective_counterterm == "goldstone_gap_equation":
        counterterm = np.asarray(ansatz.hs_counterterm(config, points, weights, amp), dtype=complex)

    return FiniteQMaterialWorkspace(
        spec=spec,
        ansatz=ansatz,
        k_points=points,
        k_weights=weights,
        config=config,
        pairing_params=amp,
        options=opts,
        collective_mode=collective_mode,
        collective_mode_disabled_reason=disabled,
        midpoint_energies=np.stack(energies_rows, axis=0),
        midpoint_states=np.stack(states_rows, axis=0),
        midpoint_occupations=np.stack(occupation_rows, axis=0),
        collective_counterterm_matrix=counterterm,
        metadata={
            "workspace_kind": "finite_q_material_vectorized",
            "q_independent": True,
            "midpoint_eigensystem_count": int(points.shape[0]),
            "goldstone_counterterm_cached_once": True,
            "unified_channel_order": ("rho", "Jx", "Jy", "eta1", "eta2"),
            "production_collective_mode": "amplitude_phase",
        },
    )


def precompute_finite_q_q_workspace(
    material: FiniteQMaterialWorkspace,
    q_model: np.ndarray,
) -> FiniteQQWorkspace:
    """Build one q-dependent cache, including the frequency-independent Ward RHS."""

    q, _, _ = validate_finite_q_inputs(
        q_model, material.k_points, material.k_weights, material.config
    )
    qx, qy = float(q[0]), float(q[1])
    shared = bool(np.linalg.norm(q) <= 1e-14)
    spec, ansatz = material.spec, material.ansatz
    amp, opts = material.pairing_params, material.options
    delta0 = _finite_scalar(getattr(amp, "delta0_eV", 0.0), "delta0_eV", positive=True)

    dim = np.asarray(
        spec.normal_hamiltonian(float(material.k_points[0, 0]), float(material.k_points[0, 1]))
    ).shape[0]
    rho = density_vertex(int(dim))

    energies_minus_rows: list[np.ndarray] = []
    energies_plus_rows: list[np.ndarray] = []
    occupations_minus_rows: list[np.ndarray] = []
    occupations_plus_rows: list[np.ndarray] = []
    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []

    direct_total = np.zeros((3, 3), dtype=complex)
    phase_direct_plus = 0.0 + 0.0j
    equal_forward = np.zeros(3, dtype=complex)
    delta_v_mid = np.zeros(3, dtype=complex)
    q_contact_mid = np.zeros(3, dtype=complex)

    for index, (weight, (kx_value, ky_value)) in enumerate(
        zip(material.k_weights, material.k_points, strict=True)
    ):
        weight_f = float(weight)
        kx, ky = float(kx_value), float(ky_value)
        midpoint_energies = material.midpoint_energies[index]
        midpoint_states = material.midpoint_states[index]
        midpoint_occupations = material.midpoint_occupations[index]

        if shared:
            energies_minus = energies_plus = midpoint_energies
            states_minus = states_plus = midpoint_states
            occupations_minus = occupations_plus = midpoint_occupations
        else:
            pairing_minus = ansatz.mean_pairing(kx - 0.5 * qx, ky - 0.5 * qy, amp)
            pairing_plus = ansatz.mean_pairing(kx + 0.5 * qx, ky + 0.5 * qy, amp)
            bands_minus = bdg_eigensystem_from_model_pairing(
                spec, kx - 0.5 * qx, ky - 0.5 * qy, pairing_minus
            )
            bands_plus = bdg_eigensystem_from_model_pairing(
                spec, kx + 0.5 * qx, ky + 0.5 * qy, pairing_plus
            )
            energies_minus, states_minus = bands_minus.energies, bands_minus.states
            energies_plus, states_plus = bands_plus.energies, bands_plus.states
            occupations_minus = fermi_function(
                energies_minus, material.config.fermi_level_eV, material.config.temperature_eV
            )
            occupations_plus = fermi_function(
                energies_plus, material.config.fermi_level_eV, material.config.temperature_eV
            )

        vx = bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "x", opts.current_vertex)
        vy = bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "y", opts.current_vertex)
        source_band = np.stack(
            [vertex_band(states_minus, vertex, states_plus) for vertex in (rho, vx, vy)],
            axis=0,
        )
        observable_band = _EM_OBSERVABLE_SIGNS[:, None, None] * source_band

        collective_vertices = tuple(ansatz.collective_vertices(kx, ky, qx, qy, amp))
        if len(collective_vertices) != _COLLECTIVE_CHANNELS:
            raise ValueError("optimized workspace requires exactly two collective channels")
        collective_band = np.stack(
            [vertex_band(states_minus, vertex, states_plus) for vertex in collective_vertices],
            axis=0,
        )
        theta = phase_vertex(ansatz.phase_pairing_matrix(kx, ky, qx, qy, amp))
        if not np.allclose(theta, delta0 * collective_vertices[1], rtol=1e-11, atol=1e-13):
            raise ValueError("phase vertex is not delta0 times the eta2 collective vertex")

        left_rows.append(np.concatenate((observable_band, collective_band), axis=0))
        right_rows.append(np.concatenate((source_band, collective_band), axis=0))
        energies_minus_rows.append(np.asarray(energies_minus, dtype=float))
        energies_plus_rows.append(np.asarray(energies_plus, dtype=float))
        occupations_minus_rows.append(np.asarray(occupations_minus, dtype=float))
        occupations_plus_rows.append(np.asarray(occupations_plus, dtype=float))

        occupation_difference = (
            np.asarray(occupations_minus, dtype=float)[:, None]
            - np.asarray(occupations_plus, dtype=float)[None, :]
        )
        rho_band = source_band[0]
        equal_forward += 0.5 * weight_f * np.einsum(
            "mn,mn,jmn->j",
            occupation_difference,
            rho_band,
            np.conjugate(source_band),
            optimize=True,
        )

        delta_theta = ansatz.phase_pairing_matrix(kx, ky, qx, qy, amp)
        theta_theta = phase_phase_direct_vertex(delta_theta)
        phase_direct_plus += weight_f * _thermal_expectation_from_bands(
            midpoint_states, midpoint_occupations, theta_theta
        )

        for i, direction_i in enumerate(("x", "y")):
            qi = qx if direction_i == "x" else qy
            for j, direction_j in enumerate(("x", "y")):
                contact = bdg_contact_vertex_from_spec(
                    spec,
                    kx,
                    ky,
                    qx,
                    qy,
                    direction_i,
                    direction_j,
                    opts.current_vertex,
                )
                direct = -weight_f * _thermal_expectation_from_bands(
                    midpoint_states, midpoint_occupations, contact
                )
                direct_total[1 + i, 1 + j] += direct
                q_contact_mid[1 + j] += qi * direct

        for j, direction in enumerate(("x", "y"), start=1):
            vertex_plus = bdg_vector_vertex_from_spec(
                spec,
                kx + 0.5 * qx,
                ky + 0.5 * qy,
                qx,
                qy,
                direction,
                opts.current_vertex,
            )
            vertex_minus = bdg_vector_vertex_from_spec(
                spec,
                kx - 0.5 * qx,
                ky - 0.5 * qy,
                qx,
                qy,
                direction,
                opts.current_vertex,
            )
            delta_v_mid[j] += weight_f * _thermal_expectation_from_bands(
                midpoint_states,
                midpoint_occupations,
                vertex_plus - vertex_minus,
            )

    ward_rhs = equal_forward - delta_v_mid + q_contact_mid
    shifted_count = 0 if shared else 2 * material.nk
    return FiniteQQWorkspace(
        material=material,
        q_model=q,
        shared_eigenbasis_q0=shared,
        energies_minus=np.stack(energies_minus_rows, axis=0),
        energies_plus=np.stack(energies_plus_rows, axis=0),
        occupations_minus=np.stack(occupations_minus_rows, axis=0),
        occupations_plus=np.stack(occupations_plus_rows, axis=0),
        left_vertices_band=np.stack(left_rows, axis=0),
        right_vertices_band=np.stack(right_rows, axis=0),
        direct_contact_contribution=direct_total,
        phase_phase_direct_plus=phase_direct_plus,
        phase_phase_direct_minus=-phase_direct_plus,
        ward_rhs_vector=ward_rhs,
        metadata={
            "workspace_kind": "finite_q_q_vectorized",
            "q_dependent": True,
            "shifted_eigensystem_count": shifted_count,
            "midpoint_eigensystems_reused": material.nk,
            "ward_rhs_cached": True,
            "ward_rhs_formula": "equal_forward - delta_v_mid + qM_mid",
            "ward_equal_forward": equal_forward.copy(),
            "ward_delta_v_mid": delta_v_mid.copy(),
            "ward_qM_mid": q_contact_mid.copy(),
            "unified_channel_count": _UNIFIED_CHANNELS,
            "phase_only_derived_from_eta2": True,
        },
    )


def _vectorized_kubo_factors(
    workspace: FiniteQQWorkspace,
    omega_values: np.ndarray,
) -> np.ndarray:
    """Return raw Kubo factors with shape (n_omega, nk, nb, nb)."""

    omega = np.asarray(omega_values, dtype=float)
    if omega.ndim != 1 or omega.size == 0 or not np.isfinite(omega).all():
        raise ValueError("omega_values must be a nonempty finite one-dimensional array")
    if np.any(omega < 0.0):
        raise ValueError("omega_values must be non-negative")

    em = workspace.energies_minus[:, :, None]
    en = workspace.energies_plus[:, None, :]
    fm = workspace.occupations_minus[:, :, None]
    fn = workspace.occupations_plus[:, None, :]
    delta_e = em - en
    occupation_difference = fm - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = occupation_difference[None, :, :, :] / (
            1j * omega[:, None, None, None] + delta_e[None, :, :, :]
        )

    eta = float(workspace.material.config.eta_eV)
    zero_indices = np.flatnonzero(omega == 0.0)
    if zero_indices.size:
        degenerate = np.abs(delta_e) < eta
        if np.any(degenerate):
            midpoint_energy = 0.5 * (em + en)
            config = workspace.material.config
            shifted = midpoint_energy - float(config.fermi_level_eV)
            if config.temperature_eV <= 0.0:
                width = max(eta, 1e-12)
                derivative = -width / (np.pi * (shifted * shifted + width * width))
            else:
                x = np.clip(
                    shifted / (2.0 * float(config.temperature_eV)), -350.0, 350.0
                )
                derivative = -1.0 / (
                    4.0 * float(config.temperature_eV) * np.cosh(x) ** 2
                )
            for index in zero_indices:
                factors[index] = np.where(degenerate, derivative, factors[index])
    return factors


def _components_from_unified_block(
    workspace: FiniteQQWorkspace,
    block: np.ndarray,
    config: KuboConfig,
) -> BdGFiniteQResponseComponents:
    bubble = np.asarray(block[:3, :3], dtype=complex)
    em_collective_left = np.asarray(block[:3, 3:5], dtype=complex)
    collective_em_right = np.asarray(block[3:5, :3], dtype=complex)
    collective_bubble = np.asarray(block[3:5, 3:5], dtype=complex)
    delta0 = float(workspace.material.pairing_params.delta0_eV)
    phase_left = delta0 * em_collective_left[:, 1]
    phase_right = delta0 * collective_em_right[1, :]
    phase_phase_bubble = np.asarray(
        [[delta0 * delta0 * collective_bubble[1, 1]]], dtype=complex
    )

    return _finalize_components(
        ansatz=workspace.material.ansatz,
        opts=workspace.material.options,
        shared_eigenbasis_q0=workspace.shared_eigenbasis_q0,
        shared_eigenbasis_q0_tolerance=1e-14,
        collective_mode=workspace.material.collective_mode,
        collective_mode_disabled_reason=workspace.material.collective_mode_disabled_reason,
        bubble=bubble,
        direct=np.asarray(workspace.direct_contact_contribution, dtype=complex),
        phase_left=phase_left,
        phase_right=phase_right,
        phase_phase_bubble_matrix=phase_phase_bubble,
        phase_phase_direct_plus=workspace.phase_phase_direct_plus,
        phase_phase_direct_minus=workspace.phase_phase_direct_minus,
        collective_bubble=collective_bubble,
        collective_counterterm_matrix=np.asarray(
            workspace.material.collective_counterterm_matrix, dtype=complex
        ),
        em_collective_left=em_collective_left,
        collective_em_right=collective_em_right,
        config=config,
        q=workspace.q_model,
        workspace_evaluation=True,
    )


def finite_q_bdg_responses_from_q_workspace(
    workspace: FiniteQQWorkspace,
    omega_eV_values: Sequence[float] | np.ndarray,
    *,
    config: KuboConfig | None = None,
) -> tuple[BdGFiniteQResponseComponents, ...]:
    """Evaluate all requested Matsubara energies in one vectorized contraction."""

    reference = workspace.material.config
    base_config = reference if config is None else config
    _compatible_config(reference, base_config)
    omega = np.asarray(omega_eV_values, dtype=float)
    raw_factors = _vectorized_kubo_factors(workspace, omega)
    weighted = 0.5 * workspace.material.k_weights[None, :, None, None] * raw_factors
    blocks = np.einsum(
        "xkmn,kamn,kbmn->xab",
        weighted,
        workspace.left_vertices_band,
        np.conjugate(workspace.right_vertices_band),
        optimize=True,
    )
    results: list[BdGFiniteQResponseComponents] = []
    for index, xi in enumerate(omega):
        eval_config = replace(base_config, omega_eV=float(xi))
        results.append(_components_from_unified_block(workspace, blocks[index], eval_config))
    return tuple(results)


def finite_q_bdg_response_from_q_workspace(
    workspace: FiniteQQWorkspace,
    omega_eV: float | None = None,
    *,
    config: KuboConfig | None = None,
) -> BdGFiniteQResponseComponents:
    """Evaluate one Matsubara energy from the vectorized q workspace."""

    xi = workspace.material.config.omega_eV if omega_eV is None else float(omega_eV)
    return finite_q_bdg_responses_from_q_workspace(
        workspace, np.asarray([xi], dtype=float), config=config
    )[0]


def primitive_ward_rhs_from_q_workspace(
    workspace: FiniteQQWorkspace,
    omega_eV: float | None = None,
) -> PrimitiveWardRHS:
    """Return the cached analytic RHS, relabeled for one Matsubara frequency."""

    xi = workspace.material.config.omega_eV if omega_eV is None else float(omega_eV)
    if not np.isfinite(xi) or xi < 0.0:
        raise ValueError("omega_eV must be finite and non-negative")
    delta0 = float(workspace.material.pairing_params.delta0_eV)
    return PrimitiveWardRHS(
        left=workspace.ward_rhs_vector,
        right=workspace.ward_rhs_vector.copy(),
        q_model=workspace.q_model,
        xi_eV=xi,
        delta0_eV=delta0,
        metadata={
            "convention": "primitive_crystal_xy_rhs_aware",
            "basis": "crystal_A0_xy",
            "formula": "R_S = equal_forward - delta_v_mid + qM_mid",
            "source": "FiniteQQWorkspace.cached_ward_rhs",
            "frequency_independent_rhs_reused": True,
            "equal_forward": workspace.metadata["ward_equal_forward"],
            "delta_v_mid": workspace.metadata["ward_delta_v_mid"],
            "qM_mid": workspace.metadata["ward_qM_mid"],
            "num_quadrature_points": workspace.nk,
        },
    )


__all__ = [
    "FiniteQMaterialWorkspace",
    "FiniteQQWorkspace",
    "finite_q_bdg_response_from_q_workspace",
    "finite_q_bdg_responses_from_q_workspace",
    "precompute_finite_q_material_workspace_from_model_ansatz",
    "precompute_finite_q_q_workspace",
    "primitive_ward_rhs_from_q_workspace",
]
