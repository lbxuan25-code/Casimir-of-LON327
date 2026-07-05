"""Model-driven finite-q BdG response assembly."""

from __future__ import annotations

import warnings

from dataclasses import dataclass, replace

import numpy as np

from lno327.bdg.finite_q import (
    bdg_finite_q_vertex_from_normal_blocks,
    density_vertex,
    phase_phase_direct_vertex,
    phase_vertex,
)
from lno327.bdg.hamiltonian import bdg_hamiltonian_from_model_pairing
from lno327.bdg.nambu import charge_current_vertex_from_model, diamagnetic_vertex_from_model
from lno327.bdg.spectrum import diagonalize_hermitian
from lno327.collective.schur import apply_amplitude_phase_schur, apply_phase_only_schur
from lno327.collective.ward import ward_metadata
from lno327.response.finite_q import (
    BdGFiniteQResponseComponents,
    add_band_bubble,
    add_bubble,
    thermal_expectation_bdg_from_hamiltonian,
)
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_finite_q_inputs


class _DefaultFiniteQOptions:
    include_phase_correction = True
    current_vertex = "peierls"
    include_phase_phase_direct = True
    phase_phase_direct_convention = "plus"
    collective_mode = "amplitude_phase"
    collective_counterterm = "goldstone_gap_equation"


@dataclass(frozen=True)
class FiniteQBdGWorkspaceEntry:
    weight: float
    kx: float
    ky: float
    qx: float
    qy: float
    shared_eigenbasis_q0: bool
    energies_minus: np.ndarray
    energies_plus: np.ndarray
    states_minus: np.ndarray
    states_plus: np.ndarray
    occupations_minus: np.ndarray
    occupations_plus: np.ndarray
    observable_vertices_band: tuple[np.ndarray, ...]
    source_vertices_band: tuple[np.ndarray, ...]
    phase_vertex_band: tuple[np.ndarray, ...]
    collective_vertices_band: tuple[np.ndarray, ...]
    phase_phase_direct_plus: complex
    phase_phase_direct_minus: complex
    direct_contact_contribution: np.ndarray


@dataclass(frozen=True)
class FiniteQBdGWorkspace:
    spec: object
    ansatz: object
    q_model: np.ndarray
    k_points: np.ndarray
    k_weights: np.ndarray
    config: object
    pairing_params: object
    options: object
    shared_eigenbasis_q0: bool
    collective_mode: str
    collective_mode_disabled_reason: str | None
    entries: tuple[FiniteQBdGWorkspaceEntry, ...]
    collective_counterterm_matrix: np.ndarray
    metadata: dict[str, object]


def require_peierls_finite_q_support(spec) -> None:
    required = (
        "hopping_terms",
        "peierls_hamiltonian_vector_vertex",
        "peierls_hamiltonian_contact_vertex",
    )
    if not all(hasattr(spec, name) for name in required):
        raise ValueError("spec must support Peierls finite-q vertices when current_vertex='peierls'")


def bdg_eigensystem_from_model_pairing(spec, kx: float, ky: float, pairing: np.ndarray):
    return diagonalize_hermitian(bdg_hamiltonian_from_model_pairing(spec, kx, ky, pairing))


def bdg_vector_vertex_from_spec(
    spec,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    current_vertex: str,
) -> np.ndarray:
    if current_vertex == "peierls":
        require_peierls_finite_q_support(spec)
        particle = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, direction)
        hole_normal = spec.peierls_hamiltonian_vector_vertex(-kx, -ky, -qx, -qy, direction)
        return bdg_finite_q_vertex_from_normal_blocks(particle, hole_normal)
    if current_vertex == "q0_velocity":
        return charge_current_vertex_from_model(spec, kx, ky, direction)
    raise ValueError("current_vertex must be 'peierls' or 'q0_velocity'")


def bdg_contact_vertex_from_spec(
    spec,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction_i: str,
    direction_j: str,
    current_vertex: str,
) -> np.ndarray:
    if current_vertex == "peierls":
        require_peierls_finite_q_support(spec)
        particle = spec.peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
        hole_normal = spec.peierls_hamiltonian_contact_vertex(-kx, -ky, -qx, -qy, direction_i, direction_j)
        return bdg_finite_q_vertex_from_normal_blocks(particle, hole_normal)
    if current_vertex == "q0_velocity":
        return diamagnetic_vertex_from_model(spec, kx, ky, direction_i, direction_j)
    raise ValueError("current_vertex must be 'peierls' or 'q0_velocity'")


def _pairing_params_from_inputs(spec, pairing_params):
    amp = pairing_params if pairing_params is not None else getattr(spec, "pairing_amplitudes", None)
    if amp is None:
        raise ValueError("pairing_params must be provided or available on spec")
    return amp


def finite_q_bdg_response_from_model_ansatz(
    spec,
    ansatz,
    omega_eV: float,
    q_model: np.ndarray,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config,
    pairing_params=None,
    options=None,
) -> BdGFiniteQResponseComponents:
    """Return finite-q BdG response components from a model spec and pairing ansatz."""

    if abs(float(config.omega_eV) - float(omega_eV)) > max(1e-14, 1e-10 * max(1.0, abs(float(omega_eV)))):
        raise ValueError("omega_eV must match config.omega_eV")
    opts = options or _DefaultFiniteQOptions()
    if opts.current_vertex not in {"peierls", "q0_velocity"}:
        raise ValueError("current_vertex must be 'peierls' or 'q0_velocity'")
    if opts.phase_phase_direct_convention not in {"plus", "minus"}:
        raise ValueError("phase_phase_direct_convention must be 'plus' or 'minus'")
    if opts.collective_mode not in {"none", "phase_only", "amplitude_phase"}:
        raise ValueError("collective_mode must be 'none', 'phase_only', or 'amplitude_phase'")
    if opts.collective_counterterm not in {"none", "goldstone_gap_equation"}:
        raise ValueError("collective_counterterm must be 'none' or 'goldstone_gap_equation'")
    q, points, weights = validate_finite_q_inputs(q_model, k_points, k_weights, config)
    if opts.current_vertex == "peierls":
        require_peierls_finite_q_support(spec)
    amp = _pairing_params_from_inputs(spec, pairing_params)
    delta0 = float(amp.delta0_eV)
    collective_mode = opts.collective_mode
    collective_mode_disabled_reason = None
    if delta0 == 0.0 and collective_mode == "amplitude_phase":
        collective_mode = "none"
        collective_mode_disabled_reason = "delta0=0 normal limit"

    qx, qy = float(q[0]), float(q[1])
    shared_eigenbasis_q0_tolerance = 1e-14
    shared_eigenbasis_q0 = bool(np.linalg.norm(q) <= shared_eigenbasis_q0_tolerance)
    sample_kx = float(points[0, 0])
    sample_ky = float(points[0, 1])
    orbital_dim = np.asarray(spec.normal_hamiltonian(sample_kx, sample_ky)).shape[0]
    rho = density_vertex(int(orbital_dim))
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    phase_left = np.zeros(3, dtype=complex)
    phase_right = np.zeros(3, dtype=complex)
    phase_phase_bubble_matrix = np.zeros((1, 1), dtype=complex)
    phase_phase_direct_plus = 0.0 + 0.0j
    phase_phase_direct_minus = 0.0 + 0.0j
    collective_bubble = np.zeros((2, 2), dtype=complex)
    em_collective_left = np.zeros((3, 2), dtype=complex)
    collective_em_right = np.zeros((2, 3), dtype=complex)
    directions = ("x", "y")

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        if shared_eigenbasis_q0:
            delta_mid = ansatz.mean_pairing(kx, ky, amp)
            bands = bdg_eigensystem_from_model_pairing(spec, kx, ky, delta_mid)
            occupations = fermi_function(bands.energies, config.fermi_level_eV, config.temperature_eV)
            energies_minus = energies_plus = bands.energies
            states_minus = states_plus = bands.states
            occupations_minus = occupations_plus = occupations
        else:
            kx_minus, ky_minus = kx - 0.5 * qx, ky - 0.5 * qy
            kx_plus, ky_plus = kx + 0.5 * qx, ky + 0.5 * qy
            delta_minus = ansatz.mean_pairing(kx_minus, ky_minus, amp)
            delta_plus = ansatz.mean_pairing(kx_plus, ky_plus, amp)
            bands_minus = bdg_eigensystem_from_model_pairing(spec, kx_minus, ky_minus, delta_minus)
            bands_plus = bdg_eigensystem_from_model_pairing(spec, kx_plus, ky_plus, delta_plus)
            energies_minus, states_minus = bands_minus.energies, bands_minus.states
            energies_plus, states_plus = bands_plus.energies, bands_plus.states
            occupations_minus = fermi_function(energies_minus, config.fermi_level_eV, config.temperature_eV)
            occupations_plus = fermi_function(energies_plus, config.fermi_level_eV, config.temperature_eV)
            delta_mid = ansatz.mean_pairing(kx, ky, amp)

        vx = bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "x", opts.current_vertex)
        vy = bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "y", opts.current_vertex)
        observable_vertices = (rho, -vx, -vy)
        source_vertices = (rho, vx, vy)
        add_bubble(
            bubble,
            observable_vertices,
            source_vertices,
            energies_minus,
            states_minus,
            occupations_minus,
            energies_plus,
            states_plus,
            occupations_plus,
            config.omega_eV,
            float(weight),
            config=config,
            static_limit=shared_eigenbasis_q0,
        )
        if collective_mode == "amplitude_phase":
            collective_vertices = ansatz.collective_vertices(kx, ky, qx, qy, amp)
            add_bubble(
                em_collective_left,
                observable_vertices,
                collective_vertices,
                energies_minus,
                states_minus,
                occupations_minus,
                energies_plus,
                states_plus,
                occupations_plus,
                config.omega_eV,
                float(weight),
                config=config,
                static_limit=shared_eigenbasis_q0,
            )
            add_bubble(
                collective_em_right,
                collective_vertices,
                source_vertices,
                energies_minus,
                states_minus,
                occupations_minus,
                energies_plus,
                states_plus,
                occupations_plus,
                config.omega_eV,
                float(weight),
                config=config,
                static_limit=shared_eigenbasis_q0,
            )
            add_bubble(
                collective_bubble,
                collective_vertices,
                collective_vertices,
                energies_minus,
                states_minus,
                occupations_minus,
                energies_plus,
                states_plus,
                occupations_plus,
                config.omega_eV,
                float(weight),
                config=config,
                static_limit=shared_eigenbasis_q0,
            )

        delta_theta = ansatz.phase_pairing_matrix(kx, ky, qx, qy, amp)
        theta = phase_vertex(delta_theta)
        tmp_left = np.zeros((3, 1), dtype=complex)
        add_bubble(
            tmp_left,
            observable_vertices,
            (theta,),
            energies_minus,
            states_minus,
            occupations_minus,
            energies_plus,
            states_plus,
            occupations_plus,
            config.omega_eV,
            float(weight),
            config=config,
            static_limit=shared_eigenbasis_q0,
        )
        phase_left += tmp_left[:, 0]
        tmp_right = np.zeros((1, 3), dtype=complex)
        add_bubble(
            tmp_right,
            (theta,),
            source_vertices,
            energies_minus,
            states_minus,
            occupations_minus,
            energies_plus,
            states_plus,
            occupations_plus,
            config.omega_eV,
            float(weight),
            config=config,
            static_limit=shared_eigenbasis_q0,
        )
        phase_right += tmp_right[0, :]
        add_bubble(
            phase_phase_bubble_matrix,
            (theta,),
            (theta,),
            energies_minus,
            states_minus,
            occupations_minus,
            energies_plus,
            states_plus,
            occupations_plus,
            config.omega_eV,
            float(weight),
            config=config,
            static_limit=shared_eigenbasis_q0,
        )
        theta_theta = phase_phase_direct_vertex(delta_theta)
        h_mid = bdg_hamiltonian_from_model_pairing(spec, kx, ky, delta_mid)
        direct_value = float(weight) * thermal_expectation_bdg_from_hamiltonian(h_mid, theta_theta, config)
        phase_phase_direct_plus += direct_value
        phase_phase_direct_minus -= direct_value

        for i, direction_i in enumerate(directions):
            for j, direction_j in enumerate(directions):
                vertex = bdg_contact_vertex_from_spec(
                    spec,
                    kx,
                    ky,
                    qx,
                    qy,
                    direction_i,
                    direction_j,
                    opts.current_vertex,
                )
                direct[1 + i, 1 + j] += -float(weight) * thermal_expectation_bdg_from_hamiltonian(
                    h_mid,
                    vertex,
                    config,
                )

    bare_total = bubble + direct
    phase_phase_bubble = complex(phase_phase_bubble_matrix[0, 0])
    selected_phase_phase_direct = (
        phase_phase_direct_plus if opts.phase_phase_direct_convention == "plus" else phase_phase_direct_minus
    )
    phase_phase_direct = selected_phase_phase_direct if opts.include_phase_phase_direct else 0.0 + 0.0j
    phase_phase_total = phase_phase_bubble + phase_phase_direct
    minus_schur_result = apply_phase_only_schur(
        bare_total,
        phase_left,
        phase_phase_total,
        phase_right,
        sign="minus",
    )
    plus_schur_result = apply_phase_only_schur(
        bare_total,
        phase_left,
        phase_phase_total,
        phase_right,
        sign="plus",
    )
    minus_schur = minus_schur_result.corrected_response
    plus_schur = plus_schur_result.corrected_response
    gauge_restored = bare_total.copy()
    phase_status = "disabled"
    warning_message = None
    threshold = max(100.0 * float(config.eta_eV), 1e-14)
    if opts.include_phase_correction:
        if abs(phase_phase_total) <= threshold:
            phase_status = "singular_phase_phase"
            warning_message = (
                f"Global phase correction skipped because |K_theta_theta|={abs(phase_phase_total):.3e} "
                f"is below threshold {threshold:.3e}."
            )
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
        else:
            gauge_restored = minus_schur
            phase_status = "applied"
    ward_bare = ward_metadata(bare_total, config.omega_eV, q)
    ward_minus = ward_metadata(minus_schur, config.omega_eV, q)
    ward_plus = ward_metadata(plus_schur, config.omega_eV, q)
    collective_counterterm_matrix = np.zeros((2, 2), dtype=complex)
    goldstone_counterterm_cg = 0.0 + 0.0j
    if collective_mode == "amplitude_phase" and opts.collective_counterterm == "goldstone_gap_equation":
        collective_counterterm_matrix = ansatz.hs_counterterm(config, points, weights, amp)
        goldstone_counterterm_cg = complex(collective_counterterm_matrix[1, 1])
    collective_total = collective_bubble + collective_counterterm_matrix
    amplitude_phase_schur = bare_total.copy()
    collective_condition: float | None = None
    collective_inverse_method = "not_used"
    amp_phase_schur_result = None
    if collective_mode == "amplitude_phase":
        amp_phase_schur_result = apply_amplitude_phase_schur(
            bare_total,
            em_collective_left,
            collective_total,
            collective_em_right,
        )
        collective_condition = amp_phase_schur_result.condition_number
        collective_inverse_method = amp_phase_schur_result.inverse_method
        amplitude_phase_schur = amp_phase_schur_result.corrected_response
        if opts.include_phase_correction:
            gauge_restored = amplitude_phase_schur
            phase_status = "amplitude_phase_applied"
    ward_amp_phase = ward_metadata(amplitude_phase_schur, config.omega_eV, q)
    selected_gauge = (
        "amplitude_phase_schur"
        if collective_mode == "amplitude_phase" and opts.include_phase_correction
        else ("minus_schur" if phase_status == "applied" else "bare_total")
    )

    return BdGFiniteQResponseComponents(
        bare_bubble=bubble,
        direct=direct,
        bare_total=bare_total,
        phase_coupling_left=phase_left,
        phase_coupling_right=phase_right,
        phase_phase_bubble=phase_phase_bubble,
        phase_phase_direct=phase_phase_direct,
        phase_phase_total=phase_phase_total,
        minus_schur=minus_schur,
        plus_schur=plus_schur,
        collective_bubble=collective_bubble,
        collective_counterterm=collective_counterterm_matrix,
        collective_total=collective_total,
        em_collective_left=em_collective_left,
        collective_em_right=collective_em_right,
        amplitude_phase_schur=amplitude_phase_schur,
        gauge_restored=gauge_restored,
        metadata={
            "nambu_basis": "(c_k, c^dagger_-k)",
            "nambu_prefactor": 0.5,
            "finite_q_routing": "k_minus=k-q/2,k_plus=k+q/2",
            "shared_eigenbasis_q0": shared_eigenbasis_q0,
            "shared_eigenbasis_q0_tolerance": shared_eigenbasis_q0_tolerance,
            "current_observable_source_convention": "J=(rho,-Vx,-Vy), P=(rho,Vx,Vy)",
            "direct_contact_convention": "D_ij=-<M_ij> with BdG Nambu 1/2",
            "effective_action_convention": "S2=1/2(a,theta)[[K_munu,K_mutheta],[K_thetanu,K_thetatheta]](a,theta)^T",
            "phase_correction_formula": "Pi_GI = Pi_bare - K_mu_theta K_theta_nu / K_theta_theta",
            "phase_correction_sign_checked": True,
            "validation_only_pairing": ansatz.name == "onsite_s",
            "finite_q_current_vertex_status": (
                "normal_state_exact_finite_q_peierls_vertex"
                if opts.current_vertex == "peierls"
                else "q0_velocity_vertex_approximation_not_gauge_closed"
            ),
            "model_input_layer": ansatz.metadata(),
            "collective_mode": collective_mode,
            "collective_channels": (
                ["amplitude_eta1", "phase_eta2"] if collective_mode == "amplitude_phase" else ["global_phase_only"]
            ),
            "collective_counterterm": opts.collective_counterterm,
            "eta2_phase_relation": "eta2 = delta0 * theta",
            "collective_mode_disabled_reason": collective_mode_disabled_reason,
            "goldstone_counterterm_Cg": goldstone_counterterm_cg,
            "goldstone_condition_target": "K_eta2_eta2(q=0, omega=0) = 0",
            "collective_total_condition_number": collective_condition,
            "collective_inverse_method": collective_inverse_method,
            "phase_only_schur_status": minus_schur_result.status,
            "amplitude_phase_schur_status": (
                amp_phase_schur_result.status if collective_mode == "amplitude_phase" else "not_used"
            ),
            "amplitude_phase_schur_formula": "Pi_GI = K_munu - K_mu_a inv(K_ab) K_b_nu",
            "phase_vertex": ansatz.phase_vertex,
            "phase_vertex_status": f"{ansatz.phase_vertex}_pair_center_of_mass_phase_not_full_gauge_closure_proof",
            "phase_vertex_convention": (
                "Gamma_theta(k) = dH_BdG/dtheta at pair center-of-mass phase; "
                f"for k-dependent Delta this implementation uses phase_vertex={ansatz.phase_vertex}, "
                "while quasiparticles are routed through k+/-q/2."
            ),
            "phase_phase_direct_included": bool(opts.include_phase_phase_direct),
            "phase_phase_total_definition": "bubble + direct",
            "phase_phase_direct_convention": opts.phase_phase_direct_convention,
            "phase_phase_direct_plus_convention": phase_phase_direct_plus,
            "phase_phase_direct_minus_convention": phase_phase_direct_minus,
            "phase_kernel_status": (
                "bubble_plus_direct"
                if opts.include_phase_phase_direct
                else "bubble_only_not_expected_to_gauge_close"
            ),
            "phase_correction_requested": bool(opts.include_phase_correction),
            "phase_correction_applied": phase_status == "applied",
            "phase_correction_status": phase_status,
            "phase_phase_bubble": phase_phase_bubble,
            "phase_phase_direct": phase_phase_direct,
            "phase_phase_total": phase_phase_total,
            "phase_phase_abs": float(abs(phase_phase_total)),
            "phase_phase_bubble_abs": float(abs(phase_phase_bubble)),
            "phase_phase_direct_abs": float(abs(phase_phase_direct)),
            "phase_phase_singular_threshold": float(threshold),
            "ward_residual_bare": ward_bare,
            "ward_residual_minus_schur": ward_minus,
            "ward_residual_plus_schur": ward_plus,
            "ward_residual_amplitude_phase_schur": ward_amp_phase,
            "selected_gauge_restored": selected_gauge,
            "gauge_restored_selected": selected_gauge,
            "normal_backend_reference_used": False,
            "valid_for_casimir_input": False,
            "casimir_gating_status": "diagnostic_finite_q_response_not_unit_converted_or_ward_validated",
            "warning": warning_message,
        },
    )


def _thermal_expectation_from_bands(energies: np.ndarray, states: np.ndarray, vertex: np.ndarray, config) -> complex:
    occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
    vertex_in_band = states.conjugate().T @ vertex @ states
    return complex(0.5 * np.sum(occupations * np.diag(vertex_in_band)))


def _compatible_workspace_config(workspace_config, eval_config) -> None:
    if float(workspace_config.temperature_eV) != float(eval_config.temperature_eV):
        raise ValueError("workspace config temperature_eV changed; rebuild the workspace")
    if float(workspace_config.fermi_level_eV) != float(eval_config.fermi_level_eV):
        raise ValueError("workspace config fermi_level_eV changed; rebuild the workspace")
    if float(workspace_config.eta_eV) != float(eval_config.eta_eV):
        raise ValueError("workspace config eta_eV changed; rebuild the workspace")
    if bool(workspace_config.output_si) != bool(eval_config.output_si):
        raise ValueError("workspace config output_si changed; rebuild the workspace")


def _assemble_finite_q_components_from_workspace(
    workspace: FiniteQBdGWorkspace,
    config,
) -> BdGFiniteQResponseComponents:
    opts = workspace.options
    q = workspace.q_model
    amp = workspace.pairing_params
    ansatz = workspace.ansatz
    collective_mode = workspace.collective_mode
    shared_eigenbasis_q0_tolerance = workspace.metadata["shared_eigenbasis_q0_tolerance"]
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    phase_left = np.zeros(3, dtype=complex)
    phase_right = np.zeros(3, dtype=complex)
    phase_phase_bubble_matrix = np.zeros((1, 1), dtype=complex)
    phase_phase_direct_plus = 0.0 + 0.0j
    phase_phase_direct_minus = 0.0 + 0.0j
    collective_bubble = np.zeros((2, 2), dtype=complex)
    em_collective_left = np.zeros((3, 2), dtype=complex)
    collective_em_right = np.zeros((2, 3), dtype=complex)

    for entry in workspace.entries:
        add_band_bubble(
            bubble,
            entry.observable_vertices_band,
            entry.source_vertices_band,
            entry.energies_minus,
            entry.occupations_minus,
            entry.energies_plus,
            entry.occupations_plus,
            config.omega_eV,
            entry.weight,
            config=config,
            static_limit=entry.shared_eigenbasis_q0,
        )
        if collective_mode == "amplitude_phase":
            add_band_bubble(
                em_collective_left,
                entry.observable_vertices_band,
                entry.collective_vertices_band,
                entry.energies_minus,
                entry.occupations_minus,
                entry.energies_plus,
                entry.occupations_plus,
                config.omega_eV,
                entry.weight,
                config=config,
                static_limit=entry.shared_eigenbasis_q0,
            )
            add_band_bubble(
                collective_em_right,
                entry.collective_vertices_band,
                entry.source_vertices_band,
                entry.energies_minus,
                entry.occupations_minus,
                entry.energies_plus,
                entry.occupations_plus,
                config.omega_eV,
                entry.weight,
                config=config,
                static_limit=entry.shared_eigenbasis_q0,
            )
            add_band_bubble(
                collective_bubble,
                entry.collective_vertices_band,
                entry.collective_vertices_band,
                entry.energies_minus,
                entry.occupations_minus,
                entry.energies_plus,
                entry.occupations_plus,
                config.omega_eV,
                entry.weight,
                config=config,
                static_limit=entry.shared_eigenbasis_q0,
            )
        tmp_left = np.zeros((3, 1), dtype=complex)
        add_band_bubble(
            tmp_left,
            entry.observable_vertices_band,
            entry.phase_vertex_band,
            entry.energies_minus,
            entry.occupations_minus,
            entry.energies_plus,
            entry.occupations_plus,
            config.omega_eV,
            entry.weight,
            config=config,
            static_limit=entry.shared_eigenbasis_q0,
        )
        phase_left += tmp_left[:, 0]
        tmp_right = np.zeros((1, 3), dtype=complex)
        add_band_bubble(
            tmp_right,
            entry.phase_vertex_band,
            entry.source_vertices_band,
            entry.energies_minus,
            entry.occupations_minus,
            entry.energies_plus,
            entry.occupations_plus,
            config.omega_eV,
            entry.weight,
            config=config,
            static_limit=entry.shared_eigenbasis_q0,
        )
        phase_right += tmp_right[0, :]
        add_band_bubble(
            phase_phase_bubble_matrix,
            entry.phase_vertex_band,
            entry.phase_vertex_band,
            entry.energies_minus,
            entry.occupations_minus,
            entry.energies_plus,
            entry.occupations_plus,
            config.omega_eV,
            entry.weight,
            config=config,
            static_limit=entry.shared_eigenbasis_q0,
        )
        phase_phase_direct_plus += entry.phase_phase_direct_plus
        phase_phase_direct_minus += entry.phase_phase_direct_minus
        direct += entry.direct_contact_contribution

    bare_total = bubble + direct
    phase_phase_bubble = complex(phase_phase_bubble_matrix[0, 0])
    selected_phase_phase_direct = (
        phase_phase_direct_plus if opts.phase_phase_direct_convention == "plus" else phase_phase_direct_minus
    )
    phase_phase_direct = selected_phase_phase_direct if opts.include_phase_phase_direct else 0.0 + 0.0j
    phase_phase_total = phase_phase_bubble + phase_phase_direct
    minus_schur_result = apply_phase_only_schur(
        bare_total,
        phase_left,
        phase_phase_total,
        phase_right,
        sign="minus",
    )
    plus_schur_result = apply_phase_only_schur(
        bare_total,
        phase_left,
        phase_phase_total,
        phase_right,
        sign="plus",
    )
    minus_schur = minus_schur_result.corrected_response
    plus_schur = plus_schur_result.corrected_response
    gauge_restored = bare_total.copy()
    phase_status = "disabled"
    warning_message = None
    threshold = max(100.0 * float(config.eta_eV), 1e-14)
    if opts.include_phase_correction:
        if abs(phase_phase_total) <= threshold:
            phase_status = "singular_phase_phase"
            warning_message = (
                f"Global phase correction skipped because |K_theta_theta|={abs(phase_phase_total):.3e} "
                f"is below threshold {threshold:.3e}."
            )
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
        else:
            gauge_restored = minus_schur
            phase_status = "applied"
    ward_bare = ward_metadata(bare_total, config.omega_eV, q)
    ward_minus = ward_metadata(minus_schur, config.omega_eV, q)
    ward_plus = ward_metadata(plus_schur, config.omega_eV, q)
    collective_counterterm_matrix = np.zeros((2, 2), dtype=complex)
    goldstone_counterterm_cg = 0.0 + 0.0j
    if collective_mode == "amplitude_phase" and opts.collective_counterterm == "goldstone_gap_equation":
        collective_counterterm_matrix = workspace.collective_counterterm_matrix
        goldstone_counterterm_cg = complex(collective_counterterm_matrix[1, 1])
    collective_total = collective_bubble + collective_counterterm_matrix
    amplitude_phase_schur = bare_total.copy()
    collective_condition: float | None = None
    collective_inverse_method = "not_used"
    amp_phase_schur_result = None
    if collective_mode == "amplitude_phase":
        amp_phase_schur_result = apply_amplitude_phase_schur(
            bare_total,
            em_collective_left,
            collective_total,
            collective_em_right,
        )
        collective_condition = amp_phase_schur_result.condition_number
        collective_inverse_method = amp_phase_schur_result.inverse_method
        amplitude_phase_schur = amp_phase_schur_result.corrected_response
        if opts.include_phase_correction:
            gauge_restored = amplitude_phase_schur
            phase_status = "amplitude_phase_applied"
    ward_amp_phase = ward_metadata(amplitude_phase_schur, config.omega_eV, q)
    selected_gauge = (
        "amplitude_phase_schur"
        if collective_mode == "amplitude_phase" and opts.include_phase_correction
        else ("minus_schur" if phase_status == "applied" else "bare_total")
    )
    return BdGFiniteQResponseComponents(
        bare_bubble=bubble,
        direct=direct,
        bare_total=bare_total,
        phase_coupling_left=phase_left,
        phase_coupling_right=phase_right,
        phase_phase_bubble=phase_phase_bubble,
        phase_phase_direct=phase_phase_direct,
        phase_phase_total=phase_phase_total,
        minus_schur=minus_schur,
        plus_schur=plus_schur,
        collective_bubble=collective_bubble,
        collective_counterterm=collective_counterterm_matrix,
        collective_total=collective_total,
        em_collective_left=em_collective_left,
        collective_em_right=collective_em_right,
        amplitude_phase_schur=amplitude_phase_schur,
        gauge_restored=gauge_restored,
        metadata={
            "nambu_basis": "(c_k, c^dagger_-k)",
            "nambu_prefactor": 0.5,
            "finite_q_routing": "k_minus=k-q/2,k_plus=k+q/2",
            "shared_eigenbasis_q0": workspace.shared_eigenbasis_q0,
            "shared_eigenbasis_q0_tolerance": shared_eigenbasis_q0_tolerance,
            "current_observable_source_convention": "J=(rho,-Vx,-Vy), P=(rho,Vx,Vy)",
            "direct_contact_convention": "D_ij=-<M_ij> with BdG Nambu 1/2",
            "effective_action_convention": "S2=1/2(a,theta)[[K_munu,K_mutheta],[K_thetanu,K_thetatheta]](a,theta)^T",
            "phase_correction_formula": "Pi_GI = Pi_bare - K_mu_theta K_theta_nu / K_theta_theta",
            "phase_correction_sign_checked": True,
            "validation_only_pairing": ansatz.name == "onsite_s",
            "finite_q_current_vertex_status": (
                "normal_state_exact_finite_q_peierls_vertex"
                if opts.current_vertex == "peierls"
                else "q0_velocity_vertex_approximation_not_gauge_closed"
            ),
            "model_input_layer": ansatz.metadata(),
            "collective_mode": collective_mode,
            "collective_channels": (
                ["amplitude_eta1", "phase_eta2"] if collective_mode == "amplitude_phase" else ["global_phase_only"]
            ),
            "collective_counterterm": opts.collective_counterterm,
            "eta2_phase_relation": "eta2 = delta0 * theta",
            "collective_mode_disabled_reason": workspace.collective_mode_disabled_reason,
            "goldstone_counterterm_Cg": goldstone_counterterm_cg,
            "goldstone_condition_target": "K_eta2_eta2(q=0, omega=0) = 0",
            "collective_total_condition_number": collective_condition,
            "collective_inverse_method": collective_inverse_method,
            "phase_only_schur_status": minus_schur_result.status,
            "amplitude_phase_schur_status": (
                amp_phase_schur_result.status if collective_mode == "amplitude_phase" else "not_used"
            ),
            "amplitude_phase_schur_formula": "Pi_GI = K_munu - K_mu_a inv(K_ab) K_b_nu",
            "phase_vertex": ansatz.phase_vertex,
            "phase_vertex_status": f"{ansatz.phase_vertex}_pair_center_of_mass_phase_not_full_gauge_closure_proof",
            "phase_vertex_convention": (
                "Gamma_theta(k) = dH_BdG/dtheta at pair center-of-mass phase; "
                f"for k-dependent Delta this implementation uses phase_vertex={ansatz.phase_vertex}, "
                "while quasiparticles are routed through k+/-q/2."
            ),
            "phase_phase_direct_included": bool(opts.include_phase_phase_direct),
            "phase_phase_total_definition": "bubble + direct",
            "phase_phase_direct_convention": opts.phase_phase_direct_convention,
            "phase_phase_direct_plus_convention": phase_phase_direct_plus,
            "phase_phase_direct_minus_convention": phase_phase_direct_minus,
            "phase_kernel_status": (
                "bubble_plus_direct"
                if opts.include_phase_phase_direct
                else "bubble_only_not_expected_to_gauge_close"
            ),
            "phase_correction_requested": bool(opts.include_phase_correction),
            "phase_correction_applied": phase_status == "applied",
            "phase_correction_status": phase_status,
            "phase_phase_bubble": phase_phase_bubble,
            "phase_phase_direct": phase_phase_direct,
            "phase_phase_total": phase_phase_total,
            "phase_phase_abs": float(abs(phase_phase_total)),
            "phase_phase_bubble_abs": float(abs(phase_phase_bubble)),
            "phase_phase_direct_abs": float(abs(phase_phase_direct)),
            "phase_phase_singular_threshold": float(threshold),
            "ward_residual_bare": ward_bare,
            "ward_residual_minus_schur": ward_minus,
            "ward_residual_plus_schur": ward_plus,
            "ward_residual_amplitude_phase_schur": ward_amp_phase,
            "selected_gauge_restored": selected_gauge,
            "gauge_restored_selected": selected_gauge,
            "normal_backend_reference_used": False,
            "valid_for_casimir_input": False,
            "casimir_gating_status": "diagnostic_finite_q_response_not_unit_converted_or_ward_validated",
            "warning": warning_message,
            "workspace_evaluation": True,
        },
    )


def precompute_finite_q_bdg_workspace_from_model_ansatz(
    spec,
    ansatz,
    q_model: np.ndarray,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config,
    pairing_params=None,
    options=None,
) -> FiniteQBdGWorkspace:
    opts = options or _DefaultFiniteQOptions()
    q, points, weights = validate_finite_q_inputs(q_model, k_points, k_weights, config)
    if opts.current_vertex not in {"peierls", "q0_velocity"}:
        raise ValueError("current_vertex must be 'peierls' or 'q0_velocity'")
    if opts.phase_phase_direct_convention not in {"plus", "minus"}:
        raise ValueError("phase_phase_direct_convention must be 'plus' or 'minus'")
    if opts.collective_mode not in {"none", "phase_only", "amplitude_phase"}:
        raise ValueError("collective_mode must be 'none', 'phase_only', or 'amplitude_phase'")
    if opts.collective_counterterm not in {"none", "goldstone_gap_equation"}:
        raise ValueError("collective_counterterm must be 'none' or 'goldstone_gap_equation'")
    if opts.current_vertex == "peierls":
        require_peierls_finite_q_support(spec)
    amp = _pairing_params_from_inputs(spec, pairing_params)
    collective_mode = opts.collective_mode
    collective_mode_disabled_reason = None
    if float(amp.delta0_eV) == 0.0 and collective_mode == "amplitude_phase":
        collective_mode = "none"
        collective_mode_disabled_reason = "delta0=0 normal limit"
    shared_tolerance = 1e-14
    shared = bool(np.linalg.norm(q) <= shared_tolerance)
    qx, qy = float(q[0]), float(q[1])
    sample_kx = float(points[0, 0])
    sample_ky = float(points[0, 1])
    orbital_dim = np.asarray(spec.normal_hamiltonian(sample_kx, sample_ky)).shape[0]
    rho = density_vertex(int(orbital_dim))
    directions = ("x", "y")
    entries: list[FiniteQBdGWorkspaceEntry] = []
    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        if shared:
            delta_mid = ansatz.mean_pairing(kx, ky, amp)
            bands = bdg_eigensystem_from_model_pairing(spec, kx, ky, delta_mid)
            occupations = fermi_function(bands.energies, config.fermi_level_eV, config.temperature_eV)
            energies_minus = energies_plus = bands.energies
            states_minus = states_plus = bands.states
            occupations_minus = occupations_plus = occupations
            midpoint_energies, midpoint_states = bands.energies, bands.states
        else:
            kx_minus, ky_minus = kx - 0.5 * qx, ky - 0.5 * qy
            kx_plus, ky_plus = kx + 0.5 * qx, ky + 0.5 * qy
            delta_minus = ansatz.mean_pairing(kx_minus, ky_minus, amp)
            delta_plus = ansatz.mean_pairing(kx_plus, ky_plus, amp)
            delta_mid = ansatz.mean_pairing(kx, ky, amp)
            bands_minus = bdg_eigensystem_from_model_pairing(spec, kx_minus, ky_minus, delta_minus)
            bands_plus = bdg_eigensystem_from_model_pairing(spec, kx_plus, ky_plus, delta_plus)
            bands_mid = bdg_eigensystem_from_model_pairing(spec, kx, ky, delta_mid)
            energies_minus, states_minus = bands_minus.energies, bands_minus.states
            energies_plus, states_plus = bands_plus.energies, bands_plus.states
            occupations_minus = fermi_function(energies_minus, config.fermi_level_eV, config.temperature_eV)
            occupations_plus = fermi_function(energies_plus, config.fermi_level_eV, config.temperature_eV)
            midpoint_energies, midpoint_states = bands_mid.energies, bands_mid.states

        vx = bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "x", opts.current_vertex)
        vy = bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "y", opts.current_vertex)
        observable_vertices = (rho, -vx, -vy)
        source_vertices = (rho, vx, vy)
        observable_vertices_band = tuple(states_minus.conjugate().T @ vertex @ states_plus for vertex in observable_vertices)
        source_vertices_band = tuple(states_minus.conjugate().T @ vertex @ states_plus for vertex in source_vertices)

        delta_theta = ansatz.phase_pairing_matrix(kx, ky, qx, qy, amp)
        theta = phase_vertex(delta_theta)
        phase_vertex_band = (states_minus.conjugate().T @ theta @ states_plus,)
        collective_vertices_band: tuple[np.ndarray, ...] = ()
        if collective_mode == "amplitude_phase":
            collective_vertices = ansatz.collective_vertices(kx, ky, qx, qy, amp)
            collective_vertices_band = tuple(
                states_minus.conjugate().T @ vertex @ states_plus for vertex in collective_vertices
            )

        theta_theta = phase_phase_direct_vertex(delta_theta)
        direct_value = float(weight) * _thermal_expectation_from_bands(
            midpoint_energies,
            midpoint_states,
            theta_theta,
            config,
        )
        direct_contact_contribution = np.zeros((3, 3), dtype=complex)
        for i, direction_i in enumerate(directions):
            for j, direction_j in enumerate(directions):
                vertex = bdg_contact_vertex_from_spec(
                    spec,
                    kx,
                    ky,
                    qx,
                    qy,
                    direction_i,
                    direction_j,
                    opts.current_vertex,
                )
                direct_contact_contribution[1 + i, 1 + j] += -float(weight) * _thermal_expectation_from_bands(
                    midpoint_energies,
                    midpoint_states,
                    vertex,
                    config,
                )

        entries.append(
            FiniteQBdGWorkspaceEntry(
                weight=float(weight),
                kx=kx,
                ky=ky,
                qx=qx,
                qy=qy,
                shared_eigenbasis_q0=shared,
                energies_minus=energies_minus,
                energies_plus=energies_plus,
                states_minus=states_minus,
                states_plus=states_plus,
                occupations_minus=occupations_minus,
                occupations_plus=occupations_plus,
                observable_vertices_band=observable_vertices_band,
                source_vertices_band=source_vertices_band,
                phase_vertex_band=phase_vertex_band,
                collective_vertices_band=collective_vertices_band,
                phase_phase_direct_plus=direct_value,
                phase_phase_direct_minus=-direct_value,
                direct_contact_contribution=direct_contact_contribution,
            )
        )

    collective_counterterm_matrix = np.zeros((2, 2), dtype=complex)
    if collective_mode == "amplitude_phase" and opts.collective_counterterm == "goldstone_gap_equation":
        collective_counterterm_matrix = ansatz.hs_counterterm(config, points, weights, amp)
    return FiniteQBdGWorkspace(
        spec=spec,
        ansatz=ansatz,
        q_model=q,
        k_points=points,
        k_weights=weights,
        config=config,
        pairing_params=amp,
        options=opts,
        shared_eigenbasis_q0=shared,
        collective_mode=collective_mode,
        collective_mode_disabled_reason=collective_mode_disabled_reason,
        entries=tuple(entries),
        collective_counterterm_matrix=collective_counterterm_matrix,
        metadata={
            "workspace_kind": "finite_q_bdg",
            "shared_eigenbasis_q0": shared,
            "shared_eigenbasis_q0_tolerance": shared_tolerance,
            "num_entries": len(entries),
            "cached_objects": [
                "bdg_eigensystems",
                "occupations",
                "em_vertices_band_basis",
                "phase_vertex_band_basis",
                "collective_vertices_band_basis",
                "phase_phase_direct",
                "direct_contact_matrix",
                "collective_counterterm",
            ],
            "valid_for_casimir_input": False,
        },
    )


def finite_q_bdg_response_from_workspace(
    workspace: FiniteQBdGWorkspace,
    omega_eV: float | None = None,
    config=None,
) -> BdGFiniteQResponseComponents:
    eval_config = config or workspace.config
    if omega_eV is not None and float(omega_eV) != float(eval_config.omega_eV):
        eval_config = replace(eval_config, omega_eV=float(omega_eV))
    _compatible_workspace_config(workspace.config, eval_config)
    return _assemble_finite_q_components_from_workspace(workspace, eval_config)
