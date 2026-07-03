"""Generic finite-q BdG response engine driven by explicit model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

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
from lno327.response.config import KuboConfig
from lno327.response.finite_q import (
    BdGFiniteQResponseComponents,
    add_bubble,
    thermal_expectation_bdg_from_hamiltonian,
)
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_finite_q_inputs
from .finite_q_primitives import ward_metadata
from .models.lno327_four_orbital.collective import PairingAnsatz, build_pairing_ansatz
from .models.lno327_four_orbital.parameters import PairingAmplitudes, PhaseVertexName
from .models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from .ward_response import normal_physical_density_current_response_components_imag_axis

PhaseDirectConvention = Literal["plus", "minus"]
CollectiveMode = Literal["none", "phase_only", "amplitude_phase"]
CollectiveCounterterm = Literal["none", "goldstone_gap_equation"]
PairingName = Literal["onsite_s", "spm", "dwave"]


@dataclass(frozen=True)
class FiniteQEngineOptions:
    include_phase_correction: bool = True
    current_vertex: Literal["peierls", "q0_velocity"] = "peierls"
    include_phase_phase_direct: bool = True
    phase_phase_direct_convention: PhaseDirectConvention = "plus"
    collective_mode: CollectiveMode = "amplitude_phase"
    collective_counterterm: CollectiveCounterterm = "goldstone_gap_equation"


@dataclass(frozen=True)
class SchurResult:
    corrected_response: np.ndarray
    condition_number: float | None
    inverse_method: Literal["inv", "pinv_diagnostic", "not_used"]
    status: str
    warning: str | None = None


class BdGPhaseCorrectionError(RuntimeError):
    """Raised when the global phase channel is singular."""


def _is_normal_limit(pairing_params: PairingAmplitudes | None) -> bool:
    return pairing_params is not None and abs(float(pairing_params.delta0_eV)) == 0.0


def _empty_phase_arrays() -> tuple[np.ndarray, np.ndarray, complex]:
    return np.zeros(3, dtype=complex), np.zeros(3, dtype=complex), 0.0 + 0.0j


def _normal_limit_components(
    q_model: np.ndarray,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: KuboConfig,
    *,
    include_phase_correction: bool,
) -> BdGFiniteQResponseComponents:
    components = normal_physical_density_current_response_components_imag_axis(
        k_points,
        config,
        q_model,
        k_weights,
    )
    left, right, phase_phase = _empty_phase_arrays()
    total = components["total"].astype(complex, copy=False)
    return BdGFiniteQResponseComponents(
        bare_bubble=components["bubble"].astype(complex, copy=False),
        direct=components["direct"].astype(complex, copy=False),
        bare_total=total,
        phase_coupling_left=left,
        phase_coupling_right=right,
        phase_phase_bubble=phase_phase,
        phase_phase_direct=phase_phase,
        phase_phase_total=phase_phase,
        minus_schur=total.copy(),
        plus_schur=total.copy(),
        collective_bubble=np.zeros((2, 2), dtype=complex),
        collective_counterterm=np.zeros((2, 2), dtype=complex),
        collective_total=np.zeros((2, 2), dtype=complex),
        em_collective_left=np.zeros((3, 2), dtype=complex),
        collective_em_right=np.zeros((2, 3), dtype=complex),
        amplitude_phase_schur=total.copy(),
        gauge_restored=total.copy(),
        metadata={
            "normal_limit_delegated_to": "normal_physical_density_current_response_components_imag_axis",
            "nambu_prefactor": 0.5,
            "finite_q_current_vertex_status": "normal_state_finite_q_peierls_backend",
            "collective_channels": ["global_phase_only"],
            "phase_correction_requested": bool(include_phase_correction),
            "phase_correction_applied": False,
            "phase_correction_status": "skipped_normal_delta0_limit",
            "valid_for_casimir_input": False,
            "casimir_gating_status": "diagnostic_normal_limit_response_not_promoted_by_finite_q_engine",
        },
    )


def _require_peierls_finite_q_support(spec) -> None:
    required = (
        "hopping_terms",
        "peierls_hamiltonian_vector_vertex",
        "peierls_hamiltonian_contact_vertex",
    )
    if not all(hasattr(spec, name) for name in required):
        raise ValueError("spec must support Peierls finite-q vertices when current_vertex='peierls'")


def _bdg_eigensystem_from_model_pairing(spec, kx: float, ky: float, pairing: np.ndarray):
    return diagonalize_hermitian(bdg_hamiltonian_from_model_pairing(spec, kx, ky, pairing))


def _bdg_vector_vertex_from_spec(
    spec,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    current_vertex: str,
) -> np.ndarray:
    if current_vertex == "peierls":
        _require_peierls_finite_q_support(spec)
        particle = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, direction)
        hole_normal = spec.peierls_hamiltonian_vector_vertex(-kx, -ky, -qx, -qy, direction)
        return bdg_finite_q_vertex_from_normal_blocks(particle, hole_normal)
    if current_vertex == "q0_velocity":
        return charge_current_vertex_from_model(spec, kx, ky, direction)
    raise ValueError("current_vertex must be 'peierls' or 'q0_velocity'")


def _bdg_contact_vertex_from_spec(
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
        _require_peierls_finite_q_support(spec)
        particle = spec.peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
        hole_normal = spec.peierls_hamiltonian_contact_vertex(-kx, -ky, -qx, -qy, direction_i, direction_j)
        return bdg_finite_q_vertex_from_normal_blocks(particle, hole_normal)
    if current_vertex == "q0_velocity":
        return diamagnetic_vertex_from_model(spec, kx, ky, direction_i, direction_j)
    raise ValueError("current_vertex must be 'peierls' or 'q0_velocity'")


def pairing_form_factor_matrix(
    pairing: PairingName,
    kx: float,
    ky: float,
    pairing_params: PairingAmplitudes,
) -> np.ndarray:
    """Return Phi(k)=Delta(k)/Delta0 for legacy diagnostics."""

    delta0 = float(pairing_params.delta0_eV)
    if delta0 == 0.0:
        raise ValueError("pairing form factor is undefined for delta0=0")
    ansatz = build_pairing_ansatz(pairing, phase_vertex="bond_endpoint_gauge")
    return ansatz.mean_pairing(kx, ky, pairing_params) / delta0


def collective_form_factor(
    pairing: PairingName,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    pairing_params: PairingAmplitudes,
    phase_vertex: PhaseVertexName,
) -> np.ndarray:
    ansatz = build_pairing_ansatz(pairing, phase_vertex=phase_vertex)
    return ansatz.collective_form_factor(kx, ky, qx, qy, pairing_params)


def collective_goldstone_counterterm(
    pairing: PairingName,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: KuboConfig,
    pairing_params: PairingAmplitudes,
    phase_vertex: PhaseVertexName,
) -> complex:
    """Return Cg=-K_22^bubble(q=0,omega=0) for eta2 normalization."""

    ansatz = build_pairing_ansatz(pairing, phase_vertex=phase_vertex)
    return complex(ansatz.hs_counterterm(config, k_points, k_weights, pairing_params)[1, 1])


def apply_phase_only_schur(
    bare_response: np.ndarray,
    phase_left: np.ndarray,
    phase_phase_total: complex,
    phase_right: np.ndarray,
    *,
    sign: Literal["minus", "plus"] = "minus",
) -> SchurResult:
    """Apply the existing one-channel phase Schur correction."""

    bare = np.asarray(bare_response, dtype=complex)
    left = np.asarray(phase_left, dtype=complex)
    right = np.asarray(phase_right, dtype=complex)
    kernel = complex(phase_phase_total)
    if sign not in {"minus", "plus"}:
        raise ValueError("sign must be 'minus' or 'plus'")
    if abs(kernel) <= 0.0:
        return SchurResult(
            corrected_response=bare.copy(),
            condition_number=None,
            inverse_method="not_used",
            status="skipped_zero_phase_kernel",
            warning="phase_phase_total is zero; phase-only Schur correction was skipped",
        )
    schur_term = np.outer(left, right) / kernel
    corrected = bare - schur_term if sign == "minus" else bare + schur_term
    return SchurResult(
        corrected_response=corrected,
        condition_number=None,
        inverse_method="not_used",
        status=f"{sign}_phase_schur_applied",
    )


def apply_amplitude_phase_schur(
    bare_response: np.ndarray,
    em_collective_left: np.ndarray,
    collective_total: np.ndarray,
    collective_em_right: np.ndarray,
    *,
    condition_threshold: float = 1e12,
) -> SchurResult:
    """Apply K_AA - K_Aeta inv(K_etaeta) K_etaA with diagnostic pinv fallback."""

    bare = np.asarray(bare_response, dtype=complex)
    left = np.asarray(em_collective_left, dtype=complex)
    kernel = np.asarray(collective_total, dtype=complex)
    right = np.asarray(collective_em_right, dtype=complex)
    condition = float(np.linalg.cond(kernel))
    if not np.isfinite(condition) or condition > condition_threshold:
        kernel_inv = np.linalg.pinv(kernel)
        inverse_method: Literal["inv", "pinv_diagnostic", "not_used"] = "pinv_diagnostic"
        status = "applied_with_pinv_diagnostic"
        warning = f"collective_total condition number {condition:.3e} exceeds threshold {condition_threshold:.3e}"
    else:
        kernel_inv = np.linalg.inv(kernel)
        inverse_method = "inv"
        status = "applied"
        warning = None
    return SchurResult(
        corrected_response=bare - left @ kernel_inv @ right,
        condition_number=condition,
        inverse_method=inverse_method,
        status=status,
        warning=warning,
    )


def finite_q_bdg_response_from_model_ansatz(
    spec,
    ansatz: PairingAnsatz,
    omega_eV: float,
    q_model: np.ndarray,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: KuboConfig,
    pairing_params: PairingAmplitudes | None = None,
    options: FiniteQEngineOptions | None = None,
) -> BdGFiniteQResponseComponents:
    """Return finite-q BdG response components from a model spec and pairing ansatz.

    The engine is intentionally generic: model-specific pairing and collective
    structure enters only through ``ansatz``.
    """

    if abs(float(config.omega_eV) - float(omega_eV)) > max(1e-14, 1e-10 * max(1.0, abs(float(omega_eV)))):
        raise ValueError("omega_eV must match config.omega_eV")
    opts = options or FiniteQEngineOptions()
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
        _require_peierls_finite_q_support(spec)
    amp = pairing_params or PairingAmplitudes()
    delta0 = float(amp.delta0_eV)
    collective_mode = opts.collective_mode
    collective_mode_disabled_reason = None
    if delta0 == 0.0 and collective_mode == "amplitude_phase":
        collective_mode = "none"
        collective_mode_disabled_reason = "delta0=0 normal limit"

    qx, qy = float(q[0]), float(q[1])
    shared_eigenbasis_q0_tolerance = 1e-14
    shared_eigenbasis_q0 = bool(np.linalg.norm(q) <= shared_eigenbasis_q0_tolerance)
    orbital_dim = np.asarray(spec.normal_hamiltonian(float(points[0, 0]), float(points[0, 1]))).shape[0]
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
            bands = _bdg_eigensystem_from_model_pairing(spec, kx, ky, delta_mid)
            occupations = fermi_function(bands.energies, config.fermi_level_eV, config.temperature_eV)
            energies_minus = energies_plus = bands.energies
            states_minus = states_plus = bands.states
            occupations_minus = occupations_plus = occupations
        else:
            kx_minus, ky_minus = kx - 0.5 * qx, ky - 0.5 * qy
            kx_plus, ky_plus = kx + 0.5 * qx, ky + 0.5 * qy
            delta_minus = ansatz.mean_pairing(kx_minus, ky_minus, amp)
            delta_plus = ansatz.mean_pairing(kx_plus, ky_plus, amp)
            bands_minus = _bdg_eigensystem_from_model_pairing(spec, kx_minus, ky_minus, delta_minus)
            bands_plus = _bdg_eigensystem_from_model_pairing(spec, kx_plus, ky_plus, delta_plus)
            energies_minus, states_minus = bands_minus.energies, bands_minus.states
            energies_plus, states_plus = bands_plus.energies, bands_plus.states
            occupations_minus = fermi_function(energies_minus, config.fermi_level_eV, config.temperature_eV)
            occupations_plus = fermi_function(energies_plus, config.fermi_level_eV, config.temperature_eV)
            delta_mid = ansatz.mean_pairing(kx, ky, amp)

        vx = _bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "x", opts.current_vertex)
        vy = _bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "y", opts.current_vertex)
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
                vertex = _bdg_contact_vertex_from_spec(
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
    # Legacy diagnostic metadata only; the engine never uses Ward residuals to
    # modify or repair the response. Use ward_validation.py for formal reports.
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
                "while quasiparticles are routed through k±q/2."
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


def finite_q_bdg_response_from_ansatz(
    ansatz: PairingAnsatz,
    omega_eV: float,
    q_model: np.ndarray,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: KuboConfig,
    pairing_params: PairingAmplitudes | None = None,
    options: FiniteQEngineOptions | None = None,
) -> BdGFiniteQResponseComponents:
    """Return finite-q BdG response components for the LNO327 four-orbital ansatz."""

    return finite_q_bdg_response_from_model_ansatz(
        LNO327FourOrbitalSpec(pairing_amplitudes=pairing_params),
        ansatz,
        omega_eV,
        q_model,
        k_points,
        k_weights,
        config,
        pairing_params,
        options,
    )


def bdg_finite_q_response_imag_axis(
    pairing: PairingName,
    omega_eV: float,
    q_model: np.ndarray,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: KuboConfig,
    pairing_params: PairingAmplitudes | None = None,
    *,
    include_phase_correction: bool = True,
    use_normal_backend_in_delta0_limit: bool = False,
    current_vertex: Literal["peierls", "q0_velocity"] = "peierls",
    phase_vertex: PhaseVertexName = "symmetric_kpm",
    include_phase_phase_direct: bool = True,
    phase_phase_direct_convention: PhaseDirectConvention = "plus",
    collective_mode: CollectiveMode = "amplitude_phase",
    collective_counterterm: CollectiveCounterterm = "goldstone_gap_equation",
) -> BdGFiniteQResponseComponents:
    """Return finite-q BdG response components through the generic engine.

    ``phase_vertex="symmetric_kpm"`` remains the legacy default for this public
    convenience entry point. New ansatz-based workflows should choose the phase
    vertex explicitly; ``PairingAnsatz`` itself defaults to
    ``bond_endpoint_gauge``.
    """

    if pairing not in {"onsite_s", "spm", "dwave"}:
        raise ValueError("pairing must be 'onsite_s', 'spm', or 'dwave'")
    q, points, weights = validate_finite_q_inputs(q_model, k_points, k_weights, config)
    if _is_normal_limit(pairing_params) and use_normal_backend_in_delta0_limit:
        return _normal_limit_components(q, points, weights, config, include_phase_correction=include_phase_correction)
    if phase_vertex not in {"midpoint", "symmetric_kpm", "bond_endpoint_gauge"}:
        raise ValueError("phase_vertex must be 'midpoint', 'symmetric_kpm', or 'bond_endpoint_gauge'")

    ansatz = build_pairing_ansatz(pairing, phase_vertex=phase_vertex)
    return finite_q_bdg_response_from_ansatz(
        ansatz,
        omega_eV,
        q,
        points,
        weights,
        config,
        pairing_params,
        FiniteQEngineOptions(
            include_phase_correction=include_phase_correction,
            current_vertex=current_vertex,
            include_phase_phase_direct=include_phase_phase_direct,
            phase_phase_direct_convention=phase_phase_direct_convention,
            collective_mode=collective_mode,
            collective_counterterm=collective_counterterm,
        ),
    )
