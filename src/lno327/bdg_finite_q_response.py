"""Finite-q BdG density/current response on the imaginary axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import warnings

import numpy as np

from .bdg_response import bdg_current_vertex, bdg_diamagnetic_vertex
from .conductivity import KuboConfig, fermi_function
from .pairing import PairingAmplitudes, bdg_hamiltonian, pairing_matrix
from .tb_fourier import peierls_hamiltonian_contact_vertex, peierls_hamiltonian_vector_vertex
from .ward_response import physical_ward_residuals
from .ward_response import normal_physical_density_current_response_components_imag_axis


PairingName = Literal["spm", "dwave"]


@dataclass(frozen=True)
class BdGFiniteQResponseComponents:
    bare_bubble: np.ndarray
    direct: np.ndarray
    bare_total: np.ndarray
    phase_coupling_left: np.ndarray
    phase_coupling_right: np.ndarray
    phase_phase: complex
    gauge_restored: np.ndarray
    metadata: dict[str, Any]


class BdGPhaseCorrectionError(RuntimeError):
    """Raised when the global phase channel is singular."""


def _validate_inputs(
    q_model: np.ndarray,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: KuboConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model must have shape (2,)")
    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("k_points must have shape (n, 2) and must not be empty")
    weights = np.asarray(k_weights, dtype=float)
    if weights.shape != (points.shape[0],):
        raise ValueError("k_weights must have shape (n,)")
    if config.omega_eV < 0.0:
        raise ValueError("omega_eV must be non-negative")
    return q, points, weights


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
        phase_phase=phase_phase,
        gauge_restored=total.copy(),
        metadata={
            "normal_limit_delegated_to": "normal_physical_density_current_response_components_imag_axis",
            "nambu_prefactor": 0.5,
            "finite_q_current_vertex_status": "normal_state_finite_q_peierls_backend",
            "collective_channels": ["global_phase_only"],
            "phase_correction_requested": bool(include_phase_correction),
            "phase_correction_applied": False,
            "phase_correction_status": "skipped_normal_delta0_limit",
        },
    )


def bdg_finite_q_vector_vertex(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
) -> np.ndarray:
    """Return BdG finite-q source vertex built from the normal Peierls vertex."""

    particle_block = peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, direction)
    hole_block = -peierls_hamiltonian_vector_vertex(-kx, -ky, -qx, -qy, direction).T
    zero = np.zeros((4, 4), dtype=complex)
    return np.block([[particle_block, zero], [zero, hole_block]]).astype(complex)


def bdg_finite_q_contact_vertex(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction_i: str,
    direction_j: str,
) -> np.ndarray:
    """Return BdG finite-q contact vertex from the normal Peierls contact."""

    particle_block = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
    hole_block = -peierls_hamiltonian_contact_vertex(-kx, -ky, -qx, -qy, direction_i, direction_j).T
    zero = np.zeros((4, 4), dtype=complex)
    return np.block([[particle_block, zero], [zero, hole_block]]).astype(complex)


def _phase_vertex(pairing: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(pairing)
    return np.block([[zero, 1j * pairing], [-1j * pairing.conjugate().T, zero]]).astype(complex)


def _density_vertex() -> np.ndarray:
    eye = np.eye(4, dtype=complex)
    return np.block([[eye, np.zeros((4, 4), dtype=complex)], [np.zeros((4, 4), dtype=complex), -eye]])


def _vertex_band(states_minus: np.ndarray, vertex: np.ndarray, states_plus: np.ndarray) -> np.ndarray:
    return states_minus.conjugate().T @ vertex @ states_plus


def _add_bubble(
    accumulator: np.ndarray,
    left_vertices: tuple[np.ndarray, ...],
    right_vertices: tuple[np.ndarray, ...],
    energies_minus: np.ndarray,
    states_minus: np.ndarray,
    occupations_minus: np.ndarray,
    energies_plus: np.ndarray,
    states_plus: np.ndarray,
    occupations_plus: np.ndarray,
    omega_eV: float,
    weight: float,
) -> None:
    left_band = tuple(_vertex_band(states_minus, vertex, states_plus) for vertex in left_vertices)
    right_band = tuple(_vertex_band(states_minus, vertex, states_plus) for vertex in right_vertices)
    for m, energy_minus in enumerate(energies_minus):
        for n, energy_plus in enumerate(energies_plus):
            occupation_diff = float(occupations_minus[m] - occupations_plus[n])
            if occupation_diff == 0.0:
                continue
            factor = 0.5 * weight * occupation_diff / (1j * omega_eV + float(energy_minus - energy_plus))
            for mu, left in enumerate(left_band):
                for nu, right in enumerate(right_band):
                    accumulator[mu, nu] += factor * left[m, n] * np.conjugate(right[m, n])


def _thermal_expectation_bdg(
    kx: float,
    ky: float,
    pairing: np.ndarray,
    vertex: np.ndarray,
    config: KuboConfig,
) -> complex:
    energies, states = np.linalg.eigh(bdg_hamiltonian(kx, ky, pairing))
    occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
    vertex_band = states.conjugate().T @ vertex @ states
    return complex(0.5 * np.sum(occupations * np.diag(vertex_band)))


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
) -> BdGFiniteQResponseComponents:
    """Return finite-q BdG response components using Stage 4/5 conventions."""

    if pairing not in {"spm", "dwave"}:
        raise ValueError("pairing must be 'spm' or 'dwave'")
    if abs(float(config.omega_eV) - float(omega_eV)) > max(1e-14, 1e-10 * max(1.0, abs(float(omega_eV)))):
        raise ValueError("omega_eV must match config.omega_eV")
    q, points, weights = _validate_inputs(q_model, k_points, k_weights, config)
    if _is_normal_limit(pairing_params) and use_normal_backend_in_delta0_limit:
        return _normal_limit_components(q, points, weights, config, include_phase_correction=include_phase_correction)
    if current_vertex not in {"peierls", "q0_velocity"}:
        raise ValueError("current_vertex must be 'peierls' or 'q0_velocity'")

    qx, qy = float(q[0]), float(q[1])
    rho = _density_vertex()
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)
    phase_left = np.zeros(3, dtype=complex)
    phase_right = np.zeros(3, dtype=complex)
    phase_phase_matrix = np.zeros((1, 1), dtype=complex)
    directions = ("x", "y")

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        kx_minus, ky_minus = kx - 0.5 * qx, ky - 0.5 * qy
        kx_plus, ky_plus = kx + 0.5 * qx, ky + 0.5 * qy
        delta_minus = pairing_matrix(pairing, kx_minus, ky_minus, pairing_params)
        delta_plus = pairing_matrix(pairing, kx_plus, ky_plus, pairing_params)
        energies_minus, states_minus = np.linalg.eigh(bdg_hamiltonian(kx_minus, ky_minus, delta_minus))
        energies_plus, states_plus = np.linalg.eigh(bdg_hamiltonian(kx_plus, ky_plus, delta_plus))
        occupations_minus = fermi_function(energies_minus, config.fermi_level_eV, config.temperature_eV)
        occupations_plus = fermi_function(energies_plus, config.fermi_level_eV, config.temperature_eV)

        if current_vertex == "peierls":
            vx = bdg_finite_q_vector_vertex(kx, ky, qx, qy, "x")
            vy = bdg_finite_q_vector_vertex(kx, ky, qx, qy, "y")
        else:
            vx = bdg_current_vertex(kx, ky, "x")
            vy = bdg_current_vertex(kx, ky, "y")
        observable_vertices = (rho, -vx, -vy)
        source_vertices = (rho, vx, vy)
        _add_bubble(
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
        )

        delta_mid = pairing_matrix(pairing, kx, ky, pairing_params)
        theta = _phase_vertex(delta_mid)
        tmp_left = np.zeros((3, 1), dtype=complex)
        _add_bubble(
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
        )
        phase_left += tmp_left[:, 0]
        tmp_right = np.zeros((1, 3), dtype=complex)
        _add_bubble(
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
        )
        phase_right += tmp_right[0, :]
        _add_bubble(
            phase_phase_matrix,
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
        )

        for i, direction_i in enumerate(directions):
            for j, direction_j in enumerate(directions):
                if current_vertex == "peierls":
                    vertex = bdg_finite_q_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
                else:
                    vertex = bdg_diamagnetic_vertex(kx, ky, direction_i, direction_j)
                direct[1 + i, 1 + j] += -float(weight) * _thermal_expectation_bdg(kx, ky, delta_mid, vertex, config)

    bare_total = bubble + direct
    phase_phase = complex(phase_phase_matrix[0, 0])
    minus_schur = bare_total.copy()
    plus_schur = bare_total.copy()
    if abs(phase_phase) > 0.0:
        schur_term = np.outer(phase_left, phase_right) / phase_phase
        minus_schur = bare_total - schur_term
        plus_schur = bare_total + schur_term
    gauge_restored = bare_total.copy()
    phase_status = "disabled"
    warning_message = None
    threshold = max(100.0 * float(config.eta_eV), 1e-14)
    if include_phase_correction:
        if abs(phase_phase) <= threshold:
            phase_status = "singular_phase_phase"
            warning_message = (
                f"Global phase correction skipped because |K_theta_theta|={abs(phase_phase):.3e} "
                f"is below threshold {threshold:.3e}."
            )
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
        else:
            gauge_restored = minus_schur
            phase_status = "applied"
    ward_bare = _ward_metadata(bare_total, config.omega_eV, q)
    ward_minus = _ward_metadata(minus_schur, config.omega_eV, q)
    ward_plus = _ward_metadata(plus_schur, config.omega_eV, q)

    return BdGFiniteQResponseComponents(
        bare_bubble=bubble,
        direct=direct,
        bare_total=bare_total,
        phase_coupling_left=phase_left,
        phase_coupling_right=phase_right,
        phase_phase=phase_phase,
        gauge_restored=gauge_restored,
        metadata={
            "nambu_basis": "(c_k, c^dagger_-k)",
            "nambu_prefactor": 0.5,
            "finite_q_routing": "k_minus=k-q/2,k_plus=k+q/2",
            "current_observable_source_convention": "J=(rho,-Vx,-Vy), P=(rho,Vx,Vy)",
            "direct_contact_convention": "D_ij=-<M_ij> with BdG Nambu 1/2",
            "effective_action_convention": "S2=1/2(a,theta)[[K_munu,K_mutheta],[K_thetanu,K_thetatheta]](a,theta)^T",
            "phase_correction_formula": "Pi_GI = Pi_bare - K_mu_theta K_theta_nu / K_theta_theta",
            "phase_correction_sign_checked": True,
            "finite_q_current_vertex_status": (
                "normal_state_exact_finite_q_peierls_vertex"
                if current_vertex == "peierls"
                else "q0_velocity_vertex_approximation_not_gauge_closed"
            ),
            "collective_channels": ["global_phase_only"],
            "phase_vertex_status": "midpoint_pair_relative_momentum_global_center_of_mass_phase_not_full_gauge_closure_proof",
            "phase_vertex_convention": (
                "Gamma_theta(k) = dH_BdG/dtheta at pair center-of-mass phase; "
                "for k-dependent Delta this implementation evaluates Delta at midpoint k, "
                "while quasiparticles are routed through k±q/2."
            ),
            "phase_correction_requested": bool(include_phase_correction),
            "phase_correction_applied": phase_status == "applied",
            "phase_correction_status": phase_status,
            "phase_phase_abs": float(abs(phase_phase)),
            "phase_phase_singular_threshold": float(threshold),
            "ward_residual_bare": ward_bare,
            "ward_residual_minus_schur": ward_minus,
            "ward_residual_plus_schur": ward_plus,
            "selected_gauge_restored": "minus_schur" if phase_status == "applied" else "bare_total",
            "normal_backend_reference_used": False,
            "warning": warning_message,
        },
    )


def _ward_metadata(response: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, float]:
    left, right = physical_ward_residuals(response, omega_eV, q)
    return {
        "left_norm": float(np.linalg.norm(left)),
        "right_norm": float(np.linalg.norm(right)),
        "max_norm": float(max(np.linalg.norm(left), np.linalg.norm(right))),
    }
