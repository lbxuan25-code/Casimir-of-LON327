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


PairingName = Literal["onsite_s", "spm", "dwave"]
PhaseVertexName = Literal["midpoint", "symmetric_kpm"]
PhaseDirectConvention = Literal["plus", "minus"]
CollectiveMode = Literal["none", "phase_only", "amplitude_phase"]
CollectiveCounterterm = Literal["none", "goldstone_gap_equation"]


@dataclass(frozen=True)
class BdGFiniteQResponseComponents:
    bare_bubble: np.ndarray
    direct: np.ndarray
    bare_total: np.ndarray

    phase_coupling_left: np.ndarray
    phase_coupling_right: np.ndarray
    phase_phase_bubble: complex
    phase_phase_direct: complex
    phase_phase_total: complex

    minus_schur: np.ndarray
    plus_schur: np.ndarray
    collective_bubble: np.ndarray
    collective_counterterm: np.ndarray
    collective_total: np.ndarray
    em_collective_left: np.ndarray
    collective_em_right: np.ndarray
    amplitude_phase_schur: np.ndarray
    gauge_restored: np.ndarray

    metadata: dict[str, Any]

    @property
    def phase_phase(self) -> complex:
        """Backward-compatible alias for the total phase kernel."""

        return self.phase_phase_total


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
        },
    )


def _pairing_matrix_for_kind(
    kind: PairingName,
    kx: float,
    ky: float,
    amp: PairingAmplitudes | None,
) -> np.ndarray:
    if kind == "onsite_s":
        amplitude = amp or PairingAmplitudes()
        return amplitude.delta0_eV * np.eye(4, dtype=complex)
    return pairing_matrix(kind, kx, ky, amp)


def pairing_form_factor_matrix(
    pairing: PairingName,
    kx: float,
    ky: float,
    pairing_params: PairingAmplitudes,
) -> np.ndarray:
    """Return Phi(k)=Delta(k)/Delta0 for collective eta fields."""

    delta0 = float(pairing_params.delta0_eV)
    if delta0 == 0.0:
        raise ValueError("pairing form factor is undefined for delta0=0")
    return _pairing_matrix_for_kind(pairing, kx, ky, pairing_params) / delta0


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


def _phase_phase_direct_vertex(delta_theta: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(delta_theta)
    return np.block([[zero, -delta_theta], [-delta_theta.conjugate().T, zero]]).astype(complex)


def _phase_pairing_matrix(
    pairing: PairingName,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    pairing_params: PairingAmplitudes | None,
    phase_vertex: PhaseVertexName,
) -> np.ndarray:
    if phase_vertex == "midpoint":
        return _pairing_matrix_for_kind(pairing, kx, ky, pairing_params)
    if phase_vertex == "symmetric_kpm":
        delta_minus = _pairing_matrix_for_kind(pairing, kx - 0.5 * qx, ky - 0.5 * qy, pairing_params)
        delta_plus = _pairing_matrix_for_kind(pairing, kx + 0.5 * qx, ky + 0.5 * qy, pairing_params)
        return 0.5 * (delta_minus + delta_plus)
    raise ValueError("phase_vertex must be 'midpoint' or 'symmetric_kpm'")


def collective_form_factor(
    pairing: PairingName,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    pairing_params: PairingAmplitudes,
    phase_vertex: PhaseVertexName,
) -> np.ndarray:
    if phase_vertex == "midpoint":
        return pairing_form_factor_matrix(pairing, kx, ky, pairing_params)
    if phase_vertex == "symmetric_kpm":
        phi_minus = pairing_form_factor_matrix(pairing, kx - 0.5 * qx, ky - 0.5 * qy, pairing_params)
        phi_plus = pairing_form_factor_matrix(pairing, kx + 0.5 * qx, ky + 0.5 * qy, pairing_params)
        return 0.5 * (phi_minus + phi_plus)
    raise ValueError("phase_vertex must be 'midpoint' or 'symmetric_kpm'")


def _amplitude_vertex(phi: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi)
    return np.block([[zero, phi], [phi.conjugate().T, zero]]).astype(complex)


def _eta2_phase_vertex(phi: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi)
    return np.block([[zero, 1j * phi], [-1j * phi.conjugate().T, zero]]).astype(complex)


def _fermi_derivative(energy_eV: float, fermi_level_eV: float, temperature_eV: float, eta_eV: float) -> float:
    shifted = float(energy_eV) - float(fermi_level_eV)
    if temperature_eV <= 0.0:
        width = max(float(eta_eV), 1e-12)
        return -float(width / (np.pi * (shifted**2 + width**2)))
    x = np.clip(shifted / (2.0 * temperature_eV), -350.0, 350.0)
    return -float(1.0 / (4.0 * temperature_eV * np.cosh(x) ** 2))


def _kubo_factor(
    em: float,
    en: float,
    fm: float,
    fn: float,
    omega_eV: float,
    *,
    static_limit: bool = False,
    fermi_level_eV: float = 0.0,
    temperature_eV: float | None = None,
    eta_eV: float = 1e-8,
) -> complex:
    delta_e = float(em) - float(en)
    if static_limit and abs(float(omega_eV)) <= eta_eV and abs(delta_e) < eta_eV:
        if temperature_eV is None:
            raise ValueError("temperature_eV is required for static degenerate Kubo factor")
        return _fermi_derivative(float(em), fermi_level_eV, temperature_eV, eta_eV)
    return (float(fm) - float(fn)) / (1j * float(omega_eV) + delta_e)


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
    config: KuboConfig | None = None,
    static_limit: bool = False,
) -> None:
    left_band = tuple(_vertex_band(states_minus, vertex, states_plus) for vertex in left_vertices)
    right_band = tuple(_vertex_band(states_minus, vertex, states_plus) for vertex in right_vertices)
    for m, energy_minus in enumerate(energies_minus):
        for n, energy_plus in enumerate(energies_plus):
            occupation_diff = float(occupations_minus[m] - occupations_plus[n])
            if occupation_diff == 0.0 and not static_limit:
                continue
            if config is None:
                raw_factor = occupation_diff / (1j * omega_eV + float(energy_minus - energy_plus))
            else:
                raw_factor = _kubo_factor(
                    float(energy_minus),
                    float(energy_plus),
                    float(occupations_minus[m]),
                    float(occupations_plus[n]),
                    omega_eV,
                    static_limit=static_limit,
                    fermi_level_eV=config.fermi_level_eV,
                    temperature_eV=config.temperature_eV,
                    eta_eV=config.eta_eV,
                )
            factor = 0.5 * weight * raw_factor
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


def collective_goldstone_counterterm(
    pairing: PairingName,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: KuboConfig,
    pairing_params: PairingAmplitudes,
    phase_vertex: PhaseVertexName,
) -> complex:
    """Return Cg=-K_22^bubble(q=0,omega=0) for eta2 Goldstone normalization."""

    points = np.asarray(k_points, dtype=float)
    weights = np.asarray(k_weights, dtype=float)
    bubble = np.zeros((2, 2), dtype=complex)
    qx = qy = 0.0
    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        delta = _pairing_matrix_for_kind(pairing, kx, ky, pairing_params)
        energies, states = np.linalg.eigh(bdg_hamiltonian(kx, ky, delta))
        occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
        phi_eta = collective_form_factor(pairing, kx, ky, qx, qy, pairing_params, phase_vertex)
        vertices = (_amplitude_vertex(phi_eta), _eta2_phase_vertex(phi_eta))
        _add_bubble(
            bubble,
            vertices,
            vertices,
            energies,
            states,
            occupations,
            energies,
            states,
            occupations,
            0.0,
            float(weight),
            config=config,
            static_limit=True,
        )
    return -complex(bubble[1, 1])


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
    """Return finite-q BdG response components using Stage 4/5 conventions."""

    if pairing not in {"onsite_s", "spm", "dwave"}:
        raise ValueError("pairing must be 'onsite_s', 'spm', or 'dwave'")
    if abs(float(config.omega_eV) - float(omega_eV)) > max(1e-14, 1e-10 * max(1.0, abs(float(omega_eV)))):
        raise ValueError("omega_eV must match config.omega_eV")
    q, points, weights = _validate_inputs(q_model, k_points, k_weights, config)
    if _is_normal_limit(pairing_params) and use_normal_backend_in_delta0_limit:
        return _normal_limit_components(q, points, weights, config, include_phase_correction=include_phase_correction)
    if current_vertex not in {"peierls", "q0_velocity"}:
        raise ValueError("current_vertex must be 'peierls' or 'q0_velocity'")
    if phase_vertex not in {"midpoint", "symmetric_kpm"}:
        raise ValueError("phase_vertex must be 'midpoint' or 'symmetric_kpm'")
    if phase_phase_direct_convention not in {"plus", "minus"}:
        raise ValueError("phase_phase_direct_convention must be 'plus' or 'minus'")
    if collective_mode not in {"none", "phase_only", "amplitude_phase"}:
        raise ValueError("collective_mode must be 'none', 'phase_only', or 'amplitude_phase'")
    if collective_counterterm not in {"none", "goldstone_gap_equation"}:
        raise ValueError("collective_counterterm must be 'none' or 'goldstone_gap_equation'")
    amp = pairing_params or PairingAmplitudes()
    delta0 = float(amp.delta0_eV)
    collective_mode_disabled_reason = None
    if delta0 == 0.0 and collective_mode == "amplitude_phase":
        collective_mode = "none"
        collective_mode_disabled_reason = "delta0=0 normal limit"

    qx, qy = float(q[0]), float(q[1])
    rho = _density_vertex()
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
        kx_minus, ky_minus = kx - 0.5 * qx, ky - 0.5 * qy
        kx_plus, ky_plus = kx + 0.5 * qx, ky + 0.5 * qy
        delta_minus = _pairing_matrix_for_kind(pairing, kx_minus, ky_minus, pairing_params)
        delta_plus = _pairing_matrix_for_kind(pairing, kx_plus, ky_plus, pairing_params)
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
        if collective_mode == "amplitude_phase":
            phi_eta = collective_form_factor(pairing, kx, ky, qx, qy, amp, phase_vertex)
            collective_vertices = (_amplitude_vertex(phi_eta), _eta2_phase_vertex(phi_eta))
            _add_bubble(
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
            )
            _add_bubble(
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
            )
            _add_bubble(
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
            )

        delta_mid = _pairing_matrix_for_kind(pairing, kx, ky, pairing_params)
        delta_theta = _phase_pairing_matrix(pairing, kx, ky, qx, qy, pairing_params, phase_vertex)
        theta = _phase_vertex(delta_theta)
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
        )
        theta_theta = _phase_phase_direct_vertex(delta_theta)
        direct_value = float(weight) * _thermal_expectation_bdg(kx, ky, delta_mid, theta_theta, config)
        phase_phase_direct_plus += direct_value
        phase_phase_direct_minus -= direct_value

        for i, direction_i in enumerate(directions):
            for j, direction_j in enumerate(directions):
                if current_vertex == "peierls":
                    vertex = bdg_finite_q_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
                else:
                    vertex = bdg_diamagnetic_vertex(kx, ky, direction_i, direction_j)
                direct[1 + i, 1 + j] += -float(weight) * _thermal_expectation_bdg(kx, ky, delta_mid, vertex, config)

    bare_total = bubble + direct
    phase_phase_bubble = complex(phase_phase_bubble_matrix[0, 0])
    selected_phase_phase_direct = (
        phase_phase_direct_plus if phase_phase_direct_convention == "plus" else phase_phase_direct_minus
    )
    phase_phase_direct = selected_phase_phase_direct if include_phase_phase_direct else 0.0 + 0.0j
    phase_phase_total = phase_phase_bubble + phase_phase_direct
    minus_schur = bare_total.copy()
    plus_schur = bare_total.copy()
    if abs(phase_phase_total) > 0.0:
        schur_term = np.outer(phase_left, phase_right) / phase_phase_total
        minus_schur = bare_total - schur_term
        plus_schur = bare_total + schur_term
    gauge_restored = bare_total.copy()
    phase_status = "disabled"
    warning_message = None
    threshold = max(100.0 * float(config.eta_eV), 1e-14)
    if include_phase_correction:
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
    ward_bare = _ward_metadata(bare_total, config.omega_eV, q)
    ward_minus = _ward_metadata(minus_schur, config.omega_eV, q)
    ward_plus = _ward_metadata(plus_schur, config.omega_eV, q)
    collective_counterterm_matrix = np.zeros((2, 2), dtype=complex)
    goldstone_counterterm_cg = 0.0 + 0.0j
    if collective_mode == "amplitude_phase" and collective_counterterm == "goldstone_gap_equation":
        goldstone_counterterm_cg = collective_goldstone_counterterm(
            pairing,
            points,
            weights,
            config,
            amp,
            phase_vertex,
        )
        collective_counterterm_matrix = goldstone_counterterm_cg * np.eye(2, dtype=complex)
    collective_total = collective_bubble + collective_counterterm_matrix
    amplitude_phase_schur = bare_total.copy()
    collective_condition: float | None = None
    collective_inverse_method = "not_used"
    if collective_mode == "amplitude_phase":
        collective_condition = float(np.linalg.cond(collective_total))
        if not np.isfinite(collective_condition) or collective_condition > 1e12:
            collective_inv = np.linalg.pinv(collective_total)
            collective_inverse_method = "pinv_diagnostic"
        else:
            collective_inv = np.linalg.inv(collective_total)
            collective_inverse_method = "inv"
        amplitude_phase_schur = bare_total - em_collective_left @ collective_inv @ collective_em_right
        if include_phase_correction:
            gauge_restored = amplitude_phase_schur
            phase_status = "amplitude_phase_applied"
    ward_amp_phase = _ward_metadata(amplitude_phase_schur, config.omega_eV, q)
    selected_gauge = (
        "amplitude_phase_schur"
        if collective_mode == "amplitude_phase" and include_phase_correction
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
            "current_observable_source_convention": "J=(rho,-Vx,-Vy), P=(rho,Vx,Vy)",
            "direct_contact_convention": "D_ij=-<M_ij> with BdG Nambu 1/2",
            "effective_action_convention": "S2=1/2(a,theta)[[K_munu,K_mutheta],[K_thetanu,K_thetatheta]](a,theta)^T",
            "phase_correction_formula": "Pi_GI = Pi_bare - K_mu_theta K_theta_nu / K_theta_theta",
            "phase_correction_sign_checked": True,
            "validation_only_pairing": pairing == "onsite_s",
            "finite_q_current_vertex_status": (
                "normal_state_exact_finite_q_peierls_vertex"
                if current_vertex == "peierls"
                else "q0_velocity_vertex_approximation_not_gauge_closed"
            ),
            "collective_mode": collective_mode,
            "collective_channels": (
                ["amplitude_eta1", "phase_eta2"] if collective_mode == "amplitude_phase" else ["global_phase_only"]
            ),
            "collective_counterterm": collective_counterterm,
            "eta2_phase_relation": "eta2 = delta0 * theta",
            "collective_mode_disabled_reason": collective_mode_disabled_reason,
            "goldstone_counterterm_Cg": goldstone_counterterm_cg,
            "goldstone_condition_target": "K_eta2_eta2(q=0, omega=0) = 0",
            "collective_total_condition_number": collective_condition,
            "collective_inverse_method": collective_inverse_method,
            "amplitude_phase_schur_formula": "Pi_GI = K_munu - K_mu_a inv(K_ab) K_b_nu",
            "phase_vertex": phase_vertex,
            "phase_vertex_status": f"{phase_vertex}_pair_center_of_mass_phase_not_full_gauge_closure_proof",
            "phase_vertex_convention": (
                "Gamma_theta(k) = dH_BdG/dtheta at pair center-of-mass phase; "
                f"for k-dependent Delta this implementation uses phase_vertex={phase_vertex}, "
                "while quasiparticles are routed through k±q/2."
            ),
            "phase_phase_direct_included": bool(include_phase_phase_direct),
            "phase_phase_total_definition": "bubble + direct",
            "phase_phase_direct_convention": phase_phase_direct_convention,
            "phase_phase_direct_plus_convention": phase_phase_direct_plus,
            "phase_phase_direct_minus_convention": phase_phase_direct_minus,
            "phase_kernel_status": (
                "bubble_plus_direct"
                if include_phase_phase_direct
                else "bubble_only_not_expected_to_gauge_close"
            ),
            "phase_correction_requested": bool(include_phase_correction),
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
