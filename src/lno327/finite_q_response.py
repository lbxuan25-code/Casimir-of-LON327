"""Finite-q response diagnostics for angular anisotropy prototypes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bdg_response import KuboConfig, bdg_current_vertex, fermi_function
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


def _finite_q_bubble(
    *,
    kind: ResponseKind,
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
    q_vector: np.ndarray,
    nk: int,
    delta0_eV: float,
) -> np.ndarray:
    mesh, weights = _uniform_bz_mesh(nk)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    omega = config.omega_eV + config.eta_eV
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
                _normal_current_vertex(float(vertex_k[0]), float(vertex_k[1]), direction)
                for direction in directions
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
                bdg_current_vertex(float(vertex_k[0]), float(vertex_k[1]), direction)
                for direction in directions
            ]
            left_states = states_minus
            right_states = states_plus

        vertex_band = [left_states.conjugate().T @ vertex @ right_states for vertex in vertices]
        reverse_vertex_band = [right_states.conjugate().T @ vertex @ left_states for vertex in vertices]

        for m, energy_m in enumerate(energies_minus):
            for n, energy_n in enumerate(energies_plus):
                occupation_diff = occ_minus[m] - occ_plus[n]
                if np.isclose(occupation_diff, 0.0):
                    continue
                energy_diff = energy_m - energy_n
                if abs(energy_diff) < eta_eV:
                    continue
                response_factor = -occupation_diff * energy_diff / (energy_diff**2 + omega**2)
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
