"""Backward-compatible facade for finite-q BdG density/current response."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .conductivity import KuboConfig
from .finite_q_primitives import (
    BdGFiniteQResponseComponents,
    add_bubble as _add_bubble,
    bdg_finite_q_contact_vertex,
    bdg_finite_q_vector_vertex,
    density_vertex as _density_vertex,
    kubo_factor as _kubo_factor,
    phase_phase_direct_vertex as _phase_phase_direct_vertex,
    phase_vertex as _phase_vertex,
    thermal_expectation_bdg as _thermal_expectation_bdg,
    validate_finite_q_inputs as _validate_inputs,
    ward_metadata as _ward_metadata,
)
from .pairing import PairingAmplitudes, pairing_matrix
from .pairing_ansatz import (
    PhaseVertexName,
    _amplitude_vertex,
    _eta2_phase_vertex,
    build_pairing_ansatz,
)
from .ward_response import normal_physical_density_current_response_components_imag_axis


PairingName = Literal["onsite_s", "spm", "dwave"]
PhaseDirectConvention = Literal["plus", "minus"]
CollectiveMode = Literal["none", "phase_only", "amplitude_phase"]
CollectiveCounterterm = Literal["none", "goldstone_gap_equation"]


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
            "casimir_gating_status": "diagnostic_normal_limit_response_not_promoted_by_finite_q_facade",
        },
    )


def _pairing_matrix_for_kind(
    kind: PairingName,
    kx: float,
    ky: float,
    amp: PairingAmplitudes | None,
) -> np.ndarray:
    return build_pairing_ansatz(kind).mean_pairing(kx, ky, amp or PairingAmplitudes())


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
    if pairing == "onsite_s":
        return np.eye(4, dtype=complex)
    return pairing_matrix(pairing, kx, ky, pairing_params) / delta0


def _phase_pairing_matrix(
    pairing: PairingName,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    pairing_params: PairingAmplitudes | None,
    phase_vertex: PhaseVertexName,
) -> np.ndarray:
    ansatz = build_pairing_ansatz(pairing, phase_vertex=phase_vertex)
    return ansatz.phase_pairing_matrix(kx, ky, qx, qy, pairing_params or PairingAmplitudes())


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

    ``phase_vertex="symmetric_kpm"`` is the legacy compatibility default for
    this wrapper. New ansatz-based workflows should choose the phase vertex
    explicitly; ``PairingAnsatz`` itself defaults to ``bond_endpoint_gauge``.
    """

    if pairing not in {"onsite_s", "spm", "dwave"}:
        raise ValueError("pairing must be 'onsite_s', 'spm', or 'dwave'")
    if abs(float(config.omega_eV) - float(omega_eV)) > max(1e-14, 1e-10 * max(1.0, abs(float(omega_eV)))):
        raise ValueError("omega_eV must match config.omega_eV")
    q, points, weights = _validate_inputs(q_model, k_points, k_weights, config)
    if _is_normal_limit(pairing_params) and use_normal_backend_in_delta0_limit:
        return _normal_limit_components(q, points, weights, config, include_phase_correction=include_phase_correction)
    if current_vertex not in {"peierls", "q0_velocity"}:
        raise ValueError("current_vertex must be 'peierls' or 'q0_velocity'")
    if phase_vertex not in {"midpoint", "symmetric_kpm", "bond_endpoint_gauge"}:
        raise ValueError("phase_vertex must be 'midpoint', 'symmetric_kpm', or 'bond_endpoint_gauge'")
    if phase_phase_direct_convention not in {"plus", "minus"}:
        raise ValueError("phase_phase_direct_convention must be 'plus' or 'minus'")
    if collective_mode not in {"none", "phase_only", "amplitude_phase"}:
        raise ValueError("collective_mode must be 'none', 'phase_only', or 'amplitude_phase'")
    if collective_counterterm not in {"none", "goldstone_gap_equation"}:
        raise ValueError("collective_counterterm must be 'none' or 'goldstone_gap_equation'")

    from .finite_q_engine import FiniteQEngineOptions, finite_q_bdg_response_from_ansatz

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
