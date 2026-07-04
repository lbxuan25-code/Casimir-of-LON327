"""Public LNO327 finite-q BdG response workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from lno327.collective.schur import (
    BdGPhaseCorrectionError,
    SchurResult,
    apply_amplitude_phase_schur,
    apply_phase_only_schur,
)
from lno327.response.config import KuboConfig
from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.finite_q_bdg import finite_q_bdg_response_from_model_ansatz
from lno327.response.validation import validate_finite_q_inputs
from .models.lno327_four_orbital.collective import PairingAnsatz, build_pairing_ansatz
from .models.lno327_four_orbital.parameters import PairingAmplitudes, PhaseVertexName
from .models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.normal_density_current import normal_physical_density_current_response_components_imag_axis

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

    amp = pairing_params or PairingAmplitudes()
    return finite_q_bdg_response_from_model_ansatz(
        LNO327FourOrbitalSpec(pairing_amplitudes=amp),
        ansatz,
        omega_eV,
        q_model,
        k_points,
        k_weights,
        config,
        amp,
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
