"""Positive-Matsubara d-wave response on an exact q orbit and adaptive transverse rule.

The adaptive variable is only the transverse torus coordinate.  Every transverse
sample contains a complete commensurate q orbit, all requested Matsubara energies
share the same nodes, and only primitive electromagnetic/collective blocks are
integrated.  The nearest-neighbour bond metric and amplitude/phase Schur are
applied once after the full Brillouin-zone integral.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.finite_q_bdg import _finalize_components
from lno327.response.finite_q_optimized import (
    _vectorized_kubo_factors,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
)
from lno327.response.phase_hessian import apply_phase_hessian_policy_to_components
from lno327.response.ward_validation import PrimitiveWardRHS
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.commensurate_orbit_adaptive import (
    CommensurateOrbitAdaptiveResult,
    integrate_commensurate_orbit_adaptive_aggregate,
)

_HEADER_WIDTH = 9 + 4 + 1 + 1 + 3
_PER_FREQUENCY_WIDTH = 9 + 4 + 6 + 6


@dataclass(frozen=True)
class DWavePositiveOrbitAdaptiveResult:
    """Integrated primitive responses and Ward RHS for one q and xi batch."""

    components: tuple[BdGFiniteQResponseComponents, ...]
    rhs: tuple[PrimitiveWardRHS, ...]
    xi_eV_values: np.ndarray
    quadrature: CommensurateOrbitAdaptiveResult

    def __post_init__(self) -> None:
        xi = np.array(self.xi_eV_values, dtype=float, copy=True)
        xi.setflags(write=False)
        object.__setattr__(self, "xi_eV_values", xi)
        if len(self.components) != xi.size or len(self.rhs) != xi.size:
            raise ValueError("components, rhs, and xi_eV_values must have equal lengths")


def _pack_orbit_primitives(*, workspace, blocks: np.ndarray) -> np.ndarray:
    direct = np.asarray(workspace.direct_contact_contribution, dtype=complex).reshape(-1)
    counterterm = np.asarray(
        workspace.material.collective_counterterm_matrix, dtype=complex
    ).reshape(-1)
    header = np.concatenate(
        (
            direct,
            counterterm,
            np.asarray(
                [workspace.phase_phase_direct_plus, workspace.phase_phase_direct_minus],
                dtype=complex,
            ),
            np.asarray(workspace.ward_rhs_vector, dtype=complex).reshape(-1),
        )
    )
    dynamic: list[np.ndarray] = []
    for block in np.asarray(blocks, dtype=complex):
        dynamic.extend(
            (
                np.asarray(block[:3, :3], dtype=complex).reshape(-1),
                np.asarray(block[3:5, 3:5], dtype=complex).reshape(-1),
                np.asarray(block[:3, 3:5], dtype=complex).reshape(-1),
                np.asarray(block[3:5, :3], dtype=complex).reshape(-1),
            )
        )
    return np.concatenate((header, *dynamic))


def _unpack_integrated_primitives(
    packed: np.ndarray,
    *,
    xi_values: np.ndarray,
    ansatz: object,
    pairing: object,
    base_config: KuboConfig,
    q_model: np.ndarray,
    options: FiniteQEngineOptions,
    quadrature: CommensurateOrbitAdaptiveResult,
) -> tuple[tuple[BdGFiniteQResponseComponents, ...], tuple[PrimitiveWardRHS, ...]]:
    vector = np.asarray(packed, dtype=complex).reshape(-1)
    expected = _HEADER_WIDTH + _PER_FREQUENCY_WIDTH * int(xi_values.size)
    if vector.size != expected:
        raise ValueError(f"packed primitive width {vector.size} does not match expected {expected}")

    offset = 0
    direct = vector[offset : offset + 9].reshape(3, 3)
    offset += 9
    counterterm = vector[offset : offset + 4].reshape(2, 2)
    offset += 4
    phase_direct_plus = complex(vector[offset])
    phase_direct_minus = complex(vector[offset + 1])
    offset += 2
    rhs_vector = vector[offset : offset + 3]
    offset += 3

    delta0 = float(getattr(pairing, "delta0_eV"))
    components_values: list[BdGFiniteQResponseComponents] = []
    rhs_values: list[PrimitiveWardRHS] = []
    common_metadata = {
        "integration_strategy": "commensurate_q_orbit_transverse_adaptive",
        "translation_by_q_is_exact_orbit_permutation": True,
        "primitive_vector_integrated_before_schur": True,
        "matsubara_batch_shared_adaptive_nodes": True,
        "commensurate_nk": int(quadrature.nk),
        "primitive_direction": tuple(int(v) for v in quadrature.primitive_direction),
        "transverse_direction": tuple(int(v) for v in quadrature.transverse_direction),
        "orbit_shift_steps": int(quadrature.orbit_shift_steps),
        "orbit_origins": tuple(float(v) for v in quadrature.orbit_origins),
        "adaptive_pilot_order": int(quadrature.pilot_order),
        "adaptive_epsabs": float(quadrature.epsabs),
        "adaptive_epsrel": float(quadrature.epsrel),
        "adaptive_limit": int(quadrature.limit),
        "adaptive_quadrature": str(quadrature.quadrature),
        "adaptive_norm": str(quadrature.norm),
        "adaptive_scaled_error_estimate": float(quadrature.scaled_error_estimate),
        "adaptive_success": bool(quadrature.success),
        "adaptive_status": int(quadrature.status),
        "adaptive_message": str(quadrature.message),
        "transverse_evaluations": int(quadrature.transverse_evaluations),
        "point_evaluations": int(quadrature.point_evaluations),
        "projection_applied": False,
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }

    for xi in xi_values:
        bubble = vector[offset : offset + 9].reshape(3, 3)
        offset += 9
        collective_bubble = vector[offset : offset + 4].reshape(2, 2)
        offset += 4
        em_collective_left = vector[offset : offset + 6].reshape(3, 2)
        offset += 6
        collective_em_right = vector[offset : offset + 6].reshape(2, 3)
        offset += 6

        phase_left = delta0 * em_collective_left[:, 1]
        phase_right = delta0 * collective_em_right[1, :]
        phase_phase_bubble = np.asarray(
            [[delta0 * delta0 * collective_bubble[1, 1]]], dtype=complex
        )
        eval_config = replace(base_config, omega_eV=float(xi))
        base = _finalize_components(
            ansatz=ansatz,
            opts=options,
            shared_eigenbasis_q0=False,
            shared_eigenbasis_q0_tolerance=1e-14,
            collective_mode="amplitude_phase",
            collective_mode_disabled_reason=None,
            bubble=bubble,
            direct=direct,
            phase_left=phase_left,
            phase_right=phase_right,
            phase_phase_bubble_matrix=phase_phase_bubble,
            phase_phase_direct_plus=phase_direct_plus,
            phase_phase_direct_minus=phase_direct_minus,
            collective_bubble=collective_bubble,
            collective_counterterm_matrix=counterterm,
            em_collective_left=em_collective_left,
            collective_em_right=collective_em_right,
            config=eval_config,
            q=np.asarray(q_model, dtype=float),
            workspace_evaluation=True,
        )
        corrected, _ = apply_phase_hessian_policy_to_components(
            base,
            ansatz,
            q_model,
            "nearest_neighbor_bond_metric",
        )
        corrected = replace(
            corrected,
            metadata={**dict(corrected.metadata), **common_metadata},
        )
        components_values.append(corrected)
        rhs_values.append(
            PrimitiveWardRHS(
                left=rhs_vector,
                right=rhs_vector.copy(),
                q_model=q_model,
                xi_eV=float(xi),
                delta0_eV=delta0,
                metadata={
                    "convention": "primitive_crystal_xy_rhs_aware",
                    "basis": "crystal_A0_xy",
                    "formula": "R_S = equal_forward - delta_v_mid + qM_mid",
                    "source": "commensurate_orbit_transverse_adaptive_integral",
                    "frequency_independent_rhs_reused": True,
                    **common_metadata,
                },
            )
        )

    if offset != vector.size:
        raise RuntimeError("integrated primitive unpack did not consume the full vector")
    return tuple(components_values), tuple(rhs_values)


def integrate_dwave_positive_orbit_adaptive(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    xi_eV_values: Sequence[float] | np.ndarray,
    temperature_K: float,
    eta_eV: float,
    nk: int,
    mx: int,
    my: int,
    shift_s: float = 0.5,
    subgrid_average: str = "auto",
    max_point_evaluations: int = 500_000,
    pilot_order: int = 16,
    epsabs: float = 2e-5,
    epsrel: float = 2e-3,
    limit: int = 60,
    quadrature: str = "gk15",
    norm: str = "max",
    scale_floor_relative: float = 1e-8,
    scale_floor_absolute: float = 1e-14,
) -> DWavePositiveOrbitAdaptiveResult:
    """Evaluate a positive-Matsubara batch with exact orbit and adaptive transverse t."""

    xi_values = np.asarray(xi_eV_values, dtype=float)
    if xi_values.ndim != 1 or xi_values.size == 0:
        raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
    if not np.isfinite(xi_values).all() or np.any(xi_values <= 0.0):
        raise ValueError("all xi_eV_values must be finite and positive")
    if getattr(ansatz, "name", None) != "dwave":
        raise ValueError("positive orbit-adaptive integration is currently d-wave only")
    if getattr(ansatz, "phase_vertex", None) != "bond_endpoint_gauge":
        raise ValueError("d-wave orbit-adaptive integration requires bond_endpoint_gauge")

    q_model = (2.0 * np.pi / float(nk)) * np.asarray([mx, my], dtype=float)
    base_config = KuboConfig.from_kelvin(
        omega_eV=float(xi_values[0]),
        temperature_K=float(temperature_K),
        eta_eV=float(eta_eV),
        output_si=False,
    )
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")

    def orbit_evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        material = precompute_finite_q_material_workspace_from_model_ansatz(
            spec,
            ansatz,
            points,
            weights,
            base_config,
            pairing,
            options,
        )
        workspace = precompute_finite_q_q_workspace(material, q_model)
        raw_factors = _vectorized_kubo_factors(workspace, xi_values)
        weighted = (
            0.5
            * workspace.material.k_weights[None, :, None, None]
            * raw_factors
        )
        blocks = np.einsum(
            "xkmn,kamn,kbmn->xab",
            weighted,
            workspace.left_vertices_band,
            np.conjugate(workspace.right_vertices_band),
            optimize=True,
        )
        return _pack_orbit_primitives(workspace=workspace, blocks=blocks)

    adaptive = integrate_commensurate_orbit_adaptive_aggregate(
        orbit_evaluator,
        nk=nk,
        mx=mx,
        my=my,
        shift_s=shift_s,
        subgrid_average=subgrid_average,
        max_point_evaluations=max_point_evaluations,
        pilot_order=pilot_order,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
        quadrature=quadrature,
        norm=norm,
        scale_floor_relative=scale_floor_relative,
        scale_floor_absolute=scale_floor_absolute,
    )
    components, rhs = _unpack_integrated_primitives(
        adaptive.value,
        xi_values=xi_values,
        ansatz=ansatz,
        pairing=pairing,
        base_config=base_config,
        q_model=q_model,
        options=options,
        quadrature=adaptive,
    )
    return DWavePositiveOrbitAdaptiveResult(
        components=components,
        rhs=rhs,
        xi_eV_values=xi_values,
        quadrature=adaptive,
    )


__all__ = [
    "DWavePositiveOrbitAdaptiveResult",
    "integrate_dwave_positive_orbit_adaptive",
]
