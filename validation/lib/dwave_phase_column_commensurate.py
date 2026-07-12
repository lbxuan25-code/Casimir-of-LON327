"""Reduced commensurate d-wave phase-column diagnostic.

Only the four integrated quantities needed to audit the finite-q phase Hessian are
computed.  The model context and exact-static divided difference are shared with
the canonical d-wave primitive evaluator.  This module remains a Ward diagnostic,
not a production integration path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lno327.response.finite_q import vertex_band
from lno327.response.finite_q_bdg import (
    bdg_eigensystem_from_model_pairing,
    bdg_vector_vertex_from_spec,
)
from lno327.response.occupations import fermi_function
from validation.lib.dwave_static_primitives import (
    DWaveStaticIntegrandContext,
    _static_factor_matrix,
)

_PHASE_COLUMN_WIDTH = 4


@dataclass(frozen=True)
class DWavePhaseColumnResult:
    q_model: tuple[float, float]
    q_norm: float
    delta0_eV: float
    left_em_collective_phase: complex
    right_collective_em_phase: complex
    phase_rotation_bubble: complex
    phase_rotation_counterterm: complex
    left_phase_defect: complex
    right_phase_defect: complex
    left_required_counterterm_multiplier: complex
    right_required_counterterm_multiplier: complex
    bond_metric_multiplier: float
    left_bond_metric_defect: complex
    right_bond_metric_defect: complex
    diagnostic_only: bool = True
    projection_applied: bool = False
    production_reference_established: bool = False
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DWavePhaseColumnContext:
    """Reduced evaluator backed by the canonical static primitive context."""

    full: DWaveStaticIntegrandContext

    def evaluate_complex(self, k_points: np.ndarray) -> np.ndarray:
        points = np.asarray(k_points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("k_points must have shape (n, 2) with finite values")

        result = np.zeros((points.shape[0], _PHASE_COLUMN_WIDTH), dtype=complex)
        qx, qy = map(float, self.full.q_model)
        spec = self.full.spec
        ansatz = self.full.ansatz
        amp = self.full.pairing_params
        config = self.full.config
        current_vertex = str(self.full.options.current_vertex)

        for index, (kx_value, ky_value) in enumerate(points):
            kx, ky = float(kx_value), float(ky_value)
            pairing_minus = ansatz.mean_pairing(kx - 0.5 * qx, ky - 0.5 * qy, amp)
            pairing_plus = ansatz.mean_pairing(kx + 0.5 * qx, ky + 0.5 * qy, amp)
            bands_minus = bdg_eigensystem_from_model_pairing(
                spec, kx - 0.5 * qx, ky - 0.5 * qy, pairing_minus
            )
            bands_plus = bdg_eigensystem_from_model_pairing(
                spec, kx + 0.5 * qx, ky + 0.5 * qy, pairing_plus
            )
            occupations_minus = fermi_function(
                bands_minus.energies,
                config.fermi_level_eV,
                config.temperature_eV,
            )
            occupations_plus = fermi_function(
                bands_plus.energies,
                config.fermi_level_eV,
                config.temperature_eV,
            )
            factor = _static_factor_matrix(
                bands_minus.energies,
                occupations_minus,
                bands_plus.energies,
                occupations_plus,
                config,
            )

            vx = bdg_vector_vertex_from_spec(
                spec, kx, ky, qx, qy, "x", current_vertex
            )
            vy = bdg_vector_vertex_from_spec(
                spec, kx, ky, qx, qy, "y", current_vertex
            )
            source_longitudinal = qx * vertex_band(
                bands_minus.states, vx, bands_plus.states
            ) + qy * vertex_band(bands_minus.states, vy, bands_plus.states)
            observable_longitudinal = -source_longitudinal
            eta2 = ansatz.collective_vertices(kx, ky, qx, qy, amp)[1]
            eta2_band = vertex_band(bands_minus.states, eta2, bands_plus.states)

            left_em = 0.5 * np.sum(
                factor * observable_longitudinal * np.conjugate(eta2_band)
            )
            right_em = 0.5 * np.sum(
                factor * eta2_band * np.conjugate(source_longitudinal)
            )
            finite_q_eta2_bubble = 0.5 * np.sum(
                factor * eta2_band * np.conjugate(eta2_band)
            )

            pairing_mid = ansatz.mean_pairing(kx, ky, amp)
            bands_mid = bdg_eigensystem_from_model_pairing(spec, kx, ky, pairing_mid)
            occupations_mid = fermi_function(
                bands_mid.energies,
                config.fermi_level_eV,
                config.temperature_eV,
            )
            midpoint_factor = _static_factor_matrix(
                bands_mid.energies,
                occupations_mid,
                bands_mid.energies,
                occupations_mid,
                config,
            )
            eta2_zero = ansatz.collective_vertices(kx, ky, 0.0, 0.0, amp)[1]
            eta2_zero_band = (
                bands_mid.states.conjugate().T
                @ np.asarray(eta2_zero, dtype=complex)
                @ bands_mid.states
            )
            q0_eta2_bubble = 0.5 * np.sum(
                midpoint_factor
                * eta2_zero_band
                * np.conjugate(eta2_zero_band)
            )
            result[index] = (
                left_em,
                right_em,
                finite_q_eta2_bubble,
                q0_eta2_bubble,
            )
        return result


def assemble_phase_column_result(
    context: DWavePhaseColumnContext,
    integrated: np.ndarray,
) -> DWavePhaseColumnResult:
    values = np.asarray(integrated, dtype=complex).reshape(-1)
    if values.shape != (_PHASE_COLUMN_WIDTH,):
        raise ValueError(
            f"integrated phase-column vector must have width {_PHASE_COLUMN_WIDTH}"
        )
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise ValueError("integrated phase-column vector must be finite")

    left_em, right_em, finite_bubble, q0_bubble = values
    delta0 = float(context.full.delta0_eV)
    q = np.asarray(context.full.q_model, dtype=float)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        raise ValueError("phase-column audit requires nonzero q")

    w_phase = complex(-2j * delta0)
    phase_rotation_bubble = w_phase * finite_bubble
    phase_rotation_counterterm = w_phase * (-q0_bubble)
    if abs(phase_rotation_counterterm) <= 1e-30:
        raise ValueError("phase counterterm rotation is at the absolute floor")

    left_defect = left_em + phase_rotation_bubble + phase_rotation_counterterm
    right_defect = right_em + phase_rotation_bubble + phase_rotation_counterterm
    left_required = -(left_em + phase_rotation_bubble) / phase_rotation_counterterm
    right_required = -(right_em + phase_rotation_bubble) / phase_rotation_counterterm
    bond_metric = float(
        0.5 * (np.cos(0.5 * q[0]) ** 2 + np.cos(0.5 * q[1]) ** 2)
    )
    return DWavePhaseColumnResult(
        q_model=(float(q[0]), float(q[1])),
        q_norm=q_norm,
        delta0_eV=delta0,
        left_em_collective_phase=complex(left_em),
        right_collective_em_phase=complex(right_em),
        phase_rotation_bubble=complex(phase_rotation_bubble),
        phase_rotation_counterterm=complex(phase_rotation_counterterm),
        left_phase_defect=complex(left_defect),
        right_phase_defect=complex(right_defect),
        left_required_counterterm_multiplier=complex(left_required),
        right_required_counterterm_multiplier=complex(right_required),
        bond_metric_multiplier=bond_metric,
        left_bond_metric_defect=complex(
            left_em + phase_rotation_bubble + bond_metric * phase_rotation_counterterm
        ),
        right_bond_metric_defect=complex(
            right_em + phase_rotation_bubble + bond_metric * phase_rotation_counterterm
        ),
    )


def phase_column_result_as_audit_payload(
    result: DWavePhaseColumnResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the compact structure consumed by phase-Hessian family analysis."""

    w_phase = complex(-2j * result.delta0_eV)
    zero = 0.0 + 0.0j
    return {
        "schema": "dwave_static_commensurate_phase_column_audit_v1",
        "audit": {
            "q_model": result.q_model,
            "q_norm": result.q_norm,
            "delta0_eV": result.delta0_eV,
            "w_left": (zero, w_phase),
            "w_right": (zero, w_phase),
            "component_sources": {
                side: {
                    "collective_defect_parts": {
                        "em_collective_contraction": (
                            zero,
                            result.left_em_collective_phase
                            if side == "left"
                            else result.right_collective_em_phase,
                        ),
                        "phase_rotation_bubble": (
                            zero,
                            result.phase_rotation_bubble,
                        ),
                        "phase_rotation_counterterm": (
                            zero,
                            result.phase_rotation_counterterm,
                        ),
                    }
                }
                for side in ("left", "right")
            },
        },
        "primitive_metadata": {},
        "phase_column_result": result.to_dict(),
        "metadata": dict(metadata or {}),
        "status": {
            "diagnostic_run_completed": True,
            "reduced_phase_column_only": True,
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }


__all__ = [
    "DWavePhaseColumnContext",
    "DWavePhaseColumnResult",
    "assemble_phase_column_result",
    "phase_column_result_as_audit_payload",
]
