"""Typed production-facing contract for primitive finite-q BdG kernels.

This module deliberately contains no electrodynamics or Casimir formulas.  It
only exposes the microscopic electromagnetic blocks already computed by the
generic finite-q BdG response engine in the primitive crystal basis
``(A0, Ax, Ay)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.response.finite_q import BdGFiniteQResponseComponents

PRIMITIVE_EM_ORDER = ("A0", "Ax", "Ay")
AMPLITUDE_PHASE_ORDER = ("amplitude_eta1", "phase_eta2")
PRIMITIVE_EM_BASIS = "crystal_A0_xy"


def _readonly_complex_matrix(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    matrix = np.array(value, dtype=complex, copy=True)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        raise ValueError(f"{name} must contain only finite values")
    matrix.setflags(write=False)
    return matrix


def _readonly_q_vector(value: np.ndarray) -> np.ndarray:
    q = np.array(value, dtype=float, copy=True)
    if q.shape != (2,):
        raise ValueError(f"q_model must have shape (2,), got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("q_model must contain only finite values")
    q.setflags(write=False)
    return q


@dataclass(frozen=True)
class EffectiveEMKernel:
    """Primitive electromagnetic and collective blocks for one ``(q, i xi)``.

    The fields implement the fixed microscopic contract

    ``K_eff = K_SS - K_Seta @ inv(K_etaeta) @ K_etaS``.

    ``K_eff`` is stored rather than recomputed here because the response engine
    owns the numerically selected inverse/solve policy for the collective block.
    All arrays are copied and made read-only at construction time so downstream
    basis and unit conversions cannot mutate the microscopic result in place.
    """

    k_ss: np.ndarray
    k_seta: np.ndarray
    k_etas: np.ndarray
    k_etaeta: np.ndarray
    k_eff: np.ndarray
    q_model: np.ndarray
    xi_eV: float
    schur_condition_number: float | None
    schur_inverse_method: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_ss", _readonly_complex_matrix(self.k_ss, (3, 3), "k_ss"))
        object.__setattr__(self, "k_seta", _readonly_complex_matrix(self.k_seta, (3, 2), "k_seta"))
        object.__setattr__(self, "k_etas", _readonly_complex_matrix(self.k_etas, (2, 3), "k_etas"))
        object.__setattr__(self, "k_etaeta", _readonly_complex_matrix(self.k_etaeta, (2, 2), "k_etaeta"))
        object.__setattr__(self, "k_eff", _readonly_complex_matrix(self.k_eff, (3, 3), "k_eff"))
        object.__setattr__(self, "q_model", _readonly_q_vector(self.q_model))

        xi = float(self.xi_eV)
        if not np.isfinite(xi) or xi < 0.0:
            raise ValueError("xi_eV must be finite and non-negative")
        object.__setattr__(self, "xi_eV", xi)

        condition = self.schur_condition_number
        if condition is not None:
            condition = float(condition)
            if not np.isfinite(condition) or condition < 0.0:
                raise ValueError("schur_condition_number must be finite and non-negative")
        object.__setattr__(self, "schur_condition_number", condition)

        method = str(self.schur_inverse_method)
        if not method:
            raise ValueError("schur_inverse_method must be non-empty")
        object.__setattr__(self, "schur_inverse_method", method)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def matrix(self) -> np.ndarray:
        """Return the full primitive ``(A0, Ax, Ay)`` effective kernel."""

        return self.k_eff

    @property
    def spatial_xy(self) -> np.ndarray:
        """Return the crystal ``(x, y)`` spatial block without copying."""

        return self.k_eff[1:3, 1:3]

    @property
    def primitive_order(self) -> tuple[str, str, str]:
        return PRIMITIVE_EM_ORDER

    @property
    def collective_order(self) -> tuple[str, str]:
        return AMPLITUDE_PHASE_ORDER


def effective_em_kernel_from_components(
    components: BdGFiniteQResponseComponents,
    *,
    q_model: np.ndarray,
    xi_eV: float,
) -> EffectiveEMKernel:
    """Extract the amplitude/phase Schur kernel from the generic BdG engine.

    This is intentionally a strict production-facing adapter.  It rejects a
    response evaluated without the full amplitude/phase Schur correction rather
    than silently falling back to a bare or phase-only kernel.
    """

    metadata = dict(components.metadata)
    if metadata.get("collective_mode") != "amplitude_phase":
        raise ValueError("effective EM kernel requires collective_mode='amplitude_phase'")
    if metadata.get("selected_gauge_restored") != "amplitude_phase_schur":
        raise ValueError("effective EM kernel requires the amplitude/phase Schur response to be selected")
    if not bool(metadata.get("phase_correction_applied", False)):
        raise ValueError("effective EM kernel requires the collective correction to be applied")

    if not np.allclose(
        np.asarray(components.gauge_restored),
        np.asarray(components.amplitude_phase_schur),
        rtol=1e-13,
        atol=1e-13,
    ):
        raise ValueError("selected gauge-restored response differs from amplitude_phase_schur")

    contract_metadata = {
        **metadata,
        "basis": PRIMITIVE_EM_BASIS,
        "primitive_order": PRIMITIVE_EM_ORDER,
        "collective_order": AMPLITUDE_PHASE_ORDER,
        "source": "BdGFiniteQResponseComponents.amplitude_phase_schur",
        "microscopic_kernel_complete": True,
        "casimir_stage": "microscopic_response_only",
    }
    return EffectiveEMKernel(
        k_ss=components.bare_total,
        k_seta=components.em_collective_left,
        k_etas=components.collective_em_right,
        k_etaeta=components.collective_total,
        k_eff=components.amplitude_phase_schur,
        q_model=q_model,
        xi_eV=xi_eV,
        schur_condition_number=metadata.get("collective_total_condition_number"),
        schur_inverse_method=str(metadata.get("collective_inverse_method", "unknown")),
        metadata=contract_metadata,
    )


__all__ = [
    "AMPLITUDE_PHASE_ORDER",
    "EffectiveEMKernel",
    "PRIMITIVE_EM_BASIS",
    "PRIMITIVE_EM_ORDER",
    "effective_em_kernel_from_components",
]
