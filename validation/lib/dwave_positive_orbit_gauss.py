"""Positive-Matsubara d-wave response with fixed transverse Gauss quadrature.

The microscopic evaluator and primitive packing are intentionally shared with the
panel-adaptive path.  The independent reference backend is a fixed Gauss-Legendre
rule that may be global or split into equal panels.  Bond-metric correction and the
amplitude/phase Schur complement are applied only after the complete Brillouin-zone
primitive integral.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.ward_validation import PrimitiveWardRHS
from validation.lib.commensurate_orbit_gauss_aggregate import (
    CommensurateOrbitGaussAggregateResult,
    integrate_commensurate_orbit_gauss_aggregate,
)
from validation.lib.dwave_orbit_primitive_evaluator import (
    DWaveOrbitEvaluatorProfile,
    DWaveOrbitPrimitiveEvaluator,
)
from validation.lib.dwave_positive_orbit_adaptive import (
    _unpack_integrated_primitives,
)


@dataclass(frozen=True)
class _AdaptiveCompatibleGaussView:
    """Duck-typed view used by the shared primitive unpacker."""

    nk: int
    primitive_direction: np.ndarray
    transverse_direction: np.ndarray
    orbit_shift_steps: int
    orbit_origins: tuple[float, ...]
    pilot_order: int
    epsabs: float
    epsrel: float
    limit: int
    quadrature: str
    norm: str
    scaled_error_estimate: float
    success: bool
    status: int
    message: str
    transverse_evaluations: int
    point_evaluations: int


@dataclass(frozen=True)
class DWavePositiveOrbitGaussResult:
    """Fixed-Gauss primitive responses and Ward RHS for one q and xi batch."""

    components: tuple[BdGFiniteQResponseComponents, ...]
    rhs: tuple[PrimitiveWardRHS, ...]
    xi_eV_values: np.ndarray
    quadrature: CommensurateOrbitGaussAggregateResult
    evaluator_profile: DWaveOrbitEvaluatorProfile

    def __post_init__(self) -> None:
        xi = np.array(self.xi_eV_values, dtype=float, copy=True)
        xi.setflags(write=False)
        object.__setattr__(self, "xi_eV_values", xi)
        if len(self.components) != xi.size or len(self.rhs) != xi.size:
            raise ValueError("components, rhs, and xi_eV_values must have equal lengths")


def _replace_gauss_metadata(
    components: tuple[BdGFiniteQResponseComponents, ...],
    rhs_values: tuple[PrimitiveWardRHS, ...],
    quadrature: CommensurateOrbitGaussAggregateResult,
    evaluator_profile: DWaveOrbitEvaluatorProfile,
) -> tuple[tuple[BdGFiniteQResponseComponents, ...], tuple[PrimitiveWardRHS, ...]]:
    common = {
        "integration_strategy": "commensurate_q_orbit_transverse_fixed_gauss",
        "matsubara_batch_shared_adaptive_nodes": False,
        "matsubara_batch_shared_gauss_nodes": True,
        "fixed_gauss_transverse_order": int(quadrature.transverse_order),
        "fixed_gauss_panel_count": int(quadrature.panel_count),
        "fixed_gauss_panel_order": int(quadrature.panel_order),
        "fixed_gauss_integration_start": float(quadrature.integration_start),
        "fixed_gauss_quadrature": str(quadrature.quadrature),
        "fixed_gauss_success": bool(quadrature.success),
        "fixed_gauss_status": int(quadrature.status),
        "fixed_gauss_message": str(quadrature.message),
        "q_workspace_implementation": str(
            evaluator_profile.q_workspace_implementation
        ),
        "full_transverse_period_integrated": bool(
            quadrature.full_transverse_period_integrated
        ),
        "symmetry_reduction_applied": bool(quadrature.symmetry_reduction_applied),
        "q_direction_special_case": bool(quadrature.q_direction_special_case),
        "transverse_evaluations": int(quadrature.transverse_evaluations),
        "point_evaluations": int(quadrature.point_evaluations),
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }

    corrected_components: list[BdGFiniteQResponseComponents] = []
    for component in components:
        metadata = {
            key: value
            for key, value in dict(component.metadata).items()
            if not key.startswith("adaptive_")
        }
        corrected_components.append(replace(component, metadata={**metadata, **common}))

    corrected_rhs: list[PrimitiveWardRHS] = []
    for rhs in rhs_values:
        metadata = {
            key: value
            for key, value in dict(rhs.metadata).items()
            if not key.startswith("adaptive_")
        }
        corrected_rhs.append(
            PrimitiveWardRHS(
                left=rhs.left,
                right=rhs.right,
                q_model=rhs.q_model,
                xi_eV=rhs.xi_eV,
                delta0_eV=rhs.delta0_eV,
                metadata={
                    **metadata,
                    **common,
                    "source": "commensurate_orbit_transverse_fixed_gauss_integral",
                },
            )
        )
    return tuple(corrected_components), tuple(corrected_rhs)


def integrate_dwave_positive_orbit_gauss(
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
    transverse_order: int,
    panel_count: int = 1,
    integration_start: float = -np.pi,
    shift_s: float = 0.5,
    subgrid_average: str = "auto",
    max_point_evaluations: int = 500_000,
) -> DWavePositiveOrbitGaussResult:
    """Evaluate one positive-Matsubara batch with global or composite Gauss t."""

    xi_values = np.asarray(xi_eV_values, dtype=float)
    if xi_values.ndim != 1 or xi_values.size == 0:
        raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
    if not np.isfinite(xi_values).all() or np.any(xi_values <= 0.0):
        raise ValueError("all xi_eV_values must be finite and positive")
    if getattr(ansatz, "name", None) != "dwave":
        raise ValueError("positive orbit fixed-Gauss integration is currently d-wave only")
    if getattr(ansatz, "phase_vertex", None) != "bond_endpoint_gauge":
        raise ValueError("d-wave fixed-Gauss integration requires bond_endpoint_gauge")

    primitive_evaluator = DWaveOrbitPrimitiveEvaluator(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_values,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        nk=nk,
        mx=mx,
        my=my,
    )

    gauss = integrate_commensurate_orbit_gauss_aggregate(
        primitive_evaluator,
        nk=nk,
        mx=mx,
        my=my,
        transverse_order=transverse_order,
        panel_count=panel_count,
        integration_start=integration_start,
        shift_s=shift_s,
        subgrid_average=subgrid_average,
        max_point_evaluations=max_point_evaluations,
    )
    evaluator_profile = primitive_evaluator.profile_snapshot()
    if evaluator_profile.q_workspace_implementation != "batched_model_capability":
        raise RuntimeError(
            "fixed/composite Gauss did not use the required batched q workspace"
        )

    view = _AdaptiveCompatibleGaussView(
        nk=int(gauss.nk),
        primitive_direction=gauss.primitive_direction,
        transverse_direction=gauss.transverse_direction,
        orbit_shift_steps=int(gauss.orbit_shift_steps),
        orbit_origins=gauss.orbit_origins,
        pilot_order=int(gauss.panel_order),
        epsabs=0.0,
        epsrel=0.0,
        limit=int(gauss.transverse_order),
        quadrature=str(gauss.quadrature),
        norm="none",
        scaled_error_estimate=0.0,
        success=bool(gauss.success),
        status=int(gauss.status),
        message=str(gauss.message),
        transverse_evaluations=int(gauss.transverse_evaluations),
        point_evaluations=int(gauss.point_evaluations),
    )
    components, rhs_values = _unpack_integrated_primitives(
        gauss.value,
        xi_values=xi_values,
        ansatz=ansatz,
        pairing=pairing,
        base_config=primitive_evaluator.base_config,
        q_model=primitive_evaluator.q_model,
        options=primitive_evaluator.options,
        quadrature=view,
    )
    components, rhs_values = _replace_gauss_metadata(
        components,
        rhs_values,
        gauss,
        evaluator_profile,
    )
    return DWavePositiveOrbitGaussResult(
        components=components,
        rhs=rhs_values,
        xi_eV_values=xi_values,
        quadrature=gauss,
        evaluator_profile=evaluator_profile,
    )


__all__ = [
    "DWavePositiveOrbitGaussResult",
    "integrate_dwave_positive_orbit_gauss",
]
