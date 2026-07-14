"""Common positive-Matsubara response with fixed/composite transverse Gauss.

Both supported superconducting ansatzes use the same complete commensurate orbit,
the same full transverse period, the same Gauss nodes and weights, and the same
parent-side complex Kahan reduction.  Pairing-specific physics enters only through
the model ansatz and the post-integral phase-Hessian policy.
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
from validation.lib.dwave_positive_orbit_adaptive import _unpack_integrated_primitives
from validation.lib.positive_orbit_primitive_evaluator import (
    PositiveOrbitEvaluatorProfile,
    PositiveOrbitPrimitiveEvaluator,
)


@dataclass(frozen=True)
class _AdaptiveCompatibleGaussView:
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
class PositiveOrbitGaussResult:
    """Fixed-Gauss primitive responses and Ward RHS for one q and xi batch."""

    pairing_name: str
    phase_hessian_policy: str
    components: tuple[BdGFiniteQResponseComponents, ...]
    rhs: tuple[PrimitiveWardRHS, ...]
    xi_eV_values: np.ndarray
    quadrature: CommensurateOrbitGaussAggregateResult
    evaluator_profile: PositiveOrbitEvaluatorProfile

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
    evaluator_profile: PositiveOrbitEvaluatorProfile,
    *,
    pairing_name: str,
    phase_hessian_policy: str,
) -> tuple[tuple[BdGFiniteQResponseComponents, ...], tuple[PrimitiveWardRHS, ...]]:
    common = {
        "pairing": str(pairing_name),
        "integration_strategy": "commensurate_q_orbit_transverse_fixed_gauss",
        "post_integral_phase_hessian_policy": str(phase_hessian_policy),
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
        "fixed_gauss_transverse_workers": int(quadrature.transverse_workers),
        "fixed_gauss_transverse_task_size": int(quadrature.transverse_task_size),
        "fixed_gauss_transverse_task_count": int(quadrature.transverse_task_count),
        "fixed_gauss_execution_strategy": str(quadrature.execution_strategy),
        "material_workspace_implementation": str(
            evaluator_profile.material_workspace_implementation
        ),
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


def integrate_positive_orbit_gauss(
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
    transverse_workers: int = 1,
    transverse_task_size: int = 1,
) -> PositiveOrbitGaussResult:
    """Evaluate one spm/d-wave positive-Matsubara batch with one Gauss method."""

    xi_values = np.asarray(xi_eV_values, dtype=float)
    if xi_values.ndim != 1 or xi_values.size == 0:
        raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
    if not np.isfinite(xi_values).all() or np.any(xi_values <= 0.0):
        raise ValueError("all xi_eV_values must be finite and positive")
    pairing_name = str(getattr(ansatz, "name", ""))
    if pairing_name not in {"spm", "dwave"}:
        raise ValueError("positive orbit fixed-Gauss supports spm and dwave")
    if getattr(ansatz, "phase_vertex", None) != "bond_endpoint_gauge":
        raise ValueError("positive orbit fixed-Gauss requires bond_endpoint_gauge")

    with PositiveOrbitPrimitiveEvaluator(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_values,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        nk=nk,
        mx=mx,
        my=my,
        process_workers=transverse_workers,
    ) as primitive_evaluator:
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
            transverse_workers=transverse_workers,
            transverse_task_size=transverse_task_size,
        )
        evaluator_profile = primitive_evaluator.profile_snapshot()
        execution_strategy = primitive_evaluator.parallel_execution_strategy
        base_config = primitive_evaluator.base_config
        q_model = np.asarray(primitive_evaluator.q_model, dtype=float)
        options = primitive_evaluator.options
        phase_hessian_policy = str(primitive_evaluator.phase_hessian_policy)

    if gauss.execution_strategy != execution_strategy:
        gauss = replace(gauss, execution_strategy=execution_strategy)
    if evaluator_profile.material_workspace_implementation != "batched_model_capability":
        raise RuntimeError("fixed Gauss did not use the batched material workspace")
    if evaluator_profile.q_workspace_implementation != "batched_model_capability":
        raise RuntimeError("fixed Gauss did not use the batched q workspace")
    if evaluator_profile.callbacks != gauss.transverse_evaluations:
        raise RuntimeError("orbit evaluator callback count does not match Gauss nodes")
    if evaluator_profile.complete_orbit_points != gauss.point_evaluations:
        raise RuntimeError("orbit evaluator point count does not match Gauss budget")

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
        base_config=base_config,
        q_model=q_model,
        options=options,
        quadrature=view,
        phase_hessian_policy=phase_hessian_policy,
    )
    components, rhs_values = _replace_gauss_metadata(
        components,
        rhs_values,
        gauss,
        evaluator_profile,
        pairing_name=pairing_name,
        phase_hessian_policy=phase_hessian_policy,
    )
    return PositiveOrbitGaussResult(
        pairing_name=pairing_name,
        phase_hessian_policy=phase_hessian_policy,
        components=components,
        rhs=rhs_values,
        xi_eV_values=xi_values,
        quadrature=gauss,
        evaluator_profile=evaluator_profile,
    )


__all__ = ["PositiveOrbitGaussResult", "integrate_positive_orbit_gauss"]
