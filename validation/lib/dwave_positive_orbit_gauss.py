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

from lno327 import KuboConfig
from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.finite_q_optimized import (
    _vectorized_kubo_factors,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
)
from lno327.response.ward_validation import PrimitiveWardRHS
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.commensurate_orbit_gauss_aggregate import (
    CommensurateOrbitGaussAggregateResult,
    integrate_commensurate_orbit_gauss_aggregate,
)
from validation.lib.dwave_positive_orbit_adaptive import (
    _pack_orbit_primitives,
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

    gauss = integrate_commensurate_orbit_gauss_aggregate(
        orbit_evaluator,
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
    )
    components, rhs_values = _replace_gauss_metadata(
        components,
        rhs_values,
        gauss,
    )
    return DWavePositiveOrbitGaussResult(
        components=components,
        rhs=rhs_values,
        xi_eV_values=xi_values,
        quadrature=gauss,
    )


__all__ = [
    "DWavePositiveOrbitGaussResult",
    "integrate_dwave_positive_orbit_gauss",
]
