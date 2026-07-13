"""Positive-Matsubara d-wave primitives with deterministic panel adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.ward_validation import PrimitiveWardRHS
from validation.lib.commensurate_orbit_panel_adaptive import (
    PanelAdaptiveResult,
    integrate_commensurate_orbit_panel_adaptive,
)
from validation.lib.dwave_orbit_primitive_evaluator import (
    DWaveOrbitEvaluatorProfile,
    DWaveOrbitPrimitiveEvaluator,
)
from validation.lib.dwave_positive_orbit_adaptive import (
    _unpack_integrated_primitives,
)


@dataclass(frozen=True)
class DWavePositiveOrbitPanelAdaptiveResult:
    """Primary/audit primitive responses from one shared panel state."""

    primary_components: tuple[BdGFiniteQResponseComponents, ...]
    primary_rhs: tuple[PrimitiveWardRHS, ...]
    audit_components: tuple[BdGFiniteQResponseComponents, ...]
    audit_rhs: tuple[PrimitiveWardRHS, ...]
    xi_eV_values: np.ndarray
    quadrature: PanelAdaptiveResult
    evaluator_profile: DWaveOrbitEvaluatorProfile

    def __post_init__(self) -> None:
        xi = np.array(self.xi_eV_values, dtype=float, copy=True)
        xi.setflags(write=False)
        object.__setattr__(self, "xi_eV_values", xi)
        for components, rhs, label in (
            (self.primary_components, self.primary_rhs, "primary"),
            (self.audit_components, self.audit_rhs, "audit"),
        ):
            if components or rhs:
                if len(components) != xi.size or len(rhs) != xi.size:
                    raise ValueError(
                        f"{label} components, rhs, and xi_eV_values must have equal lengths"
                    )

    @property
    def components(self) -> tuple[BdGFiniteQResponseComponents, ...]:
        return self.audit_components or self.primary_components

    @property
    def rhs(self) -> tuple[PrimitiveWardRHS, ...]:
        return self.audit_rhs or self.primary_rhs


def integrate_dwave_positive_orbit_panel_adaptive(
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
    max_unique_transverse_evaluations: int = 256,
    epsabs: float = 2e-5,
    epsrel: float = 2e-3,
    audit_tolerance_factor: float = 0.25,
    norm: str = "max",
    scale_floor_relative: float = 1e-8,
    scale_floor_absolute: float = 1e-14,
) -> DWavePositiveOrbitPanelAdaptiveResult:
    """Evaluate one positive-Matsubara batch with nested panel adaptation."""

    xi_values = np.asarray(xi_eV_values, dtype=float)
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
    quadrature = integrate_commensurate_orbit_panel_adaptive(
        primitive_evaluator,
        nk=nk,
        mx=mx,
        my=my,
        shift_s=shift_s,
        subgrid_average=subgrid_average,
        max_unique_transverse_evaluations=max_unique_transverse_evaluations,
        epsabs=epsabs,
        epsrel=epsrel,
        audit_tolerance_factor=audit_tolerance_factor,
        norm=norm,
        scale_floor_relative=scale_floor_relative,
        scale_floor_absolute=scale_floor_absolute,
    )

    primary_components: tuple[BdGFiniteQResponseComponents, ...] = ()
    primary_rhs: tuple[PrimitiveWardRHS, ...] = ()
    audit_components: tuple[BdGFiniteQResponseComponents, ...] = ()
    audit_rhs: tuple[PrimitiveWardRHS, ...] = ()
    if quadrature.primary.value is not None:
        primary_components, primary_rhs = _unpack_integrated_primitives(
            quadrature.primary.value,
            xi_values=xi_values,
            ansatz=ansatz,
            pairing=pairing,
            base_config=primitive_evaluator.base_config,
            q_model=primitive_evaluator.q_model,
            options=primitive_evaluator.options,
            quadrature=quadrature,
        )
    if quadrature.audit is not None and quadrature.audit.value is not None:
        audit_components, audit_rhs = _unpack_integrated_primitives(
            quadrature.audit.value,
            xi_values=xi_values,
            ansatz=ansatz,
            pairing=pairing,
            base_config=primitive_evaluator.base_config,
            q_model=primitive_evaluator.q_model,
            options=primitive_evaluator.options,
            quadrature=quadrature,
        )

    return DWavePositiveOrbitPanelAdaptiveResult(
        primary_components=primary_components,
        primary_rhs=primary_rhs,
        audit_components=audit_components,
        audit_rhs=audit_rhs,
        xi_eV_values=xi_values,
        quadrature=quadrature,
        evaluator_profile=primitive_evaluator.profile_snapshot(),
    )


__all__ = [
    "DWavePositiveOrbitPanelAdaptiveResult",
    "integrate_dwave_positive_orbit_panel_adaptive",
]
