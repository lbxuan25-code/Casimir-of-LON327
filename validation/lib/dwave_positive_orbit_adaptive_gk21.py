"""Positive-Matsubara d-wave primitives with the production-candidate GK21 rule."""

from __future__ import annotations

from dataclasses import dataclass
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
from validation.lib.commensurate_orbit_adaptive_gk21 import (
    AdaptiveGK21Result,
    integrate_commensurate_orbit_adaptive_gk21,
)
from validation.lib.dwave_positive_orbit_adaptive import (
    _pack_orbit_primitives,
    _unpack_integrated_primitives,
)


@dataclass(frozen=True)
class DWavePositiveOrbitAdaptiveGK21Result:
    """Primary/audit primitive responses sharing one complete-orbit cache."""

    primary_components: tuple[BdGFiniteQResponseComponents, ...]
    primary_rhs: tuple[PrimitiveWardRHS, ...]
    audit_components: tuple[BdGFiniteQResponseComponents, ...]
    audit_rhs: tuple[PrimitiveWardRHS, ...]
    xi_eV_values: np.ndarray
    quadrature: AdaptiveGK21Result

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
        """Return the tightened-audit result when available, else the primary result."""

        return self.audit_components or self.primary_components

    @property
    def rhs(self) -> tuple[PrimitiveWardRHS, ...]:
        return self.audit_rhs or self.primary_rhs


def integrate_dwave_positive_orbit_adaptive_gk21(
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
    limit: int = 60,
    norm: str = "max",
    scale_floor_relative: float = 1e-8,
    scale_floor_absolute: float = 1e-14,
) -> DWavePositiveOrbitAdaptiveGK21Result:
    """Evaluate one positive-Matsubara batch with complete-orbit adaptive GK21."""

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

    quadrature = integrate_commensurate_orbit_adaptive_gk21(
        orbit_evaluator,
        nk=nk,
        mx=mx,
        my=my,
        shift_s=shift_s,
        subgrid_average=subgrid_average,
        max_unique_transverse_evaluations=max_unique_transverse_evaluations,
        epsabs=epsabs,
        epsrel=epsrel,
        audit_tolerance_factor=audit_tolerance_factor,
        limit=limit,
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
            base_config=base_config,
            q_model=q_model,
            options=options,
            quadrature=quadrature,
        )
    if quadrature.audit is not None and quadrature.audit.value is not None:
        audit_components, audit_rhs = _unpack_integrated_primitives(
            quadrature.audit.value,
            xi_values=xi_values,
            ansatz=ansatz,
            pairing=pairing,
            base_config=base_config,
            q_model=q_model,
            options=options,
            quadrature=quadrature,
        )

    return DWavePositiveOrbitAdaptiveGK21Result(
        primary_components=primary_components,
        primary_rhs=primary_rhs,
        audit_components=audit_components,
        audit_rhs=audit_rhs,
        xi_eV_values=xi_values,
        quadrature=quadrature,
    )


__all__ = [
    "DWavePositiveOrbitAdaptiveGK21Result",
    "integrate_dwave_positive_orbit_adaptive_gk21",
]
