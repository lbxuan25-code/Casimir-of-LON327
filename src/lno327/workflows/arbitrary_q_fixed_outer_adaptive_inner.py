"""Reference fixed-outer/adaptive-inner arbitrary-q d-wave workflow.

This generalized implementation uses the current all-Matsubara pointwise
primitive kernel and applies all nonlinear collective processing only after the
complete BZ integral.

The workflow is diagnostic/reference-only until outer-order, orientation,
inner-tolerance, commensurate-orbit, periodic-grid, and two-plate gates exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.numerics.fixed_outer_adaptive_inner import (
    FixedOuterAdaptiveInnerOptions,
    FixedOuterAdaptiveInnerOrientationResult,
    IntegrationOrder,
    integrate_fixed_outer_adaptive_inner_orientation,
)
from lno327.response.arbitrary_q_accumulator import combine_operator_ward_reports
from lno327.response.arbitrary_q_formal_policy import validate_q_domain
from lno327.response.arbitrary_q_pointwise_primitives import (
    evaluate_arbitrary_q_pointwise_primitives,
    pack_complex_density_to_real,
    unpack_real_integral_to_complex,
)
from lno327.response.periodic_bz_grid import exact_float64_key
from lno327.response.primitive_kernel_v2 import (
    OperatorWardReport,
    unpack_integrated_primitives,
)
from lno327.response.ward_validation import PrimitiveWardRHS
from lno327.workflows.finite_q_engine import FiniteQEngineOptions

_BZ_NORMALIZATION = 1.0 / (2.0 * np.pi) ** 2
METHOD_ID = "DWaveFixedOuterAdaptiveInner-v1"
EXECUTOR_ID = "scipy_quad_vec_reference-v1"


@dataclass(frozen=True)
class PointwiseContextProfile:
    callback_count: int
    unique_point_count: int
    cache_hit_count: int
    material_workspace_seconds: float
    q_workspace_seconds: float
    counterterm_q0_workspace_seconds: float
    kubo_factor_seconds: float
    counterterm_factor_seconds: float
    primitive_density_seconds: float
    shifted_eigensystem_build_count: int
    q_workspace_build_count: int
    counterterm_q0_workspace_build_count: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "schema": "FixedOuterAdaptiveInnerPointwiseContextProfile-v1",
            "callback_count": int(self.callback_count),
            "unique_point_count": int(self.unique_point_count),
            "cache_hit_count": int(self.cache_hit_count),
            "material_workspace_seconds": float(self.material_workspace_seconds),
            "q_workspace_seconds": float(self.q_workspace_seconds),
            "counterterm_q0_workspace_seconds": float(
                self.counterterm_q0_workspace_seconds
            ),
            "kubo_factor_seconds": float(self.kubo_factor_seconds),
            "counterterm_factor_seconds": float(self.counterterm_factor_seconds),
            "primitive_density_seconds": float(self.primitive_density_seconds),
            "shifted_eigensystem_build_count": int(
                self.shifted_eigensystem_build_count
            ),
            "q_workspace_build_count": int(self.q_workspace_build_count),
            "counterterm_q0_workspace_build_count": int(
                self.counterterm_q0_workspace_build_count
            ),
        }


class ArbitraryQPointwisePrimitiveContext:
    """Exact-float memoized scalar callback over the shared pointwise kernel."""

    def __init__(
        self,
        *,
        spec: object,
        ansatz: object,
        pairing: object,
        q_model: np.ndarray,
        xi_eV_values: np.ndarray,
        config: KuboConfig,
        options: FiniteQEngineOptions,
        operator_ward_atol: float,
        operator_ward_rtol: float,
    ) -> None:
        self.spec = spec
        self.ansatz = ansatz
        self.pairing = pairing
        self.q_model = np.asarray(q_model, dtype=float)
        self.xi_eV_values = np.asarray(xi_eV_values, dtype=float)
        self.config = config
        self.options = options
        self.operator_ward_atol = float(operator_ward_atol)
        self.operator_ward_rtol = float(operator_ward_rtol)
        self.reset_orientation()

    def reset_orientation(self) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self._reports: list[OperatorWardReport] = []
        self._callback_count = 0
        self._cache_hits = 0
        self._material_seconds = 0.0
        self._q_workspace_seconds = 0.0
        self._q0_workspace_seconds = 0.0
        self._kubo_seconds = 0.0
        self._q0_factor_seconds = 0.0
        self._density_seconds = 0.0
        self._shifted_eigh = 0
        self._q_workspace_builds = 0
        self._q0_workspace_builds = 0

    def evaluate_real(self, kx: float, ky: float) -> np.ndarray:
        self._callback_count += 1
        point = np.asarray([float(kx), float(ky)], dtype=float)
        key = exact_float64_key(point)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return np.array(cached, dtype=float, copy=True)

        batch = evaluate_arbitrary_q_pointwise_primitives(
            spec=self.spec,
            ansatz=self.ansatz,
            pairing=self.pairing,
            config=self.config,
            q_model=self.q_model,
            xi_eV_values=self.xi_eV_values,
            k_points=point.reshape(1, 2),
            options=self.options,
            operator_ward_atol=self.operator_ward_atol,
            operator_ward_rtol=self.operator_ward_rtol,
        )
        metrics = batch.metrics
        self._material_seconds += metrics.material_workspace_seconds
        self._q_workspace_seconds += metrics.q_workspace_seconds
        self._q0_workspace_seconds += metrics.counterterm_q0_workspace_seconds
        self._kubo_seconds += metrics.kubo_factor_seconds
        self._q0_factor_seconds += metrics.counterterm_factor_seconds
        self._density_seconds += metrics.primitive_density_seconds
        self._shifted_eigh += metrics.shifted_eigensystem_build_count
        self._q_workspace_builds += metrics.q_workspace_build_count
        self._q0_workspace_builds += metrics.counterterm_q0_workspace_build_count
        self._reports.append(batch.operator_ward)
        value = np.asarray(
            pack_complex_density_to_real(batch.densities[0]),
            dtype=float,
        )
        value.setflags(write=False)
        self._cache[key] = value
        return np.array(value, dtype=float, copy=True)

    def profile(self) -> PointwiseContextProfile:
        return PointwiseContextProfile(
            callback_count=int(self._callback_count),
            unique_point_count=len(self._cache),
            cache_hit_count=int(self._cache_hits),
            material_workspace_seconds=float(self._material_seconds),
            q_workspace_seconds=float(self._q_workspace_seconds),
            counterterm_q0_workspace_seconds=float(self._q0_workspace_seconds),
            kubo_factor_seconds=float(self._kubo_seconds),
            counterterm_factor_seconds=float(self._q0_factor_seconds),
            primitive_density_seconds=float(self._density_seconds),
            shifted_eigensystem_build_count=int(self._shifted_eigh),
            q_workspace_build_count=int(self._q_workspace_builds),
            counterterm_q0_workspace_build_count=int(self._q0_workspace_builds),
        )

    def operator_ward(self) -> OperatorWardReport:
        if not self._reports:
            raise RuntimeError("orientation produced no operator Ward reports")
        return combine_operator_ward_reports(self._reports)


@dataclass(frozen=True)
class FixedOuterAdaptiveInnerOrientationResponse:
    order: IntegrationOrder
    packed_primitives: np.ndarray
    components: tuple[object, ...]
    rhs: tuple[PrimitiveWardRHS, ...]
    operator_ward: OperatorWardReport
    quadrature: FixedOuterAdaptiveInnerOrientationResult
    pointwise_profile: PointwiseContextProfile
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        packed = np.array(self.packed_primitives, dtype=complex, copy=True)
        packed.setflags(write=False)
        object.__setattr__(self, "packed_primitives", packed)


@dataclass(frozen=True)
class ArbitraryQFixedOuterAdaptiveInnerResult:
    q_model: np.ndarray
    xi_eV_values: np.ndarray
    primary_order: IntegrationOrder
    orientations: tuple[FixedOuterAdaptiveInnerOrientationResponse, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        q = np.array(self.q_model, dtype=float, copy=True)
        xi = np.array(self.xi_eV_values, dtype=float, copy=True)
        q.setflags(write=False)
        xi.setflags(write=False)
        object.__setattr__(self, "q_model", q)
        object.__setattr__(self, "xi_eV_values", xi)
        orders = tuple(item.order for item in self.orientations)
        if self.primary_order not in orders:
            raise ValueError("primary order is absent from orientation results")

    @property
    def primary(self) -> FixedOuterAdaptiveInnerOrientationResponse:
        return next(item for item in self.orientations if item.order == self.primary_order)


def _validate_xi(values: Sequence[float] | np.ndarray) -> np.ndarray:
    xi = np.asarray(values, dtype=float)
    if xi.ndim != 1 or xi.size == 0:
        raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
    if not np.isfinite(xi).all() or np.any(xi < 0.0):
        raise ValueError("xi_eV_values must be finite and non-negative")
    if np.count_nonzero(xi == 0.0) > 1:
        raise ValueError("exact zero may appear at most once")
    return xi


def integrate_arbitrary_q_fixed_outer_adaptive_inner(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    xi_eV_values: Sequence[float] | np.ndarray,
    temperature_K: float,
    eta_eV: float,
    q_model: np.ndarray,
    outer_order: int = 96,
    inner_options: FixedOuterAdaptiveInnerOptions | None = None,
    orders: Sequence[IntegrationOrder] = ("xy", "yx"),
    primary_order: IntegrationOrder = "xy",
    operator_ward_atol: float = 512.0 * np.finfo(float).eps,
    operator_ward_rtol: float = 512.0 * np.finfo(float).eps,
) -> ArbitraryQFixedOuterAdaptiveInnerResult:
    """Run the retained d-wave method on exact arbitrary q and all frequencies."""

    xi = _validate_xi(xi_eV_values)
    q = validate_q_domain(np.asarray(q_model, dtype=float))
    if str(getattr(ansatz, "name", "")) != "dwave":
        raise ValueError(
            "fixed-outer/adaptive-inner primary workflow currently supports d-wave only"
        )
    if getattr(ansatz, "phase_vertex", None) != "bond_endpoint_gauge":
        raise ValueError("d-wave workflow requires bond_endpoint_gauge")
    requested_orders = tuple(str(value) for value in orders)
    if not requested_orders or len(set(requested_orders)) != len(requested_orders):
        raise ValueError("orders must be nonempty and distinct")
    if any(value not in {"xy", "yx"} for value in requested_orders):
        raise ValueError("orders may contain only 'xy' and 'yx'")
    if primary_order not in requested_orders:
        raise ValueError("primary_order must be included in orders")

    base_config = KuboConfig.from_kelvin(
        omega_eV=float(xi[0]),
        temperature_K=float(temperature_K),
        eta_eV=float(eta_eV),
        output_si=False,
    )
    engine_options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    adaptive = inner_options or FixedOuterAdaptiveInnerOptions()
    context = ArbitraryQPointwisePrimitiveContext(
        spec=spec,
        ansatz=ansatz,
        pairing=pairing,
        q_model=q,
        xi_eV_values=xi,
        config=base_config,
        options=engine_options,
        operator_ward_atol=operator_ward_atol,
        operator_ward_rtol=operator_ward_rtol,
    )

    orientation_results: list[FixedOuterAdaptiveInnerOrientationResponse] = []
    for order_value in requested_orders:
        order = order_value
        context.reset_orientation()
        quadrature = integrate_fixed_outer_adaptive_inner_orientation(
            lambda kx, ky: _BZ_NORMALIZATION * context.evaluate_real(kx, ky),
            order=order,
            outer_order=int(outer_order),
            options=adaptive,
        )
        packed = unpack_real_integral_to_complex(quadrature.value)
        integration_metadata: dict[str, object] = {
            "integration_strategy": (
                "fixed_global_gauss_outer_shared_vector_adaptive_inner"
            ),
            "method_id": METHOD_ID,
            "executor_id": EXECUTOR_ID,
            "integration_order": order,
            "primary_order": str(primary_order),
            "outer_order": int(outer_order),
            "inner_epsabs": float(adaptive.epsabs),
            "inner_epsrel": float(adaptive.epsrel),
            "inner_limit": int(adaptive.inner_limit),
            "inner_quadrature": str(adaptive.quadrature),
            "inner_norm": str(adaptive.norm),
            "inner_split_points": list(adaptive.split_points),
            "inner_error_estimate": float(quadrature.error_estimate),
            "outer_discretization_error_estimated": False,
            "exact_q_used_without_rounding": True,
            "matsubara_batch_shared_nodes": True,
            "zero_and_positive_frequencies_share_nodes": bool(
                np.any(xi == 0.0) and np.any(xi > 0.0)
            ),
            "post_integral_phase_hessian_policy": "nearest_neighbor_bond_metric",
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        }
        components, rhs = unpack_integrated_primitives(
            packed,
            xi_values=xi,
            ansatz=ansatz,
            pairing=pairing,
            base_config=base_config,
            q_model=q,
            options=engine_options,
            phase_hessian_policy="nearest_neighbor_bond_metric",
            integration_metadata=integration_metadata,
            rhs_source="fixed_outer_adaptive_inner_pointwise_integral",
        )
        orientation_results.append(
            FixedOuterAdaptiveInnerOrientationResponse(
                order=order,
                packed_primitives=packed,
                components=components,
                rhs=rhs,
                operator_ward=context.operator_ward(),
                quadrature=quadrature,
                pointwise_profile=context.profile(),
                metadata=integration_metadata,
            )
        )

    return ArbitraryQFixedOuterAdaptiveInnerResult(
        q_model=q,
        xi_eV_values=xi,
        primary_order=primary_order,
        orientations=tuple(orientation_results),
        metadata={
            "schema": "ArbitraryQFixedOuterAdaptiveInnerResult-v1",
            "method_id": METHOD_ID,
            "executor_id": EXECUTOR_ID,
            "primary_value_is_orientation_average": False,
            "required_orientation_audit": "yx",
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    )


__all__ = [
    "ArbitraryQFixedOuterAdaptiveInnerResult",
    "ArbitraryQPointwisePrimitiveContext",
    "EXECUTOR_ID",
    "FixedOuterAdaptiveInnerOrientationResponse",
    "METHOD_ID",
    "PointwiseContextProfile",
    "integrate_arbitrary_q_fixed_outer_adaptive_inner",
]
