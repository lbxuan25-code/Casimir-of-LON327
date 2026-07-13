"""Complete-orbit adaptive GK21 transverse integration.

Every transverse callback evaluates a complete exact commensurate q orbit and
returns only packed primitive response blocks.  Global metric/Schur/sheet/
reflection/logdet operations remain outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Sequence

import numpy as np
import scipy
from scipy.integrate import quad_vec

from validation.lib.commensurate_orbit_groups import (
    component_scales_from_groups,
    group_layout,
    group_point_scales,
    vector_norm,
)
from validation.lib.dwave_commensurate_orbit_gauss import (
    commensurate_orbit_basis,
    complementary_orbit_origins,
)

OrbitAggregateEvaluator = Callable[[np.ndarray, np.ndarray], np.ndarray]
_GK21_POSITIVE = np.asarray(
    [
        0.9956571630258081,
        0.9739065285171717,
        0.9301574913557082,
        0.8650633666889845,
        0.7808177265864169,
        0.6794095682990244,
        0.5627571346686047,
        0.4333953941292472,
        0.2943928627014602,
        0.1488743389816312,
    ]
)
GK21_ROOT_NODES = np.concatenate(
    (-_GK21_POSITIVE, np.asarray([0.0]), _GK21_POSITIVE[::-1])
)


class TransverseEvaluationBudgetExceeded(RuntimeError):
    def __init__(self, maximum: int, attempted: int) -> None:
        self.maximum = int(maximum)
        self.attempted = int(attempted)
        super().__init__(
            "unique transverse evaluation budget exceeded: "
            f"maximum={self.maximum}, attempted={self.attempted}"
        )


def _readonly(value: np.ndarray, *, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _real_norm(value: np.ndarray, norm: str) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if norm == "max":
        return float(np.max(np.abs(array))) if array.size else 0.0
    if norm == "2":
        return float(np.linalg.norm(array))
    raise ValueError("norm must be 'max' or '2'")


@dataclass
class CompleteOrbitAggregateWorkspace:
    evaluator: OrbitAggregateEvaluator
    nk: int
    mx: int
    my: int
    shift_s: float = 0.5
    subgrid_average: str = "auto"
    max_unique_transverse_evaluations: int = 256

    primitive_direction: np.ndarray = field(init=False)
    transverse_direction: np.ndarray = field(init=False)
    orbit_shift_steps: int = field(init=False)
    orbit_origins: tuple[float, ...] = field(init=False)
    q_model: np.ndarray = field(init=False)
    points_per_t: int = field(init=False)
    cache: dict[float, np.ndarray] = field(init=False, default_factory=dict)
    cache_hits: int = field(init=False, default=0)
    point_evaluations: int = field(init=False, default=0)
    geometry_wall_seconds: float = field(init=False, default=0.0)
    evaluator_wall_seconds: float = field(init=False, default=0.0)
    _orbit_base: np.ndarray = field(init=False, repr=False)
    _points: np.ndarray = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)
    _width: int | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.nk, self.mx, self.my = int(self.nk), int(self.mx), int(self.my)
        self.max_unique_transverse_evaluations = int(
            self.max_unique_transverse_evaluations
        )
        if self.nk <= 0 or self.max_unique_transverse_evaluations <= 0:
            raise ValueError("nk and transverse budget must be positive")
        if self.mx == 0 and self.my == 0:
            raise ValueError("at least one of mx,my must be nonzero")
        if abs(self.mx) > self.nk // 2 or abs(self.my) > self.nk // 2:
            raise ValueError("mx and my must lie in the principal periodic range")
        if self.subgrid_average not in {"auto", "none"}:
            raise ValueError("subgrid_average must be 'auto' or 'none'")

        primitive, transverse, steps = commensurate_orbit_basis(self.mx, self.my)
        origins = complementary_orbit_origins(
            steps, float(self.shift_s), self.subgrid_average
        )
        spacing = 2.0 * np.pi / float(self.nk)
        s_values = np.concatenate(
            [
                -np.pi
                + (np.arange(self.nk, dtype=float) + float(origin)) * spacing
                for origin in origins
            ]
        )
        self.primitive_direction = _readonly(primitive, dtype=float)
        self.transverse_direction = _readonly(transverse, dtype=float)
        self.orbit_shift_steps = int(steps)
        self.orbit_origins = tuple(float(value) for value in origins)
        self.q_model = _readonly(
            spacing * np.asarray([self.mx, self.my], dtype=float), dtype=float
        )
        self.points_per_t = int(s_values.size)
        self._orbit_base = s_values[:, None] * self.primitive_direction[None, :]
        self._points = np.empty_like(self._orbit_base)
        self._weights = _readonly(
            np.full(self.points_per_t, 1.0 / self.points_per_t), dtype=float
        )

    def evaluate_phase(self, phase: float) -> np.ndarray:
        canonical = float(phase % 1.0)
        key = round(canonical, 14)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        attempted = len(self.cache) + 1
        if attempted > self.max_unique_transverse_evaluations:
            raise TransverseEvaluationBudgetExceeded(
                self.max_unique_transverse_evaluations, attempted
            )

        started = time.perf_counter()
        t_value = -np.pi + 2.0 * np.pi * canonical
        np.add(
            self._orbit_base,
            t_value * self.transverse_direction[None, :],
            out=self._points,
        )
        self._points += np.pi
        np.remainder(self._points, 2.0 * np.pi, out=self._points)
        self._points -= np.pi
        self.geometry_wall_seconds += time.perf_counter() - started

        started = time.perf_counter()
        value = np.asarray(
            self.evaluator(self._points, self._weights), dtype=complex
        ).reshape(-1)
        self.evaluator_wall_seconds += time.perf_counter() - started
        if not value.size or not np.isfinite(value.real).all() or not np.isfinite(value.imag).all():
            raise ValueError("aggregate evaluator returned an empty or non-finite vector")
        if self._width is None:
            self._width = int(value.size)
        elif value.size != self._width:
            raise ValueError("aggregate evaluator vector width changed between calls")
        stored = _readonly(value, dtype=complex)
        self.cache[key] = stored
        self.point_evaluations += self.points_per_t
        return stored

    def evaluate_t(self, t_value: float) -> np.ndarray:
        return self.evaluate_phase((float(t_value) + np.pi) / (2.0 * np.pi))

    @property
    def transverse_evaluations_unique(self) -> int:
        return len(self.cache)

    @property
    def cached_values(self) -> np.ndarray:
        return np.stack(tuple(self.cache.values()), axis=0)


@dataclass(frozen=True)
class AdaptiveGK21PassResult:
    value: np.ndarray | None
    integral_error_estimate: float
    integral_tolerance: float
    integral_error_ratio: float
    success: bool
    status: int
    message: str
    subinterval_count: int
    scipy_neval: int
    unique_evaluations_added: int
    cache_hits_added: int
    worst_intervals: tuple[tuple[float, float, float], ...] = ()
    budget_exceeded: bool = False

    def __post_init__(self) -> None:
        if self.value is not None:
            object.__setattr__(self, "value", _readonly(self.value, dtype=complex))


@dataclass(frozen=True)
class AdaptiveGK21Result:
    primary: AdaptiveGK21PassResult
    audit: AdaptiveGK21PassResult | None
    component_scales: np.ndarray
    frozen_group_scales: np.ndarray
    integration_scales: np.ndarray
    q_model: np.ndarray
    primitive_direction: np.ndarray
    transverse_direction: np.ndarray
    orbit_shift_steps: int
    orbit_origins: tuple[float, ...]
    nk: int
    epsabs: float
    epsrel: float
    audit_tolerance_factor: float
    limit: int
    norm: str
    max_unique_transverse_evaluations: int
    transverse_evaluations: int
    cache_hits: int
    point_evaluations: int
    chunk_size: int
    wall_seconds: float
    geometry_wall_seconds: float
    evaluator_wall_seconds: float
    group_names: tuple[str, ...]
    control_group_names: tuple[str, ...]
    monitor_group_names: tuple[str, ...]
    primary_audit_group_ratios: tuple[float, ...]
    observed_to_frozen_scale_ratios: tuple[float, ...]
    primitive_group_agreement_passed: bool
    success: bool
    status: int
    message: str
    failure_reason: str
    scipy_version: str
    strategy: str = "adaptive_gk21"
    quadrature: str = "gk21"
    pilot_order: int = 21
    summation_method: str = "complete_q_orbit_adaptive_gk21_shared_cache_audit"

    def __post_init__(self) -> None:
        for name in (
            "component_scales",
            "frozen_group_scales",
            "integration_scales",
            "q_model",
            "primitive_direction",
            "transverse_direction",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def value(self) -> np.ndarray | None:
        return self.audit.value if self.audit and self.audit.value is not None else self.primary.value

    @property
    def chunks(self) -> int:
        return self.transverse_evaluations

    @property
    def scaled_error_estimate(self) -> float:
        return float(self.primary.integral_error_ratio)

    @property
    def required_consecutive_levels(self) -> int:
        return 1

    @property
    def consecutive_converged_levels(self) -> int:
        return int(self.success)

    @property
    def final_transverse_order(self) -> int:
        return self.transverse_evaluations

    @property
    def max_transverse_order(self) -> int:
        return self.max_unique_transverse_evaluations

    @property
    def order_history(self) -> tuple[int, ...]:
        return (self.transverse_evaluations,)

    @property
    def convergence_ratio_history(self) -> tuple[float, ...]:
        return (
            self.primary.integral_error_ratio,
            self.audit.integral_error_ratio if self.audit else float("inf"),
        )

    @property
    def control_group_ratios(self) -> tuple[float, ...]:
        return self.primary_audit_group_ratios

    @property
    def monitor_group_ratios(self) -> tuple[float, ...]:
        return ()


def _failed_pass(
    message: str,
    *,
    unique_added: int,
    hits_added: int,
) -> AdaptiveGK21PassResult:
    return AdaptiveGK21PassResult(
        value=None,
        integral_error_estimate=float("inf"),
        integral_tolerance=float("nan"),
        integral_error_ratio=float("inf"),
        success=False,
        status=2,
        message=message,
        subinterval_count=0,
        scipy_neval=0,
        unique_evaluations_added=unique_added,
        cache_hits_added=hits_added,
        budget_exceeded=True,
    )


def _worst_intervals(info: object) -> tuple[tuple[float, float, float], ...]:
    intervals = np.asarray(getattr(info, "intervals", np.empty((0, 2))), dtype=float)
    errors = np.asarray(getattr(info, "errors", np.empty(0)), dtype=float).reshape(-1)
    if intervals.ndim != 2 or intervals.shape != (errors.size, 2):
        return ()
    indices = np.argsort(errors)[::-1][:8]
    return tuple(
        (
            float(intervals[index, 0]),
            float(intervals[index, 1]),
            float(errors[index]) / (2.0 * np.pi),
        )
        for index in indices
    )


def _run_pass(
    workspace: CompleteOrbitAggregateWorkspace,
    *,
    scales: np.ndarray,
    epsabs: float,
    epsrel: float,
    limit: int,
    norm: str,
) -> AdaptiveGK21PassResult:
    unique_before, hits_before = workspace.transverse_evaluations_unique, workspace.cache_hits
    width = int(scales.size)

    def integrand(t_value: float) -> np.ndarray:
        value = workspace.evaluate_t(t_value) / scales
        return np.concatenate((value.real, value.imag))

    try:
        packed, raw_error, info = quad_vec(
            integrand,
            -np.pi,
            np.pi,
            epsabs=float(epsabs) * (2.0 * np.pi),
            epsrel=float(epsrel),
            norm=norm,
            limit=int(limit),
            quadrature="gk21",
            full_output=True,
        )
    except TransverseEvaluationBudgetExceeded as exc:
        return _failed_pass(
            str(exc),
            unique_added=workspace.transverse_evaluations_unique - unique_before,
            hits_added=workspace.cache_hits - hits_before,
        )

    packed = np.asarray(packed, dtype=float).reshape(-1)
    if packed.shape != (2 * width,):
        raise RuntimeError(f"unexpected quad_vec shape {packed.shape}")
    scaled_average = packed / (2.0 * np.pi)
    scaled_complex = scaled_average[:width] + 1j * scaled_average[width:]
    error = float(raw_error) / (2.0 * np.pi)
    tolerance = float(epsabs) + float(epsrel) * _real_norm(scaled_average, norm)
    ratio = error / max(tolerance, np.finfo(float).tiny)
    scipy_success = bool(getattr(info, "success", False))
    success = bool(scipy_success and np.isfinite(ratio) and ratio <= 1.0)
    message = str(getattr(info, "message", ""))
    intervals = np.asarray(getattr(info, "intervals", np.empty((0, 2))))
    return AdaptiveGK21PassResult(
        value=scaled_complex * scales,
        integral_error_estimate=error,
        integral_tolerance=tolerance,
        integral_error_ratio=ratio,
        success=success,
        status=int(getattr(info, "status", -1)),
        message=message,
        subinterval_count=int(intervals.shape[0]) if intervals.ndim == 2 else 0,
        scipy_neval=int(getattr(info, "neval", 0)),
        unique_evaluations_added=workspace.transverse_evaluations_unique - unique_before,
        cache_hits_added=workspace.cache_hits - hits_before,
        worst_intervals=_worst_intervals(info),
    )


def _audit_group_ratios(
    primary: np.ndarray,
    audit: np.ndarray,
    *,
    group_ids: np.ndarray,
    scales: np.ndarray,
    epsabs: float,
    epsrel: float,
    norm: str,
) -> np.ndarray:
    ratios = np.zeros(scales.size)
    for group in range(scales.size):
        mask = group_ids == group
        scale = float(scales[group])
        left = vector_norm(primary[mask], norm) / scale
        right = vector_norm(audit[mask], norm) / scale
        delta = vector_norm(audit[mask] - primary[mask], norm) / scale
        tolerance = float(epsabs) + float(epsrel) * max(left, right)
        ratios[group] = delta / max(tolerance, np.finfo(float).tiny)
    return ratios


def integrate_commensurate_orbit_adaptive_gk21(
    evaluator: OrbitAggregateEvaluator,
    *,
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
    component_group_ids: Sequence[int] | np.ndarray | None = None,
    group_names: Sequence[str] | None = None,
    group_control_weights: Sequence[float] | np.ndarray | None = None,
) -> AdaptiveGK21Result:
    for value, name in (
        (epsabs, "epsabs"),
        (epsrel, "epsrel"),
        (scale_floor_relative, "scale_floor_relative"),
        (scale_floor_absolute, "scale_floor_absolute"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0.0 < audit_tolerance_factor < 1.0:
        raise ValueError("audit_tolerance_factor must lie between zero and one")
    if limit <= 0 or norm not in {"max", "2"}:
        raise ValueError("limit must be positive and norm must be 'max' or '2'")

    started = time.perf_counter()
    workspace = CompleteOrbitAggregateWorkspace(
        evaluator,
        nk,
        mx,
        my,
        shift_s,
        subgrid_average,
        max_unique_transverse_evaluations,
    )
    try:
        seed = np.stack(
            [workspace.evaluate_phase(0.5 * (node + 1.0)) for node in GK21_ROOT_NODES]
        )
    except TransverseEvaluationBudgetExceeded as exc:
        primary = _failed_pass(
            str(exc),
            unique_added=workspace.transverse_evaluations_unique,
            hits_added=workspace.cache_hits,
        )
        empty = np.empty(0)
        return AdaptiveGK21Result(
            primary,
            None,
            empty,
            empty,
            empty,
            workspace.q_model,
            workspace.primitive_direction,
            workspace.transverse_direction,
            workspace.orbit_shift_steps,
            workspace.orbit_origins,
            workspace.nk,
            epsabs,
            epsrel,
            audit_tolerance_factor,
            limit,
            norm,
            max_unique_transverse_evaluations,
            workspace.transverse_evaluations_unique,
            workspace.cache_hits,
            workspace.point_evaluations,
            workspace.points_per_t,
            time.perf_counter() - started,
            workspace.geometry_wall_seconds,
            workspace.evaluator_wall_seconds,
            (),
            (),
            (),
            (),
            (),
            False,
            False,
            2,
            str(exc),
            "transverse_evaluation_budget_exceeded",
            scipy.__version__,
        )

    ids, names, weights = group_layout(
        seed.shape[1],
        component_group_ids=component_group_ids,
        group_names=group_names,
        group_control_weights=group_control_weights,
    )
    frozen = group_point_scales(
        seed,
        ids,
        norm=norm,
        scale_floor_relative=scale_floor_relative,
        scale_floor_absolute=scale_floor_absolute,
    )
    component_scales = component_scales_from_groups(ids, frozen)
    integration_scales = component_scales / np.maximum(weights[ids], 1e-6)
    primary = _run_pass(
        workspace,
        scales=integration_scales,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
        norm=norm,
    )
    audit = None
    ratios = np.full(frozen.size, np.inf)
    agreement = False
    if primary.success and primary.value is not None:
        factor = float(audit_tolerance_factor)
        audit = _run_pass(
            workspace,
            scales=integration_scales,
            epsabs=factor * epsabs,
            epsrel=factor * epsrel,
            limit=limit,
            norm=norm,
        )
        if audit.success and audit.value is not None:
            ratios = _audit_group_ratios(
                primary.value,
                audit.value,
                group_ids=ids,
                scales=frozen,
                epsabs=epsabs,
                epsrel=epsrel,
                norm=norm,
            )
            agreement = bool(np.all(ratios[weights > 0.0] <= 1.0))

    observed = group_point_scales(
        workspace.cached_values,
        ids,
        norm=norm,
        scale_floor_relative=scale_floor_relative,
        scale_floor_absolute=scale_floor_absolute,
    )
    control_mask = weights > 0.0
    success = bool(primary.success and audit and audit.success and agreement)
    if success:
        status, message, reason = 0, "primary and tightened GK21 audit passed", ""
    elif primary.budget_exceeded or (audit and audit.budget_exceeded):
        status, reason = 2, "transverse_evaluation_budget_exceeded"
        message = audit.message if audit and audit.budget_exceeded else primary.message
    elif not primary.success:
        status, message, reason = 1, primary.message, "primary_adaptive_error_not_converged"
    elif not audit or not audit.success:
        status, message, reason = 1, "tightened GK21 audit did not converge", "tightened_audit_not_converged"
    else:
        status, message, reason = 1, "primitive group audit disagreement", "primitive_group_audit_disagreement"

    return AdaptiveGK21Result(
        primary,
        audit,
        component_scales,
        frozen,
        integration_scales,
        workspace.q_model,
        workspace.primitive_direction,
        workspace.transverse_direction,
        workspace.orbit_shift_steps,
        workspace.orbit_origins,
        workspace.nk,
        epsabs,
        epsrel,
        audit_tolerance_factor,
        limit,
        norm,
        max_unique_transverse_evaluations,
        workspace.transverse_evaluations_unique,
        workspace.cache_hits,
        workspace.point_evaluations,
        workspace.points_per_t,
        time.perf_counter() - started,
        workspace.geometry_wall_seconds,
        workspace.evaluator_wall_seconds,
        tuple(names),
        tuple(name for name, keep in zip(names, control_mask, strict=True) if keep),
        tuple(name for name, keep in zip(names, ~control_mask, strict=True) if keep),
        tuple(float(value) for value in ratios[control_mask]),
        tuple(float(value) for value in observed / np.maximum(frozen, np.finfo(float).tiny)),
        agreement,
        success,
        status,
        message,
        reason,
        scipy.__version__,
    )


__all__ = [
    "AdaptiveGK21PassResult",
    "AdaptiveGK21Result",
    "CompleteOrbitAggregateWorkspace",
    "GK21_ROOT_NODES",
    "TransverseEvaluationBudgetExceeded",
    "integrate_commensurate_orbit_adaptive_gk21",
]
