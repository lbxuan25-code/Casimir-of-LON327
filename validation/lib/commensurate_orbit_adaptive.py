"""Adaptive transverse integration on an exact commensurate q orbit.

Translation by q is kept as an exact permutation by evaluating a complete,
equally weighted orbit at every transverse coordinate.  The default transverse
integrator is a nested periodic trapezoid sequence.  This is natural for the
torus coordinate, reuses every previously evaluated orbit, and controls errors
on physically meaningful blocks instead of normalizing every near-zero scalar
component independently.  The older local ``quad_vec`` route remains available
as a diagnostic backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.integrate import quad_vec

from validation.lib.dwave_commensurate_orbit_gauss import (
    OrbitEvaluationBudgetExceeded,
    commensurate_orbit_basis,
    complementary_orbit_origins,
)

OrbitAggregateEvaluator = Callable[[np.ndarray, np.ndarray], np.ndarray]
SubgridAverageMode = Literal["auto", "none"]
AdaptiveStrategy = Literal["periodic_nested", "quad_vec"]


def _wrap_periodic_bz(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    return np.asarray((array + np.pi) % (2.0 * np.pi) - np.pi, dtype=float)


def _vector_norm(values: np.ndarray, norm: str) -> float:
    array = np.asarray(values, dtype=complex).reshape(-1)
    if array.size == 0:
        return 0.0
    if norm == "max":
        return float(np.max(np.abs(array)))
    if norm == "2":
        return float(np.linalg.norm(array))
    raise ValueError("norm must be 'max' or '2'")


def _compensated_mean(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=complex)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("values must have shape (n,width) with n>0")
    total = np.zeros(array.shape[1], dtype=complex)
    compensation = np.zeros_like(total)
    for row in array:
        corrected = row - compensation
        updated = total + corrected
        compensation = (updated - total) - corrected
        total = updated
    return total / float(array.shape[0])


@dataclass(frozen=True)
class CommensurateOrbitAdaptiveResult:
    """Complete q-orbit average with an iterative transverse quadrature."""

    value: np.ndarray
    component_scales: np.ndarray
    q_model: np.ndarray
    primitive_direction: np.ndarray
    transverse_direction: np.ndarray
    orbit_shift_steps: int
    orbit_origins: tuple[float, ...]
    nk: int
    pilot_order: int
    epsabs: float
    epsrel: float
    limit: int
    quadrature: str
    norm: str
    scaled_error_estimate: float
    transverse_evaluations: int
    point_evaluations: int
    chunks: int
    chunk_size: int
    wall_seconds: float
    success: bool
    status: int
    message: str
    summation_method: str
    strategy: str = "periodic_nested"
    final_transverse_order: int = 0
    max_transverse_order: int = 0
    required_consecutive_levels: int = 1
    consecutive_converged_levels: int = 0
    order_history: tuple[int, ...] = ()
    convergence_ratio_history: tuple[float, ...] = ()
    control_group_names: tuple[str, ...] = ()
    control_group_ratios: tuple[float, ...] = ()
    monitor_group_names: tuple[str, ...] = ()
    monitor_group_ratios: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "value",
            "component_scales",
            "q_model",
            "primitive_direction",
            "transverse_direction",
        ):
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def _validate_common(
    *,
    nk: int,
    mx: int,
    my: int,
    maximum: int,
    pilot: int,
    subdivision_limit: int,
    epsabs: float,
    epsrel: float,
    quadrature: str,
    norm: str,
    scale_floor_relative: float,
    scale_floor_absolute: float,
    strategy: str,
    max_transverse_order: int,
    required_consecutive_levels: int,
) -> None:
    if nk <= 0 or maximum <= 0 or pilot <= 0 or subdivision_limit <= 0:
        raise ValueError("nk, budget, pilot_order, and limit must be positive")
    if mx == 0 and my == 0:
        raise ValueError("at least one of mx,my must be nonzero")
    if abs(mx) > nk // 2 or abs(my) > nk // 2:
        raise ValueError("mx and my must lie in the principal periodic range")
    for value, name in (
        (epsabs, "epsabs"),
        (epsrel, "epsrel"),
        (scale_floor_relative, "scale_floor_relative"),
        (scale_floor_absolute, "scale_floor_absolute"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if quadrature not in {"gk15", "gk21", "trapezoid"}:
        raise ValueError("quadrature must be 'gk15', 'gk21', or 'trapezoid'")
    if norm not in {"max", "2"}:
        raise ValueError("norm must be 'max' or '2'")
    if strategy not in {"periodic_nested", "quad_vec"}:
        raise ValueError("strategy must be 'periodic_nested' or 'quad_vec'")
    if max_transverse_order < pilot:
        raise ValueError("max_transverse_order must be at least pilot_order")
    if required_consecutive_levels <= 0:
        raise ValueError("required_consecutive_levels must be positive")


def _group_layout(
    width: int,
    *,
    component_group_ids: Sequence[int] | np.ndarray | None,
    group_names: Sequence[str] | None,
    group_control_weights: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    if component_group_ids is None and width >= 43 and (width - 18) % 25 == 0:
        # Canonical positive-d-wave primitive payload:
        #   header = EM direct(9), collective static(6), Ward RHS(3)
        #   per xi = EM bubble(9), collective bubble(4), mixed left/right(12).
        # Near-zero scalar entries share their parent block scale.  The Ward RHS is
        # integrated on the same nodes but does not independently drive refinement;
        # the final Ward gate remains authoritative.
        n_frequencies = (width - 18) // 25
        ids = np.empty(width, dtype=int)
        inferred_names: list[str] = []
        inferred_weights: list[float] = []
        group = 0
        ids[0:9] = group
        inferred_names.append("em_direct")
        inferred_weights.append(1.0)
        group += 1
        ids[9:15] = group
        inferred_names.append("collective_static")
        inferred_weights.append(1.0)
        group += 1
        ids[15:18] = group
        inferred_names.append("ward_rhs_monitor")
        inferred_weights.append(0.0)
        group += 1
        offset = 18
        for frequency in range(n_frequencies):
            ids[offset : offset + 9] = group
            inferred_names.append(f"xi_{frequency}_em")
            inferred_weights.append(1.0)
            group += 1
            offset += 9
            ids[offset : offset + 4] = group
            inferred_names.append(f"xi_{frequency}_collective")
            inferred_weights.append(1.0)
            group += 1
            offset += 4
            ids[offset : offset + 12] = group
            inferred_names.append(f"xi_{frequency}_mixed")
            inferred_weights.append(1.0)
            group += 1
            offset += 12
        if offset != width:
            raise RuntimeError("internal positive-d-wave block layout mismatch")
        if group_names is None:
            group_names = tuple(inferred_names)
        if group_control_weights is None:
            group_control_weights = np.asarray(inferred_weights, dtype=float)
    elif component_group_ids is None:
        ids = np.arange(width, dtype=int)
    else:
        ids = np.asarray(component_group_ids, dtype=int).reshape(-1)
        if ids.size != width:
            raise ValueError(
                f"component_group_ids has width {ids.size}; expected {width}"
            )
        if np.any(ids < 0):
            raise ValueError("component_group_ids must be non-negative")
    unique = np.unique(ids)
    if not np.array_equal(unique, np.arange(unique.size, dtype=int)):
        raise ValueError("component_group_ids must form contiguous ids 0..n_groups-1")
    n_groups = int(unique.size)

    if group_names is None:
        names = tuple(f"component_{index}" for index in range(n_groups))
    else:
        names = tuple(str(value) for value in group_names)
        if len(names) != n_groups:
            raise ValueError(f"group_names has length {len(names)}; expected {n_groups}")
        if len(set(names)) != len(names):
            raise ValueError("group_names must be unique")

    if group_control_weights is None:
        weights = np.ones(n_groups, dtype=float)
    else:
        weights = np.asarray(group_control_weights, dtype=float).reshape(-1)
        if weights.size != n_groups:
            raise ValueError(
                f"group_control_weights has length {weights.size}; expected {n_groups}"
            )
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("group_control_weights must be finite and non-negative")
    if not np.any(weights > 0.0):
        raise ValueError("at least one control group must have positive weight")
    return ids, names, weights


def _group_point_scales(
    sampled_values: np.ndarray,
    group_ids: np.ndarray,
    *,
    norm: str,
    scale_floor_relative: float,
    scale_floor_absolute: float,
) -> np.ndarray:
    values = np.asarray(sampled_values, dtype=complex)
    n_groups = int(np.max(group_ids)) + 1
    raw = np.zeros(n_groups, dtype=float)
    for group in range(n_groups):
        mask = group_ids == group
        raw[group] = max(_vector_norm(row[mask], norm) for row in values)
    global_scale = max(float(np.max(raw)), 1.0)
    floor = max(float(scale_floor_absolute), float(scale_floor_relative) * global_scale)
    return np.maximum(raw, floor)


def _component_scales_from_groups(
    group_ids: np.ndarray,
    group_scales: np.ndarray,
) -> np.ndarray:
    return np.asarray(group_scales[group_ids], dtype=float)


def _group_convergence_ratios(
    current: np.ndarray,
    previous: np.ndarray,
    *,
    group_ids: np.ndarray,
    group_scales: np.ndarray,
    group_control_weights: np.ndarray,
    epsabs: float,
    epsrel: float,
    norm: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_groups = int(group_scales.size)
    raw_ratios = np.zeros(n_groups, dtype=float)
    weighted_ratios = np.zeros(n_groups, dtype=float)
    for group in range(n_groups):
        mask = group_ids == group
        scale = float(group_scales[group])
        current_scaled = _vector_norm(current[mask], norm) / scale
        previous_scaled = _vector_norm(previous[mask], norm) / scale
        delta_scaled = _vector_norm(current[mask] - previous[mask], norm) / scale
        threshold = float(epsabs) + float(epsrel) * max(
            current_scaled, previous_scaled
        )
        ratio = delta_scaled / max(threshold, np.finfo(float).tiny)
        raw_ratios[group] = ratio
        weighted_ratios[group] = float(group_control_weights[group]) * ratio
    return raw_ratios, weighted_ratios


def _build_orbit_aggregate(
    evaluator: OrbitAggregateEvaluator,
    *,
    nk: int,
    mx: int,
    my: int,
    shift_s: float,
    subgrid_average: SubgridAverageMode,
    maximum: int,
):
    primitive, transverse, orbit_shift_steps = commensurate_orbit_basis(mx, my)
    origins = complementary_orbit_origins(orbit_shift_steps, shift_s, subgrid_average)
    points_per_t = int(nk) * len(origins)
    if points_per_t > maximum:
        raise OrbitEvaluationBudgetExceeded(maximum, points_per_t)
    step = 2.0 * np.pi / float(nk)
    q_model = step * np.asarray([mx, my], dtype=float)
    cache: dict[float, np.ndarray] = {}
    expected_width: int | None = None
    point_evaluations = 0

    def aggregate_phase(phase: float) -> np.ndarray:
        nonlocal expected_width, point_evaluations
        canonical_phase = float(phase % 1.0)
        key = round(canonical_phase, 15)
        cached = cache.get(key)
        if cached is not None:
            return cached
        attempted = point_evaluations + points_per_t
        if attempted > maximum:
            raise OrbitEvaluationBudgetExceeded(maximum, attempted)

        t_value = -np.pi + 2.0 * np.pi * canonical_phase
        point_sets: list[np.ndarray] = []
        for origin in origins:
            indices = np.arange(nk, dtype=float)
            s_values = -np.pi + (indices + float(origin)) * step
            points = (
                s_values[:, None] * primitive[None, :]
                + float(t_value) * transverse[None, :]
            )
            point_sets.append(_wrap_periodic_bz(points))
        all_points = np.concatenate(point_sets, axis=0)
        weights = np.full(all_points.shape[0], 1.0 / float(all_points.shape[0]))
        value = np.asarray(evaluator(all_points, weights), dtype=complex).reshape(-1)
        if value.size == 0:
            raise ValueError("aggregate evaluator vector width must be positive")
        if not np.isfinite(value.real).all() or not np.isfinite(value.imag).all():
            raise ValueError("aggregate evaluator returned non-finite values")
        if expected_width is None:
            expected_width = int(value.size)
        elif int(value.size) != expected_width:
            raise ValueError("aggregate evaluator vector width changed between calls")
        point_evaluations = attempted
        cache[key] = value
        return value

    return (
        aggregate_phase,
        cache,
        lambda: point_evaluations,
        points_per_t,
        q_model,
        primitive,
        transverse,
        orbit_shift_steps,
        origins,
    )


def _integrate_periodic_nested(
    evaluator: OrbitAggregateEvaluator,
    *,
    nk: int,
    mx: int,
    my: int,
    shift_s: float,
    subgrid_average: SubgridAverageMode,
    maximum: int,
    pilot: int,
    epsabs: float,
    epsrel: float,
    limit: int,
    norm: str,
    scale_floor_relative: float,
    scale_floor_absolute: float,
    max_transverse_order: int,
    required_consecutive_levels: int,
    transverse_shift_fraction: float | None,
    component_group_ids: Sequence[int] | np.ndarray | None,
    group_names: Sequence[str] | None,
    group_control_weights: Sequence[float] | np.ndarray | None,
) -> CommensurateOrbitAdaptiveResult:
    started = time.perf_counter()
    (
        aggregate_phase,
        cache,
        point_counter,
        points_per_t,
        q_model,
        primitive,
        transverse,
        orbit_shift_steps,
        origins,
    ) = _build_orbit_aggregate(
        evaluator,
        nk=nk,
        mx=mx,
        my=my,
        shift_s=shift_s,
        subgrid_average=subgrid_average,
        maximum=maximum,
    )

    initial = int(pilot)
    max_order_by_budget = maximum // points_per_t
    effective_max_order = min(int(max_transverse_order), int(max_order_by_budget))
    if initial > effective_max_order:
        raise OrbitEvaluationBudgetExceeded(maximum, initial * points_per_t)
    shift_fraction = (
        0.5 / float(initial)
        if transverse_shift_fraction is None
        else float(transverse_shift_fraction) % 1.0
    )

    order_history: list[int] = []
    convergence_history: list[float] = []
    previous: np.ndarray | None = None
    current: np.ndarray | None = None
    group_ids: np.ndarray | None = None
    names: tuple[str, ...] = ()
    control_weights: np.ndarray | None = None
    group_scales: np.ndarray | None = None
    final_raw_ratios: np.ndarray | None = None
    final_weighted_ratios: np.ndarray | None = None
    consecutive = 0
    success = False
    status = 1
    message = "maximum transverse order reached before convergence"
    order = initial

    while order <= effective_max_order:
        phases = (shift_fraction + np.arange(order, dtype=float) / float(order)) % 1.0
        values = np.stack([aggregate_phase(float(phase)) for phase in phases], axis=0)
        current = _compensated_mean(values)
        order_history.append(order)

        if group_ids is None:
            group_ids, names, control_weights = _group_layout(
                int(current.size),
                component_group_ids=component_group_ids,
                group_names=group_names,
                group_control_weights=group_control_weights,
            )
        assert control_weights is not None
        candidate_scales = _group_point_scales(
            np.stack(tuple(cache.values()), axis=0),
            group_ids,
            norm=norm,
            scale_floor_relative=scale_floor_relative,
            scale_floor_absolute=scale_floor_absolute,
        )
        group_scales = (
            candidate_scales
            if group_scales is None
            else np.maximum(group_scales, candidate_scales)
        )

        if previous is None:
            convergence_history.append(float("nan"))
        else:
            final_raw_ratios, final_weighted_ratios = _group_convergence_ratios(
                current,
                previous,
                group_ids=group_ids,
                group_scales=group_scales,
                group_control_weights=control_weights,
                epsabs=epsabs,
                epsrel=epsrel,
                norm=norm,
            )
            active = control_weights > 0.0
            max_ratio = float(np.max(final_weighted_ratios[active]))
            convergence_history.append(max_ratio)
            if max_ratio <= 1.0:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= int(required_consecutive_levels):
                success = True
                status = 0
                message = (
                    "nested periodic transverse integral converged for "
                    f"{consecutive} consecutive refinements"
                )
                break
        previous = current
        next_order = 2 * order
        if next_order > effective_max_order:
            if next_order > max_order_by_budget:
                status = 2
                message = (
                    "next complete nested transverse level would exceed "
                    "max_point_evaluations; returning the last complete level"
                )
            break
        order = next_order

    if current is None or group_ids is None or control_weights is None or group_scales is None:
        raise RuntimeError("periodic nested integration produced no complete level")
    if final_raw_ratios is None:
        final_raw_ratios = np.full(group_scales.size, np.inf, dtype=float)
        final_weighted_ratios = control_weights * final_raw_ratios

    control_mask = control_weights > 0.0
    monitor_mask = ~control_mask
    component_scales = _component_scales_from_groups(group_ids, group_scales)
    finite_history = [value for value in convergence_history if np.isfinite(value)]
    scaled_error = float(finite_history[-1]) if finite_history else float("inf")

    return CommensurateOrbitAdaptiveResult(
        value=np.asarray(current, dtype=complex),
        component_scales=component_scales,
        q_model=q_model,
        primitive_direction=primitive,
        transverse_direction=transverse,
        orbit_shift_steps=orbit_shift_steps,
        orbit_origins=origins,
        nk=int(nk),
        pilot_order=initial,
        epsabs=float(epsabs),
        epsrel=float(epsrel),
        limit=int(limit),
        quadrature="periodic_trapezoid",
        norm=str(norm),
        scaled_error_estimate=scaled_error,
        transverse_evaluations=len(cache),
        point_evaluations=int(point_counter()),
        chunks=len(cache),
        chunk_size=points_per_t,
        wall_seconds=float(time.perf_counter() - started),
        success=success,
        status=status,
        message=message,
        summation_method=(
            "equal_complete_q_orbit_aggregate_with_complementary_half_step_if_needed_"
            "plus_nested_shifted_periodic_trapezoid_with_node_reuse_and_block_control"
        ),
        strategy="periodic_nested",
        final_transverse_order=int(order_history[-1]),
        max_transverse_order=int(effective_max_order),
        required_consecutive_levels=int(required_consecutive_levels),
        consecutive_converged_levels=int(consecutive),
        order_history=tuple(order_history),
        convergence_ratio_history=tuple(float(value) for value in convergence_history),
        control_group_names=tuple(name for name, keep in zip(names, control_mask, strict=True) if keep),
        control_group_ratios=tuple(
            float(value)
            for value, keep in zip(final_weighted_ratios, control_mask, strict=True)
            if keep
        ),
        monitor_group_names=tuple(name for name, keep in zip(names, monitor_mask, strict=True) if keep),
        monitor_group_ratios=tuple(
            float(value)
            for value, keep in zip(final_raw_ratios, monitor_mask, strict=True)
            if keep
        ),
    )


def _integrate_quad_vec(
    evaluator: OrbitAggregateEvaluator,
    *,
    nk: int,
    mx: int,
    my: int,
    shift_s: float,
    subgrid_average: SubgridAverageMode,
    maximum: int,
    pilot: int,
    epsabs: float,
    epsrel: float,
    limit: int,
    quadrature: str,
    norm: str,
    scale_floor_relative: float,
    scale_floor_absolute: float,
    component_group_ids: Sequence[int] | np.ndarray | None,
    group_names: Sequence[str] | None,
    group_control_weights: Sequence[float] | np.ndarray | None,
) -> CommensurateOrbitAdaptiveResult:
    started = time.perf_counter()
    (
        aggregate_phase,
        cache,
        point_counter,
        points_per_t,
        q_model,
        primitive,
        transverse,
        orbit_shift_steps,
        origins,
    ) = _build_orbit_aggregate(
        evaluator,
        nk=nk,
        mx=mx,
        my=my,
        shift_s=shift_s,
        subgrid_average=subgrid_average,
        maximum=maximum,
    )

    pilot_nodes, _pilot_weights = np.polynomial.legendre.leggauss(int(pilot))
    pilot_phases = 0.5 * (np.asarray(pilot_nodes, dtype=float) + 1.0)
    pilot_values = np.stack([aggregate_phase(float(value)) for value in pilot_phases], axis=0)
    group_ids, names, control_weights = _group_layout(
        int(pilot_values.shape[1]),
        component_group_ids=component_group_ids,
        group_names=group_names,
        group_control_weights=group_control_weights,
    )
    group_scales = _group_point_scales(
        pilot_values,
        group_ids,
        norm=norm,
        scale_floor_relative=scale_floor_relative,
        scale_floor_absolute=scale_floor_absolute,
    )
    physical_scales = _component_scales_from_groups(group_ids, group_scales)
    effective_weights = np.maximum(control_weights[group_ids], 1e-6)
    integration_scales = physical_scales / effective_weights
    width = int(integration_scales.size)

    def scaled_real_integrand(t_value: float) -> np.ndarray:
        phase = (float(t_value) + np.pi) / (2.0 * np.pi)
        scaled = aggregate_phase(phase) / integration_scales
        return np.concatenate((scaled.real, scaled.imag))

    scaled_integral, error, info = quad_vec(
        scaled_real_integrand,
        -np.pi,
        np.pi,
        epsabs=float(epsabs),
        epsrel=float(epsrel),
        norm=norm,
        limit=int(limit),
        quadrature=quadrature,
        full_output=True,
    )
    packed = np.asarray(scaled_integral, dtype=float)
    if packed.shape != (2 * width,):
        raise RuntimeError(
            f"quad_vec returned unexpected shape {packed.shape}; expected {(2 * width,)}"
        )
    scaled_complex = packed[:width] + 1j * packed[width:]
    value = scaled_complex * integration_scales / (2.0 * np.pi)
    success = bool(getattr(info, "success", False))
    status = int(getattr(info, "status", -1))
    message = str(getattr(info, "message", ""))

    return CommensurateOrbitAdaptiveResult(
        value=np.asarray(value, dtype=complex),
        component_scales=physical_scales,
        q_model=q_model,
        primitive_direction=primitive,
        transverse_direction=transverse,
        orbit_shift_steps=orbit_shift_steps,
        orbit_origins=origins,
        nk=int(nk),
        pilot_order=int(pilot),
        epsabs=float(epsabs),
        epsrel=float(epsrel),
        limit=int(limit),
        quadrature=str(quadrature),
        norm=str(norm),
        scaled_error_estimate=float(error) / (2.0 * np.pi),
        transverse_evaluations=len(cache),
        point_evaluations=int(point_counter()),
        chunks=len(cache),
        chunk_size=points_per_t,
        wall_seconds=float(time.perf_counter() - started),
        success=success,
        status=status,
        message=message,
        summation_method=(
            "equal_complete_q_orbit_aggregate_with_complementary_half_step_if_needed_"
            "plus_block_scaled_scipy_quad_vec_transverse"
        ),
        strategy="quad_vec",
        final_transverse_order=len(cache),
        max_transverse_order=len(cache),
        required_consecutive_levels=1,
        consecutive_converged_levels=int(success),
        order_history=(len(cache),),
        convergence_ratio_history=(float(error) / (2.0 * np.pi),),
        control_group_names=tuple(
            name for name, weight in zip(names, control_weights, strict=True) if weight > 0.0
        ),
        control_group_ratios=(),
        monitor_group_names=tuple(
            name for name, weight in zip(names, control_weights, strict=True) if weight == 0.0
        ),
        monitor_group_ratios=(),
    )


def integrate_commensurate_orbit_adaptive_aggregate(
    evaluator: OrbitAggregateEvaluator,
    *,
    nk: int,
    mx: int,
    my: int,
    shift_s: float = 0.5,
    subgrid_average: SubgridAverageMode = "auto",
    max_point_evaluations: int = 500_000,
    pilot_order: int = 16,
    epsabs: float = 2e-5,
    epsrel: float = 2e-3,
    limit: int = 60,
    quadrature: str = "gk15",
    norm: str = "max",
    scale_floor_relative: float = 1e-8,
    scale_floor_absolute: float = 1e-14,
    strategy: AdaptiveStrategy = "periodic_nested",
    max_transverse_order: int = 512,
    required_consecutive_levels: int = 2,
    transverse_shift_fraction: float | None = None,
    component_group_ids: Sequence[int] | np.ndarray | None = None,
    group_names: Sequence[str] | None = None,
    group_control_weights: Sequence[float] | np.ndarray | None = None,
) -> CommensurateOrbitAdaptiveResult:
    """Integrate one complex primitive vector on exact orbits.

    ``periodic_nested`` is the default because the transverse variable is a torus
    coordinate.  Orders double from ``pilot_order`` and reuse all old nodes.  Error
    control is blockwise: ``component_group_ids`` assigns scalar components to
    physical groups, while zero-weight groups are integrated on the same nodes but
    do not independently trigger refinement.  ``quad_vec`` is retained for direct
    diagnostic comparisons.
    """

    nk_value = int(nk)
    mx_value, my_value = int(mx), int(my)
    maximum = int(max_point_evaluations)
    pilot = int(pilot_order)
    subdivision_limit = int(limit)
    _validate_common(
        nk=nk_value,
        mx=mx_value,
        my=my_value,
        maximum=maximum,
        pilot=pilot,
        subdivision_limit=subdivision_limit,
        epsabs=epsabs,
        epsrel=epsrel,
        quadrature=quadrature,
        norm=norm,
        scale_floor_relative=scale_floor_relative,
        scale_floor_absolute=scale_floor_absolute,
        strategy=strategy,
        max_transverse_order=int(max_transverse_order),
        required_consecutive_levels=int(required_consecutive_levels),
    )

    common = dict(
        evaluator=evaluator,
        nk=nk_value,
        mx=mx_value,
        my=my_value,
        shift_s=float(shift_s),
        subgrid_average=subgrid_average,
        maximum=maximum,
        pilot=pilot,
        epsabs=float(epsabs),
        epsrel=float(epsrel),
        limit=subdivision_limit,
        norm=str(norm),
        scale_floor_relative=float(scale_floor_relative),
        scale_floor_absolute=float(scale_floor_absolute),
        component_group_ids=component_group_ids,
        group_names=group_names,
        group_control_weights=group_control_weights,
    )
    if strategy == "periodic_nested":
        return _integrate_periodic_nested(
            **common,
            max_transverse_order=int(max_transverse_order),
            required_consecutive_levels=int(required_consecutive_levels),
            transverse_shift_fraction=transverse_shift_fraction,
        )
    return _integrate_quad_vec(
        **common,
        quadrature=str(quadrature),
    )


__all__ = [
    "CommensurateOrbitAdaptiveResult",
    "integrate_commensurate_orbit_adaptive_aggregate",
]
