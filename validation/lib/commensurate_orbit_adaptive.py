"""Adaptive transverse integration on an exact commensurate q orbit.

The orbit coordinate is always evaluated on a complete equally weighted periodic
set so translation by q remains an exact index permutation.  Only the transverse
torus coordinate is selected adaptively.  Complex vector components are scaled
from a deterministic pilot rule before ``scipy.integrate.quad_vec`` is called.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Literal

import numpy as np
from scipy.integrate import quad_vec

from validation.lib.dwave_commensurate_orbit_gauss import (
    OrbitEvaluationBudgetExceeded,
    commensurate_orbit_basis,
    complementary_orbit_origins,
)

ComplexVectorEvaluator = Callable[[np.ndarray], np.ndarray]
OrbitAggregateEvaluator = Callable[[np.ndarray, np.ndarray], np.ndarray]
SubgridAverageMode = Literal["auto", "none"]


def _wrap_periodic_bz(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    return np.asarray((array + np.pi) % (2.0 * np.pi) - np.pi, dtype=float)


@dataclass(frozen=True)
class CommensurateOrbitAdaptiveResult:
    """Complete q-orbit average with adaptive transverse quadrature."""

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


def _component_scales(
    pilot_values: np.ndarray,
    pilot_weights: np.ndarray,
    *,
    scale_floor_relative: float,
    scale_floor_absolute: float,
) -> np.ndarray:
    pilot_integral = np.tensordot(
        pilot_weights / (2.0 * np.pi), pilot_values, axes=(0, 0)
    )
    max_component = np.max(np.abs(pilot_values), axis=0)
    global_scale = max(
        float(np.max(max_component)), float(np.max(np.abs(pilot_integral))), 1.0
    )
    floor = max(float(scale_floor_absolute), float(scale_floor_relative) * global_scale)
    return np.maximum.reduce(
        [
            max_component,
            np.abs(pilot_integral),
            np.full_like(max_component, floor, dtype=float),
        ]
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
) -> CommensurateOrbitAdaptiveResult:
    """Adaptive transverse integral whose evaluator consumes one complete orbit.

    This variant is intended for workspace-based response engines that need all
    orbit points and normalized weights at once.  ``evaluator`` returns one complex
    vector for the complete orbit; no nonlinear operation is performed by the
    integrator itself.
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
    )

    primitive, transverse, orbit_shift_steps = commensurate_orbit_basis(
        mx_value, my_value
    )
    origins = complementary_orbit_origins(
        orbit_shift_steps, shift_s, subgrid_average
    )
    points_per_t = nk_value * len(origins)
    if points_per_t > maximum:
        raise OrbitEvaluationBudgetExceeded(maximum, points_per_t)

    step = 2.0 * np.pi / float(nk_value)
    q_model = step * np.asarray([mx_value, my_value], dtype=float)
    cache: dict[float, np.ndarray] = {}
    expected_width: int | None = None
    point_evaluations = 0
    started = time.perf_counter()

    def aggregate(t_value: float) -> np.ndarray:
        nonlocal expected_width, point_evaluations
        key = float(t_value)
        cached = cache.get(key)
        if cached is not None:
            return cached
        attempted = point_evaluations + points_per_t
        if attempted > maximum:
            raise OrbitEvaluationBudgetExceeded(maximum, attempted)

        point_sets: list[np.ndarray] = []
        for origin in origins:
            indices = np.arange(nk_value, dtype=float)
            s_values = -np.pi + (indices + float(origin)) * step
            points = (
                s_values[:, None] * primitive[None, :]
                + key * transverse[None, :]
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

    pilot_nodes, pilot_weights = np.polynomial.legendre.leggauss(pilot)
    pilot_t = np.pi * np.asarray(pilot_nodes, dtype=float)
    pilot_w = np.pi * np.asarray(pilot_weights, dtype=float)
    pilot_values = np.stack([aggregate(float(t)) for t in pilot_t], axis=0)
    scales = _component_scales(
        pilot_values,
        pilot_w,
        scale_floor_relative=scale_floor_relative,
        scale_floor_absolute=scale_floor_absolute,
    )
    width = int(scales.size)

    def scaled_real_integrand(t_value: float) -> np.ndarray:
        scaled = aggregate(float(t_value)) / scales
        return np.concatenate((scaled.real, scaled.imag))

    scaled_integral, error, info = quad_vec(
        scaled_real_integrand,
        -np.pi,
        np.pi,
        epsabs=float(epsabs),
        epsrel=float(epsrel),
        norm=norm,
        limit=subdivision_limit,
        quadrature=quadrature,
        full_output=True,
    )
    packed = np.asarray(scaled_integral, dtype=float)
    if packed.shape != (2 * width,):
        raise RuntimeError(
            f"quad_vec returned unexpected shape {packed.shape}; expected {(2 * width,)}"
        )
    scaled_complex = packed[:width] + 1j * packed[width:]
    value = scaled_complex * scales / (2.0 * np.pi)

    return CommensurateOrbitAdaptiveResult(
        value=np.asarray(value, dtype=complex),
        component_scales=np.asarray(scales, dtype=float),
        q_model=q_model,
        primitive_direction=primitive,
        transverse_direction=transverse,
        orbit_shift_steps=orbit_shift_steps,
        orbit_origins=origins,
        nk=nk_value,
        pilot_order=pilot,
        epsabs=float(epsabs),
        epsrel=float(epsrel),
        limit=subdivision_limit,
        quadrature=str(quadrature),
        norm=str(norm),
        scaled_error_estimate=float(error) / (2.0 * np.pi),
        transverse_evaluations=len(cache),
        point_evaluations=point_evaluations,
        chunks=len(cache),
        chunk_size=points_per_t,
        wall_seconds=float(time.perf_counter() - started),
        success=bool(getattr(info, "success", False)),
        status=int(getattr(info, "status", -1)),
        message=str(getattr(info, "message", "")),
        summation_method=(
            "equal_complete_q_orbit_aggregate_with_complementary_half_step_if_needed_"
            "plus_component_scaled_scipy_quad_vec_transverse"
        ),
    )


__all__ = [
    "CommensurateOrbitAdaptiveResult",
    "integrate_commensurate_orbit_adaptive_aggregate",
]
