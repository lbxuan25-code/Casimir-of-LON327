"""Fixed Gauss-Legendre outer rule with adaptive vector inner integration.

This is a bounded-cost alternative to fully nested ``quad_vec``.  The outer
Brillouin-zone coordinate is integrated by a global Gauss-Legendre rule, while
for every outer node the orthogonal coordinate is integrated by one shared
vector-valued ``quad_vec`` rule.  All primitive response channels therefore
share the same inner adaptive nodes and weights.

The returned error estimate contains the weighted inner ``quad_vec`` errors
only.  Outer discretization must be assessed independently by changing
``outer_order`` and by comparing the ``xy`` and ``yx`` orientations.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.integrate import quad_vec

from validation.lib.iterated_adaptive import (
    EvaluationBudgetExceeded,
    IntegrationOrder,
    IteratedAdaptiveOptions,
    IteratedAdaptiveResult,
)

VectorIntegrand2D = Callable[[float, float], np.ndarray]


def _interior_points(points: tuple[float, ...], lower: float, upper: float) -> list[float] | None:
    selected = sorted({float(value) for value in points if lower < float(value) < upper})
    return selected or None


def gauss_outer_adaptive_integral(
    integrand: VectorIntegrand2D,
    *,
    order: IntegrationOrder,
    outer_order: int,
    options: IteratedAdaptiveOptions,
    x_bounds: tuple[float, float] = (-np.pi, np.pi),
    y_bounds: tuple[float, float] = (-np.pi, np.pi),
) -> IteratedAdaptiveResult:
    """Integrate a real vector function with fixed outer and adaptive inner rules."""

    if order not in {"xy", "yx"}:
        raise ValueError("order must be 'xy' or 'yx'")
    outer_count = int(outer_order)
    if outer_count <= 0:
        raise ValueError("outer_order must be positive")

    x0, x1 = map(float, x_bounds)
    y0, y1 = map(float, y_bounds)
    if not all(np.isfinite(value) for value in (x0, x1, y0, y1)):
        raise ValueError("integration bounds must be finite")
    if not x0 < x1 or not y0 < y1:
        raise ValueError("integration bounds must be strictly increasing")

    if order == "xy":
        outer_bounds, inner_bounds = (x0, x1), (y0, y1)
    else:
        outer_bounds, inner_bounds = (y0, y1), (x0, x1)

    nodes, weights = np.polynomial.legendre.leggauss(outer_count)
    half = 0.5 * (outer_bounds[1] - outer_bounds[0])
    center = 0.5 * (outer_bounds[1] + outer_bounds[0])
    outer_nodes = center + half * nodes
    outer_weights = half * weights
    inner_points = _interior_points(tuple(options.split_points), *inner_bounds)

    point_evaluations = 0
    inner_integrals = 0
    max_inner_error = 0.0
    sum_inner_error = 0.0
    weighted_inner_error = 0.0
    expected_shape: tuple[int, ...] | None = None
    total: np.ndarray | None = None
    success = True
    messages: list[str] = []
    started = time.perf_counter()

    def microscopic_value(outer_value: float, inner_value: float) -> np.ndarray:
        nonlocal point_evaluations, expected_shape
        attempted = point_evaluations + 1
        if attempted > int(options.max_point_evaluations):
            raise EvaluationBudgetExceeded(int(options.max_point_evaluations), attempted)
        point_evaluations = attempted
        if order == "xy":
            value = np.asarray(integrand(float(outer_value), float(inner_value)), dtype=float)
        else:
            value = np.asarray(integrand(float(inner_value), float(outer_value)), dtype=float)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError("adaptive integrand must return a finite one-dimensional real vector")
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise ValueError(
                "adaptive integrand changed vector shape: "
                f"expected {expected_shape}, got {value.shape}"
            )
        return value

    for outer_value, outer_weight in zip(outer_nodes, outer_weights, strict=True):
        def inner_integrand(inner_value: float) -> np.ndarray:
            return microscopic_value(float(outer_value), float(inner_value))

        value, error, info = quad_vec(
            inner_integrand,
            inner_bounds[0],
            inner_bounds[1],
            epsabs=float(options.epsabs),
            epsrel=float(options.epsrel),
            norm=str(options.norm),
            cache_size=int(options.cache_size_bytes),
            limit=int(options.inner_limit),
            workers=1,
            points=inner_points,
            quadrature=str(options.quadrature),
            full_output=True,
        )
        inner_integrals += 1
        error_value = float(error)
        max_inner_error = max(max_inner_error, error_value)
        sum_inner_error += error_value
        weighted_inner_error += abs(float(outer_weight)) * error_value
        current_success = bool(getattr(info, "success", True))
        success = bool(success and current_success)
        if not current_success and len(messages) < 8:
            messages.append(str(getattr(info, "message", "inner quad_vec failed")))
        contribution = float(outer_weight) * np.asarray(value, dtype=float)
        total = contribution.copy() if total is None else total + contribution

    if total is None:
        raise RuntimeError("fixed-outer integration produced no outer nodes")
    messages.insert(
        0,
        "fixed Gauss-Legendre outer rule; reported error excludes outer discretization",
    )
    return IteratedAdaptiveResult(
        value=np.asarray(total, dtype=float),
        error_estimate=float(weighted_inner_error),
        order=order,
        success=bool(success),
        message=" | ".join(item for item in messages if item),
        point_evaluations=int(point_evaluations),
        outer_evaluations=int(outer_count),
        inner_integrals=int(inner_integrals),
        max_inner_error=float(max_inner_error),
        sum_inner_error=float(sum_inner_error),
        wall_seconds=float(time.perf_counter() - started),
    )


__all__ = ["gauss_outer_adaptive_integral"]
