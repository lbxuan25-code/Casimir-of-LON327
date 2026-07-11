"""Vector-valued globally iterated adaptive integration on a rectangular domain.

The implementation deliberately uses one shared ``quad_vec`` partition for every
component of the vector integrand.  In the d-wave validation path this means all
primitive electromagnetic blocks, collective blocks, Goldstone counterterms and
Ward-RHS terms are evaluated on exactly the same adaptive nodes and weights before
one final Schur complement is formed.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.integrate import quad_vec

IntegrationOrder = Literal["xy", "yx"]
VectorIntegrand2D = Callable[[float, float], np.ndarray]


class EvaluationBudgetExceeded(RuntimeError):
    """Raised when an adaptive run exceeds its microscopic point budget."""

    def __init__(self, maximum: int, attempted: int):
        super().__init__(
            "iterated adaptive integration exceeded max_point_evaluations: "
            f"maximum={int(maximum)}, attempted={int(attempted)}"
        )
        self.maximum = int(maximum)
        self.attempted = int(attempted)


@dataclass(frozen=True)
class IteratedAdaptiveOptions:
    """Numerical controls for one nested vector-valued adaptive integral."""

    epsabs: float = 1e-7
    epsrel: float = 5e-4
    inner_limit: int = 160
    outer_limit: int = 160
    max_point_evaluations: int = 200_000
    cache_size_bytes: int = 64_000_000
    quadrature: Literal["gk15", "gk21", "trapezoid"] = "gk15"
    norm: Literal["max", "2"] = "max"
    split_points: tuple[float, ...] = (0.0,)

    def __post_init__(self) -> None:
        for name in ("epsabs", "epsrel"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "inner_limit",
            "outer_limit",
            "max_point_evaluations",
            "cache_size_bytes",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        points = np.asarray(self.split_points, dtype=float)
        if points.ndim != 1 or not np.isfinite(points).all():
            raise ValueError("split_points must be a finite one-dimensional sequence")


@dataclass(frozen=True)
class IteratedAdaptiveResult:
    """One integration-order result and its audit diagnostics."""

    value: np.ndarray
    error_estimate: float
    order: IntegrationOrder
    success: bool
    message: str
    point_evaluations: int
    outer_evaluations: int
    inner_integrals: int
    max_inner_error: float
    sum_inner_error: float
    wall_seconds: float


def _interior_points(
    points: Sequence[float], lower: float, upper: float
) -> list[float] | None:
    selected = sorted(
        {
            float(value)
            for value in points
            if float(lower) < float(value) < float(upper)
        }
    )
    return selected or None


def iterated_adaptive_integral(
    integrand: VectorIntegrand2D,
    *,
    x_bounds: tuple[float, float] = (-np.pi, np.pi),
    y_bounds: tuple[float, float] = (-np.pi, np.pi),
    order: IntegrationOrder = "xy",
    options: IteratedAdaptiveOptions | None = None,
) -> IteratedAdaptiveResult:
    """Integrate a real vector function by nested global ``quad_vec`` calls.

    ``order='xy'`` means that ``y`` is integrated first at fixed ``x`` and the
    resulting vector is then integrated over ``x``.  ``order='yx'`` reverses the
    nesting.  Running both orders is an important independent diagnostic for narrow
    ridges because the two adaptive trees are geometrically different.
    """

    opts = options or IteratedAdaptiveOptions()
    if order not in {"xy", "yx"}:
        raise ValueError("order must be 'xy' or 'yx'")

    x0, x1 = (float(x_bounds[0]), float(x_bounds[1]))
    y0, y1 = (float(y_bounds[0]), float(y_bounds[1]))
    if not all(np.isfinite(value) for value in (x0, x1, y0, y1)):
        raise ValueError("integration bounds must be finite")
    if not x0 < x1 or not y0 < y1:
        raise ValueError("integration bounds must be strictly increasing")

    if order == "xy":
        outer_bounds, inner_bounds = (x0, x1), (y0, y1)
    else:
        outer_bounds, inner_bounds = (y0, y1), (x0, x1)

    outer_points = _interior_points(opts.split_points, *outer_bounds)
    inner_points = _interior_points(opts.split_points, *inner_bounds)
    point_evaluations = 0
    inner_integrals = 0
    max_inner_error = 0.0
    sum_inner_error = 0.0
    inner_success = True
    inner_messages: list[str] = []
    expected_shape: tuple[int, ...] | None = None
    started = time.perf_counter()

    def microscopic_value(outer_value: float, inner_value: float) -> np.ndarray:
        nonlocal point_evaluations, expected_shape
        attempted = point_evaluations + 1
        if attempted > int(opts.max_point_evaluations):
            raise EvaluationBudgetExceeded(int(opts.max_point_evaluations), attempted)
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

    def outer_integrand(outer_value: float) -> np.ndarray:
        nonlocal inner_integrals, max_inner_error, sum_inner_error, inner_success

        def inner_integrand(inner_value: float) -> np.ndarray:
            return microscopic_value(float(outer_value), float(inner_value))

        value, error, info = quad_vec(
            inner_integrand,
            inner_bounds[0],
            inner_bounds[1],
            epsabs=float(opts.epsabs),
            epsrel=float(opts.epsrel),
            norm=str(opts.norm),
            cache_size=int(opts.cache_size_bytes),
            limit=int(opts.inner_limit),
            workers=1,
            points=inner_points,
            quadrature=str(opts.quadrature),
            full_output=True,
        )
        inner_integrals += 1
        error_value = float(error)
        max_inner_error = max(max_inner_error, error_value)
        sum_inner_error += error_value
        current_success = bool(getattr(info, "success", True))
        inner_success = bool(inner_success and current_success)
        if not current_success and len(inner_messages) < 8:
            inner_messages.append(str(getattr(info, "message", "inner quad_vec failed")))
        return np.asarray(value, dtype=float)

    value, error, info = quad_vec(
        outer_integrand,
        outer_bounds[0],
        outer_bounds[1],
        epsabs=float(opts.epsabs),
        epsrel=float(opts.epsrel),
        norm=str(opts.norm),
        cache_size=int(opts.cache_size_bytes),
        limit=int(opts.outer_limit),
        workers=1,
        points=outer_points,
        quadrature=str(opts.quadrature),
        full_output=True,
    )
    outer_success = bool(getattr(info, "success", True))
    success = bool(outer_success and inner_success)
    messages = [str(getattr(info, "message", ""))]
    messages.extend(inner_messages)
    message = " | ".join(item for item in messages if item)
    return IteratedAdaptiveResult(
        value=np.asarray(value, dtype=float),
        error_estimate=float(error),
        order=order,
        success=success,
        message=message,
        point_evaluations=int(point_evaluations),
        outer_evaluations=int(getattr(info, "neval", 0)),
        inner_integrals=int(inner_integrals),
        max_inner_error=float(max_inner_error),
        sum_inner_error=float(sum_inner_error),
        wall_seconds=float(time.perf_counter() - started),
    )


__all__ = [
    "EvaluationBudgetExceeded",
    "IntegrationOrder",
    "IteratedAdaptiveOptions",
    "IteratedAdaptiveResult",
    "iterated_adaptive_integral",
]
