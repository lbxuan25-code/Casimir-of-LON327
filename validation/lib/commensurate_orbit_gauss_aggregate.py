"""Fixed Gauss-Legendre transverse integration of complete commensurate q orbits.

Unlike the pointwise helper in :mod:`dwave_commensurate_orbit_gauss`, this
module calls the supplied evaluator once per complete orbit.  Workspace-based
finite-q response engines therefore see the same complete orbit, normalized
weights, contact terms, and Ward RHS as the periodic adaptive path.  Only the
transverse quadrature nodes and weights differ.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Literal

import numpy as np

from validation.lib.dwave_commensurate_orbit_gauss import (
    OrbitEvaluationBudgetExceeded,
    commensurate_orbit_basis,
    complementary_orbit_origins,
)

OrbitAggregateEvaluator = Callable[[np.ndarray, np.ndarray], np.ndarray]
SubgridAverageMode = Literal["auto", "none"]


def _wrap_periodic_bz(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    return np.asarray((array + np.pi) % (2.0 * np.pi) - np.pi, dtype=float)


def _compensated_add(
    total: np.ndarray,
    compensation: np.ndarray,
    increment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corrected = np.asarray(increment, dtype=complex) - compensation
    updated = total + corrected
    return updated, (updated - total) - corrected


@dataclass(frozen=True)
class CommensurateOrbitGaussAggregateResult:
    """Complete q-orbit aggregate followed by fixed transverse Gauss quadrature."""

    value: np.ndarray
    q_model: np.ndarray
    primitive_direction: np.ndarray
    transverse_direction: np.ndarray
    orbit_shift_steps: int
    orbit_origins: tuple[float, ...]
    nk: int
    transverse_order: int
    transverse_evaluations: int
    point_evaluations: int
    chunks: int
    chunk_size: int
    wall_seconds: float
    success: bool
    status: int
    message: str
    quadrature: str
    summation_method: str

    def __post_init__(self) -> None:
        for name in (
            "value",
            "q_model",
            "primitive_direction",
            "transverse_direction",
        ):
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def integrate_commensurate_orbit_gauss_aggregate(
    evaluator: OrbitAggregateEvaluator,
    *,
    nk: int,
    mx: int,
    my: int,
    transverse_order: int,
    shift_s: float = 0.5,
    subgrid_average: SubgridAverageMode = "auto",
    max_point_evaluations: int = 500_000,
) -> CommensurateOrbitGaussAggregateResult:
    """Integrate a complete-orbit complex vector with fixed Gauss-Legendre nodes.

    The evaluator receives all orbit points and normalized weights at once.  No
    nonlinear response operation is performed inside the integrator.
    """

    nk_value = int(nk)
    mx_value, my_value = int(mx), int(my)
    order = int(transverse_order)
    maximum = int(max_point_evaluations)
    if nk_value <= 0 or order <= 0 or maximum <= 0:
        raise ValueError("nk, transverse_order, and budget must be positive")
    if mx_value == 0 and my_value == 0:
        raise ValueError("at least one of mx,my must be nonzero")
    if abs(mx_value) > nk_value // 2 or abs(my_value) > nk_value // 2:
        raise ValueError("mx and my must lie in the principal periodic range")

    primitive, transverse, orbit_shift_steps = commensurate_orbit_basis(
        mx_value, my_value
    )
    origins = complementary_orbit_origins(
        orbit_shift_steps,
        shift_s,
        subgrid_average,
    )
    points_per_t = nk_value * len(origins)
    expected_points = order * points_per_t
    if expected_points > maximum:
        raise OrbitEvaluationBudgetExceeded(maximum, expected_points)

    step = 2.0 * np.pi / float(nk_value)
    q_model = step * np.asarray([mx_value, my_value], dtype=float)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    transverse_nodes = np.pi * np.asarray(nodes, dtype=float)
    transverse_weights = np.pi * np.asarray(weights, dtype=float)

    total: np.ndarray | None = None
    compensation: np.ndarray | None = None
    expected_width: int | None = None
    points_seen = 0
    started = time.perf_counter()

    indices = np.arange(nk_value, dtype=float)
    for t_value, t_weight in zip(
        transverse_nodes,
        transverse_weights,
        strict=True,
    ):
        point_sets: list[np.ndarray] = []
        for origin in origins:
            s_values = -np.pi + (indices + float(origin)) * step
            points = (
                s_values[:, None] * primitive[None, :]
                + float(t_value) * transverse[None, :]
            )
            point_sets.append(_wrap_periodic_bz(points))

        all_points = np.concatenate(point_sets, axis=0)
        orbit_weights = np.full(
            all_points.shape[0],
            1.0 / float(all_points.shape[0]),
            dtype=float,
        )
        value = np.asarray(evaluator(all_points, orbit_weights), dtype=complex).reshape(-1)
        if value.size == 0:
            raise ValueError("aggregate evaluator vector width must be positive")
        if not np.isfinite(value.real).all() or not np.isfinite(value.imag).all():
            raise ValueError("aggregate evaluator returned non-finite values")
        if expected_width is None:
            expected_width = int(value.size)
        elif int(value.size) != expected_width:
            raise ValueError("aggregate evaluator vector width changed between calls")

        contribution = (float(t_weight) / (2.0 * np.pi)) * value
        if total is None:
            total = np.zeros_like(contribution)
            compensation = np.zeros_like(contribution)
        assert compensation is not None
        total, compensation = _compensated_add(total, compensation, contribution)
        points_seen += int(all_points.shape[0])

    if total is None or points_seen != expected_points:
        raise RuntimeError(
            "incomplete fixed-Gauss orbit integration: "
            f"seen={points_seen}, expected={expected_points}"
        )

    return CommensurateOrbitGaussAggregateResult(
        value=np.asarray(total, dtype=complex),
        q_model=q_model,
        primitive_direction=primitive,
        transverse_direction=transverse,
        orbit_shift_steps=int(orbit_shift_steps),
        orbit_origins=origins,
        nk=nk_value,
        transverse_order=order,
        transverse_evaluations=order,
        point_evaluations=points_seen,
        chunks=order,
        chunk_size=points_per_t,
        wall_seconds=float(time.perf_counter() - started),
        success=True,
        status=0,
        message="fixed Gauss-Legendre transverse integration completed",
        quadrature="fixed_gauss_legendre",
        summation_method=(
            "equal_complete_q_orbit_aggregate_with_complementary_half_step_if_needed_"
            "plus_complex_kahan_fixed_gauss_legendre_transverse"
        ),
    )


__all__ = [
    "CommensurateOrbitGaussAggregateResult",
    "integrate_commensurate_orbit_gauss_aggregate",
]
