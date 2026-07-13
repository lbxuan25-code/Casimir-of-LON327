"""Fixed or composite Gauss-Legendre integration of complete commensurate q orbits.

The supplied evaluator is called once per transverse node with one complete exact
commensurate q orbit.  ``transverse_order`` always means the total number of
transverse nodes.  ``panel_count=1`` preserves the historical global Gauss rule;
``panel_count>1`` partitions one full interval of length ``2*pi`` into equal panels
and applies an independent Gauss-Legendre rule on each panel.

Only periodicity is used.  Moving ``integration_start`` is an exact change of the
periodic cut, not an even, C4, axis/diagonal, or q-direction symmetry reduction.
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


def _composite_gauss_rule(
    *,
    total_order: int,
    panel_count: int,
    integration_start: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return one full-period equal-panel Gauss rule.

    The result contains exactly ``total_order`` nodes.  Panel boundaries are not
    sampled, so adjacent panels do not create duplicated transverse evaluations.
    """

    order = int(total_order)
    panels = int(panel_count)
    start = float(integration_start)
    if order <= 0 or panels <= 0:
        raise ValueError("total_order and panel_count must be positive")
    if order % panels != 0:
        raise ValueError("transverse_order must be divisible by panel_count")
    if not np.isfinite(start):
        raise ValueError("integration_start must be finite")

    panel_order = order // panels
    local_nodes, local_weights = np.polynomial.legendre.leggauss(panel_order)
    boundaries = start + np.linspace(0.0, 2.0 * np.pi, panels + 1)

    node_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        midpoint = 0.5 * (float(left) + float(right))
        half_width = 0.5 * (float(right) - float(left))
        node_blocks.append(midpoint + half_width * local_nodes)
        weight_blocks.append(half_width * local_weights)

    nodes = np.concatenate(node_blocks)
    weights = np.concatenate(weight_blocks)
    if nodes.size != order or weights.size != order:
        raise RuntimeError("internal composite Gauss rule size mismatch")
    return nodes, weights, panel_order


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
    panel_count: int
    panel_order: int
    integration_start: float
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
    full_transverse_period_integrated: bool = True
    symmetry_reduction_applied: bool = False
    q_direction_special_case: bool = False

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
    panel_count: int = 1,
    integration_start: float = -np.pi,
    shift_s: float = 0.5,
    subgrid_average: SubgridAverageMode = "auto",
    max_point_evaluations: int = 500_000,
) -> CommensurateOrbitGaussAggregateResult:
    """Integrate a complete-orbit complex vector with a full-period Gauss rule.

    ``transverse_order`` is the total node count across all panels.  No nonlinear
    response operation is performed inside the integrator.
    """

    nk_value = int(nk)
    mx_value, my_value = int(mx), int(my)
    order = int(transverse_order)
    panels = int(panel_count)
    start = float(integration_start)
    maximum = int(max_point_evaluations)
    if nk_value <= 0 or order <= 0 or panels <= 0 or maximum <= 0:
        raise ValueError(
            "nk, transverse_order, panel_count, and budget must be positive"
        )
    if order % panels != 0:
        raise ValueError("transverse_order must be divisible by panel_count")
    if not np.isfinite(start):
        raise ValueError("integration_start must be finite")
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
    transverse_nodes, transverse_weights, panel_order = _composite_gauss_rule(
        total_order=order,
        panel_count=panels,
        integration_start=start,
    )

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

    quadrature = (
        "fixed_gauss_legendre"
        if panels == 1
        else "composite_fixed_gauss_legendre"
    )
    summation = (
        "equal_complete_q_orbit_aggregate_with_complementary_half_step_if_needed_"
        "plus_complex_kahan_"
        + (
            "global_fixed_gauss_legendre_transverse"
            if panels == 1
            else "equal_panel_composite_fixed_gauss_legendre_transverse"
        )
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
        panel_count=panels,
        panel_order=panel_order,
        integration_start=start,
        transverse_evaluations=order,
        point_evaluations=points_seen,
        chunks=order,
        chunk_size=points_per_t,
        wall_seconds=float(time.perf_counter() - started),
        success=True,
        status=0,
        message=(
            "global fixed Gauss-Legendre transverse integration completed"
            if panels == 1
            else "composite fixed Gauss-Legendre transverse integration completed"
        ),
        quadrature=quadrature,
        summation_method=summation,
    )


__all__ = [
    "CommensurateOrbitGaussAggregateResult",
    "integrate_commensurate_orbit_gauss_aggregate",
]
