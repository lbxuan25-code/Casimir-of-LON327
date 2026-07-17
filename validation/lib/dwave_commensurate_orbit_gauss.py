"""One-dimensional commensurate q-orbit with transverse Gauss quadrature.

For ``q = (2 pi / nk) (mx, my)``, an integer unimodular change of torus
coordinates is chosen so that translation by q acts only on the periodic orbit
coordinate.  A complete equally weighted orbit therefore preserves the finite-q
translation permutation exactly, while the transverse coordinate is integrated
with a deterministic Gauss-Legendre rule.

When the reduced orbit shift is odd, two complementary half-step orbit grids are
averaged before any nonlinear response operation.  This keeps the ``k +/- q/2``
endpoint sampling symmetric without paying for a complete ``nk x nk`` tensor grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Literal

import numpy as np

ComplexVectorEvaluator = Callable[[np.ndarray], np.ndarray]
SubgridAverageMode = Literal["auto", "none"]


class OrbitEvaluationBudgetExceeded(RuntimeError):
    """Raised when the microscopic orbit evaluation budget is exceeded."""

    def __init__(self, maximum: int, attempted: int):
        super().__init__(
            "commensurate-orbit Gauss integration exceeded max_point_evaluations: "
            f"maximum={int(maximum)}, attempted={int(attempted)}"
        )
        self.maximum = int(maximum)
        self.attempted = int(attempted)


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return ``(g,x,y)`` with positive ``g`` and ``a*x+b*y=g``."""

    old_r, r = int(a), int(b)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return int(old_r), int(old_s), int(old_t)


def commensurate_orbit_basis(
    mx: int,
    my: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a minimal-shear unimodular basis aligned with integer q.

    The first vector is the primitive integer direction ``p=(mx,my)/g``.  The
    second integer vector ``n`` is chosen so that ``det[p,n]=1``.  Translation by
    ``q`` is then a shift of ``g`` uniform orbit steps in the first coordinate.
    """

    mx_value, my_value = int(mx), int(my)
    if mx_value == 0 and my_value == 0:
        raise ValueError("at least one of mx,my must be nonzero")
    common = math.gcd(abs(mx_value), abs(my_value))
    px, py = mx_value // common, my_value // common
    gcd_value, x_value, y_value = _extended_gcd(px, py)
    if gcd_value != 1:
        raise RuntimeError("reduced commensurate direction must be primitive")

    # px*x + py*y = 1 implies det[(px,py),(-y,x)] = 1.
    transverse = np.asarray([-y_value, x_value], dtype=int)
    primitive = np.asarray([px, py], dtype=int)

    # n -> n + ell p preserves the determinant.  Choose the shortest nearby
    # representative to avoid an unnecessarily sheared transverse integrand.
    denominator = int(primitive @ primitive)
    center = int(round(-float(transverse @ primitive) / float(denominator)))
    candidates = [transverse + offset * primitive for offset in range(center - 2, center + 3)]
    transverse = min(
        candidates,
        key=lambda value: (int(value @ value), abs(int(value[0])) + abs(int(value[1]))),
    )
    determinant = int(primitive[0] * transverse[1] - primitive[1] * transverse[0])
    if determinant != 1:
        raise RuntimeError(f"internal unimodular basis failure: determinant={determinant}")
    primitive.setflags(write=False)
    transverse.setflags(write=False)
    return primitive, transverse, int(common)


def complementary_orbit_origins(
    orbit_shift_steps: int,
    shift_s: float,
    mode: SubgridAverageMode = "auto",
) -> tuple[float, ...]:
    """Return orbit origins needed for symmetric half-q endpoint sampling."""

    shift = float(shift_s)
    if not np.isfinite(shift) or not 0.0 <= shift < 1.0:
        raise ValueError("shift_s must lie in [0,1)")
    if mode not in {"auto", "none"}:
        raise ValueError("subgrid_average must be 'auto' or 'none'")
    if mode == "none" or int(orbit_shift_steps) % 2 == 0:
        return (shift,)
    return (shift, float((shift + 0.5) % 1.0))


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
class CommensurateOrbitGaussResult:
    """Complete q-orbit average followed by transverse Gauss integration."""

    value: np.ndarray
    q_model: np.ndarray
    primitive_direction: np.ndarray
    transverse_direction: np.ndarray
    orbit_shift_steps: int
    orbit_origins: tuple[float, ...]
    nk: int
    transverse_order: int
    point_evaluations: int
    chunks: int
    chunk_size: int
    wall_seconds: float
    summation_method: str

    def __post_init__(self) -> None:
        for name in ("value", "q_model", "primitive_direction", "transverse_direction"):
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def integrate_commensurate_orbit_gauss_vector(
    evaluator: ComplexVectorEvaluator,
    *,
    nk: int,
    mx: int,
    my: int,
    transverse_order: int,
    shift_s: float = 0.5,
    subgrid_average: SubgridAverageMode = "auto",
    chunk_size: int = 1024,
    max_point_evaluations: int = 200_000,
) -> CommensurateOrbitGaussResult:
    """Integrate a complex BZ vector on an exact q orbit and Gauss transverse rule."""

    nk_value = int(nk)
    mx_value, my_value = int(mx), int(my)
    order = int(transverse_order)
    size = int(chunk_size)
    maximum = int(max_point_evaluations)
    if nk_value <= 0 or order <= 0 or size <= 0 or maximum <= 0:
        raise ValueError("nk, transverse_order, chunk_size, and budget must be positive")
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
    expected_points = nk_value * order * len(origins)
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
    chunks = 0
    started = time.perf_counter()

    for t_value, t_weight in zip(transverse_nodes, transverse_weights, strict=True):
        origin_values: list[np.ndarray] = []
        for origin in origins:
            orbit_total: np.ndarray | None = None
            orbit_compensation: np.ndarray | None = None
            for start in range(0, nk_value, size):
                stop = min(start + size, nk_value)
                indices = np.arange(start, stop, dtype=float)
                s_values = -np.pi + (indices + float(origin)) * step
                points = (
                    s_values[:, None] * primitive[None, :]
                    + float(t_value) * transverse[None, :]
                )
                points = _wrap_periodic_bz(points)
                values = np.asarray(evaluator(points), dtype=complex)
                if values.ndim != 2 or values.shape[0] != points.shape[0]:
                    raise ValueError(
                        "evaluator must return shape (n_points,width), got "
                        f"{values.shape} for {points.shape[0]} points"
                    )
                if values.shape[1] == 0:
                    raise ValueError("evaluator vector width must be positive")
                if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
                    raise ValueError("evaluator returned non-finite values")
                if expected_width is None:
                    expected_width = int(values.shape[1])
                elif values.shape[1] != expected_width:
                    raise ValueError("evaluator vector width changed between chunks")
                chunk_sum = np.sum(values, axis=0, dtype=complex)
                if orbit_total is None:
                    orbit_total = np.zeros_like(chunk_sum)
                    orbit_compensation = np.zeros_like(chunk_sum)
                assert orbit_compensation is not None
                orbit_total, orbit_compensation = _compensated_add(
                    orbit_total,
                    orbit_compensation,
                    chunk_sum,
                )
                points_seen += int(points.shape[0])
                chunks += 1
            if orbit_total is None:
                raise RuntimeError("commensurate orbit produced no points")
            origin_values.append(np.asarray(orbit_total / float(nk_value), dtype=complex))

        orbit_average = np.mean(np.stack(origin_values, axis=0), axis=0)
        contribution = (float(t_weight) / (2.0 * np.pi)) * orbit_average
        if total is None:
            total = np.zeros_like(contribution)
            compensation = np.zeros_like(contribution)
        assert compensation is not None
        total, compensation = _compensated_add(total, compensation, contribution)

    if total is None or points_seen != expected_points:
        raise RuntimeError(
            "incomplete commensurate-orbit integration: "
            f"seen={points_seen}, expected={expected_points}"
        )
    return CommensurateOrbitGaussResult(
        value=np.asarray(total, dtype=complex),
        q_model=q_model,
        primitive_direction=primitive,
        transverse_direction=transverse,
        orbit_shift_steps=orbit_shift_steps,
        orbit_origins=origins,
        nk=nk_value,
        transverse_order=order,
        point_evaluations=points_seen,
        chunks=chunks,
        chunk_size=size,
        wall_seconds=float(time.perf_counter() - started),
        summation_method=(
            "equal_complete_q_orbit_average_with_complementary_half_step_if_needed_"
            "plus_complex_kahan_transverse_gauss"
        ),
    )


__all__ = [
    "CommensurateOrbitGaussResult",
    "OrbitEvaluationBudgetExceeded",
    "commensurate_orbit_basis",
    "complementary_orbit_origins",
    "integrate_commensurate_orbit_gauss_vector",
]
