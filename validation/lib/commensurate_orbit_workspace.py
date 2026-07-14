"""Shared complete-orbit cache for transverse quadrature backends.

The workspace owns commensurate-orbit geometry, complementary origins, normalized
microscopic weights, periodic transverse-key caching, and a hard unique-node budget.
It contains no quadrature rule and no response postprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np

from validation.lib.dwave_commensurate_orbit_gauss import (
    commensurate_orbit_basis,
    complementary_orbit_origins,
)

OrbitAggregateEvaluator = Callable[[np.ndarray, np.ndarray], np.ndarray]


class TransverseEvaluationBudgetExceeded(RuntimeError):
    """Raised before evaluating a transverse coordinate beyond the hard cap."""

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


@dataclass
class CompleteOrbitAggregateWorkspace:
    """Cache complete exact q-orbit evaluations by periodic transverse phase."""

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
        if (
            not value.size
            or not np.isfinite(value.real).all()
            or not np.isfinite(value.imag).all()
        ):
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


__all__ = [
    "CompleteOrbitAggregateWorkspace",
    "TransverseEvaluationBudgetExceeded",
]
