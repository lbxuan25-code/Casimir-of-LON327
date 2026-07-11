"""Chunked complete-periodic tensor grids with exactly commensurate wavevectors.

For a grid with ``nk`` points per reciprocal direction and integer shift vector
``m=(mx,my)``, the external momentum is constructed as

    q = (2 pi / nk) * m.

Translation by q therefore permutes the complete tensor lattice exactly at the
index level.  The implementation streams lexicographic point chunks and applies
one compensated complex-vector sum, so large diagnostic grids do not require a
full finite-q workspace in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterator

import numpy as np


@dataclass(frozen=True)
class CommensuratePeriodicGrid:
    """One shifted complete periodic ``nk x nk`` lattice and integer q shift."""

    nk: int
    mx: int
    my: int
    shift_x: float = 0.5
    shift_y: float = 0.5
    max_points: int = 500_000

    def __post_init__(self) -> None:
        nk = int(self.nk)
        mx = int(self.mx)
        my = int(self.my)
        max_points = int(self.max_points)
        shift_x = float(self.shift_x)
        shift_y = float(self.shift_y)
        if nk <= 0:
            raise ValueError("nk must be positive")
        if mx == 0 and my == 0:
            raise ValueError("at least one of mx,my must be nonzero")
        if abs(mx) > nk // 2 or abs(my) > nk // 2:
            raise ValueError("mx and my must lie in the principal periodic range")
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        if nk * nk > max_points:
            raise RuntimeError(
                "commensurate periodic grid exceeded max_points: "
                f"requested={nk * nk}, maximum={max_points}"
            )
        for name, value in (("shift_x", shift_x), ("shift_y", shift_y)):
            if not np.isfinite(value) or value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        object.__setattr__(self, "nk", nk)
        object.__setattr__(self, "mx", mx)
        object.__setattr__(self, "my", my)
        object.__setattr__(self, "shift_x", shift_x)
        object.__setattr__(self, "shift_y", shift_y)
        object.__setattr__(self, "max_points", max_points)

    @property
    def step(self) -> float:
        return 2.0 * np.pi / float(self.nk)

    @property
    def q_model(self) -> np.ndarray:
        return self.step * np.asarray([self.mx, self.my], dtype=float)

    @property
    def num_points(self) -> int:
        return self.nk * self.nk

    @property
    def translation_permutation_exact(self) -> bool:
        """The integer index map is a bijection for every integer shift."""

        return True

    def shifted_index(self, ix: int, iy: int) -> tuple[int, int]:
        """Return the periodic index reached by translation by ``q``."""

        return ((int(ix) + self.mx) % self.nk, (int(iy) + self.my) % self.nk)

    def iter_point_chunks(self, chunk_size: int) -> Iterator[np.ndarray]:
        """Yield the complete lattice in deterministic lexicographic chunks."""

        size = int(chunk_size)
        if size <= 0:
            raise ValueError("chunk_size must be positive")
        step = self.step
        total = self.num_points
        for start in range(0, total, size):
            stop = min(start + size, total)
            flat = np.arange(start, stop, dtype=np.int64)
            ix = flat // self.nk
            iy = flat % self.nk
            kx = -np.pi + (ix.astype(float) + self.shift_x) * step
            ky = -np.pi + (iy.astype(float) + self.shift_y) * step
            yield np.column_stack((kx, ky))


@dataclass(frozen=True)
class CommensuratePeriodicIntegral:
    """Result of one streamed equally weighted complete-periodic integral."""

    value: np.ndarray
    point_evaluations: int
    chunks: int
    chunk_size: int
    wall_seconds: float
    summation_method: str


def _compensated_add(
    total: np.ndarray,
    compensation: np.ndarray,
    increment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corrected = np.asarray(increment, dtype=complex) - compensation
    updated = total + corrected
    new_compensation = (updated - total) - corrected
    return updated, new_compensation


def integrate_commensurate_periodic_vector(
    grid: CommensuratePeriodicGrid,
    evaluator: Callable[[np.ndarray], np.ndarray],
    *,
    chunk_size: int = 1024,
) -> CommensuratePeriodicIntegral:
    """Integrate a complex vector density using one streamed periodic average.

    ``evaluator(points)`` must return an array of shape ``(n_points, width)``.
    The same lattice and equal weights are therefore shared by every returned
    primitive channel.
    """

    started = time.perf_counter()
    total: np.ndarray | None = None
    compensation: np.ndarray | None = None
    points_seen = 0
    chunks = 0

    for points in grid.iter_point_chunks(chunk_size):
        values = np.asarray(evaluator(points), dtype=complex)
        if values.ndim != 2 or values.shape[0] != points.shape[0]:
            raise ValueError(
                "evaluator must return shape (n_points, width), got "
                f"{values.shape} for {points.shape[0]} points"
            )
        if values.shape[1] == 0:
            raise ValueError("evaluator vector width must be positive")
        if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
            raise ValueError("evaluator returned non-finite values")
        chunk_sum = np.sum(values, axis=0, dtype=complex)
        if total is None:
            total = np.zeros_like(chunk_sum)
            compensation = np.zeros_like(chunk_sum)
        elif chunk_sum.shape != total.shape:
            raise ValueError("evaluator vector width changed between chunks")
        assert compensation is not None
        total, compensation = _compensated_add(total, compensation, chunk_sum)
        points_seen += int(points.shape[0])
        chunks += 1

    if total is None or points_seen != grid.num_points:
        raise RuntimeError(
            "incomplete commensurate periodic integration: "
            f"seen={points_seen}, expected={grid.num_points}"
        )
    value = np.asarray(total / float(grid.num_points), dtype=complex)
    value.setflags(write=False)
    return CommensuratePeriodicIntegral(
        value=value,
        point_evaluations=points_seen,
        chunks=chunks,
        chunk_size=int(chunk_size),
        wall_seconds=float(time.perf_counter() - started),
        summation_method="pairwise_chunk_sum_plus_complex_kahan_across_chunks",
    )


__all__ = [
    "CommensuratePeriodicGrid",
    "CommensuratePeriodicIntegral",
    "integrate_commensurate_periodic_vector",
]
