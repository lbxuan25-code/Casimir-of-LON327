"""Fixed shifted periodic Brillouin-zone lattices for exact arbitrary q."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Iterable

import numpy as np


def _readonly(value: np.ndarray, dtype=None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _normalize_shift(value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError("grid shifts must be finite")
    normalized = scalar % 1.0
    if normalized == 0.0:
        normalized = 0.0
    return normalized


def _has_internal_inversion(shift: tuple[float, float]) -> bool:
    return all(
        abs(2.0 * s - round(2.0 * s)) <= 64.0 * np.finfo(float).eps
        for s in shift
    )


def _partner_original_index(
    n: int,
    ix: int,
    iy: int,
    shift: tuple[float, float],
) -> int:
    sx2 = int(round(2.0 * shift[0]))
    sy2 = int(round(2.0 * shift[1]))
    jx = (-ix - sx2) % n
    jy = (-iy - sy2) % n
    return int(jx * n + jy)


def _pair_order(
    n: int,
    shift: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    total = n * n
    visited = np.zeros(total, dtype=bool)
    order: list[int] = []
    for original in range(total):
        if visited[original]:
            continue
        ix, iy = divmod(original, n)
        partner = _partner_original_index(n, ix, iy, shift)
        order.append(original)
        visited[original] = True
        if partner != original:
            order.append(partner)
            visited[partner] = True
    order_array = np.asarray(order, dtype=np.int64)
    inverse_position = np.empty(total, dtype=np.int64)
    inverse_position[order_array] = np.arange(total, dtype=np.int64)
    partner_positions = np.empty(total, dtype=np.int64)
    for position, original in enumerate(order_array):
        ix, iy = divmod(int(original), n)
        partner_original = _partner_original_index(n, ix, iy, shift)
        partner_positions[position] = inverse_position[partner_original]
    return order_array, partner_positions


@dataclass(frozen=True)
class PeriodicBZGrid:
    n: int
    shift: tuple[float, float]
    points: np.ndarray
    weights: np.ndarray
    original_indices: np.ndarray
    inversion_partner: np.ndarray
    internally_inversion_symmetric: bool
    ordering: str
    bz_convention: str = "[-pi,pi)_x_[-pi,pi)"

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _readonly(self.points, float))
        object.__setattr__(self, "weights", _readonly(self.weights, float))
        object.__setattr__(
            self,
            "original_indices",
            _readonly(self.original_indices, np.int64),
        )
        object.__setattr__(
            self,
            "inversion_partner",
            _readonly(self.inversion_partner, np.int64),
        )
        total = int(self.n) ** 2
        if self.points.shape != (total, 2):
            raise ValueError("periodic BZ points must have shape (N^2,2)")
        if self.weights.shape != (total,):
            raise ValueError("periodic BZ weights must have shape (N^2,)")
        if self.original_indices.shape != (total,):
            raise ValueError("original_indices must have shape (N^2,)")
        if self.inversion_partner.shape != (total,):
            raise ValueError("inversion_partner must have shape (N^2,)")
        if not np.all(self.weights > 0.0):
            raise ValueError("periodic BZ weights must be positive")
        if abs(float(np.sum(self.weights)) - 1.0) > 1e-13:
            raise ValueError("periodic BZ weights must sum to one")

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def fingerprint(self) -> str:
        payload = (
            f"PeriodicBZGrid:v1:N={self.n}:shift={self.shift[0].hex()},",
            f"{self.shift[1].hex()}:ordering={self.ordering}:bz={self.bz_convention}",
        )
        return hashlib.sha256("".join(payload).encode("utf-8")).hexdigest()

    def metadata(self) -> dict[str, object]:
        return {
            "grid_contract": "ArbitraryQPeriodicBZContract-v1",
            "N": int(self.n),
            "point_count": self.point_count,
            "shift": [float(self.shift[0]), float(self.shift[1])],
            "shift_hex": [self.shift[0].hex(), self.shift[1].hex()],
            "weights_equal": True,
            "weight": float(1.0 / self.point_count),
            "weight_sum": float(np.sum(self.weights)),
            "bz_convention": self.bz_convention,
            "ordering": self.ordering,
            "internally_inversion_symmetric": bool(
                self.internally_inversion_symmetric
            ),
            "fingerprint": self.fingerprint,
        }


def build_periodic_bz_grid(
    n: int,
    shift: tuple[float, float] = (0.5, 0.5),
    *,
    pair_inversion: bool = True,
) -> PeriodicBZGrid:
    size = int(n)
    if size <= 0:
        raise ValueError("N must be positive")
    if size % 2 != 0:
        raise ValueError("ArbitraryQPeriodicBZContract-v1 requires even N")
    normalized = (_normalize_shift(shift[0]), _normalize_shift(shift[1]))
    axis_x = -np.pi + (np.arange(size, dtype=float) + normalized[0]) * (
        2.0 * np.pi / size
    )
    axis_y = -np.pi + (np.arange(size, dtype=float) + normalized[1]) * (
        2.0 * np.pi / size
    )
    kx, ky = np.meshgrid(axis_x, axis_y, indexing="ij")
    points = np.column_stack((kx.reshape(-1), ky.reshape(-1)))
    total = size * size
    internally_symmetric = _has_internal_inversion(normalized)
    if pair_inversion and internally_symmetric:
        order, partners = _pair_order(size, normalized)
        points = points[order]
        ordering = "adjacent_k_minus_k_pairs_then_lexicographic"
    else:
        order = np.arange(total, dtype=np.int64)
        partners = np.full(total, -1, dtype=np.int64)
        ordering = "lexicographic_kx_then_ky"
    weights = np.full(total, 1.0 / total, dtype=float)
    return PeriodicBZGrid(
        n=size,
        shift=normalized,
        points=points,
        weights=weights,
        original_indices=order,
        inversion_partner=partners,
        internally_inversion_symmetric=internally_symmetric,
        ordering=ordering,
    )


def audit_shift_pair(
    n: int,
    first: tuple[float, float] = (0.25, 0.75),
    second: tuple[float, float] = (0.75, 0.25),
) -> tuple[PeriodicBZGrid, PeriodicBZGrid, np.ndarray]:
    grid_a = build_periodic_bz_grid(n, first, pair_inversion=False)
    grid_b = build_periodic_bz_grid(n, second, pair_inversion=False)
    expected = tuple((-s) % 1.0 for s in grid_a.shift)
    if any(
        abs(a - b) > 64.0 * np.finfo(float).eps
        for a, b in zip(expected, grid_b.shift)
    ):
        raise ValueError("audit shifts must be related by inversion")
    size = int(n)
    mapping = np.empty(size * size, dtype=np.int64)
    sx_sum = int(round(grid_a.shift[0] + grid_b.shift[0]))
    sy_sum = int(round(grid_a.shift[1] + grid_b.shift[1]))
    for original in range(size * size):
        ix, iy = divmod(original, size)
        jx = (-ix - sx_sum) % size
        jy = (-iy - sy_sum) % size
        mapping[original] = jx * size + jy
    mapping.setflags(write=False)
    return grid_a, grid_b, mapping


def exact_float64_key(values: Iterable[float]) -> str:
    """Return an exact IEEE-754 key after canonicalizing signed zero."""
    encoded = bytearray()
    for value in values:
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError("exact float keys reject NaN and infinity")
        if scalar == 0.0:
            scalar = 0.0
        encoded.extend(struct.pack(">d", scalar))
    return bytes(encoded).hex()


__all__ = [
    "PeriodicBZGrid",
    "audit_shift_pair",
    "build_periodic_bz_grid",
    "exact_float64_key",
]
