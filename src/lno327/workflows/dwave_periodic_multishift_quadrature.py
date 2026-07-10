"""Ward-compatible periodic multi-shift quadrature for d-wave response.

Local cell refinement can improve nodal sampling while destroying the discrete
translation structure needed by the primitive finite-q Ward identity.  This
module therefore refines only in *shift space*: every selected shift contributes
one complete periodic tensor lattice over the Brillouin zone.  The union is
merged before any primitive response block or Schur complement is formed.

For ``shift_order=m`` the shifts and their weights are the tensor product of an
m-point Gauss-Legendre rule mapped to the unit cell.  ``shift_order=1`` is the
ordinary midpoint lattice.  Higher orders are composite Gauss rules represented
as weighted complete periodic lattices, so no local parent/child replacement is
performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DWavePeriodicMultishiftOptions:
    """Controls for Ward-compatible periodic multi-shift quadrature."""

    base_nk: int = 64
    shift_order: int = 2
    max_quadrature_points: int = 400_000


def _validate_options(options: DWavePeriodicMultishiftOptions) -> None:
    if int(options.base_nk) <= 0:
        raise ValueError("base_nk must be positive")
    if int(options.shift_order) <= 0:
        raise ValueError("shift_order must be positive")
    if int(options.max_quadrature_points) <= 0:
        raise ValueError("max_quadrature_points must be positive")
    requested = int(options.base_nk) ** 2 * int(options.shift_order) ** 2
    if requested > int(options.max_quadrature_points):
        raise RuntimeError(
            "periodic multishift quadrature exceeded max_quadrature_points: "
            f"requested={requested}, maximum={options.max_quadrature_points}"
        )


def _periodic_shift_mesh(nk: int, shift_x: float, shift_y: float) -> np.ndarray:
    if int(nk) <= 0:
        raise ValueError("nk must be positive")
    shifts = np.asarray([shift_x, shift_y], dtype=float)
    if not np.isfinite(shifts).all() or np.any(shifts < 0.0) or np.any(shifts >= 1.0):
        raise ValueError("grid shifts must be finite and lie in [0, 1)")
    step = 2.0 * np.pi / float(nk)
    kx = -np.pi + (np.arange(int(nk), dtype=float) + float(shift_x)) * step
    ky = -np.pi + (np.arange(int(nk), dtype=float) + float(shift_y)) * step
    gx, gy = np.meshgrid(kx, ky, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])


def _gauss_shifts(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    mapped_nodes = 0.5 * (np.asarray(nodes, dtype=float) + 1.0)
    mapped_weights = 0.5 * np.asarray(weights, dtype=float)
    if np.any(mapped_nodes < 0.0) or np.any(mapped_nodes >= 1.0):
        raise RuntimeError("mapped Gauss shifts must lie in [0, 1)")
    if not np.isclose(np.sum(mapped_weights), 1.0, rtol=0.0, atol=1e-14):
        raise RuntimeError("one-dimensional shift weights must sum to one")
    return mapped_nodes, mapped_weights


def build_dwave_periodic_multishift_quadrature(
    q_model: np.ndarray,
    options: DWavePeriodicMultishiftOptions,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return weighted complete periodic lattices for one finite-q response.

    ``q_model`` is recorded for auditability but does not alter the shift rule.
    Keeping the rule independent of the integrand avoids selecting a biased set
    of shifts after inspecting the d-wave response.
    """

    _validate_options(options)
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")

    base_nk = int(options.base_nk)
    shift_order = int(options.shift_order)
    shift_nodes, shift_weights_1d = _gauss_shifts(shift_order)

    point_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    shifts: list[list[float]] = []
    shift_weights: list[float] = []
    points_per_shift = base_nk * base_nk

    for ix, iy in product(range(shift_order), repeat=2):
        sx = float(shift_nodes[ix])
        sy = float(shift_nodes[iy])
        shift_weight = float(shift_weights_1d[ix] * shift_weights_1d[iy])
        block = _periodic_shift_mesh(base_nk, sx, sy)
        point_blocks.append(block)
        weight_blocks.append(
            np.full(points_per_shift, shift_weight / float(points_per_shift), dtype=float)
        )
        shifts.append([sx, sy])
        shift_weights.append(shift_weight)

    points = np.concatenate(point_blocks, axis=0)
    weights = np.concatenate(weight_blocks, axis=0)
    expected = base_nk * base_nk * shift_order * shift_order
    if points.shape != (expected, 2) or weights.shape != (expected,):
        raise RuntimeError("unexpected periodic multishift quadrature shape")
    if not np.all(weights > 0.0):
        raise RuntimeError("periodic multishift weights must be positive")
    if np.any(points < -np.pi) or np.any(points >= np.pi):
        raise RuntimeError("periodic multishift points must lie in [-pi, pi)")
    weight_sum = float(np.sum(weights))
    if abs(weight_sum - 1.0) >= 2e-12:
        raise RuntimeError("periodic multishift weights do not sum to one")

    metadata = {
        "integration_strategy": "dwave_periodic_gauss_multishift",
        "q_model": [float(q[0]), float(q[1])],
        "base_nk": base_nk,
        "shift_order": shift_order,
        "num_grid_shifts": shift_order * shift_order,
        "num_points_per_shift": points_per_shift,
        "num_quadrature_points": expected,
        "effective_points_per_axis": base_nk * shift_order,
        "grid_shifts": shifts,
        "grid_shift_weights": shift_weights,
        "weight_sum": weight_sum,
        "full_periodic_lattice_per_shift": True,
        "local_cell_refinement": False,
        "parent_child_double_counting": False,
        "primitive_merge_before_schur_required": True,
        "ward_compatibility_semantics": (
            "weighted union of complete periodic tensor lattices; no local cell replacement"
        ),
        "quadrature_rule": (
            "tensor Gauss-Legendre rule in shift space, with one complete periodic "
            "base lattice per shift"
        ),
    }
    return points, weights, metadata


__all__ = [
    "DWavePeriodicMultishiftOptions",
    "build_dwave_periodic_multishift_quadrature",
]
