"""Finite-q Brillouin-zone quadrature for main Casimir workflows.

This module provides numerical integration grids only.  It does not change the
finite-q response formula, Peierls vertices, pairing ansatz, Ward diagnostics,
or Casimir trace-log formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models.lno327_four_orbital.normal import normal_state_hamiltonian
from .numerics.grids import uniform_bz_mesh
from .numerics.weights import k_weights


@dataclass(frozen=True)
class FiniteQQuadratureOptions:
    """Options for q-specific finite-q Brillouin-zone quadrature."""

    integration_strategy: str = "best_available_adaptive"
    coarse_grid: int = 32
    adaptive_level: int = 5
    gauss_order: int = 5
    fermi_window_eV: float = 0.12
    q_specific_adaptive_grid: bool = True
    fermi_level_eV: float = 0.0


def _cell_centers(nk: int) -> tuple[np.ndarray, float]:
    if nk <= 0:
        raise ValueError("coarse_grid must be positive")
    step = 2.0 * np.pi / float(nk)
    centers_1d = -np.pi + (np.arange(nk, dtype=float) + 0.5) * step
    kx, ky = np.meshgrid(centers_1d, centers_1d, indexing="ij")
    return np.column_stack([kx.ravel(), ky.ravel()]), step


def _cell_sample_points(x0: float, x1: float, y0: float, y1: float) -> tuple[tuple[float, float], ...]:
    """Return the stage4_15 corner-plus-center cell samples."""

    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    return ((x0, y0), (x0, y1), (x1, y0), (x1, y1), (xm, ym))


def _cell_in_fermi_window(
    cell: tuple[float, float, float, float],
    q: np.ndarray,
    fermi_window_eV: float,
    fermi_level_eV: float,
) -> bool:
    """Return whether a cell intersects the q-shifted Fermi window."""

    x0, x1, y0, y1 = cell
    qx, qy = float(q[0]), float(q[1])
    for kx, ky in _cell_sample_points(x0, x1, y0, y1):
        for sx, sy in ((0.0, 0.0), (0.5 * qx, 0.5 * qy), (-0.5 * qx, -0.5 * qy)):
            energies = np.linalg.eigvalsh(normal_state_hamiltonian(kx + sx, ky + sy))
            if np.any(np.abs(energies - float(fermi_level_eV)) < float(fermi_window_eV)):
                return True
    return False


def build_adaptive_cells(
    q: np.ndarray,
    *,
    coarse_grid: int,
    refinement_level: int,
    fermi_window_eV: float,
    fermi_level_eV: float = 0.0,
) -> tuple[list[tuple[float, float, float, float]], int, int]:
    """Return stage4_15 final cells, refined parent count, and flagged base cells."""

    q_array = np.asarray(q, dtype=float)
    if q_array.shape != (2,):
        raise ValueError("q must have shape (2,)")
    if int(coarse_grid) <= 0:
        raise ValueError("coarse_grid must be positive")
    if int(refinement_level) < 0:
        raise ValueError("refinement_level must be non-negative")
    if float(fermi_window_eV) <= 0.0:
        raise ValueError("fermi_window_eV must be positive")

    edges = np.linspace(-np.pi, np.pi, int(coarse_grid) + 1)
    cells = [
        (float(edges[ix]), float(edges[ix + 1]), float(edges[iy]), float(edges[iy + 1]))
        for ix in range(int(coarse_grid))
        for iy in range(int(coarse_grid))
    ]
    flagged_base = sum(
        1 for cell in cells if _cell_in_fermi_window(cell, q_array, float(fermi_window_eV), float(fermi_level_eV))
    )
    refined_count = 0
    for _level in range(int(refinement_level)):
        next_cells: list[tuple[float, float, float, float]] = []
        for cell in cells:
            if _cell_in_fermi_window(cell, q_array, float(fermi_window_eV), float(fermi_level_eV)):
                x0, x1, y0, y1 = cell
                xm = 0.5 * (x0 + x1)
                ym = 0.5 * (y0 + y1)
                next_cells.extend(
                    [
                        (x0, xm, y0, ym),
                        (x0, xm, ym, y1),
                        (xm, x1, y0, ym),
                        (xm, x1, ym, y1),
                    ]
                )
                refined_count += 1
            else:
                next_cells.append(cell)
        cells = next_cells
    return cells, refined_count, flagged_base


def quadrature_points_for_cells(
    cells: list[tuple[float, float, float, float]],
    gauss_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre points and normalized BZ-average weights."""

    if int(gauss_order) <= 0:
        raise ValueError("gauss_order must be positive")
    nodes, node_weights = np.polynomial.legendre.leggauss(int(gauss_order))
    points: list[tuple[float, float]] = []
    weights: list[float] = []
    bz_area = (2.0 * np.pi) ** 2
    for x0, x1, y0, y1 in cells:
        x_mid = 0.5 * (x0 + x1)
        y_mid = 0.5 * (y0 + y1)
        x_half = 0.5 * (x1 - x0)
        y_half = 0.5 * (y1 - y0)
        for i, node_x in enumerate(nodes):
            for j, node_y in enumerate(nodes):
                points.append((float(x_mid + x_half * node_x), float(y_mid + y_half * node_y)))
                weights.append(float(node_weights[i] * node_weights[j] * x_half * y_half / bz_area))
    point_array = np.asarray(points, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("finite-q quadrature points must have shape (N, 2)")
    return point_array, weight_array


def _validate_weights(points: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if np.asarray(points).ndim != 2 or np.asarray(points).shape[1] != 2:
        raise ValueError("finite-q quadrature points must have shape (N, 2)")
    if np.asarray(weights).shape != (len(points),):
        raise ValueError("finite-q quadrature weights must have shape (N,)")
    weight_sum = float(np.sum(weights))
    abs_error = float(abs(weight_sum - 1.0))
    if abs_error >= 1e-12:
        raise ValueError("finite-q quadrature weights do not sum to one")
    if not np.all(weights > 0.0):
        raise ValueError("finite-q quadrature weights must be positive")
    return weight_sum, abs_error


def finite_q_quadrature_points(
    q_model: np.ndarray,
    options: FiniteQQuadratureOptions,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return k-points, weights, and metadata for a finite-q response integral.

    ``best_available_adaptive`` refines coarse cells whose shifted normal-state
    bands approach the configured Fermi window.  Refined parent cells are
    replaced by exact child quadrature nodes; the parent center is not retained.
    """

    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model must have shape (2,)")
    if int(options.coarse_grid) <= 0:
        raise ValueError("coarse_grid must be positive")
    if int(options.adaptive_level) < 0:
        raise ValueError("adaptive_level must be non-negative")
    if int(options.gauss_order) <= 0:
        raise ValueError("gauss_order must be positive")
    if float(options.fermi_window_eV) <= 0.0:
        raise ValueError("fermi_window_eV must be positive")
    strategy = str(options.integration_strategy)
    if strategy not in {"uniform", "best_available_adaptive"}:
        raise ValueError("integration_strategy must be 'uniform' or 'best_available_adaptive'")

    if strategy == "uniform" or (strategy == "best_available_adaptive" and not bool(options.q_specific_adaptive_grid)):
        points = uniform_bz_mesh(int(options.coarse_grid))
        weights = k_weights(points)
        weight_sum, abs_weight_sum_minus_one = _validate_weights(points, weights)
        metadata = {
            "integration_strategy": "uniform",
            "requested_integration_strategy": strategy,
            "coarse_grid": int(options.coarse_grid),
            "adaptive_level": None,
            "refinement_level": None,
            "gauss_order": None,
            "fermi_window_eV": None,
            "q_specific_adaptive_grid": False,
            "fermi_level_eV": float(options.fermi_level_eV),
            "num_base_cells": int(options.coarse_grid) ** 2,
            "num_flagged_base_cells": 0,
            "num_cells_total": int(options.coarse_grid) ** 2,
            "num_cells_refined": 0,
            "num_cells_unrefined": int(options.coarse_grid) ** 2,
            "num_quadrature_points": int(len(points)),
            "q_model_used_for_quadrature": [float(q[0]), float(q[1])],
            "weight_sum": weight_sum,
            "abs_weight_sum_minus_one": abs_weight_sum_minus_one,
            "quadrature_rule": "uniform midpoint Brillouin-zone mesh",
            "validation_semantics": None,
            "parent_child_double_counting": False,
        }
        return points, weights, metadata

    cells, refined_count, flagged_base = build_adaptive_cells(
        q,
        coarse_grid=int(options.coarse_grid),
        refinement_level=int(options.adaptive_level),
        fermi_window_eV=float(options.fermi_window_eV),
        fermi_level_eV=float(options.fermi_level_eV),
    )
    points, weights = quadrature_points_for_cells(cells, int(options.gauss_order))
    weight_sum, abs_weight_sum_minus_one = _validate_weights(points, weights)
    num_base_cells = int(options.coarse_grid) ** 2

    metadata = {
        "integration_strategy": "best_available_adaptive",
        "requested_integration_strategy": strategy,
        "coarse_grid": int(options.coarse_grid),
        "adaptive_level": int(options.adaptive_level),
        "refinement_level": int(options.adaptive_level),
        "gauss_order": int(options.gauss_order),
        "fermi_window_eV": float(options.fermi_window_eV),
        "q_specific_adaptive_grid": bool(options.q_specific_adaptive_grid),
        "fermi_level_eV": float(options.fermi_level_eV),
        "num_base_cells": num_base_cells,
        "num_flagged_base_cells": int(flagged_base),
        "num_cells_total": int(len(cells)),
        "num_cells_refined": int(refined_count),
        "num_cells_unrefined": int(max(len(cells) - refined_count, 0)),
        "num_quadrature_points": int(len(points)),
        "q_model_used_for_quadrature": [float(q[0]), float(q[1])],
        "weight_sum": weight_sum,
        "abs_weight_sum_minus_one": abs_weight_sum_minus_one,
        "quadrature_rule": "recursive q-specific Fermi-window adaptive cells with Gauss-Legendre quadrature on final cells",
        "validation_semantics": "stage4_15_build_adaptive_cells_and_quadrature_points_for_cells",
        "parent_child_double_counting": False,
    }
    return points, weights, metadata
