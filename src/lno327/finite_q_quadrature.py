"""Finite-q Brillouin-zone quadrature for main Casimir workflows.

This module provides numerical integration grids only.  It does not change the
finite-q response formula, Peierls vertices, pairing ansatz, Ward diagnostics,
or Casimir trace-log formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .conductivity import k_weights, uniform_bz_mesh
from .model import normal_state_hamiltonian


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


def _minimum_shifted_band_distance_to_fermi(k_point: np.ndarray, q_model: np.ndarray, fermi_level_eV: float) -> float:
    distances = []
    for sign in (-0.5, 0.5):
        shifted = np.asarray(k_point, dtype=float) + sign * np.asarray(q_model, dtype=float)
        energies = np.linalg.eigvalsh(normal_state_hamiltonian(float(shifted[0]), float(shifted[1])))
        distances.append(float(np.min(np.abs(energies - float(fermi_level_eV)))))
    return min(distances)


def _adaptive_subcell_nodes(
    center: np.ndarray,
    cell_step: float,
    adaptive_level: int,
    gauss_order: int,
    parent_weight: float,
) -> tuple[list[list[float]], list[float]]:
    subdivisions = max(1, int(adaptive_level))
    order = max(1, int(gauss_order))
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)
    subcell_step = float(cell_step) / float(subdivisions)
    lower = np.asarray(center, dtype=float) - 0.5 * float(cell_step)
    points: list[list[float]] = []
    weights: list[float] = []
    for ix in range(subdivisions):
        for iy in range(subdivisions):
            sub_lower = lower + np.array([ix * subcell_step, iy * subcell_step])
            sub_center = sub_lower + 0.5 * subcell_step
            for ax, wx in zip(nodes_1d, weights_1d, strict=True):
                for ay, wy in zip(nodes_1d, weights_1d, strict=True):
                    point = sub_center + 0.5 * subcell_step * np.array([ax, ay])
                    weight = parent_weight * (float(wx) * float(wy)) / (4.0 * subdivisions * subdivisions)
                    points.append([float(point[0]), float(point[1])])
                    weights.append(float(weight))
    if abs(float(np.sum(weights)) - float(parent_weight)) >= 1e-12:
        raise ValueError("refined child weights do not sum to parent cell weight")
    return points, weights


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
    strategy = str(options.integration_strategy)
    if strategy not in {"uniform", "best_available_adaptive"}:
        raise ValueError("integration_strategy must be 'uniform' or 'best_available_adaptive'")

    if strategy == "uniform":
        points = uniform_bz_mesh(int(options.coarse_grid))
        weights = k_weights(points)
        metadata = {
            "integration_strategy": "uniform",
            "coarse_grid": int(options.coarse_grid),
            "adaptive_level": None,
            "gauss_order": None,
            "fermi_window_eV": None,
            "q_specific_adaptive_grid": False,
            "num_cells_total": int(options.coarse_grid) ** 2,
            "num_cells_refined": 0,
            "num_cells_unrefined": int(options.coarse_grid) ** 2,
            "num_quadrature_points": int(len(points)),
            "q_model_used_for_quadrature": [float(q[0]), float(q[1])],
            "weight_sum": float(np.sum(weights)),
            "abs_weight_sum_minus_one": float(abs(np.sum(weights) - 1.0)),
            "quadrature_rule": "uniform midpoint Brillouin-zone mesh",
        }
        return points, weights, metadata

    centers, cell_step = _cell_centers(int(options.coarse_grid))
    parent_weight = 1.0 / float(len(centers))
    refined_flags = []
    if bool(options.q_specific_adaptive_grid):
        for center in centers:
            distance = _minimum_shifted_band_distance_to_fermi(center, q, float(options.fermi_level_eV))
            refined_flags.append(distance <= float(options.fermi_window_eV))
    else:
        refined_flags = [False] * len(centers)

    points_list: list[list[float]] = []
    weights_list: list[float] = []
    refined_count = 0
    children_per_refined_cell = 0
    for center, refine in zip(centers, refined_flags, strict=True):
        if refine:
            refined_count += 1
            child_points, child_weights = _adaptive_subcell_nodes(
                center,
                cell_step,
                int(options.adaptive_level),
                int(options.gauss_order),
                parent_weight,
            )
            children_per_refined_cell = max(children_per_refined_cell, len(child_points))
            points_list.extend(child_points)
            weights_list.extend(child_weights)
        else:
            points_list.append([float(center[0]), float(center[1])])
            weights_list.append(parent_weight)

    points = np.asarray(points_list, dtype=float)
    weights = np.asarray(weights_list, dtype=float)
    weight_sum = float(np.sum(weights))
    if abs(weight_sum - 1.0) >= 1e-12:
        raise ValueError("finite-q quadrature weights do not sum to one")
    if not np.all(weights > 0.0):
        raise ValueError("finite-q quadrature weights must be positive")

    metadata = {
        "integration_strategy": "best_available_adaptive",
        "coarse_grid": int(options.coarse_grid),
        "adaptive_level": int(options.adaptive_level),
        "gauss_order": int(options.gauss_order),
        "fermi_window_eV": float(options.fermi_window_eV),
        "q_specific_adaptive_grid": bool(options.q_specific_adaptive_grid),
        "fermi_level_eV": float(options.fermi_level_eV),
        "num_cells_total": int(len(centers)),
        "num_cells_refined": int(refined_count),
        "num_cells_unrefined": int(len(centers) - refined_count),
        "children_per_refined_cell": int(children_per_refined_cell),
        "num_quadrature_points": int(len(points)),
        "q_model_used_for_quadrature": [float(q[0]), float(q[1])],
        "weight_sum": weight_sum,
        "abs_weight_sum_minus_one": float(abs(weight_sum - 1.0)),
        "quadrature_rule": (
            "q-specific adaptive cell quadrature; refined parent cells are replaced by "
            "Gauss subcell nodes and parent/children are not double-counted"
        ),
    }
    return points, weights, metadata
