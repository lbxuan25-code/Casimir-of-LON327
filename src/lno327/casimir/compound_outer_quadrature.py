"""Composite radial outer-q quadrature with nested cumulative panels.

The physical measure remains

    d^2Q/(2pi)^2 = u du dphi / (16 pi^2 d^2),

with ``u = 2 Q d`` and the complete angular interval retained.  Unlike one global
Gauss-Legendre rule on ``[0, u_max]``, this builder applies one fixed Gauss rule to
each radial panel.  Extending the final cutoff therefore preserves every node and
weight in earlier panels exactly and adds only the new tail panel.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from lno327.casimir.outer_quadrature import OuterQPolarGrid

_TWO_PI = 2.0 * np.pi


def _validated_panel_edges(values: Sequence[float]) -> tuple[float, ...]:
    edges = tuple(float(value) for value in values)
    if len(edges) < 2:
        raise ValueError("radial_panel_edges must contain at least two values")
    if not np.isfinite(edges).all():
        raise ValueError("radial_panel_edges must be finite")
    if edges[0] != 0.0:
        raise ValueError("radial_panel_edges must start at zero")
    if any(right <= left for left, right in zip(edges[:-1], edges[1:], strict=True)):
        raise ValueError("radial_panel_edges must be strictly increasing")
    return edges


def build_compound_outer_q_polar_grid(
    *,
    separation_m: float,
    lattice_a_x_m: float,
    lattice_a_y_m: float,
    radial_panel_edges: Sequence[float],
    radial_panel_order: int,
    angular_order: int,
    angular_offset_fraction: float = 0.5,
) -> OuterQPolarGrid:
    """Build a nested composite-Gauss outer-q grid.

    ``radial_panel_order`` is the Gauss-Legendre order used independently on every
    interval in ``radial_panel_edges``.  The resulting ``OuterQPolarGrid.radial_order``
    stores the total number of radial nodes across all panels.
    """

    d = float(separation_m)
    ax = float(lattice_a_x_m)
    ay = float(lattice_a_y_m)
    panel_order = int(radial_panel_order)
    nphi = int(angular_order)
    offset = float(angular_offset_fraction)
    edges = _validated_panel_edges(radial_panel_edges)

    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("separation_m must be finite and positive")
    if not np.isfinite(ax) or ax <= 0.0 or not np.isfinite(ay) or ay <= 0.0:
        raise ValueError("lattice constants must be finite and positive")
    if panel_order <= 0 or nphi <= 0:
        raise ValueError("radial_panel_order and angular_order must be positive")
    if not np.isfinite(offset) or not 0.0 <= offset < 1.0:
        raise ValueError("angular_offset_fraction must lie in [0, 1)")

    roots, root_weights = np.polynomial.legendre.leggauss(panel_order)
    radial_u_parts: list[np.ndarray] = []
    radial_du_weight_parts: list[np.ndarray] = []
    panel_index_parts: list[np.ndarray] = []
    for panel_index, (left, right) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        half_width = 0.5 * (right - left)
        midpoint = 0.5 * (right + left)
        radial_u_parts.append(midpoint + half_width * roots)
        radial_du_weight_parts.append(half_width * root_weights)
        panel_index_parts.append(np.full(panel_order, panel_index, dtype=int))

    radial_u = np.concatenate(radial_u_parts)
    radial_du_weights = np.concatenate(radial_du_weight_parts)
    radial_panel_index = np.concatenate(panel_index_parts)
    angular_phi = _TWO_PI * (np.arange(nphi, dtype=float) + offset) / float(nphi)
    angular_weight = _TWO_PI / float(nphi)

    u_mesh, phi_mesh = np.meshgrid(radial_u, angular_phi, indexing="ij")
    radial_weight_mesh, _ = np.meshgrid(
        radial_du_weights,
        angular_phi,
        indexing="ij",
    )
    q_radius = u_mesh / (2.0 * d)
    qx = q_radius * np.cos(phi_mesh)
    qy = q_radius * np.sin(phi_mesh)
    q_si = np.column_stack([qx.ravel(), qy.ravel()])
    q_model = np.column_stack([(ax * qx).ravel(), (ay * qy).ravel()])
    weights = (
        u_mesh
        * radial_weight_mesh
        * angular_weight
        / (16.0 * np.pi**2 * d**2)
    ).ravel()

    upper = edges[-1]
    exact_measure = upper**2 / (16.0 * np.pi * d**2)
    weight_sum = float(np.sum(weights))
    weight_error = abs(weight_sum - exact_measure)
    tolerance = 128.0 * np.finfo(float).eps * max(exact_measure, 1.0)
    if weight_error > tolerance:
        raise RuntimeError("compound outer-q weights fail the exact disk-measure check")

    total_radial_order = len(radial_u)
    return OuterQPolarGrid(
        u=u_mesh.ravel(),
        phi_rad=phi_mesh.ravel(),
        q_si_m_inv=q_si,
        q_model=q_model,
        measure_weights_m_inv2=weights,
        separation_m=d,
        lattice_a_x_m=ax,
        lattice_a_y_m=ay,
        u_max=upper,
        radial_order=total_radial_order,
        angular_order=nphi,
        angular_offset_fraction=offset,
        metadata={
            "schema": "outer-q-composite-polar-grid-v1",
            "radial_variable": "u = 2 Q d",
            "measure_formula": "d^2Q/(2pi)^2 = u du dphi/(16 pi^2 d^2)",
            "model_momentum_formula": "q_model = (a_x Q_x, a_y Q_y)",
            "radial_rule": "composite Gauss-Legendre on nested cumulative panels",
            "radial_panel_edges": list(edges),
            "radial_panel_count": len(edges) - 1,
            "radial_panel_order": panel_order,
            "radial_panel_index_by_radial_node": radial_panel_index.tolist(),
            "total_radial_nodes": total_radial_order,
            "nested_cutoff_node_reuse": True,
            "angular_rule": "full-period equal-weight trapezoidal",
            "angular_symmetry_reduction": False,
            "q_zero_node_present": False,
            "q_max_m_inv": upper / (2.0 * d),
            "weight_sum_m_inv2": weight_sum,
            "exact_disk_measure_m_inv2": exact_measure,
            "absolute_weight_sum_error_m_inv2": weight_error,
            "max_abs_q_model_x": float(np.max(np.abs(q_model[:, 0]))),
            "max_abs_q_model_y": float(np.max(np.abs(q_model[:, 1]))),
        },
    )


__all__ = ["build_compound_outer_q_polar_grid"]
