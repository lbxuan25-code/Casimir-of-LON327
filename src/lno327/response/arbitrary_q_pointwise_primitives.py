"""Quadrature-independent pointwise arbitrary-q primitive densities.

The periodic-BZ backend already owns the microscopic q-workspace, exact zero-
frequency divided differences, positive Matsubara factors, Peierls vertices,
collective vertices, contact terms, phase-direct term, and analytic Ward RHS.
This module exposes those same linear quantities before k integration so a
quadrature may supply arbitrary nodes and weights without duplicating physics.

Counterterm semantics are also pointwise: the q=0 eta2 bubble density is evaluated
on the same nodes and the full-BZ Goldstone counterterm is obtained only after the
quadrature weights are applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from lno327.response.finite_q_material_workspace_batched import (
    precompute_finite_q_material_workspace_batched,
)
from lno327.response.finite_q_optimized import _vectorized_kubo_factors
from lno327.response.finite_q_q_workspace_batched import (
    _integrated_linear_terms_from_workspace_slice,
    precompute_finite_q_q_workspace_batched,
)
from lno327.response.primitive_kernel_v2 import (
    OperatorWardReport,
    operator_ward_report_from_workspace,
    pack_integrated_primitives,
    primitive_vector_width,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions


@dataclass(frozen=True)
class PointwisePrimitiveMetrics:
    k_point_count: int
    frequency_count: int
    material_workspace_seconds: float
    q_workspace_seconds: float
    counterterm_q0_workspace_seconds: float
    kubo_factor_seconds: float
    counterterm_factor_seconds: float
    primitive_density_seconds: float
    shifted_eigensystem_build_count: int
    q_workspace_build_count: int
    counterterm_q0_workspace_build_count: int

    @property
    def total_seconds(self) -> float:
        return float(
            self.material_workspace_seconds
            + self.q_workspace_seconds
            + self.counterterm_q0_workspace_seconds
            + self.kubo_factor_seconds
            + self.counterterm_factor_seconds
            + self.primitive_density_seconds
        )

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "schema": "ArbitraryQPointwisePrimitiveMetrics-v1",
            "k_point_count": int(self.k_point_count),
            "frequency_count": int(self.frequency_count),
            "material_workspace_seconds": float(self.material_workspace_seconds),
            "q_workspace_seconds": float(self.q_workspace_seconds),
            "counterterm_q0_workspace_seconds": float(
                self.counterterm_q0_workspace_seconds
            ),
            "kubo_factor_seconds": float(self.kubo_factor_seconds),
            "counterterm_factor_seconds": float(self.counterterm_factor_seconds),
            "primitive_density_seconds": float(self.primitive_density_seconds),
            "shifted_eigensystem_build_count": int(
                self.shifted_eigensystem_build_count
            ),
            "q_workspace_build_count": int(self.q_workspace_build_count),
            "counterterm_q0_workspace_build_count": int(
                self.counterterm_q0_workspace_build_count
            ),
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True)
class PointwisePrimitiveBatch:
    densities: np.ndarray
    operator_ward: OperatorWardReport
    metrics: PointwisePrimitiveMetrics

    def __post_init__(self) -> None:
        values = np.array(self.densities, dtype=complex, copy=True)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("pointwise primitive densities must be a finite matrix")
        values.setflags(write=False)
        object.__setattr__(self, "densities", values)


def pack_complex_density_to_real(value: np.ndarray) -> np.ndarray:
    """Pack complex primitive densities for a real-vector adaptive integrator."""

    array = np.asarray(value, dtype=complex)
    if array.ndim < 1:
        raise ValueError("complex primitive density must have at least one dimension")
    return np.concatenate((array.real, array.imag), axis=-1)


def unpack_real_integral_to_complex(value: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_complex_density_to_real` for one integrated vector."""

    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0 or array.size % 2:
        raise ValueError("packed real primitive integral must have even nonzero width")
    half = array.size // 2
    return np.asarray(array[:half] + 1j * array[half:], dtype=complex)


def _validate_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise ValueError("k_points must have shape (n,2) and be nonempty")
    if not np.isfinite(array).all():
        raise ValueError("k_points must be finite")
    return array


def _validate_frequencies(values: Sequence[float] | np.ndarray) -> np.ndarray:
    xi = np.asarray(values, dtype=float)
    if xi.ndim != 1 or xi.size == 0:
        raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
    if not np.isfinite(xi).all() or np.any(xi < 0.0):
        raise ValueError("xi_eV_values must be finite and non-negative")
    if np.count_nonzero(xi == 0.0) > 1:
        raise ValueError("exact zero may appear at most once")
    return xi


def _pointwise_counterterm_density(
    workspace_q0: object,
    factor_q0: np.ndarray,
    index: int,
) -> np.ndarray:
    """Return the q-independent Goldstone counterterm density at one k point."""

    eta2 = 4
    factor = np.asarray(factor_q0, dtype=complex)[0, index]
    left = np.asarray(workspace_q0.left_vertices_band, dtype=complex)[index, eta2]
    right = np.asarray(workspace_q0.right_vertices_band, dtype=complex)[index, eta2]
    eta2_bubble_density = 0.5 * np.einsum(
        "mn,mn,mn->",
        factor,
        left,
        np.conjugate(right),
        optimize=True,
    )
    return -complex(eta2_bubble_density) * np.eye(2, dtype=complex)


def evaluate_arbitrary_q_pointwise_primitives(
    *,
    spec: object,
    ansatz: object,
    pairing: object,
    config: object,
    q_model: np.ndarray,
    xi_eV_values: Sequence[float] | np.ndarray,
    k_points: np.ndarray,
    options: object | None = None,
    operator_ward_atol: float = 512.0 * np.finfo(float).eps,
    operator_ward_rtol: float = 512.0 * np.finfo(float).eps,
) -> PointwisePrimitiveBatch:
    """Evaluate unweighted linear primitive densities on arbitrary BZ points.

    A normalized probe weight is supplied only because the existing material/q
    workspace validates BZ weights. Every integrated linear term is divided by
    that probe weight before return, so the result is quadrature independent.
    """

    points = _validate_points(k_points)
    xi_values = _validate_frequencies(xi_eV_values)
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")
    opts = options or FiniteQEngineOptions(phase_hessian_policy="q_independent")
    probe_weight = 1.0 / float(points.shape[0])
    probe_weights = np.full(points.shape[0], probe_weight, dtype=float)

    started = perf_counter()
    material = precompute_finite_q_material_workspace_batched(
        spec,
        ansatz,
        points,
        probe_weights,
        config,
        pairing,
        opts,
    )
    material_seconds = perf_counter() - started

    started = perf_counter()
    workspace = precompute_finite_q_q_workspace_batched(
        material,
        q,
        operator_diagnostics=True,
    )
    q_workspace_seconds = perf_counter() - started
    operator_ward = operator_ward_report_from_workspace(
        workspace,
        atol=operator_ward_atol,
        rtol=operator_ward_rtol,
    )

    started = perf_counter()
    factors = _vectorized_kubo_factors(workspace, xi_values)
    kubo_factor_seconds = perf_counter() - started

    shared_q0 = bool(np.linalg.norm(q) <= 1e-14)
    if shared_q0:
        workspace_q0 = workspace
        q0_workspace_seconds = 0.0
    else:
        started = perf_counter()
        workspace_q0 = precompute_finite_q_q_workspace_batched(
            material,
            np.zeros(2, dtype=float),
            operator_diagnostics=False,
        )
        q0_workspace_seconds = perf_counter() - started

    started = perf_counter()
    if shared_q0 and np.any(xi_values == 0.0):
        zero_index = int(np.flatnonzero(xi_values == 0.0)[0])
        factors_q0 = np.asarray(factors[zero_index : zero_index + 1], dtype=complex)
    else:
        factors_q0 = _vectorized_kubo_factors(
            workspace_q0,
            np.asarray([0.0], dtype=float),
        )
    counterterm_factor_seconds = perf_counter() - started

    width = primitive_vector_width(int(xi_values.size))
    densities = np.zeros((points.shape[0], width), dtype=complex)
    started = perf_counter()
    left_vertices = np.asarray(workspace.left_vertices_band, dtype=complex)
    right_vertices = np.asarray(workspace.right_vertices_band, dtype=complex)
    factor_array = np.asarray(factors, dtype=complex)
    for index in range(points.shape[0]):
        direct, phase_plus, ward_rhs = _integrated_linear_terms_from_workspace_slice(
            workspace,
            index,
            index + 1,
        )
        direct = np.asarray(direct, dtype=complex) / probe_weight
        phase_plus = complex(phase_plus) / probe_weight
        ward_rhs = np.asarray(ward_rhs, dtype=complex) / probe_weight

        blocks = 0.5 * np.einsum(
            "xmn,amn,bmn->xab",
            factor_array[:, index],
            left_vertices[index],
            np.conjugate(right_vertices[index]),
            optimize=True,
        )
        counterterm = _pointwise_counterterm_density(
            workspace_q0,
            factors_q0,
            index,
        )
        point_workspace = SimpleNamespace(
            direct_contact_contribution=direct,
            phase_phase_direct_plus=phase_plus,
            phase_phase_direct_minus=-phase_plus,
            ward_rhs_vector=ward_rhs,
            material=SimpleNamespace(collective_counterterm_matrix=counterterm),
        )
        densities[index] = pack_integrated_primitives(
            workspace=point_workspace,
            blocks=blocks,
            include_counterterm=True,
        )
    primitive_density_seconds = perf_counter() - started

    metrics = PointwisePrimitiveMetrics(
        k_point_count=int(points.shape[0]),
        frequency_count=int(xi_values.size),
        material_workspace_seconds=float(material_seconds),
        q_workspace_seconds=float(q_workspace_seconds),
        counterterm_q0_workspace_seconds=float(q0_workspace_seconds),
        kubo_factor_seconds=float(kubo_factor_seconds),
        counterterm_factor_seconds=float(counterterm_factor_seconds),
        primitive_density_seconds=float(primitive_density_seconds),
        shifted_eigensystem_build_count=int(
            workspace.metadata.get("shifted_eigh_call_count", 0)
        ),
        q_workspace_build_count=1,
        counterterm_q0_workspace_build_count=(0 if shared_q0 else 1),
    )
    return PointwisePrimitiveBatch(
        densities=densities,
        operator_ward=operator_ward,
        metrics=metrics,
    )


__all__ = [
    "PointwisePrimitiveBatch",
    "PointwisePrimitiveMetrics",
    "evaluate_arbitrary_q_pointwise_primitives",
    "pack_complex_density_to_real",
    "unpack_real_integral_to_complex",
]
