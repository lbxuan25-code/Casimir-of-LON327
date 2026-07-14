"""Quadrature-independent finite-q primitive evaluation and packing.

The functions in this module own the shared microscopic execution path used by
both commensurate complete-orbit validation and the arbitrary-q periodic-BZ
backend. They deliberately stop at linear primitive blocks. Collective Schur,
phase-Hessian, sheet, reflection, and logdet operations happen only after the
full Brillouin-zone integral is complete.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.response.finite_q_bdg import _finalize_components
from lno327.response.finite_q_optimized import (
    FiniteQMaterialWorkspace,
    _vectorized_kubo_factors,
)
from lno327.response.finite_q_q_workspace_batched import (
    precompute_finite_q_q_workspace_batched,
)
from lno327.response.phase_hessian import apply_phase_hessian_policy_to_components
from lno327.response.ward_validation import PrimitiveWardRHS

_HEADER_WIDTH = 9 + 4 + 1 + 1 + 3
_PER_FREQUENCY_WIDTH = 9 + 4 + 6 + 6
_FLOAT_EPS = np.finfo(float).eps


def primitive_vector_width(frequency_count: int) -> int:
    count = int(frequency_count)
    if count <= 0:
        raise ValueError("frequency_count must be positive")
    return _HEADER_WIDTH + _PER_FREQUENCY_WIDTH * count


def counterterm_primitive_vector(
    counterterm: np.ndarray,
    *,
    frequency_count: int,
) -> np.ndarray:
    vector = np.zeros(primitive_vector_width(frequency_count), dtype=complex)
    matrix = np.asarray(counterterm, dtype=complex)
    if matrix.shape != (2, 2):
        raise ValueError("counterterm must have shape (2,2)")
    vector[9:13] = matrix.reshape(-1)
    return vector


def pack_integrated_primitives(
    *,
    workspace: object,
    blocks: np.ndarray,
    include_counterterm: bool = True,
) -> np.ndarray:
    """Pack one already-integrated linear primitive result.

    The layout is stable and intentionally shared with the historical orbit
    evaluator. ``include_counterterm=False`` is used by streamed arbitrary-q
    chunks so the full-BZ Goldstone/HS counterterm can be added exactly once.
    """

    block_array = np.asarray(blocks, dtype=complex)
    if block_array.ndim != 3 or block_array.shape[1:] != (5, 5):
        raise ValueError("blocks must have shape (n_frequency,5,5)")

    direct = np.asarray(workspace.direct_contact_contribution, dtype=complex)
    if direct.shape != (3, 3):
        raise ValueError("direct_contact_contribution must have shape (3,3)")
    counterterm = np.asarray(
        workspace.material.collective_counterterm_matrix,
        dtype=complex,
    )
    if counterterm.shape != (2, 2):
        raise ValueError("collective_counterterm_matrix must have shape (2,2)")
    if not include_counterterm:
        counterterm = np.zeros_like(counterterm)

    header = np.concatenate(
        (
            direct.reshape(-1),
            counterterm.reshape(-1),
            np.asarray(
                [
                    workspace.phase_phase_direct_plus,
                    workspace.phase_phase_direct_minus,
                ],
                dtype=complex,
            ),
            np.asarray(workspace.ward_rhs_vector, dtype=complex).reshape(-1),
        )
    )
    dynamic: list[np.ndarray] = []
    for block in block_array:
        dynamic.extend(
            (
                block[:3, :3].reshape(-1),
                block[3:5, 3:5].reshape(-1),
                block[:3, 3:5].reshape(-1),
                block[3:5, :3].reshape(-1),
            )
        )
    return np.concatenate((header, *dynamic))


@dataclass(frozen=True)
class OperatorWardReport:
    """Peierls operator identity diagnostics on one k batch.

    This is deliberately not named a pointwise response Ward identity. It
    checks only exact operator-level algebra for which machine-scale residuals
    are expected before any k integration.
    """

    point_count: int
    max_absolute_error: float
    max_relative_error: float
    max_mixed_ratio: float
    failed_points: int
    atol: float
    rtol: float
    passed: bool

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "identity": "normal_peierls_operator_qV_equals_hplus_minus_hminus",
            "point_count": int(self.point_count),
            "max_absolute_error": float(self.max_absolute_error),
            "max_relative_error": float(self.max_relative_error),
            "max_mixed_ratio": float(self.max_mixed_ratio),
            "failed_points": int(self.failed_points),
            "atol": float(self.atol),
            "rtol": float(self.rtol),
            "passed": bool(self.passed),
        }


def evaluate_peierls_operator_identity_batch(
    spec: object,
    k_points: np.ndarray,
    q_model: np.ndarray,
    *,
    atol: float = 512.0 * _FLOAT_EPS,
    rtol: float = 512.0 * _FLOAT_EPS,
) -> OperatorWardReport:
    points = np.asarray(k_points, dtype=float)
    q = np.asarray(q_model, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("k_points must have shape (n,2) and be nonempty")
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")
    if (
        not np.isfinite(atol)
        or not np.isfinite(rtol)
        or atol <= 0.0
        or rtol <= 0.0
    ):
        raise ValueError("operator Ward tolerances must be finite and positive")

    vector_fn = getattr(spec, "peierls_hamiltonian_vector_vertices_batch", None)
    normal_fn = getattr(spec, "normal_hamiltonian_batch", None)
    if not callable(vector_fn) or not callable(normal_fn):
        raise ValueError(
            "spec must provide batched normal Hamiltonians and Peierls vector vertices"
        )

    vertices = np.asarray(vector_fn(points, q), dtype=complex)
    if (
        vertices.ndim != 4
        or vertices.shape[0] != points.shape[0]
        or vertices.shape[1] != 2
    ):
        raise ValueError(
            "batched Peierls vector vertices must have shape (nk,2,nb,nb)"
        )
    lhs = np.einsum("i,kiab->kab", q, vertices, optimize=True)
    q_half = 0.5 * q
    h_plus = np.asarray(normal_fn(points + q_half), dtype=complex)
    h_minus = np.asarray(normal_fn(points - q_half), dtype=complex)
    rhs = h_plus - h_minus
    if lhs.shape != rhs.shape:
        raise ValueError("operator Ward lhs/rhs shapes do not match")

    delta_norm = np.linalg.norm(lhs - rhs, axis=(-2, -1))
    lhs_norm = np.linalg.norm(lhs, axis=(-2, -1))
    rhs_norm = np.linalg.norm(rhs, axis=(-2, -1))
    scale = np.maximum(lhs_norm, rhs_norm)
    threshold = float(atol) + float(rtol) * scale
    mixed = delta_norm / np.maximum(threshold, np.finfo(float).tiny)
    relative = delta_norm / np.maximum(scale, np.finfo(float).tiny)
    finite = np.isfinite(mixed)
    failed = (~finite) | (mixed > 1.0)
    return OperatorWardReport(
        point_count=int(points.shape[0]),
        max_absolute_error=float(np.max(delta_norm)),
        max_relative_error=float(np.max(relative)),
        max_mixed_ratio=float(np.max(mixed)),
        failed_points=int(np.count_nonzero(failed)),
        atol=float(atol),
        rtol=float(rtol),
        passed=bool(not np.any(failed)),
    )


@dataclass(frozen=True)
class PrimitiveBatchMetrics:
    k_point_count: int
    frequency_count: int
    q_workspace_seconds: float
    kubo_factor_seconds: float
    kubo_contraction_seconds: float
    primitive_pack_seconds: float
    shifted_eigensystem_build_count: int
    q_workspace_implementation: str

    @property
    def total_seconds(self) -> float:
        return float(
            self.q_workspace_seconds
            + self.kubo_factor_seconds
            + self.kubo_contraction_seconds
            + self.primitive_pack_seconds
        )

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "k_point_count": int(self.k_point_count),
            "frequency_count": int(self.frequency_count),
            "q_workspace_seconds": float(self.q_workspace_seconds),
            "kubo_factor_seconds": float(self.kubo_factor_seconds),
            "kubo_contraction_seconds": float(self.kubo_contraction_seconds),
            "primitive_pack_seconds": float(self.primitive_pack_seconds),
            "shifted_eigensystem_build_count": int(
                self.shifted_eigensystem_build_count
            ),
            "q_workspace_implementation": str(self.q_workspace_implementation),
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True)
class PrimitiveBatchResult:
    packed: np.ndarray
    operator_ward: OperatorWardReport
    metrics: PrimitiveBatchMetrics

    def __post_init__(self) -> None:
        value = np.array(self.packed, dtype=complex, copy=True)
        value.setflags(write=False)
        object.__setattr__(self, "packed", value)


def evaluate_primitive_batch_from_material(
    material: FiniteQMaterialWorkspace | object,
    q_model: np.ndarray,
    xi_eV_values: Sequence[float] | np.ndarray,
    *,
    include_counterterm: bool = True,
    operator_ward_atol: float = 512.0 * _FLOAT_EPS,
    operator_ward_rtol: float = 512.0 * _FLOAT_EPS,
) -> PrimitiveBatchResult:
    """Evaluate one k batch for exact q and all requested Matsubara frequencies."""

    xi_values = np.asarray(xi_eV_values, dtype=float)
    if xi_values.ndim != 1 or xi_values.size == 0:
        raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
    if not np.isfinite(xi_values).all() or np.any(xi_values < 0.0):
        raise ValueError("xi_eV_values must be finite and non-negative")
    if np.count_nonzero(xi_values == 0.0) > 1:
        raise ValueError("exact zero may appear at most once")

    operator_ward = evaluate_peierls_operator_identity_batch(
        material.spec,
        material.k_points,
        q_model,
        atol=operator_ward_atol,
        rtol=operator_ward_rtol,
    )

    started = perf_counter()
    workspace = precompute_finite_q_q_workspace_batched(material, q_model)
    q_workspace_seconds = perf_counter() - started

    started = perf_counter()
    raw_factors = _vectorized_kubo_factors(workspace, xi_values)
    kubo_factor_seconds = perf_counter() - started

    started = perf_counter()
    weighted = (
        0.5
        * np.asarray(workspace.material.k_weights, dtype=float)[None, :, None, None]
        * raw_factors
    )
    blocks = np.einsum(
        "xkmn,kamn,kbmn->xab",
        weighted,
        workspace.left_vertices_band,
        np.conjugate(workspace.right_vertices_band),
        optimize=True,
    )
    kubo_contraction_seconds = perf_counter() - started

    started = perf_counter()
    packed = pack_integrated_primitives(
        workspace=workspace,
        blocks=blocks,
        include_counterterm=include_counterterm,
    )
    primitive_pack_seconds = perf_counter() - started
    q_nonzero = bool(np.linalg.norm(np.asarray(q_model, dtype=float)) > 1e-14)
    metrics = PrimitiveBatchMetrics(
        k_point_count=int(material.k_points.shape[0]),
        frequency_count=int(xi_values.size),
        q_workspace_seconds=float(q_workspace_seconds),
        kubo_factor_seconds=float(kubo_factor_seconds),
        kubo_contraction_seconds=float(kubo_contraction_seconds),
        primitive_pack_seconds=float(primitive_pack_seconds),
        shifted_eigensystem_build_count=(2 if q_nonzero else 0),
        q_workspace_implementation=str(
            workspace.metadata.get("q_workspace_implementation", "unknown")
        ),
    )
    return PrimitiveBatchResult(
        packed=np.asarray(packed, dtype=complex),
        operator_ward=operator_ward,
        metrics=metrics,
    )


def unpack_integrated_primitives(
    packed: np.ndarray,
    *,
    xi_values: Sequence[float] | np.ndarray,
    ansatz: object,
    pairing: object,
    base_config: object,
    q_model: np.ndarray,
    options: object,
    phase_hessian_policy: str,
    integration_metadata: Mapping[str, Any],
    rhs_source: str,
) -> tuple[tuple[object, ...], tuple[PrimitiveWardRHS, ...]]:
    """Finalize one full-BZ primitive vector after all linear accumulation."""

    xi = np.asarray(xi_values, dtype=float)
    vector = np.asarray(packed, dtype=complex).reshape(-1)
    expected = primitive_vector_width(int(xi.size))
    if vector.size != expected:
        raise ValueError(
            f"packed primitive width {vector.size} does not match expected {expected}"
        )
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")

    offset = 0
    direct = vector[offset : offset + 9].reshape(3, 3)
    offset += 9
    counterterm = vector[offset : offset + 4].reshape(2, 2)
    offset += 4
    phase_direct_plus = complex(vector[offset])
    phase_direct_minus = complex(vector[offset + 1])
    offset += 2
    rhs_vector = vector[offset : offset + 3]
    offset += 3

    delta0 = float(getattr(pairing, "delta0_eV"))
    components_values: list[object] = []
    rhs_values: list[PrimitiveWardRHS] = []
    common_metadata = dict(integration_metadata)
    common_metadata.update(
        {
            "primitive_vector_integrated_before_schur": True,
            "post_integral_phase_hessian_policy": str(phase_hessian_policy),
            "projection_applied": False,
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        }
    )

    for frequency in xi:
        bubble = vector[offset : offset + 9].reshape(3, 3)
        offset += 9
        collective_bubble = vector[offset : offset + 4].reshape(2, 2)
        offset += 4
        em_collective_left = vector[offset : offset + 6].reshape(3, 2)
        offset += 6
        collective_em_right = vector[offset : offset + 6].reshape(2, 3)
        offset += 6

        phase_left = delta0 * em_collective_left[:, 1]
        phase_right = delta0 * collective_em_right[1, :]
        phase_phase_bubble = np.asarray(
            [[delta0 * delta0 * collective_bubble[1, 1]]],
            dtype=complex,
        )
        eval_config = replace(base_config, omega_eV=float(frequency))
        base = _finalize_components(
            ansatz=ansatz,
            opts=options,
            shared_eigenbasis_q0=bool(np.linalg.norm(q) <= 1e-14),
            shared_eigenbasis_q0_tolerance=1e-14,
            collective_mode="amplitude_phase",
            collective_mode_disabled_reason=None,
            bubble=bubble,
            direct=direct,
            phase_left=phase_left,
            phase_right=phase_right,
            phase_phase_bubble_matrix=phase_phase_bubble,
            phase_phase_direct_plus=phase_direct_plus,
            phase_phase_direct_minus=phase_direct_minus,
            collective_bubble=collective_bubble,
            collective_counterterm_matrix=counterterm,
            em_collective_left=em_collective_left,
            collective_em_right=collective_em_right,
            config=eval_config,
            q=q,
            workspace_evaluation=True,
        )
        corrected, _ = apply_phase_hessian_policy_to_components(
            base,
            ansatz,
            q,
            phase_hessian_policy,
        )
        corrected = replace(
            corrected,
            metadata={**dict(corrected.metadata), **common_metadata},
        )
        components_values.append(corrected)
        rhs_values.append(
            PrimitiveWardRHS(
                left=rhs_vector,
                right=rhs_vector.copy(),
                q_model=q,
                xi_eV=float(frequency),
                delta0_eV=delta0,
                metadata={
                    "convention": "primitive_crystal_xy_rhs_aware",
                    "basis": "crystal_A0_xy",
                    "formula": "R_S = equal_forward - delta_v_mid + qM_mid",
                    "source": str(rhs_source),
                    "frequency_independent_rhs_reused": True,
                    **common_metadata,
                },
            )
        )

    if offset != vector.size:
        raise RuntimeError("integrated primitive unpack did not consume the full vector")
    return tuple(components_values), tuple(rhs_values)


__all__ = [
    "OperatorWardReport",
    "PrimitiveBatchMetrics",
    "PrimitiveBatchResult",
    "counterterm_primitive_vector",
    "evaluate_peierls_operator_identity_batch",
    "evaluate_primitive_batch_from_material",
    "pack_integrated_primitives",
    "primitive_vector_width",
    "unpack_integrated_primitives",
]
