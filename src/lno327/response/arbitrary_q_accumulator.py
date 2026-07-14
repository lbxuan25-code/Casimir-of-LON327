"""Deterministic streamed primitive accumulation for exact arbitrary q."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np

from lno327.response.arbitrary_q_material_cache import MaterialGridCache
from lno327.response.primitive_kernel import (
    OperatorWardReport,
    counterterm_primitive_vector,
    evaluate_primitive_batch_from_material,
    primitive_vector_width,
)


class ComplexKahanVector:
    """Component-wise complex Kahan accumulator with a fixed addition order."""

    def __init__(self, width: int) -> None:
        if int(width) <= 0:
            raise ValueError("Kahan width must be positive")
        self._sum = np.zeros(int(width), dtype=complex)
        self._compensation = np.zeros(int(width), dtype=complex)

    def add(self, value: np.ndarray) -> None:
        vector = np.asarray(value, dtype=complex).reshape(-1)
        if vector.shape != self._sum.shape:
            raise ValueError("Kahan vector width mismatch")
        corrected = vector - self._compensation
        updated = self._sum + corrected
        self._compensation = (updated - self._sum) - corrected
        self._sum = updated

    def value(self) -> np.ndarray:
        result = np.array(self._sum, dtype=complex, copy=True)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class ArbitraryQAccumulationProfile:
    k_point_count: int
    frequency_count: int
    canonical_reduction_block_size: int
    runtime_chunk_size: int
    canonical_block_count: int
    runtime_chunk_count: int
    shifted_eigensystem_build_count: int
    q_workspace_seconds: float
    kubo_factor_seconds: float
    kubo_contraction_seconds: float
    primitive_pack_seconds: float
    operator_ward_seconds: float
    accumulation_seconds: float
    total_seconds: float
    counterterm_add_count: int
    material_cache_fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "k_point_count": int(self.k_point_count),
            "frequency_count": int(self.frequency_count),
            "canonical_reduction_block_size": int(
                self.canonical_reduction_block_size
            ),
            "runtime_chunk_size": int(self.runtime_chunk_size),
            "canonical_block_count": int(self.canonical_block_count),
            "runtime_chunk_count": int(self.runtime_chunk_count),
            "shifted_eigensystem_build_count": int(
                self.shifted_eigensystem_build_count
            ),
            "q_workspace_seconds": float(self.q_workspace_seconds),
            "kubo_factor_seconds": float(self.kubo_factor_seconds),
            "kubo_contraction_seconds": float(self.kubo_contraction_seconds),
            "primitive_pack_seconds": float(self.primitive_pack_seconds),
            "operator_ward_seconds": float(self.operator_ward_seconds),
            "accumulation_seconds": float(self.accumulation_seconds),
            "total_seconds": float(self.total_seconds),
            "counterterm_add_count": int(self.counterterm_add_count),
            "material_cache_fingerprint": self.material_cache_fingerprint,
        }


@dataclass(frozen=True)
class ArbitraryQAccumulationResult:
    packed: np.ndarray
    operator_ward: OperatorWardReport
    profile: ArbitraryQAccumulationProfile

    def __post_init__(self) -> None:
        array = np.array(self.packed, dtype=complex, copy=True)
        array.setflags(write=False)
        object.__setattr__(self, "packed", array)


def _combine_operator_reports(reports: list[OperatorWardReport]) -> OperatorWardReport:
    if not reports:
        raise ValueError("at least one operator Ward report is required")
    atol = reports[0].atol
    rtol = reports[0].rtol
    if any(report.atol != atol or report.rtol != rtol for report in reports):
        raise ValueError("operator Ward tolerances changed across blocks")
    return OperatorWardReport(
        point_count=sum(report.point_count for report in reports),
        max_absolute_error=max(report.max_absolute_error for report in reports),
        max_relative_error=max(report.max_relative_error for report in reports),
        max_mixed_ratio=max(report.max_mixed_ratio for report in reports),
        failed_points=sum(report.failed_points for report in reports),
        atol=atol,
        rtol=rtol,
        passed=all(report.passed for report in reports),
    )


def accumulate_arbitrary_q_primitives(
    cache: MaterialGridCache,
    q_model: np.ndarray,
    xi_eV_values: Sequence[float] | np.ndarray,
    *,
    canonical_reduction_block_size: int = 4096,
    runtime_chunk_size: int = 16384,
    operator_ward_atol: float = 512.0 * np.finfo(float).eps,
    operator_ward_rtol: float = 512.0 * np.finfo(float).eps,
) -> ArbitraryQAccumulationResult:
    """Integrate exact q on the cached full BZ using fixed reduction blocks.

    ``runtime_chunk_size`` groups work for scheduling/profile purposes only.  The
    numerical reduction boundaries are fixed by
    ``canonical_reduction_block_size`` and therefore do not change when runtime
    memory tuning changes.
    """

    xi_values = np.asarray(xi_eV_values, dtype=float)
    if xi_values.ndim != 1 or xi_values.size == 0:
        raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
    if not np.isfinite(xi_values).all() or np.any(xi_values < 0.0):
        raise ValueError("xi_eV_values must be finite and non-negative")
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")

    block_size = int(canonical_reduction_block_size)
    runtime_size = int(runtime_chunk_size)
    if block_size <= 0 or runtime_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if block_size % 2 != 0:
        raise ValueError("canonical reduction block size must be even")
    if runtime_size < block_size or runtime_size % block_size != 0:
        raise ValueError(
            "runtime_chunk_size must be an integer multiple of canonical block size"
        )

    width = primitive_vector_width(int(xi_values.size))
    accumulator = ComplexKahanVector(width)
    reports: list[OperatorWardReport] = []
    point_count = cache.grid.point_count
    canonical_block_count = 0
    runtime_chunk_count = 0
    shifted_builds = 0
    q_workspace_seconds = 0.0
    kubo_factor_seconds = 0.0
    kubo_contraction_seconds = 0.0
    primitive_pack_seconds = 0.0
    operator_ward_seconds = 0.0
    accumulation_seconds = 0.0
    total_started = perf_counter()

    for runtime_start in range(0, point_count, runtime_size):
        runtime_chunk_count += 1
        runtime_stop = min(runtime_start + runtime_size, point_count)
        for block_start in range(runtime_start, runtime_stop, block_size):
            block_stop = min(block_start + block_size, runtime_stop)
            material = cache.chunk_view(block_start, block_stop)
            started = perf_counter()
            result = evaluate_primitive_batch_from_material(
                material,
                q,
                xi_values,
                include_counterterm=False,
                operator_ward_atol=operator_ward_atol,
                operator_ward_rtol=operator_ward_rtol,
            )
            elapsed = perf_counter() - started
            known = result.metrics.total_seconds
            operator_ward_seconds += max(float(elapsed - known), 0.0)
            q_workspace_seconds += result.metrics.q_workspace_seconds
            kubo_factor_seconds += result.metrics.kubo_factor_seconds
            kubo_contraction_seconds += result.metrics.kubo_contraction_seconds
            primitive_pack_seconds += result.metrics.primitive_pack_seconds
            shifted_builds += result.metrics.shifted_eigensystem_build_count
            reports.append(result.operator_ward)
            started = perf_counter()
            accumulator.add(result.packed)
            accumulation_seconds += perf_counter() - started
            canonical_block_count += 1

    # The full-BZ Goldstone/HS counterterm is q independent and is added exactly
    # once after every streamed linear contribution has been accumulated.
    accumulator.add(
        counterterm_primitive_vector(
            cache.counterterm,
            frequency_count=int(xi_values.size),
        )
    )
    total_seconds = perf_counter() - total_started
    profile = ArbitraryQAccumulationProfile(
        k_point_count=point_count,
        frequency_count=int(xi_values.size),
        canonical_reduction_block_size=block_size,
        runtime_chunk_size=runtime_size,
        canonical_block_count=canonical_block_count,
        runtime_chunk_count=runtime_chunk_count,
        shifted_eigensystem_build_count=shifted_builds,
        q_workspace_seconds=float(q_workspace_seconds),
        kubo_factor_seconds=float(kubo_factor_seconds),
        kubo_contraction_seconds=float(kubo_contraction_seconds),
        primitive_pack_seconds=float(primitive_pack_seconds),
        operator_ward_seconds=float(operator_ward_seconds),
        accumulation_seconds=float(accumulation_seconds),
        total_seconds=float(total_seconds),
        counterterm_add_count=1,
        material_cache_fingerprint=cache.fingerprint,
    )
    return ArbitraryQAccumulationResult(
        packed=accumulator.value(),
        operator_ward=_combine_operator_reports(reports),
        profile=profile,
    )


__all__ = [
    "ArbitraryQAccumulationProfile",
    "ArbitraryQAccumulationResult",
    "ComplexKahanVector",
    "accumulate_arbitrary_q_primitives",
]
