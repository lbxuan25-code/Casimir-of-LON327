"""Diagnostic equivalence checks for TODO 4 geometry assembly.

The core planner and executor never import the legacy point engine. This module
is the only quarantine boundary allowed to compare the new persisted-response
geometry path with the archived geometry-specific point path.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.casimir.material_geometry_batch import (
    GeometryBatchResult,
)
from lno327.casimir.material_two_plate import (
    TwoPlateGeometryPolicy,
    assemble_two_plate_logdet,
)
from lno327.electrodynamics.basis import q_lab_to_crystal

MATERIAL_GEOMETRY_QUALIFICATION_SCHEMA = "material-geometry-qualification-v1"


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _comparison(
    reference: float,
    candidate: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    left = float(reference)
    right = float(candidate)
    finite = bool(np.isfinite(left) and np.isfinite(right))
    absolute = abs(right - left) if finite else float("inf")
    scale = max(abs(left), abs(right), np.finfo(float).tiny) if finite else 1.0
    relative = absolute / scale if finite else float("inf")
    absolute_passed = bool(finite and absolute <= absolute_tolerance)
    relative_passed = bool(finite and relative <= relative_tolerance)
    return {
        "reference": left,
        "candidate": right,
        "finite": finite,
        "absolute": absolute,
        "relative": relative,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "passed_by": (
            "absolute"
            if absolute_passed
            else "relative"
            if relative_passed
            else "failed"
        ),
        "passed": bool(absolute_passed or relative_passed),
    }


@dataclass(frozen=True)
class GeometryEquivalencePolicy:
    """Uniform scalar comparison policy for diagnostic qualification."""

    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-10
    matrix_absolute_tolerance: float = 1e-12
    matrix_relative_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        for name in (
            "absolute_tolerance",
            "relative_tolerance",
            "matrix_absolute_tolerance",
            "matrix_relative_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )


@dataclass(frozen=True)
class GeometryEquivalenceReport:
    """Diagnostic-only old/new or scalar/batch comparison report."""

    mode: str
    point_id: str
    comparisons: Mapping[str, Any]
    passed: bool
    metadata: Mapping[str, Any]
    schema: str = MATERIAL_GEOMETRY_QUALIFICATION_SCHEMA
    production_casimir_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_GEOMETRY_QUALIFICATION_SCHEMA:
            raise ValueError("unsupported geometry qualification schema")
        mode = str(self.mode)
        if mode not in {"scalar_vs_batch", "legacy_vs_persisted_batch"}:
            raise ValueError("unsupported geometry qualification mode")
        object.__setattr__(self, "mode", mode)
        point_id = str(self.point_id)
        if not point_id:
            raise ValueError("point_id must be nonempty")
        object.__setattr__(self, "point_id", point_id)
        object.__setattr__(
            self,
            "comparisons",
            MappingProxyType(dict(self.comparisons)),
        )
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if bool(self.production_casimir_allowed):
            raise ValueError("TODO 4 qualification cannot admit production")
        object.__setattr__(self, "production_casimir_allowed", False)


def qualify_batch_point_against_scalar(
    batch: GeometryBatchResult,
    *,
    point_id: str,
    policy: GeometryEquivalencePolicy | None = None,
) -> GeometryEquivalenceReport:
    """Recompute each distance through the scalar assembler and compare."""

    if not isinstance(batch, GeometryBatchResult):
        raise TypeError("batch must be a GeometryBatchResult")
    comparison_policy = GeometryEquivalencePolicy() if policy is None else policy
    if not isinstance(comparison_policy, GeometryEquivalencePolicy):
        raise TypeError("policy must be a GeometryEquivalencePolicy")
    candidate = batch.points[str(point_id)]
    spec = candidate.spec
    first = batch.preflight.snapshots[spec.plate_1_requirement]
    second = batch.preflight.snapshots[spec.plate_2_requirement]

    comparisons: dict[str, Any] = {}
    for distance, batch_point in zip(candidate.distances_m, candidate.lifshitz_points):
        scalar = assemble_two_plate_logdet(
            first,
            second,
            q_lab=spec.q_lab,
            theta_1_rad=spec.theta_1_rad,
            theta_2_rad=spec.theta_2_rad,
            policy=TwoPlateGeometryPolicy(
                separation_m=distance,
                reflection_policy=batch.plan.policy.reflection_policy,
                compatibility_tolerance=batch.plan.policy.compatibility_tolerance,
                eigenvalue_imag_tolerance=batch.plan.policy.eigenvalue_imag_tolerance,
                eigenvalue_lower_tolerance=batch.plan.policy.eigenvalue_lower_tolerance,
            ),
        )
        row = _comparison(
            scalar.logdet,
            batch_point.logdet,
            absolute_tolerance=comparison_policy.absolute_tolerance,
            relative_tolerance=comparison_policy.relative_tolerance,
        )
        row["trace_log_matrix_close"] = bool(
            np.allclose(
                scalar.point.trace_log_matrix,
                batch_point.trace_log_matrix,
                rtol=comparison_policy.matrix_relative_tolerance,
                atol=comparison_policy.matrix_absolute_tolerance,
            )
        )
        row["passed"] = bool(row["passed"] and row["trace_log_matrix_close"])
        comparisons[float(distance).hex()] = row

    return GeometryEquivalenceReport(
        mode="scalar_vs_batch",
        point_id=spec.point_id,
        comparisons=comparisons,
        passed=all(bool(row["passed"]) for row in comparisons.values()),
        metadata={
            "reference_route": "assemble_two_plate_logdet_scalar",
            "candidate_route": "prepared_read_only_geometry_batch",
            "same_persisted_responses": True,
            "microscopic_calls": 0,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
    )


def qualify_matched_legacy_point(
    batch: GeometryBatchResult,
    *,
    point_id: str,
    distance_m: float,
    legacy_batch: object,
    legacy_frequency_index: int,
    legacy_n: int,
    legacy_xi_eV: float,
    legacy_args: object,
    policy: GeometryEquivalencePolicy | None = None,
) -> GeometryEquivalenceReport:
    """Compare a matched-N/shift archived point with one persisted batch point.

    ``legacy_batch`` must be the result of the archived
    ``integrate_two_plate_angle_batch`` path at exactly the same q, angles,
    frequency, N, shift, and physical policy. The archived private plate helper
    is intentionally imported only inside this qualification boundary.
    """

    if not isinstance(batch, GeometryBatchResult):
        raise TypeError("batch must be a GeometryBatchResult")
    comparison_policy = GeometryEquivalencePolicy() if policy is None else policy
    if not isinstance(comparison_policy, GeometryEquivalencePolicy):
        raise TypeError("policy must be a GeometryEquivalencePolicy")
    point = batch.points[str(point_id)]
    spec = point.spec
    distance = float(distance_m)
    try:
        distance_index = point.distances_m.index(distance)
    except ValueError as exc:
        raise ValueError("distance_m is absent from the geometry batch") from exc

    angles = tuple(float(value) for value in getattr(legacy_args, "plate_angles_rad"))
    if angles != (spec.theta_1_rad, spec.theta_2_rad):
        raise ValueError("legacy plate angles differ from geometry point")
    q_lab = np.asarray(getattr(legacy_batch, "q_lab"), dtype=float)
    if not np.array_equal(q_lab, spec.q_lab):
        raise ValueError("legacy q_lab differs from geometry point")

    from lno327.casimir import fixed_transverse_point_engine as legacy_engine

    reflection_1, legacy_plate_1 = legacy_engine._plate_state(
        legacy_batch.plate_1,
        frequency_index=int(legacy_frequency_index),
        q_lab=q_lab,
        theta_rad=angles[0],
        xi_eV=float(legacy_xi_eV),
        args=legacy_args,
    )
    reflection_2, legacy_plate_2 = legacy_engine._plate_state(
        legacy_batch.plate_2[0],
        frequency_index=int(legacy_frequency_index),
        q_lab=q_lab,
        theta_rad=angles[1],
        xi_eV=float(legacy_xi_eV),
        args=legacy_args,
    )
    legacy_point = passive_sheet_logdet(
        reflection_1,
        reflection_2,
        separation_m=distance,
        compatibility_tolerance=batch.plan.policy.compatibility_tolerance,
        eigenvalue_imag_tolerance=batch.plan.policy.eigenvalue_imag_tolerance,
        eigenvalue_lower_tolerance=batch.plan.policy.eigenvalue_lower_tolerance,
    )
    candidate = point.lifshitz_points[distance_index]
    logdet = _comparison(
        legacy_point.logdet,
        candidate.logdet,
        absolute_tolerance=comparison_policy.absolute_tolerance,
        relative_tolerance=comparison_policy.relative_tolerance,
    )
    product_close = bool(
        np.allclose(
            reflection_1.matrix_lt @ reflection_2.matrix_lt,
            point.prepared_pair.product_matrix,
            rtol=comparison_policy.matrix_relative_tolerance,
            atol=comparison_policy.matrix_absolute_tolerance,
        )
    )
    eigenvalues_close = bool(
        np.allclose(
            np.sort(legacy_point.product_eigenvalues),
            np.sort(candidate.product_eigenvalues),
            rtol=comparison_policy.matrix_relative_tolerance,
            atol=comparison_policy.matrix_absolute_tolerance,
        )
    )
    expected_q_1 = q_lab_to_crystal(q_lab, spec.theta_1_rad)
    expected_q_2 = q_lab_to_crystal(q_lab, spec.theta_2_rad)
    legacy_q_exact = bool(
        np.array_equal(np.asarray(legacy_batch.plate_1.q_model), expected_q_1)
        and np.array_equal(np.asarray(legacy_batch.plate_2[0].q_model), expected_q_2)
    )
    comparisons = {
        "logdet": logdet,
        "round_trip_product_matrix_close": product_close,
        "product_eigenvalues_close": eigenvalues_close,
        "legacy_exact_q_mapping": legacy_q_exact,
    }
    passed = bool(
        logdet["passed"]
        and product_close
        and eigenvalues_close
        and legacy_q_exact
        and legacy_plate_1["hard_physical_passed"]
        and legacy_plate_2["hard_physical_passed"]
    )
    return GeometryEquivalenceReport(
        mode="legacy_vs_persisted_batch",
        point_id=spec.point_id,
        comparisons=comparisons,
        passed=passed,
        metadata={
            "legacy_route": "fixed_transverse_point_engine._plate_state",
            "candidate_route": "persistent_response_geometry_batch",
            "legacy_N": int(legacy_n),
            "matched_N_shift_policy_required": True,
            "qualification_boundary_only_imports_legacy_engine": True,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
    )


__all__ = [
    "GeometryEquivalencePolicy",
    "GeometryEquivalenceReport",
    "MATERIAL_GEOMETRY_QUALIFICATION_SCHEMA",
    "qualify_batch_point_against_scalar",
    "qualify_matched_legacy_point",
]
