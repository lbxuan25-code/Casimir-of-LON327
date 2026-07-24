"""Reduced fixed-outer replay for TODO 4 old/new point arrays.

The function in this module does not evaluate material responses or geometry.
It verifies that two already matched logdet arrays remain equivalent after the
same fixed outer-Q and finite Matsubara reduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.casimir.material_geometry_qualification import GeometryEquivalencePolicy
from lno327.casimir.outer_quadrature import (
    MatsubaraFreeEnergyPerArea,
    OuterQPolarGrid,
    free_energy_per_area_from_logdet,
)

MATERIAL_GEOMETRY_OUTER_QUALIFICATION_SCHEMA = (
    "material-geometry-outer-qualification-v1"
)


def _comparison(
    reference: float,
    candidate: float,
    *,
    policy: GeometryEquivalencePolicy,
) -> dict[str, float | bool | str]:
    left = float(reference)
    right = float(candidate)
    finite = bool(np.isfinite(left) and np.isfinite(right))
    absolute = abs(right - left) if finite else float("inf")
    scale = max(abs(left), abs(right), np.finfo(float).tiny) if finite else 1.0
    relative = absolute / scale if finite else float("inf")
    absolute_passed = bool(finite and absolute <= policy.absolute_tolerance)
    relative_passed = bool(finite and relative <= policy.relative_tolerance)
    return {
        "reference": left,
        "candidate": right,
        "absolute": absolute,
        "relative": relative,
        "absolute_tolerance": policy.absolute_tolerance,
        "relative_tolerance": policy.relative_tolerance,
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
class FixedOuterGeometryQualificationReport:
    """Diagnostic comparison before and after one fixed outer-Q reduction."""

    reference: MatsubaraFreeEnergyPerArea
    candidate: MatsubaraFreeEnergyPerArea
    node_comparison: Mapping[str, Any]
    outer_integral_comparisons: tuple[Mapping[str, Any], ...]
    contribution_comparisons: tuple[Mapping[str, Any], ...]
    total_comparison: Mapping[str, Any]
    passed: bool
    metadata: Mapping[str, Any]
    schema: str = MATERIAL_GEOMETRY_OUTER_QUALIFICATION_SCHEMA
    production_casimir_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_GEOMETRY_OUTER_QUALIFICATION_SCHEMA:
            raise ValueError("unsupported fixed-outer geometry qualification schema")
        if not isinstance(self.reference, MatsubaraFreeEnergyPerArea):
            raise TypeError("reference must be a MatsubaraFreeEnergyPerArea")
        if not isinstance(self.candidate, MatsubaraFreeEnergyPerArea):
            raise TypeError("candidate must be a MatsubaraFreeEnergyPerArea")
        object.__setattr__(
            self,
            "node_comparison",
            MappingProxyType(dict(self.node_comparison)),
        )
        object.__setattr__(
            self,
            "outer_integral_comparisons",
            tuple(MappingProxyType(dict(row)) for row in self.outer_integral_comparisons),
        )
        object.__setattr__(
            self,
            "contribution_comparisons",
            tuple(MappingProxyType(dict(row)) for row in self.contribution_comparisons),
        )
        object.__setattr__(
            self,
            "total_comparison",
            MappingProxyType(dict(self.total_comparison)),
        )
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if bool(self.production_casimir_allowed):
            raise ValueError("TODO 4 outer qualification cannot admit production")
        object.__setattr__(self, "production_casimir_allowed", False)


def qualify_fixed_outer_geometry_replay(
    *,
    reference_logdet_by_n_and_node: np.ndarray,
    candidate_logdet_by_n_and_node: np.ndarray,
    matsubara_indices: Sequence[int],
    temperature_K: float,
    grid: OuterQPolarGrid,
    policy: GeometryEquivalencePolicy | None = None,
) -> FixedOuterGeometryQualificationReport:
    """Compare matched logdet arrays and their identical fixed outer reduction."""

    if not isinstance(grid, OuterQPolarGrid):
        raise TypeError("grid must be an OuterQPolarGrid")
    comparison_policy = GeometryEquivalencePolicy() if policy is None else policy
    if not isinstance(comparison_policy, GeometryEquivalencePolicy):
        raise TypeError("policy must be a GeometryEquivalencePolicy")
    indices = tuple(int(value) for value in matsubara_indices)
    reference_values = np.asarray(reference_logdet_by_n_and_node, dtype=float)
    candidate_values = np.asarray(candidate_logdet_by_n_and_node, dtype=float)
    expected_shape = (len(indices), grid.node_count)
    if reference_values.shape != expected_shape or candidate_values.shape != expected_shape:
        raise ValueError(
            "reference and candidate logdet arrays must match Matsubara and grid shape"
        )
    if not np.isfinite(reference_values).all() or not np.isfinite(candidate_values).all():
        raise ValueError("fixed-outer qualification logdet arrays must be finite")

    absolute = np.abs(candidate_values - reference_values)
    scale = np.maximum(
        np.maximum(np.abs(reference_values), np.abs(candidate_values)),
        np.finfo(float).tiny,
    )
    relative = absolute / scale
    node_pass = np.logical_or(
        absolute <= comparison_policy.absolute_tolerance,
        relative <= comparison_policy.relative_tolerance,
    )
    node_comparison = {
        "maximum_absolute": float(np.max(absolute)),
        "maximum_relative": float(np.max(relative)),
        "failed_node_count": int(np.size(node_pass) - np.count_nonzero(node_pass)),
        "total_node_count": int(np.size(node_pass)),
        "passed": bool(np.all(node_pass)),
    }

    reference = free_energy_per_area_from_logdet(
        reference_values,
        matsubara_indices=indices,
        temperature_K=temperature_K,
        grid=grid,
    )
    candidate = free_energy_per_area_from_logdet(
        candidate_values,
        matsubara_indices=indices,
        temperature_K=temperature_K,
        grid=grid,
    )
    outer = tuple(
        _comparison(left, right, policy=comparison_policy)
        for left, right in zip(
            reference.outer_q_integrals_m_inv2,
            candidate.outer_q_integrals_m_inv2,
        )
    )
    contributions = tuple(
        _comparison(left, right, policy=comparison_policy)
        for left, right in zip(
            reference.contributions_J_m2,
            candidate.contributions_J_m2,
        )
    )
    total = _comparison(
        reference.total_J_m2,
        candidate.total_J_m2,
        policy=comparison_policy,
    )
    passed = bool(
        node_comparison["passed"]
        and all(bool(row["passed"]) for row in outer)
        and all(bool(row["passed"]) for row in contributions)
        and total["passed"]
    )
    return FixedOuterGeometryQualificationReport(
        reference=reference,
        candidate=candidate,
        node_comparison=node_comparison,
        outer_integral_comparisons=outer,
        contribution_comparisons=contributions,
        total_comparison=total,
        passed=passed,
        metadata={
            "casimir_stage": "reduced_fixed_outer_geometry_replay",
            "same_grid_object_used": True,
            "matsubara_indices": list(indices),
            "temperature_K": float(temperature_K),
            "outer_grid_schema": grid.metadata.get("schema"),
            "angular_symmetry_reduction": False,
            "tail_included": False,
            "partial_sum_only": True,
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
    )


__all__ = [
    "FixedOuterGeometryQualificationReport",
    "MATERIAL_GEOMETRY_OUTER_QUALIFICATION_SCHEMA",
    "qualify_fixed_outer_geometry_replay",
]
