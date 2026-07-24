"""Reduced fixed-outer replay for TODO 4 old/new point arrays.

The function in this module does not evaluate material responses or geometry.
It verifies that two already matched logdet arrays remain equivalent after the
same fixed outer-Q and finite Matsubara reduction. Absolute tolerances are
unit-specific; one number is never reused across dimensionless logdet,
``m^-2`` outer integrals, and ``J/m^2`` energy quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.casimir.outer_quadrature import (
    MatsubaraFreeEnergyPerArea,
    OuterQPolarGrid,
    free_energy_per_area_from_logdet,
)

MATERIAL_GEOMETRY_OUTER_QUALIFICATION_SCHEMA = (
    "material-geometry-outer-qualification-v1"
)


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


@dataclass(frozen=True)
class FixedOuterEquivalencePolicy:
    """Unit-aware comparison tolerances for a reduced fixed-outer replay."""

    node_logdet_absolute: float = 1e-12
    node_logdet_relative: float = 1e-10
    outer_integral_absolute_m_inv2: float = 0.0
    outer_integral_relative: float = 1e-10
    contribution_absolute_J_m2: float = 1e-15
    contribution_relative: float = 1e-10
    total_absolute_J_m2: float = 1e-15
    total_relative: float = 1e-10

    def __post_init__(self) -> None:
        for name in (
            "node_logdet_absolute",
            "node_logdet_relative",
            "outer_integral_absolute_m_inv2",
            "outer_integral_relative",
            "contribution_absolute_J_m2",
            "contribution_relative",
            "total_absolute_J_m2",
            "total_relative",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )

    def as_dict(self) -> dict[str, float | str]:
        return {
            "schema": "fixed-outer-equivalence-policy-v1",
            "node_logdet_absolute": self.node_logdet_absolute,
            "node_logdet_relative": self.node_logdet_relative,
            "outer_integral_absolute_m_inv2": self.outer_integral_absolute_m_inv2,
            "outer_integral_relative": self.outer_integral_relative,
            "contribution_absolute_J_m2": self.contribution_absolute_J_m2,
            "contribution_relative": self.contribution_relative,
            "total_absolute_J_m2": self.total_absolute_J_m2,
            "total_relative": self.total_relative,
        }


def _comparison(
    reference: float,
    candidate: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    unit: str,
) -> dict[str, float | bool | str]:
    left = float(reference)
    right = float(candidate)
    finite = bool(np.isfinite(left) and np.isfinite(right))
    absolute = abs(right - left) if finite else float("inf")
    scale = max(abs(left), abs(right), np.finfo(float).tiny) if finite else 1.0
    relative = absolute / scale if finite else float("inf")
    absolute_passed = bool(finite and absolute <= float(absolute_tolerance))
    relative_passed = bool(finite and relative <= float(relative_tolerance))
    return {
        "reference": left,
        "candidate": right,
        "unit": str(unit),
        "absolute": absolute,
        "relative": relative,
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
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
    policy: FixedOuterEquivalencePolicy
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
        if not isinstance(self.policy, FixedOuterEquivalencePolicy):
            raise TypeError("policy must be a FixedOuterEquivalencePolicy")
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
    policy: FixedOuterEquivalencePolicy | None = None,
) -> FixedOuterGeometryQualificationReport:
    """Compare matched logdet arrays and their identical fixed outer reduction."""

    if not isinstance(grid, OuterQPolarGrid):
        raise TypeError("grid must be an OuterQPolarGrid")
    comparison_policy = FixedOuterEquivalencePolicy() if policy is None else policy
    if not isinstance(comparison_policy, FixedOuterEquivalencePolicy):
        raise TypeError("policy must be a FixedOuterEquivalencePolicy")
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
        absolute <= comparison_policy.node_logdet_absolute,
        relative <= comparison_policy.node_logdet_relative,
    )
    node_comparison = {
        "unit": "dimensionless_logdet",
        "maximum_absolute": float(np.max(absolute)),
        "maximum_relative": float(np.max(relative)),
        "absolute_tolerance": comparison_policy.node_logdet_absolute,
        "relative_tolerance": comparison_policy.node_logdet_relative,
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
        _comparison(
            left,
            right,
            absolute_tolerance=comparison_policy.outer_integral_absolute_m_inv2,
            relative_tolerance=comparison_policy.outer_integral_relative,
            unit="m^-2",
        )
        for left, right in zip(
            reference.outer_q_integrals_m_inv2,
            candidate.outer_q_integrals_m_inv2,
        )
    )
    contributions = tuple(
        _comparison(
            left,
            right,
            absolute_tolerance=comparison_policy.contribution_absolute_J_m2,
            relative_tolerance=comparison_policy.contribution_relative,
            unit="J/m^2",
        )
        for left, right in zip(
            reference.contributions_J_m2,
            candidate.contributions_J_m2,
        )
    )
    total = _comparison(
        reference.total_J_m2,
        candidate.total_J_m2,
        absolute_tolerance=comparison_policy.total_absolute_J_m2,
        relative_tolerance=comparison_policy.total_relative,
        unit="J/m^2",
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
        policy=comparison_policy,
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
            "comparison_policy": comparison_policy.as_dict(),
            "unit_specific_absolute_tolerances": True,
            "angular_symmetry_reduction": False,
            "tail_included": False,
            "partial_sum_only": True,
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
    )


__all__ = [
    "FixedOuterEquivalencePolicy",
    "FixedOuterGeometryQualificationReport",
    "MATERIAL_GEOMETRY_OUTER_QUALIFICATION_SCHEMA",
    "qualify_fixed_outer_geometry_replay",
]
