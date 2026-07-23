"""Response-level N/shift certification independent of Casimir geometry.

The historical transverse certifier compares a geometry-specific two-plate
logdet. This module instead compares static susceptibility channels or positive-
Matsubara crystal-frame sheet tensors before plate rotation, reflection,
separation, or outer quadrature enters the calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.casimir.material_response import MaterialResponseSample

MATERIAL_RESPONSE_CERTIFICATION_SCHEMA = "material-response-certification-v1"


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


@dataclass(frozen=True)
class MaterialResponseConvergencePolicy:
    """Provisional response-space tolerances for TODO 2 qualification."""

    relative_tolerance: float = 1e-3
    absolute_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_tolerance",
            _finite_nonnegative(self.relative_tolerance, "relative_tolerance"),
        )
        object.__setattr__(
            self,
            "absolute_tolerance",
            _finite_nonnegative(self.absolute_tolerance, "absolute_tolerance"),
        )

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "schema": "material-response-convergence-policy-v1",
            "comparison_order": "absolute_first_then_relative_fallback",
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "observable_error_budget_calibrated": False,
            "production_admission": False,
        }


def _absolute_then_relative(
    absolute: float,
    scale: float,
    *,
    policy: MaterialResponseConvergencePolicy,
) -> dict[str, float | bool | str]:
    absolute_value = float(absolute)
    scale_value = float(scale)
    finite = bool(
        np.isfinite(absolute_value)
        and absolute_value >= 0.0
        and np.isfinite(scale_value)
        and scale_value >= 0.0
    )
    if not finite:
        return {
            "finite": False,
            "absolute": float("nan"),
            "relative": float("nan"),
            "scale": float("nan"),
            "absolute_tolerance": policy.absolute_tolerance,
            "relative_tolerance": policy.relative_tolerance,
            "absolute_passed": False,
            "relative_passed": False,
            "passed_by": "failed",
            "passed": False,
        }

    relative = absolute_value / max(scale_value, np.finfo(float).tiny)
    absolute_passed = bool(absolute_value <= policy.absolute_tolerance)
    relative_passed = bool(relative <= policy.relative_tolerance)
    passed_by = (
        "absolute"
        if absolute_passed
        else "relative"
        if relative_passed
        else "failed"
    )
    return {
        "finite": True,
        "absolute": absolute_value,
        "relative": relative,
        "scale": scale_value,
        "absolute_tolerance": policy.absolute_tolerance,
        "relative_tolerance": policy.relative_tolerance,
        "absolute_passed": absolute_passed,
        "relative_passed": relative_passed,
        "passed_by": passed_by,
        "passed": bool(absolute_passed or relative_passed),
    }


def _require_compatible(
    left: MaterialResponseSample,
    right: MaterialResponseSample,
) -> None:
    if not isinstance(left, MaterialResponseSample) or not isinstance(
        right, MaterialResponseSample
    ):
        raise TypeError("response comparison requires MaterialResponseSample objects")
    if left.frequency_sector != right.frequency_sector:
        raise ValueError("response samples have different frequency sectors")
    if left.xi_eV != right.xi_eV:
        raise ValueError("response samples have different xi_eV")
    if not np.array_equal(left.q_crystal, right.q_crystal):
        raise ValueError("response samples have different q_crystal")


def compare_material_responses(
    left: MaterialResponseSample,
    right: MaterialResponseSample,
    *,
    policy: MaterialResponseConvergencePolicy,
) -> dict[str, Any]:
    """Compare two compatible crystal-frame responses without geometry."""

    _require_compatible(left, right)
    if not isinstance(policy, MaterialResponseConvergencePolicy):
        raise TypeError("policy must be a MaterialResponseConvergencePolicy")

    if left.frequency_sector == "zero_matsubara":
        left_values = np.asarray(
            [left.response.chi_bar, left.response.dbar_t], dtype=float
        )
        right_values = np.asarray(
            [right.response.chi_bar, right.response.dbar_t], dtype=float
        )
        channels = {
            name: _absolute_then_relative(
                abs(float(right_values[index] - left_values[index])),
                max(abs(float(left_values[index])), abs(float(right_values[index]))),
                policy=policy,
            )
            for index, name in enumerate(("chi_bar", "dbar_t"))
        }
        return {
            "frequency_sector": left.frequency_sector,
            "comparison_basis": "static_channels_chi_bar_dbar_t",
            "channels": channels,
            "passed": all(bool(row["passed"]) for row in channels.values()),
        }

    left_matrix = np.asarray(left.response.matrix_tilde, dtype=complex)
    right_matrix = np.asarray(right.response.matrix_tilde, dtype=complex)
    difference = float(np.linalg.norm(right_matrix - left_matrix, ord=2))
    scale = max(
        float(np.linalg.norm(left_matrix, ord=2)),
        float(np.linalg.norm(right_matrix, ord=2)),
    )
    matrix = _absolute_then_relative(difference, scale, policy=policy)
    return {
        "frequency_sector": left.frequency_sector,
        "comparison_basis": "crystal_xy_sigma_tilde_spectral_norm",
        "matrix": matrix,
        "passed": bool(matrix["passed"]),
    }


def _comparison_map(
    samples: Sequence[tuple[str, MaterialResponseSample]],
    *,
    policy: MaterialResponseConvergencePolicy,
) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    for (left_label, left), (right_label, right) in combinations(samples, 2):
        comparisons[f"{left_label}|{right_label}"] = compare_material_responses(
            left,
            right,
            policy=policy,
        )
    return comparisons


@dataclass(frozen=True)
class MaterialResponseLevelAssessment:
    """Cross-shift and adjacent-N evidence for one N level."""

    hard_physical_closure_across_shifts: bool
    cross_shift_comparisons: Mapping[str, Mapping[str, Any]]
    cross_shift_all_passed: bool
    adjacent_N_by_shift: Mapping[str, Mapping[str, Any]] | None
    adjacent_N_all_shifts_passed: bool
    accepted_transition: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hard_physical_closure_across_shifts",
            bool(self.hard_physical_closure_across_shifts),
        )
        object.__setattr__(
            self,
            "cross_shift_comparisons",
            MappingProxyType(dict(self.cross_shift_comparisons)),
        )
        adjacent = self.adjacent_N_by_shift
        object.__setattr__(
            self,
            "adjacent_N_by_shift",
            None if adjacent is None else MappingProxyType(dict(adjacent)),
        )
        object.__setattr__(
            self,
            "cross_shift_all_passed",
            bool(self.cross_shift_all_passed),
        )
        object.__setattr__(
            self,
            "adjacent_N_all_shifts_passed",
            bool(self.adjacent_N_all_shifts_passed),
        )
        object.__setattr__(self, "accepted_transition", bool(self.accepted_transition))

    def as_dict(self) -> dict[str, Any]:
        return {
            "hard_physical_closure_across_shifts": (
                self.hard_physical_closure_across_shifts
            ),
            "material_response_cross_shift": {
                key: dict(value) for key, value in self.cross_shift_comparisons.items()
            },
            "material_response_cross_shift_all_passed": self.cross_shift_all_passed,
            "adjacent_N_by_shift": (
                None
                if self.adjacent_N_by_shift is None
                else {
                    key: dict(value)
                    for key, value in self.adjacent_N_by_shift.items()
                }
            ),
            "adjacent_N_all_shifts_passed": self.adjacent_N_all_shifts_passed,
            "accepted_transition": self.accepted_transition,
        }


def assess_material_response_level(
    *,
    current_by_shift: Mapping[str, MaterialResponseSample],
    previous_by_shift: Mapping[str, MaterialResponseSample] | None,
    policy: MaterialResponseConvergencePolicy,
) -> MaterialResponseLevelAssessment:
    """Assess one N level entirely in material-response space."""

    labels = tuple(current_by_shift)
    if len(labels) < 2:
        raise ValueError("material response certification requires at least two shifts")
    current = {label: current_by_shift[label] for label in labels}
    reference = current[labels[0]]
    for sample in current.values():
        _require_compatible(reference, sample)

    hard = all(sample.hard_physical_passed for sample in current.values())
    cross_shift = _comparison_map(tuple(current.items()), policy=policy)
    cross_passed = all(bool(row["passed"]) for row in cross_shift.values())

    adjacent: dict[str, dict[str, Any]] | None = None
    adjacent_passed = False
    if previous_by_shift is not None:
        if tuple(previous_by_shift) != labels:
            raise ValueError("current and previous shift labels/order differ")
        adjacent = {
            label: compare_material_responses(
                previous_by_shift[label],
                current[label],
                policy=policy,
            )
            for label in labels
        }
        adjacent_passed = all(bool(row["passed"]) for row in adjacent.values())

    accepted = bool(
        previous_by_shift is not None and hard and cross_passed and adjacent_passed
    )
    return MaterialResponseLevelAssessment(
        hard_physical_closure_across_shifts=hard,
        cross_shift_comparisons=cross_shift,
        cross_shift_all_passed=cross_passed,
        adjacent_N_by_shift=adjacent,
        adjacent_N_all_shifts_passed=adjacent_passed,
        accepted_transition=accepted,
    )


@dataclass(frozen=True)
class MaterialResponseLevelRecord:
    """One evaluated N level and its immutable response-space assessment."""

    n_grid: int
    samples_by_shift: Mapping[str, MaterialResponseSample]
    assessment: MaterialResponseLevelAssessment

    def __post_init__(self) -> None:
        n_grid = int(self.n_grid)
        samples = dict(self.samples_by_shift)
        if n_grid <= 0:
            raise ValueError("n_grid must be positive")
        if len(samples) < 2:
            raise ValueError("a level record requires at least two shifts")
        reference = next(iter(samples.values()))
        for sample in samples.values():
            _require_compatible(reference, sample)
        object.__setattr__(self, "n_grid", n_grid)
        object.__setattr__(self, "samples_by_shift", MappingProxyType(samples))


def assess_material_response_envelope(
    history: Sequence[MaterialResponseLevelRecord],
    *,
    policy: MaterialResponseConvergencePolicy,
    levels: int = 3,
) -> dict[str, Any]:
    """Assess the maximum pairwise spread over the final N/shift window."""

    count = int(levels)
    if count < 3:
        raise ValueError("oscillatory envelope requires at least three N levels")
    if len(history) < count:
        return {
            "available": False,
            "levels": count,
            "N_window": [],
            "hard_physical_closure": False,
            "cross_shift_all_levels_passed": False,
            "joint_response_envelope": {},
            "pairwise_complete": True,
            "passed": False,
        }

    window = list(history[-count:])
    n_values = [row.n_grid for row in window]
    if n_values != sorted(n_values) or len(set(n_values)) != len(n_values):
        raise ValueError("response history N values must be strictly increasing")
    hard = all(
        row.assessment.hard_physical_closure_across_shifts for row in window
    )
    cross = all(row.assessment.cross_shift_all_passed for row in window)
    flattened = tuple(
        (f"N{row.n_grid}:{shift_label}", sample)
        for row in window
        for shift_label, sample in row.samples_by_shift.items()
    )
    comparisons = _comparison_map(flattened, policy=policy)
    envelope_passed = all(bool(row["passed"]) for row in comparisons.values())
    return {
        "available": True,
        "levels": count,
        "N_window": n_values,
        "hard_physical_closure": hard,
        "cross_shift_all_levels_passed": cross,
        "joint_response_envelope": comparisons,
        "pairwise_complete": True,
        "comparison_count": len(comparisons),
        "passed": bool(hard and cross and envelope_passed),
    }


@dataclass(frozen=True)
class CertifiedMaterialResponse:
    """Diagnostic response certification; never a production-admission token."""

    working_N: int
    audit_N: int
    primary_shift: str
    audit_samples_by_shift: Mapping[str, MaterialResponseSample]
    establishment_mode: str
    evidence: Mapping[str, Any]
    schema: str = MATERIAL_RESPONSE_CERTIFICATION_SCHEMA
    production_casimir_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_RESPONSE_CERTIFICATION_SCHEMA:
            raise ValueError(
                f"schema must be {MATERIAL_RESPONSE_CERTIFICATION_SCHEMA!r}"
            )
        working = int(self.working_N)
        audit = int(self.audit_N)
        if working <= 0 or audit <= 0 or working >= audit:
            raise ValueError("certification requires 0 < working_N < audit_N")
        samples = dict(self.audit_samples_by_shift)
        if len(samples) < 2:
            raise ValueError("certification requires at least two audit shifts")
        primary = str(self.primary_shift)
        if primary not in samples:
            raise ValueError("primary_shift is absent from audit_samples_by_shift")
        reference = samples[primary]
        for sample in samples.values():
            _require_compatible(reference, sample)
            if not sample.hard_physical_passed:
                raise ValueError("certified audit sample failed a hard physical gate")
        if bool(self.production_casimir_allowed):
            raise ValueError("TODO 2 response certification cannot admit production")
        object.__setattr__(self, "working_N", working)
        object.__setattr__(self, "audit_N", audit)
        object.__setattr__(self, "primary_shift", primary)
        object.__setattr__(
            self,
            "audit_samples_by_shift",
            MappingProxyType(samples),
        )
        object.__setattr__(self, "establishment_mode", str(self.establishment_mode))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        object.__setattr__(self, "production_casimir_allowed", False)

    @property
    def primary_response(self) -> MaterialResponseSample:
        return self.audit_samples_by_shift[self.primary_shift]

    @property
    def status(self) -> str:
        return "response_certified_diagnostic"


def certify_material_response_history(
    history: Sequence[MaterialResponseLevelRecord],
    *,
    policy: MaterialResponseConvergencePolicy,
    required_consecutive_passes: int = 2,
    envelope_levels: int = 3,
) -> CertifiedMaterialResponse | None:
    """Establish a deterministic audit response from an evaluated N history."""

    required = int(required_consecutive_passes)
    if required <= 0:
        raise ValueError("required_consecutive_passes must be positive")
    if len(history) < 2:
        return None

    n_values = [row.n_grid for row in history]
    if n_values != sorted(n_values) or len(set(n_values)) != len(n_values):
        raise ValueError("response history N values must be strictly increasing")
    consecutive = 0
    for row in history:
        consecutive = consecutive + 1 if row.assessment.accepted_transition else 0
    envelope = assess_material_response_envelope(
        history,
        policy=policy,
        levels=envelope_levels,
    )
    strict_ready = consecutive >= required
    envelope_ready = bool(envelope["passed"])
    if not strict_ready and not envelope_ready:
        return None

    working = history[-2]
    audit = history[-1]
    primary_shift = sorted(audit.samples_by_shift)[0]
    mode = (
        "strict_consecutive_adjacent"
        if strict_ready
        else "three_level_oscillatory_envelope"
    )
    evidence = {
        "convergence_policy": policy.as_dict(),
        "required_consecutive_passes": required,
        "consecutive_accepted_transitions": consecutive,
        "audit_level_assessment": audit.assessment.as_dict(),
        "oscillatory_envelope": envelope,
        "observable_error_budget_calibrated": False,
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
    }
    return CertifiedMaterialResponse(
        working_N=working.n_grid,
        audit_N=audit.n_grid,
        primary_shift=primary_shift,
        audit_samples_by_shift=audit.samples_by_shift,
        establishment_mode=mode,
        evidence=evidence,
    )


__all__ = [
    "MATERIAL_RESPONSE_CERTIFICATION_SCHEMA",
    "CertifiedMaterialResponse",
    "MaterialResponseConvergencePolicy",
    "MaterialResponseLevelAssessment",
    "MaterialResponseLevelRecord",
    "assess_material_response_envelope",
    "assess_material_response_level",
    "certify_material_response_history",
    "compare_material_responses",
]
