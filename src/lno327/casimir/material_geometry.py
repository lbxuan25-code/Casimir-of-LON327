"""Geometry-only assembly from certified or diagnostic material responses.

The material-response builder owns microscopic integration, Ward validation, and
sheet-response conversion in the crystal frame. This module starts only after
that boundary: it rotates a supplied sheet response into a requested plate
geometry and constructs the single-plate reflection operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from lno327.casimir.material_response import MaterialResponseSample
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.electrodynamics.reflection import (
    SheetReflection,
    positive_matsubara_sheet_response_to_reflection,
)
from lno327.electrodynamics.static_sheet import (
    StaticSheetReflection,
    StaticSheetResponse,
    static_sheet_response_to_reflection,
)

PlateReflection: TypeAlias = StaticSheetReflection | SheetReflection


def _finite_positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


@dataclass(frozen=True)
class ReflectionGeometryPolicy:
    """Numerical policy for mapping one material response to one plate."""

    lattice_constant_m: float = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m
    q_match_tolerance: float = 1e-10
    require_physical: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "lattice_constant_m",
            _finite_positive(self.lattice_constant_m, "lattice_constant_m"),
        )
        object.__setattr__(
            self,
            "q_match_tolerance",
            _finite_nonnegative(self.q_match_tolerance, "q_match_tolerance"),
        )
        object.__setattr__(self, "require_physical", bool(self.require_physical))

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "schema": "reflection-geometry-policy-v1",
            "lattice_constant_m": self.lattice_constant_m,
            "q_match_tolerance": self.q_match_tolerance,
            "require_physical": self.require_physical,
        }


def material_response_to_reflection(
    sample: MaterialResponseSample,
    *,
    q_lab: np.ndarray,
    theta_rad: float,
    policy: ReflectionGeometryPolicy | None = None,
) -> tuple[PlateReflection, dict[str, object]]:
    """Construct one plate reflection without recomputing microscopic response."""

    if not isinstance(sample, MaterialResponseSample):
        raise TypeError("sample must be a MaterialResponseSample")
    geometry = ReflectionGeometryPolicy() if policy is None else policy
    if not isinstance(geometry, ReflectionGeometryPolicy):
        raise TypeError("policy must be a ReflectionGeometryPolicy")

    q = np.asarray(q_lab, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_lab must be a finite vector with shape (2,)")
    if float(np.linalg.norm(q)) == 0.0:
        raise ValueError("q_lab must be nonzero")
    theta = float(theta_rad)
    if not np.isfinite(theta):
        raise ValueError("theta_rad must be finite")

    # The material layer owns the physical validation policy.  In particular,
    # positive-frequency responses may have been validated with tolerances that
    # differ from the electrodynamics adapter defaults.  Geometry must therefore
    # consume the recorded MaterialResponseSample gate rather than silently
    # revalidating the response under a second policy.
    if geometry.require_physical and not sample.hard_physical_passed:
        raise ValueError(
            "material response failed its recorded hard physical validation policy"
        )

    if isinstance(sample.response, StaticSheetResponse):
        reflection: PlateReflection = static_sheet_response_to_reflection(
            sample.response,
            q_lab_model=q,
            theta_rad=theta,
            lattice_constant_m=geometry.lattice_constant_m,
            q_match_tolerance=geometry.q_match_tolerance,
            require_physical=False,
        )
    else:
        reflection = positive_matsubara_sheet_response_to_reflection(
            sample.response,
            q_lab_model=q,
            theta_rad=theta,
            lattice_constant_m=geometry.lattice_constant_m,
            q_match_tolerance=geometry.q_match_tolerance,
            require_physical=False,
        )

    diagnostics = sample.diagnostics()
    diagnostics.update(
        {
            "geometry_schema": "material-response-reflection-v1",
            "q_lab": q.tolist(),
            "theta_rad": theta,
            "reflection_constructed": True,
            "reflection_norm": float(np.linalg.norm(reflection.matrix_lt)),
            "hard_physical_passed": bool(sample.hard_physical_passed),
            "material_validation_gate_required": geometry.require_physical,
            "material_validation_gate_source": "MaterialResponseSample",
            "adapter_default_policy_revalidation_performed": False,
        }
    )
    return reflection, diagnostics


__all__ = [
    "PlateReflection",
    "ReflectionGeometryPolicy",
    "material_response_to_reflection",
]
