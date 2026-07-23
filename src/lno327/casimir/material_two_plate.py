"""Fast two-plate geometry assembly from precomputed material responses.

No microscopic integration or response certification is performed here. The
caller supplies one crystal-frame material response per plate; this module only
constructs lab-frame reflections, propagation, and the signed passive logdet.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.casimir.lifshitz_integrand import LifshitzPoint, passive_sheet_logdet
from lno327.casimir.material_geometry import (
    ReflectionGeometryPolicy,
    material_response_to_reflection,
)
from lno327.casimir.material_response import MaterialResponseSample


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
class TwoPlateGeometryPolicy:
    """Geometry and trace-log policy, deliberately separate from material policy."""

    separation_m: float
    reflection_policy: ReflectionGeometryPolicy = ReflectionGeometryPolicy()
    compatibility_tolerance: float = 1e-11
    eigenvalue_imag_tolerance: float = 1e-9
    eigenvalue_lower_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "separation_m",
            _finite_positive(self.separation_m, "separation_m"),
        )
        if not isinstance(self.reflection_policy, ReflectionGeometryPolicy):
            raise TypeError("reflection_policy must be a ReflectionGeometryPolicy")
        for name in (
            "compatibility_tolerance",
            "eigenvalue_imag_tolerance",
            "eigenvalue_lower_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "two-plate-geometry-policy-v1",
            "separation_m": self.separation_m,
            "reflection_policy": self.reflection_policy.as_dict(),
            "compatibility_tolerance": self.compatibility_tolerance,
            "eigenvalue_imag_tolerance": self.eigenvalue_imag_tolerance,
            "eigenvalue_lower_tolerance": self.eigenvalue_lower_tolerance,
        }


@dataclass(frozen=True)
class TwoPlateAssembly:
    """One geometry-specific logdet assembled from two material samples."""

    point: LifshitzPoint
    plate_1_diagnostics: Mapping[str, Any]
    plate_2_diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.point, LifshitzPoint):
            raise TypeError("point must be a LifshitzPoint")
        object.__setattr__(
            self,
            "plate_1_diagnostics",
            MappingProxyType(dict(self.plate_1_diagnostics)),
        )
        object.__setattr__(
            self,
            "plate_2_diagnostics",
            MappingProxyType(dict(self.plate_2_diagnostics)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def logdet(self) -> float:
        return self.point.logdet


def assemble_two_plate_logdet(
    plate_1: MaterialResponseSample,
    plate_2: MaterialResponseSample,
    *,
    q_lab: np.ndarray,
    theta_1_rad: float,
    theta_2_rad: float,
    policy: TwoPlateGeometryPolicy,
) -> TwoPlateAssembly:
    """Assemble reflection, propagation, and logdet without a microscopic fallback."""

    if not isinstance(plate_1, MaterialResponseSample) or not isinstance(
        plate_2, MaterialResponseSample
    ):
        raise TypeError("both plates must be MaterialResponseSample objects")
    if not isinstance(policy, TwoPlateGeometryPolicy):
        raise TypeError("policy must be a TwoPlateGeometryPolicy")
    if plate_1.frequency_sector != plate_2.frequency_sector:
        raise ValueError("plate material responses use different frequency sectors")
    if plate_1.xi_eV != plate_2.xi_eV:
        raise ValueError("plate material responses use different xi_eV")

    q = np.asarray(q_lab, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_lab must be a finite vector with shape (2,)")
    if float(np.linalg.norm(q)) == 0.0:
        raise ValueError("q_lab must be nonzero")

    reflection_1, diagnostics_1 = material_response_to_reflection(
        plate_1,
        q_lab=q,
        theta_rad=float(theta_1_rad),
        policy=policy.reflection_policy,
    )
    reflection_2, diagnostics_2 = material_response_to_reflection(
        plate_2,
        q_lab=q,
        theta_rad=float(theta_2_rad),
        policy=policy.reflection_policy,
    )
    point = passive_sheet_logdet(
        reflection_1,
        reflection_2,
        separation_m=policy.separation_m,
        compatibility_tolerance=policy.compatibility_tolerance,
        eigenvalue_imag_tolerance=policy.eigenvalue_imag_tolerance,
        eigenvalue_lower_tolerance=policy.eigenvalue_lower_tolerance,
    )
    metadata = {
        "schema": "two-plate-material-assembly-v1",
        "source": "two precomputed MaterialResponseSample objects",
        "microscopic_integration_performed": False,
        "response_certification_performed": False,
        "geometry_assembly_only": True,
        "frequency_sector": plate_1.frequency_sector,
        "xi_eV": plate_1.xi_eV,
        "q_lab": q.tolist(),
        "theta_1_rad": float(theta_1_rad),
        "theta_2_rad": float(theta_2_rad),
        "separation_m": policy.separation_m,
        "production_casimir_allowed": False,
    }
    return TwoPlateAssembly(
        point=point,
        plate_1_diagnostics=diagnostics_1,
        plate_2_diagnostics=diagnostics_2,
        metadata=metadata,
    )


__all__ = [
    "TwoPlateAssembly",
    "TwoPlateGeometryPolicy",
    "assemble_two_plate_logdet",
]
