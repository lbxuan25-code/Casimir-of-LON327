"""Exact geometry-batch planning for persisted material responses.

The planner is geometry-aware but does not load cache files, construct
reflections, evaluate propagation, or call the microscopic solver. It maps
laboratory momenta to each plate's exact crystal-frame momentum and builds the
strict TODO 3 cache identities required by a later read-only assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.casimir.material_geometry import ReflectionGeometryPolicy
from lno327.casimir.material_response_cache_identity import MaterialResponseCacheIdentity
from lno327.casimir.material_response_cache_request import (
    build_material_response_cache_identity,
    build_material_response_identity_context,
)
from lno327.casimir.material_response_engine import MaterialResponseEngineConfig
from lno327.electrodynamics.basis import q_lab_to_crystal

MATERIAL_GEOMETRY_BATCH_PLAN_SCHEMA = "material-geometry-batch-plan-v1"


def _readonly_q(value: np.ndarray, name: str) -> np.ndarray:
    q = np.array(value, dtype=float, copy=True)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError(f"{name} must be a finite vector with shape (2,)")
    if float(np.linalg.norm(q)) == 0.0:
        raise ValueError(f"{name} must be nonzero")
    q.setflags(write=False)
    return q


def _finite_angle(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


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


def _q_hex(value: np.ndarray) -> tuple[str, str]:
    q = np.asarray(value, dtype=float)
    return (float(q[0]).hex(), float(q[1]).hex())


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class GeometryBatchPolicy:
    """Geometry and trace-log policy independent of material identity."""

    reflection_policy: ReflectionGeometryPolicy = ReflectionGeometryPolicy()
    compatibility_tolerance: float = 1e-11
    eigenvalue_imag_tolerance: float = 1e-9
    eigenvalue_lower_tolerance: float = 1e-10

    def __post_init__(self) -> None:
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
            "schema": "geometry-batch-policy-v1",
            "reflection_policy": self.reflection_policy.as_dict(),
            "compatibility_tolerance": self.compatibility_tolerance,
            "eigenvalue_imag_tolerance": self.eigenvalue_imag_tolerance,
            "eigenvalue_lower_tolerance": self.eigenvalue_lower_tolerance,
        }


@dataclass(frozen=True)
class PlateResponseRequirement:
    """One exact persisted response needed by one or more plate consumers."""

    identity: MaterialResponseCacheIdentity
    q_crystal: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.identity, MaterialResponseCacheIdentity):
            raise TypeError("identity must be a MaterialResponseCacheIdentity")
        q = _readonly_q(self.q_crystal, "q_crystal")
        if _q_hex(q) != _q_hex(self.identity.q_crystal):
            raise ValueError("requirement q_crystal differs from cache identity")
        object.__setattr__(self, "q_crystal", q)

    @property
    def key(self) -> str:
        return self.identity.sha256


@dataclass(frozen=True)
class GeometryPointSpec:
    """One distance-independent q/frequency/angle-pair scattering request."""

    point_id: str
    q_label: str
    q_lab: np.ndarray
    matsubara_index: int
    theta_1_rad: float
    theta_2_rad: float
    plate_1_requirement: str
    plate_2_requirement: str

    def __post_init__(self) -> None:
        point_id = str(self.point_id)
        q_label = str(self.q_label)
        if not point_id or not q_label:
            raise ValueError("point_id and q_label must be nonempty")
        object.__setattr__(self, "point_id", point_id)
        object.__setattr__(self, "q_label", q_label)
        object.__setattr__(self, "q_lab", _readonly_q(self.q_lab, "q_lab"))
        index = int(self.matsubara_index)
        if index < 0:
            raise ValueError("matsubara_index must be non-negative")
        object.__setattr__(self, "matsubara_index", index)
        object.__setattr__(
            self,
            "theta_1_rad",
            _finite_angle(self.theta_1_rad, "theta_1_rad"),
        )
        object.__setattr__(
            self,
            "theta_2_rad",
            _finite_angle(self.theta_2_rad, "theta_2_rad"),
        )
        for name in ("plate_1_requirement", "plate_2_requirement"):
            value = str(getattr(self, name))
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 cache key")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GeometryBatchPlan:
    """Deterministic exact-response plan for angle and distance assembly."""

    response_config: MaterialResponseEngineConfig
    q_lab_by_label: Mapping[str, np.ndarray]
    angle_pairs_rad: tuple[tuple[float, float], ...]
    separations_m: tuple[float, ...]
    policy: GeometryBatchPolicy
    requirements: Mapping[str, PlateResponseRequirement]
    points: tuple[GeometryPointSpec, ...]
    schema: str = MATERIAL_GEOMETRY_BATCH_PLAN_SCHEMA
    production_casimir_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_GEOMETRY_BATCH_PLAN_SCHEMA:
            raise ValueError("unsupported geometry batch plan schema")
        if not isinstance(self.response_config, MaterialResponseEngineConfig):
            raise TypeError("response_config must be a MaterialResponseEngineConfig")
        if not isinstance(self.policy, GeometryBatchPolicy):
            raise TypeError("policy must be a GeometryBatchPolicy")

        q_map: dict[str, np.ndarray] = {}
        for label, value in self.q_lab_by_label.items():
            name = str(label)
            if not name or name in q_map:
                raise ValueError("q labels must be nonempty and unique")
            q_map[name] = _readonly_q(value, f"q_lab[{name}]")
        if not q_map:
            raise ValueError("at least one q_lab point is required")
        object.__setattr__(self, "q_lab_by_label", MappingProxyType(q_map))

        angles = tuple(
            (
                _finite_angle(first, "theta_1_rad"),
                _finite_angle(second, "theta_2_rad"),
            )
            for first, second in self.angle_pairs_rad
        )
        if not angles:
            raise ValueError("at least one angle pair is required")
        if len(set(angles)) != len(angles):
            raise ValueError("angle pairs must be unique")
        object.__setattr__(self, "angle_pairs_rad", angles)

        separations = tuple(
            _finite_positive(value, "separation_m") for value in self.separations_m
        )
        if not separations:
            raise ValueError("at least one separation is required")
        if tuple(sorted(set(separations))) != separations:
            raise ValueError("separations_m must be strictly increasing and unique")
        object.__setattr__(self, "separations_m", separations)

        requirements = dict(self.requirements)
        if not requirements:
            raise ValueError("geometry plan requires at least one response")
        for key, requirement in requirements.items():
            if not isinstance(requirement, PlateResponseRequirement):
                raise TypeError("requirements must contain PlateResponseRequirement values")
            if str(key) != requirement.key:
                raise ValueError("requirement mapping key differs from identity SHA")
        object.__setattr__(
            self,
            "requirements",
            MappingProxyType(dict(sorted(requirements.items()))),
        )

        points = tuple(self.points)
        if not points:
            raise ValueError("geometry plan requires at least one point")
        ids: set[str] = set()
        for point in points:
            if not isinstance(point, GeometryPointSpec):
                raise TypeError("points must contain GeometryPointSpec values")
            if point.point_id in ids:
                raise ValueError("geometry point ids must be unique")
            ids.add(point.point_id)
            if point.q_label not in q_map:
                raise ValueError("geometry point refers to an unknown q label")
            if not np.array_equal(point.q_lab, q_map[point.q_label]):
                raise ValueError("geometry point q_lab differs from q label mapping")
            if point.matsubara_index not in self.response_config.matsubara_indices:
                raise ValueError("geometry point Matsubara index is not requested")
            if point.plate_1_requirement not in requirements:
                raise ValueError("plate-1 requirement is absent from plan")
            if point.plate_2_requirement not in requirements:
                raise ValueError("plate-2 requirement is absent from plan")
        object.__setattr__(self, "points", points)

        if bool(self.production_casimir_allowed):
            raise ValueError("TODO 4 geometry plans cannot admit production")
        object.__setattr__(self, "production_casimir_allowed", False)

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "response_identity_context": {
                key: value
                for key, value in self.response_config.as_dict().items()
                if key != "runtime_chunk_size"
            },
            "q_lab_by_label_hex": {
                label: list(_q_hex(value))
                for label, value in sorted(self.q_lab_by_label.items())
            },
            "angle_pairs_rad_hex": [
                [float(first).hex(), float(second).hex()]
                for first, second in self.angle_pairs_rad
            ],
            "separations_m_hex": [float(value).hex() for value in self.separations_m],
            "policy": self.policy.as_dict(),
            "requirement_sha256": list(self.requirements),
            "point_ids": [point.point_id for point in self.points],
            "exact_q_mapping": "q_crystal=R(-theta_plate)@q_lab",
            "nearest_q_reuse": False,
            "angle_rounding": False,
            "symmetry_q_reduction": False,
            "production_casimir_allowed": False,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.identity_payload)).hexdigest()


def _normalized_q_points(
    values: Mapping[str, np.ndarray] | Sequence[tuple[str, np.ndarray]],
) -> tuple[tuple[str, np.ndarray], ...]:
    source = values.items() if isinstance(values, Mapping) else values
    points: list[tuple[str, np.ndarray]] = []
    labels: set[str] = set()
    for raw_label, raw_q in source:
        label = str(raw_label)
        if not label or label in labels:
            raise ValueError("q labels must be nonempty and unique")
        labels.add(label)
        points.append((label, _readonly_q(raw_q, f"q_lab[{label}]")))
    if not points:
        raise ValueError("at least one q_lab point is required")
    return tuple(points)


def build_geometry_batch_plan(
    response_config: MaterialResponseEngineConfig,
    *,
    q_lab_points: Mapping[str, np.ndarray] | Sequence[tuple[str, np.ndarray]],
    angle_pairs_rad: Sequence[tuple[float, float]],
    separations_m: Sequence[float],
    policy: GeometryBatchPolicy | None = None,
) -> GeometryBatchPlan:
    """Build a deterministic plan with exact crystal-q cache requirements."""

    if not isinstance(response_config, MaterialResponseEngineConfig):
        raise TypeError("response_config must be a MaterialResponseEngineConfig")
    q_points = _normalized_q_points(q_lab_points)
    angles = tuple(
        (
            _finite_angle(first, "theta_1_rad"),
            _finite_angle(second, "theta_2_rad"),
        )
        for first, second in angle_pairs_rad
    )
    if not angles or len(set(angles)) != len(angles):
        raise ValueError("angle_pairs_rad must be nonempty and unique")
    distances = tuple(_finite_positive(value, "separation_m") for value in separations_m)
    if tuple(sorted(set(distances))) != distances or not distances:
        raise ValueError("separations_m must be strictly increasing and unique")
    geometry_policy = GeometryBatchPolicy() if policy is None else policy
    if not isinstance(geometry_policy, GeometryBatchPolicy):
        raise TypeError("policy must be a GeometryBatchPolicy")

    identity_context = build_material_response_identity_context(response_config)
    requirements: dict[str, PlateResponseRequirement] = {}
    points: list[GeometryPointSpec] = []
    for q_label, q_lab in q_points:
        for matsubara_index in response_config.matsubara_indices:
            for angle_index, (theta_1, theta_2) in enumerate(angles):
                identities: list[MaterialResponseCacheIdentity] = []
                for theta in (theta_1, theta_2):
                    q_crystal = q_lab_to_crystal(q_lab, theta)
                    identity = build_material_response_cache_identity(
                        response_config,
                        q_crystal=q_crystal,
                        matsubara_index=matsubara_index,
                        context=identity_context,
                    )
                    requirements.setdefault(
                        identity.sha256,
                        PlateResponseRequirement(
                            identity=identity,
                            q_crystal=q_crystal,
                        ),
                    )
                    identities.append(identity)
                point_id = (
                    f"{q_label}:n={matsubara_index}:angles={angle_index}:"
                    f"{float(theta_1).hex()}:{float(theta_2).hex()}"
                )
                points.append(
                    GeometryPointSpec(
                        point_id=point_id,
                        q_label=q_label,
                        q_lab=q_lab,
                        matsubara_index=matsubara_index,
                        theta_1_rad=theta_1,
                        theta_2_rad=theta_2,
                        plate_1_requirement=identities[0].sha256,
                        plate_2_requirement=identities[1].sha256,
                    )
                )

    return GeometryBatchPlan(
        response_config=response_config,
        q_lab_by_label={label: q for label, q in q_points},
        angle_pairs_rad=angles,
        separations_m=distances,
        policy=geometry_policy,
        requirements=requirements,
        points=tuple(points),
    )


__all__ = [
    "GeometryBatchPlan",
    "GeometryBatchPolicy",
    "GeometryPointSpec",
    "MATERIAL_GEOMETRY_BATCH_PLAN_SCHEMA",
    "PlateResponseRequirement",
    "build_geometry_batch_plan",
]
