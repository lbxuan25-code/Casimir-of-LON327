"""Strict read-only angle and multi-distance geometry assembly.

This module consumes a :class:`GeometryBatchPlan` and the TODO 3 persistent
response store. It never imports or calls the microscopic response engine.
Every exact response is loaded at most once, every plate reflection is prepared
at most once per response/q/angle requirement, and every two-plate product is
prepared once before evaluating all requested distances.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.casimir.lifshitz_integrand import (
    LifshitzPoint,
    PreparedPassiveSheetPair,
    evaluate_prepared_passive_sheet_pair,
    prepare_passive_sheet_pair,
)
from lno327.casimir.material_geometry import (
    PlateReflection,
    material_response_to_reflection,
)
from lno327.casimir.material_geometry_plan import (
    GeometryBatchPlan,
    GeometryPointSpec,
)
from lno327.casimir.material_response_cache_artifact import (
    CachedCertifiedMaterialResponse,
)
from lno327.casimir.material_response_cache_errors import MaterialResponseCacheMiss
from lno327.casimir.material_response_cache_store import MaterialResponseCacheStore
from lno327.casimir.material_response_snapshot import MaterialResponseSnapshot

MATERIAL_GEOMETRY_BATCH_RESULT_SCHEMA = "material-geometry-batch-result-v1"


class GeometryBatchCacheIncomplete(MaterialResponseCacheMiss):
    """The strict read-only geometry plan has one or more missing responses."""


def _reflection_key(
    requirement_sha256: str,
    q_lab: np.ndarray,
    theta_rad: float,
) -> tuple[str, str, str, str]:
    q = np.asarray(q_lab, dtype=float)
    return (
        str(requirement_sha256),
        float(q[0]).hex(),
        float(q[1]).hex(),
        float(theta_rad).hex(),
    )


@dataclass(frozen=True)
class GeometryCachePreflight:
    """Validated cache availability and loaded artifacts for one plan."""

    plan_sha256: str
    artifacts: Mapping[str, CachedCertifiedMaterialResponse]
    hits: tuple[str, ...]
    misses: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        plan_sha = str(self.plan_sha256)
        if len(plan_sha) != 64:
            raise ValueError("plan_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "plan_sha256", plan_sha)
        artifacts = dict(self.artifacts)
        for key, artifact in artifacts.items():
            if len(str(key)) != 64:
                raise ValueError("artifact keys must be SHA-256 digests")
            if not isinstance(artifact, CachedCertifiedMaterialResponse):
                raise TypeError(
                    "preflight artifacts must be CachedCertifiedMaterialResponse values"
                )
            if artifact.identity.sha256 != str(key):
                raise ValueError("preflight artifact key differs from cache identity")
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(dict(sorted(artifacts.items()))),
        )
        hits = tuple(str(value) for value in self.hits)
        misses = tuple(str(value) for value in self.misses)
        if len(set(hits)) != len(hits) or len(set(misses)) != len(misses):
            raise ValueError("preflight hit and miss keys must be unique")
        if set(hits).intersection(misses):
            raise ValueError("a response cannot be both a hit and a miss")
        if set(hits) != set(artifacts):
            raise ValueError("preflight hit keys differ from loaded artifacts")
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "misses", misses)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def complete(self) -> bool:
        return not self.misses

    @property
    def snapshots(self) -> Mapping[str, MaterialResponseSnapshot]:
        return MappingProxyType(
            {key: artifact.snapshot for key, artifact in self.artifacts.items()}
        )


@dataclass(frozen=True)
class GeometryBatchPointResult:
    """All requested separations for one q/frequency/angle pair."""

    spec: GeometryPointSpec
    prepared_pair: PreparedPassiveSheetPair
    distances_m: tuple[float, ...]
    lifshitz_points: tuple[LifshitzPoint, ...]
    plate_1_diagnostics: Mapping[str, Any]
    plate_2_diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, GeometryPointSpec):
            raise TypeError("spec must be a GeometryPointSpec")
        if not isinstance(self.prepared_pair, PreparedPassiveSheetPair):
            raise TypeError("prepared_pair must be a PreparedPassiveSheetPair")
        distances = tuple(float(value) for value in self.distances_m)
        points = tuple(self.lifshitz_points)
        if not distances or len(distances) != len(points):
            raise ValueError("distance and Lifshitz point counts differ")
        if tuple(sorted(set(distances))) != distances:
            raise ValueError("distances_m must be strictly increasing and unique")
        for distance, point in zip(distances, points):
            if not isinstance(point, LifshitzPoint):
                raise TypeError("lifshitz_points must contain LifshitzPoint values")
            if point.separation_m != distance:
                raise ValueError("Lifshitz point separation differs from result distance")
        object.__setattr__(self, "distances_m", distances)
        object.__setattr__(self, "lifshitz_points", points)
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

    @property
    def logdets(self) -> np.ndarray:
        values = np.asarray([point.logdet for point in self.lifshitz_points], dtype=float)
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class GeometryBatchResult:
    """Deterministic geometry-only batch result."""

    plan: GeometryBatchPlan
    points: Mapping[str, GeometryBatchPointResult]
    preflight: GeometryCachePreflight
    metadata: Mapping[str, Any]
    schema: str = MATERIAL_GEOMETRY_BATCH_RESULT_SCHEMA
    production_casimir_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_GEOMETRY_BATCH_RESULT_SCHEMA:
            raise ValueError("unsupported geometry batch result schema")
        if not isinstance(self.plan, GeometryBatchPlan):
            raise TypeError("plan must be a GeometryBatchPlan")
        if not isinstance(self.preflight, GeometryCachePreflight):
            raise TypeError("preflight must be a GeometryCachePreflight")
        if self.preflight.plan_sha256 != self.plan.sha256 or not self.preflight.complete:
            raise ValueError("geometry batch requires a complete matching preflight")
        point_map = dict(self.points)
        expected = [point.point_id for point in self.plan.points]
        if set(point_map) != set(expected):
            raise ValueError("geometry result point ids differ from plan")
        for point_id, result in point_map.items():
            if not isinstance(result, GeometryBatchPointResult):
                raise TypeError("points must contain GeometryBatchPointResult values")
            if result.spec.point_id != point_id:
                raise ValueError("geometry result mapping key differs from point spec")
            if result.distances_m != self.plan.separations_m:
                raise ValueError("geometry point distances differ from plan")
        object.__setattr__(
            self,
            "points",
            MappingProxyType({point_id: point_map[point_id] for point_id in expected}),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if bool(self.production_casimir_allowed):
            raise ValueError("TODO 4 geometry results cannot admit production")
        object.__setattr__(self, "production_casimir_allowed", False)


def preflight_geometry_batch(
    plan: GeometryBatchPlan,
    *,
    cache: MaterialResponseCacheStore,
) -> GeometryCachePreflight:
    """Load and validate all exact requirements without microscopic fallback."""

    if not isinstance(plan, GeometryBatchPlan):
        raise TypeError("plan must be a GeometryBatchPlan")
    if not isinstance(cache, MaterialResponseCacheStore):
        raise TypeError("cache must be a MaterialResponseCacheStore")
    if cache.mode != "read_only":
        raise ValueError("geometry preflight requires a strict read_only cache")

    artifacts: dict[str, CachedCertifiedMaterialResponse] = {}
    hits: list[str] = []
    misses: list[str] = []
    for key, requirement in plan.requirements.items():
        try:
            artifact = cache.get(requirement.identity)
        except MaterialResponseCacheMiss:
            misses.append(key)
            continue
        if artifact is None:
            raise RuntimeError("read_only cache returned None instead of a typed miss")
        artifacts[key] = artifact
        hits.append(key)

    return GeometryCachePreflight(
        plan_sha256=plan.sha256,
        artifacts=artifacts,
        hits=tuple(hits),
        misses=tuple(misses),
        metadata={
            "schema": "geometry-cache-preflight-v1",
            "cache_mode": cache.mode,
            "requested_response_count": len(plan.requirements),
            "cache_hit_count": len(hits),
            "cache_miss_count": len(misses),
            "microscopic_fallback_attempted": False,
            "cache_write_attempted": False,
            "exact_identity_only": True,
            "production_casimir_allowed": False,
        },
    )


def require_complete_geometry_preflight(
    preflight: GeometryCachePreflight,
) -> GeometryCachePreflight:
    if not isinstance(preflight, GeometryCachePreflight):
        raise TypeError("preflight must be a GeometryCachePreflight")
    if preflight.misses:
        joined = ", ".join(preflight.misses)
        raise GeometryBatchCacheIncomplete(
            f"geometry batch is missing {len(preflight.misses)} exact responses: {joined}"
        )
    return preflight


def execute_geometry_batch(
    plan: GeometryBatchPlan,
    *,
    cache: MaterialResponseCacheStore,
) -> GeometryBatchResult:
    """Execute exact read-only angle and multi-distance geometry assembly."""

    preflight = require_complete_geometry_preflight(
        preflight_geometry_batch(plan, cache=cache)
    )
    reflection_cache: dict[
        tuple[str, str, str, str],
        tuple[PlateReflection, Mapping[str, Any]],
    ] = {}
    prepared_pairs: dict[str, PreparedPassiveSheetPair] = {}
    results: dict[str, GeometryBatchPointResult] = {}

    def reflection_for(
        requirement_key: str,
        *,
        q_lab: np.ndarray,
        theta_rad: float,
    ) -> tuple[PlateReflection, Mapping[str, Any]]:
        key = _reflection_key(requirement_key, q_lab, theta_rad)
        existing = reflection_cache.get(key)
        if existing is not None:
            return existing
        reflection, diagnostics = material_response_to_reflection(
            preflight.snapshots[requirement_key],
            q_lab=q_lab,
            theta_rad=theta_rad,
            policy=plan.policy.reflection_policy,
        )
        stored = (reflection, MappingProxyType(dict(diagnostics)))
        reflection_cache[key] = stored
        return stored

    for spec in plan.points:
        reflection_1, diagnostics_1 = reflection_for(
            spec.plate_1_requirement,
            q_lab=spec.q_lab,
            theta_rad=spec.theta_1_rad,
        )
        reflection_2, diagnostics_2 = reflection_for(
            spec.plate_2_requirement,
            q_lab=spec.q_lab,
            theta_rad=spec.theta_2_rad,
        )
        prepared = prepare_passive_sheet_pair(
            reflection_1,
            reflection_2,
            compatibility_tolerance=plan.policy.compatibility_tolerance,
            eigenvalue_imag_tolerance=plan.policy.eigenvalue_imag_tolerance,
            eigenvalue_lower_tolerance=plan.policy.eigenvalue_lower_tolerance,
        )
        prepared_pairs[spec.point_id] = prepared
        points = tuple(
            evaluate_prepared_passive_sheet_pair(
                prepared,
                separation_m=distance,
            )
            for distance in plan.separations_m
        )
        results[spec.point_id] = GeometryBatchPointResult(
            spec=spec,
            prepared_pair=prepared,
            distances_m=plan.separations_m,
            lifshitz_points=points,
            plate_1_diagnostics=diagnostics_1,
            plate_2_diagnostics=diagnostics_2,
        )

    return GeometryBatchResult(
        plan=plan,
        points=results,
        preflight=preflight,
        metadata={
            "casimir_stage": "geometry_only_exact_response_batch",
            "plan_sha256": plan.sha256,
            "response_load_count": len(preflight.hits),
            "unique_response_identity_count": len(plan.requirements),
            "reflection_build_count": len(reflection_cache),
            "prepared_pair_count": len(prepared_pairs),
            "distance_update_count": len(plan.points) * len(plan.separations_m),
            "requested_geometry_point_count": len(plan.points),
            "requested_distance_count": len(plan.separations_m),
            "microscopic_integration_call_count": 0,
            "response_certification_call_count": 0,
            "cache_write_count": 0,
            "strict_read_only_cache": True,
            "nearest_q_reuse": False,
            "angle_rounding": False,
            "symmetry_q_reduction": False,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
    )


__all__ = [
    "GeometryBatchCacheIncomplete",
    "GeometryBatchPointResult",
    "GeometryBatchResult",
    "GeometryCachePreflight",
    "MATERIAL_GEOMETRY_BATCH_RESULT_SCHEMA",
    "execute_geometry_batch",
    "preflight_geometry_batch",
    "require_complete_geometry_preflight",
]
