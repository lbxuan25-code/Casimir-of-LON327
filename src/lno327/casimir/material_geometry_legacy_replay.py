"""Narrow matched-N/shift legacy replay for TODO 4 qualification.

This module is deliberately outside the core geometry planner and executor. It
may rebuild one archived geometry-specific microscopic point for direct
qualification, but it is never imported by the read-only geometry batch route
and cannot act as a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

import numpy as np

from lno327 import KuboConfig
from lno327.casimir.material_geometry_batch import GeometryBatchResult
from lno327.casimir.material_geometry_qualification import (
    GeometryEquivalencePolicy,
    GeometryEquivalenceReport,
    qualify_matched_legacy_point,
)
from lno327.casimir.microscopic_model import get_finite_q_microscopic_model
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.response.arbitrary_q_material_cache import build_material_grid_cache
from lno327.response.periodic_bz_grid import build_periodic_bz_grid
from lno327.workflows.arbitrary_q_matsubara import integrate_two_plate_angle_batch
from lno327.workflows.finite_q_engine import FiniteQEngineOptions

MATERIAL_GEOMETRY_LEGACY_REPLAY_SCHEMA = "material-geometry-legacy-replay-v1"


@dataclass(frozen=True)
class LegacyGeometryReplayResult:
    """One real archived-point replay and its direct equivalence report."""

    report: GeometryEquivalenceReport
    point_id: str
    distance_m: float
    working_N: int
    primary_shift: tuple[float, float]
    metadata: Mapping[str, Any]
    schema: str = MATERIAL_GEOMETRY_LEGACY_REPLAY_SCHEMA
    production_casimir_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_GEOMETRY_LEGACY_REPLAY_SCHEMA:
            raise ValueError("unsupported legacy geometry replay schema")
        if not isinstance(self.report, GeometryEquivalenceReport):
            raise TypeError("report must be a GeometryEquivalenceReport")
        point_id = str(self.point_id)
        if not point_id or point_id != self.report.point_id:
            raise ValueError("point_id must match the equivalence report")
        object.__setattr__(self, "point_id", point_id)
        distance = float(self.distance_m)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("distance_m must be finite and positive")
        object.__setattr__(self, "distance_m", distance)
        working = int(self.working_N)
        if working <= 0 or working % 2:
            raise ValueError("working_N must be a positive even integer")
        object.__setattr__(self, "working_N", working)
        shift = tuple(float(value) for value in self.primary_shift)
        if len(shift) != 2 or not np.isfinite(shift).all():
            raise ValueError("primary_shift must be a finite two-component shift")
        object.__setattr__(self, "primary_shift", shift)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if bool(self.production_casimir_allowed):
            raise ValueError("TODO 4 legacy replay cannot admit production")
        object.__setattr__(self, "production_casimir_allowed", False)


def _shift_label(index: int, shift: tuple[float, float]) -> str:
    return f"shift_{index}:{float(shift[0]).hex()}:{float(shift[1]).hex()}"


def _primary_shift(artifact: object) -> tuple[float, float]:
    for index, shift in enumerate(artifact.identity.shifts):
        normalized = tuple(float(value) for value in shift)
        if _shift_label(index, normalized) == artifact.primary_shift:
            return normalized
    raise ValueError("artifact primary_shift is absent from its certification identity")


def _require_plate_identity_compatibility(first: object, second: object) -> None:
    """Require one common material/frequency contract apart from exact q."""

    names = (
        "pairing_name",
        "temperature_K",
        "matsubara_index",
        "xi_eV",
        "microscopic_model_name",
        "material_state_fingerprint",
        "response_policy_fingerprint",
        "primitive_contract_version",
        "phase_hessian_policy",
        "basis",
        "certification_policy_fingerprint",
        "canonical_reduction_block_size",
    )
    mismatched = [
        name
        for name in names
        if getattr(first.identity, name) != getattr(second.identity, name)
    ]
    if mismatched:
        raise ValueError(
            "plate response identities differ outside q_crystal: "
            + ", ".join(mismatched)
        )


def _require_legacy_policy_compatibility(batch: GeometryBatchResult) -> None:
    """Fail if the archived helper cannot express the requested material policy."""

    material = batch.plan.response_config.material_policy
    hardcoded_static = {
        "static_primitive_tolerance": 1e-6,
        "static_amplitude_tolerance": 1e-6,
        "static_phase_tolerance": 1e-6,
        "static_effective_direct_tolerance": 1e-6,
        "static_effective_residual_tolerance": 1e-6,
    }
    for name, expected in hardcoded_static.items():
        if float(getattr(material, name)) != expected:
            raise ValueError(
                f"legacy point helper hardcodes {name}={expected}; requested policy differs"
            )
    positive_defaults = {
        "positive_reality_tolerance": 1e-9,
        "positive_symmetry_tolerance": 1e-9,
        "positive_passivity_tolerance": 1e-10,
    }
    for name, expected in positive_defaults.items():
        if float(getattr(material, name)) != expected:
            raise ValueError(
                f"legacy positive response helper hardcodes {name}={expected}; "
                "requested policy differs"
            )
    reflection = batch.plan.policy.reflection_policy
    expected_lattice = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m
    if float(reflection.lattice_constant_m) != float(expected_lattice):
        raise ValueError("legacy point helper uses the fixed LNO327 in-plane lattice")
    if not reflection.require_physical:
        raise ValueError("legacy replay qualification requires the hard physical gate")


def _legacy_args(batch: GeometryBatchResult, point: object) -> SimpleNamespace:
    material = batch.plan.response_config.material_policy
    return SimpleNamespace(
        plate_angles_rad=(float(point.theta_1_rad), float(point.theta_2_rad)),
        ward_tolerance=float(material.ward_tolerance),
        ward_absolute_tolerance=float(material.ward_absolute_tolerance),
        condition_max=float(material.condition_max),
        static_energy_scale_eV=float(material.static_energy_scale_eV),
        static_reality_tolerance=float(material.static_reality_tolerance),
        static_longitudinal_tolerance=float(material.static_longitudinal_tolerance),
        static_mixing_tolerance=float(material.static_mixing_tolerance),
        static_passivity_tolerance=float(material.static_passivity_tolerance),
        degeneracy=float(material.degeneracy),
    )


def _require_legacy_result_contract(
    result: object,
    *,
    artifact: object,
    plate_name: str,
) -> None:
    metadata = dict(getattr(result, "metadata", {}))
    expected = {
        "material_state_fingerprint": artifact.identity.material_state_fingerprint,
        "primitive_contract_version": artifact.identity.primitive_contract_version,
        "post_integral_phase_hessian_policy": artifact.identity.phase_hessian_policy,
        "canonical_reduction_block_size": (
            artifact.identity.canonical_reduction_block_size
        ),
    }
    mismatched = {
        name: {"expected": value, "actual": metadata.get(name)}
        for name, value in expected.items()
        if metadata.get(name) != value
    }
    if mismatched:
        raise ValueError(
            f"legacy replay {plate_name} contract differs from cache identity: "
            f"{mismatched}"
        )


def run_matched_legacy_geometry_replay(
    batch: GeometryBatchResult,
    *,
    point_id: str,
    distance_m: float,
    policy: GeometryEquivalencePolicy | None = None,
) -> LegacyGeometryReplayResult:
    """Rebuild and compare one archived point under exact persisted evidence.

    The function executes exactly one working-N, primary-shift replay. It does
    not search an N ladder, populate response caches, or extend the requested
    q/angle/distance set.
    """

    if not isinstance(batch, GeometryBatchResult):
        raise TypeError("batch must be a GeometryBatchResult")
    _require_legacy_policy_compatibility(batch)
    point = batch.points[str(point_id)]
    spec = point.spec
    first_artifact = batch.preflight.artifacts[spec.plate_1_requirement]
    second_artifact = batch.preflight.artifacts[spec.plate_2_requirement]
    _require_plate_identity_compatibility(first_artifact, second_artifact)
    if first_artifact.working_N != second_artifact.working_N:
        raise ValueError("plate responses use different working N values")
    shift_1 = _primary_shift(first_artifact)
    shift_2 = _primary_shift(second_artifact)
    if tuple(float(value).hex() for value in shift_1) != tuple(
        float(value).hex() for value in shift_2
    ):
        raise ValueError("plate responses use different primary shifts")
    if first_artifact.identity.matsubara_index != spec.matsubara_index:
        raise ValueError("plate-1 artifact Matsubara index differs from geometry point")

    config = batch.plan.response_config
    model = get_finite_q_microscopic_model(config.microscopic_model_name)
    ansatz = model.build_ansatz(
        config.pairing_name,
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(config.delta0_eV)
    xi = float(first_artifact.identity.xi_eV)
    base_config = KuboConfig.from_kelvin(
        omega_eV=xi,
        temperature_K=config.temperature_K,
        eta_eV=config.eta_eV,
        output_si=False,
    )
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    grid = build_periodic_bz_grid(first_artifact.working_N, shift_1)
    material_cache = build_material_grid_cache(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        config=base_config,
        options=options,
        grid=grid,
    )
    legacy_batch = integrate_two_plate_angle_batch(
        q_lab=spec.q_lab,
        theta_1_rad=spec.theta_1_rad,
        theta_2_rad_values=(spec.theta_2_rad,),
        material_cache=material_cache,
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=(xi,),
        temperature_K=config.temperature_K,
        eta_eV=config.eta_eV,
        canonical_reduction_block_size=config.canonical_reduction_block_size,
        runtime_chunk_size=config.runtime_chunk_size,
    )
    _require_legacy_result_contract(
        legacy_batch.plate_1,
        artifact=first_artifact,
        plate_name="plate_1",
    )
    _require_legacy_result_contract(
        legacy_batch.plate_2[0],
        artifact=second_artifact,
        plate_name="plate_2",
    )

    report = qualify_matched_legacy_point(
        batch,
        point_id=spec.point_id,
        distance_m=distance_m,
        legacy_batch=legacy_batch,
        legacy_frequency_index=0,
        legacy_n=first_artifact.working_N,
        legacy_xi_eV=xi,
        legacy_args=_legacy_args(batch, spec),
        policy=policy,
    )
    return LegacyGeometryReplayResult(
        report=report,
        point_id=spec.point_id,
        distance_m=distance_m,
        working_N=first_artifact.working_N,
        primary_shift=shift_1,
        metadata={
            "casimir_stage": "narrow_matched_legacy_geometry_replay",
            "model_name": config.microscopic_model_name,
            "pairing_name": config.pairing_name,
            "temperature_K": config.temperature_K,
            "matsubara_index": spec.matsubara_index,
            "xi_eV": xi,
            "q_lab": spec.q_lab.tolist(),
            "theta_1_rad": spec.theta_1_rad,
            "theta_2_rad": spec.theta_2_rad,
            "working_N": first_artifact.working_N,
            "primary_shift": list(shift_1),
            "primitive_contract_version": (
                first_artifact.identity.primitive_contract_version
            ),
            "phase_hessian_policy": first_artifact.identity.phase_hessian_policy,
            "canonical_reduction_block_size": config.canonical_reduction_block_size,
            "all_material_and_numerical_contracts_matched": True,
            "n_ladder_search_performed": False,
            "response_cache_write_performed": False,
            "geometry_fallback_enabled": False,
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
    )


__all__ = [
    "LegacyGeometryReplayResult",
    "MATERIAL_GEOMETRY_LEGACY_REPLAY_SCHEMA",
    "run_matched_legacy_geometry_replay",
]
