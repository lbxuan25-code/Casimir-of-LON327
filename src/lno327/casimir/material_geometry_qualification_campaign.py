"""Frozen TODO 4 representative qualification campaign construction.

This module is pure planning/serialization support. It never reads or writes the
response cache, evaluates microscopic responses, constructs reflections, or runs
legacy qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.casimir.material_geometry import ReflectionGeometryPolicy
from lno327.casimir.material_geometry_plan import (
    GeometryBatchPlan,
    GeometryBatchPolicy,
    build_geometry_batch_plan,
)
from lno327.casimir.material_geometry_qualification import GeometryEquivalencePolicy
from lno327.casimir.material_geometry_outer_qualification import (
    FixedOuterEquivalencePolicy,
)
from lno327.casimir.material_response import MaterialResponsePolicy
from lno327.casimir.material_response_certification import (
    MaterialResponseConvergencePolicy,
)
from lno327.casimir.material_response_engine import MaterialResponseEngineConfig
from lno327.casimir.outer_quadrature import OuterQPolarGrid, build_outer_q_polar_grid
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE

TODO4_QUALIFICATION_MANIFEST_SCHEMA = (
    "todo4-representative-qualification-manifest-v1"
)
TODO4_QUALIFICATION_PLAN_SCHEMA = "todo4-representative-qualification-plan-v1"


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _finite_positive(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_angle_pair(value: object, name: str) -> tuple[float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain two finite angles")
    return values


def _q_vector(value: object, name: str) -> np.ndarray:
    q = np.asarray(value, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError(f"{name} must contain two finite components")
    if float(np.linalg.norm(q)) == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return q


def load_todo4_qualification_manifest(path: Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    manifest = _require_mapping(payload, "manifest")
    if manifest.get("schema") != TODO4_QUALIFICATION_MANIFEST_SCHEMA:
        raise ValueError("unsupported TODO 4 qualification manifest schema")
    if not str(manifest.get("campaign_id", "")):
        raise ValueError("campaign_id must be nonempty")
    for name in (
        "diagnostic_only",
        "valid_for_casimir_input",
        "production_casimir_allowed",
    ):
        expected = name == "diagnostic_only"
        if bool(manifest.get(name)) is not expected:
            raise ValueError(f"manifest {name} has an unsafe value")
    direct = manifest.get("direct_cases")
    if not isinstance(direct, list) or not direct:
        raise ValueError("manifest requires nonempty direct_cases")
    return manifest


def _response_config(
    manifest: Mapping[str, Any],
    pairing_name: str,
) -> MaterialResponseEngineConfig:
    response = _require_mapping(manifest["response"], "response")
    material = MaterialResponsePolicy(
        **_require_mapping(response["material_policy"], "material_policy")
    )
    convergence = MaterialResponseConvergencePolicy(
        **_require_mapping(response["convergence_policy"], "convergence_policy")
    )
    return MaterialResponseEngineConfig(
        pairing_name=str(pairing_name),
        temperature_K=float(response["temperature_K"]),
        delta0_eV=float(response["delta0_eV"]),
        eta_eV=float(response["eta_eV"]),
        matsubara_indices=tuple(int(value) for value in response["matsubara_indices"]),
        n_candidates=tuple(int(value) for value in response["n_candidates"]),
        shifts=tuple(
            tuple(float(component) for component in shift)
            for shift in response["shifts"]
        ),
        required_consecutive_passes=int(response["required_consecutive_passes"]),
        envelope_levels=int(response["envelope_levels"]),
        canonical_reduction_block_size=int(
            response["canonical_reduction_block_size"]
        ),
        runtime_chunk_size=int(response["runtime_chunk_size"]),
        microscopic_model_name=str(response["microscopic_model_name"]),
        material_policy=material,
        convergence_policy=convergence,
    )


def _geometry_policy(manifest: Mapping[str, Any]) -> GeometryBatchPolicy:
    payload = _require_mapping(manifest["geometry_policy"], "geometry_policy")
    reflection = ReflectionGeometryPolicy(
        lattice_constant_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        q_match_tolerance=float(payload["q_match_tolerance"]),
        require_physical=bool(payload["require_physical"]),
    )
    return GeometryBatchPolicy(
        reflection_policy=reflection,
        compatibility_tolerance=float(payload["compatibility_tolerance"]),
        eigenvalue_imag_tolerance=float(payload["eigenvalue_imag_tolerance"]),
        eigenvalue_lower_tolerance=float(payload["eigenvalue_lower_tolerance"]),
    )


def geometry_equivalence_policy(
    manifest: Mapping[str, Any],
) -> GeometryEquivalencePolicy:
    payload = _require_mapping(manifest["equivalence_policy"], "equivalence_policy")
    return GeometryEquivalencePolicy(
        absolute_tolerance=float(payload["logdet_absolute"]),
        relative_tolerance=float(payload["logdet_relative"]),
        matrix_absolute_tolerance=float(payload["matrix_absolute"]),
        matrix_relative_tolerance=float(payload["matrix_relative"]),
    )


def fixed_outer_equivalence_policy(
    manifest: Mapping[str, Any],
) -> FixedOuterEquivalencePolicy:
    payload = _require_mapping(manifest["equivalence_policy"], "equivalence_policy")
    return FixedOuterEquivalencePolicy(
        node_logdet_absolute=float(payload["logdet_absolute"]),
        node_logdet_relative=float(payload["logdet_relative"]),
        outer_integral_absolute_m_inv2=float(
            payload["outer_integral_absolute_m_inv2"]
        ),
        outer_integral_relative=float(payload["outer_integral_relative"]),
        contribution_absolute_J_m2=float(payload["contribution_absolute_J_m2"]),
        contribution_relative=float(payload["contribution_relative"]),
        total_absolute_J_m2=float(payload["total_absolute_J_m2"]),
        total_relative=float(payload["total_relative"]),
    )


@dataclass(frozen=True)
class Todo4QualificationPlanEntry:
    plan_id: str
    kind: str
    pairing_name: str
    geometry_plan: GeometryBatchPlan
    legacy_distance_m: float
    outer_grid: OuterQPolarGrid | None = None

    def __post_init__(self) -> None:
        plan_id = str(self.plan_id)
        if not plan_id:
            raise ValueError("plan_id must be nonempty")
        object.__setattr__(self, "plan_id", plan_id)
        kind = str(self.kind)
        if kind not in {"direct", "fixed_outer"}:
            raise ValueError("kind must be direct or fixed_outer")
        object.__setattr__(self, "kind", kind)
        if self.geometry_plan.response_config.pairing_name != self.pairing_name:
            raise ValueError("pairing_name differs from geometry plan")
        distance = _finite_positive(self.legacy_distance_m, "legacy_distance_m")
        if distance not in self.geometry_plan.separations_m:
            raise ValueError("legacy_distance_m is absent from geometry plan")
        object.__setattr__(self, "legacy_distance_m", distance)
        if kind == "fixed_outer" and self.outer_grid is None:
            raise ValueError("fixed_outer entry requires outer_grid")
        if kind == "direct" and self.outer_grid is not None:
            raise ValueError("direct entry cannot carry outer_grid")


@dataclass(frozen=True)
class Todo4QualificationCampaign:
    manifest: Mapping[str, Any]
    manifest_sha256: str
    entries: tuple[Todo4QualificationPlanEntry, ...]

    def __post_init__(self) -> None:
        manifest = dict(self.manifest)
        if _sha256(manifest) != str(self.manifest_sha256):
            raise ValueError("manifest SHA does not match manifest content")
        object.__setattr__(self, "manifest", MappingProxyType(manifest))
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("qualification campaign requires at least one plan")
        ids = [entry.plan_id for entry in entries]
        if len(set(ids)) != len(ids):
            raise ValueError("qualification plan ids must be unique")
        object.__setattr__(self, "entries", entries)

    @property
    def campaign_id(self) -> str:
        return str(self.manifest["campaign_id"])

    @property
    def unique_requirements(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for entry in self.entries:
            result.update(entry.geometry_plan.requirements)
        return dict(sorted(result.items()))

    @property
    def populate_groups(self) -> tuple[tuple[str, tuple[str, str], np.ndarray], ...]:
        groups: dict[tuple[str, str, str], np.ndarray] = {}
        for entry in self.entries:
            pairing = entry.pairing_name
            for requirement in entry.geometry_plan.requirements.values():
                q = np.asarray(requirement.q_crystal, dtype=float)
                key = (pairing, float(q[0]).hex(), float(q[1]).hex())
                groups.setdefault(key, q)
        return tuple(
            (key[0], (key[1], key[2]), groups[key])
            for key in sorted(groups)
        )


def build_todo4_qualification_campaign(
    manifest: Mapping[str, Any],
) -> Todo4QualificationCampaign:
    source = dict(manifest)
    if source.get("schema") != TODO4_QUALIFICATION_MANIFEST_SCHEMA:
        raise ValueError("unsupported TODO 4 qualification manifest schema")
    geometry_policy = _geometry_policy(source)
    entries: list[Todo4QualificationPlanEntry] = []

    for raw_case in source["direct_cases"]:
        case = _require_mapping(raw_case, "direct case")
        case_id = str(case["case_id"])
        q = _q_vector(case["q_lab"], f"{case_id}.q_lab")
        angles = _finite_angle_pair(
            case["angle_pair_rad"],
            f"{case_id}.angle_pair_rad",
        )
        separations = tuple(
            _finite_positive(value, f"{case_id}.separation_nm") * 1e-9
            for value in case["separations_nm"]
        )
        legacy_distance = (
            _finite_positive(
                case["legacy_distance_nm"],
                f"{case_id}.legacy_distance_nm",
            )
            * 1e-9
        )
        for pairing in case["pairings"]:
            config = _response_config(source, str(pairing))
            plan = build_geometry_batch_plan(
                config,
                q_lab_points={case_id: q},
                angle_pairs_rad=(angles,),
                separations_m=separations,
                policy=geometry_policy,
            )
            entries.append(
                Todo4QualificationPlanEntry(
                    plan_id=f"direct/{case_id}/{pairing}",
                    kind="direct",
                    pairing_name=str(pairing),
                    geometry_plan=plan,
                    legacy_distance_m=legacy_distance,
                )
            )

    outer = _require_mapping(source["fixed_outer"], "fixed_outer")
    if bool(outer.get("enabled")):
        material = LNO327_THIN_FILM_SLAO_IN_PLANE
        distance = _finite_positive(
            outer["separation_nm"],
            "fixed_outer.separation_nm",
        ) * 1e-9
        grid = build_outer_q_polar_grid(
            separation_m=distance,
            lattice_a_x_m=material.lattice_a_x_m,
            lattice_a_y_m=material.lattice_a_y_m,
            u_max=_finite_positive(outer["u_max"], "fixed_outer.u_max"),
            radial_order=int(outer["radial_order"]),
            angular_order=int(outer["angular_order"]),
            angular_offset_fraction=float(outer["angular_offset_fraction"]),
        )
        labels = {
            f"outer_{index:04d}": grid.q_model[index]
            for index in range(grid.node_count)
        }
        angles = _finite_angle_pair(
            outer["angle_pair_rad"],
            "fixed_outer.angle_pair_rad",
        )
        case_id = str(outer["case_id"])
        for pairing in outer["pairings"]:
            config = _response_config(source, str(pairing))
            plan = build_geometry_batch_plan(
                config,
                q_lab_points=labels,
                angle_pairs_rad=(angles,),
                separations_m=(distance,),
                policy=geometry_policy,
            )
            entries.append(
                Todo4QualificationPlanEntry(
                    plan_id=f"fixed_outer/{case_id}/{pairing}",
                    kind="fixed_outer",
                    pairing_name=str(pairing),
                    geometry_plan=plan,
                    legacy_distance_m=distance,
                    outer_grid=grid,
                )
            )

    return Todo4QualificationCampaign(
        manifest=source,
        manifest_sha256=_sha256(source),
        entries=tuple(entries),
    )


def _requirement_payload(requirement: object) -> dict[str, Any]:
    identity = requirement.identity
    q = np.asarray(requirement.q_crystal, dtype=float)
    return {
        "sha256": identity.sha256,
        "identity": identity.payload,
        "q_crystal_hex": [float(q[0]).hex(), float(q[1]).hex()],
    }


def qualification_plan_payload(
    campaign: Todo4QualificationCampaign,
    *,
    source_commit: str,
) -> dict[str, Any]:
    plans: list[dict[str, Any]] = []
    for entry in campaign.entries:
        plan = entry.geometry_plan
        record: dict[str, Any] = {
            "plan_id": entry.plan_id,
            "kind": entry.kind,
            "pairing_name": entry.pairing_name,
            "plan_sha256": plan.sha256,
            "legacy_distance_m_hex": float(entry.legacy_distance_m).hex(),
            "q_labels": list(plan.q_lab_by_label),
            "matsubara_indices": list(plan.response_config.matsubara_indices),
            "angle_pairs_rad_hex": [
                [float(first).hex(), float(second).hex()]
                for first, second in plan.angle_pairs_rad
            ],
            "separations_m_hex": [
                float(value).hex() for value in plan.separations_m
            ],
            "requirement_count": len(plan.requirements),
            "geometry_point_count": len(plan.points),
            "requirements": [
                _requirement_payload(plan.requirements[key])
                for key in plan.requirements
            ],
            "points": [
                {
                    "point_id": point.point_id,
                    "q_label": point.q_label,
                    "matsubara_index": point.matsubara_index,
                    "theta_1_rad_hex": float(point.theta_1_rad).hex(),
                    "theta_2_rad_hex": float(point.theta_2_rad).hex(),
                    "plate_1_requirement": point.plate_1_requirement,
                    "plate_2_requirement": point.plate_2_requirement,
                }
                for point in plan.points
            ],
        }
        if entry.outer_grid is not None:
            record["outer_grid"] = {
                "schema": entry.outer_grid.metadata.get("schema"),
                "node_count": entry.outer_grid.node_count,
                "u_max": entry.outer_grid.u_max,
                "radial_order": entry.outer_grid.radial_order,
                "angular_order": entry.outer_grid.angular_order,
                "angular_offset_fraction": entry.outer_grid.angular_offset_fraction,
                "q_model_hex": [
                    [float(qx).hex(), float(qy).hex()]
                    for qx, qy in entry.outer_grid.q_model
                ],
            }
        plans.append(record)

    payload = {
        "schema": TODO4_QUALIFICATION_PLAN_SCHEMA,
        "campaign_id": campaign.campaign_id,
        "source_commit": str(source_commit),
        "manifest_sha256": campaign.manifest_sha256,
        "manifest": dict(campaign.manifest),
        "plans": plans,
        "summary": {
            "plan_count": len(campaign.entries),
            "direct_plan_count": sum(
                entry.kind == "direct" for entry in campaign.entries
            ),
            "fixed_outer_plan_count": sum(
                entry.kind == "fixed_outer" for entry in campaign.entries
            ),
            "geometry_point_count": sum(
                len(entry.geometry_plan.points) for entry in campaign.entries
            ),
            "unique_response_identity_count": len(campaign.unique_requirements),
            "populate_group_count": len(campaign.populate_groups),
            "expected_reflection_build_count": sum(
                len(entry.geometry_plan.requirements) for entry in campaign.entries
            ),
            "expected_prepared_pair_count": sum(
                len(entry.geometry_plan.points) for entry in campaign.entries
            ),
            "expected_distance_update_count": sum(
                len(entry.geometry_plan.points)
                * len(entry.geometry_plan.separations_m)
                for entry in campaign.entries
            ),
        },
        "contract": {
            "exact_q_mapping": "q_crystal=R(-theta_plate)@q_lab",
            "nearest_q_reuse": False,
            "q_rounding": False,
            "angle_rounding": False,
            "symmetry_q_reduction": False,
            "interpolation": False,
            "geometry_cache_mode": "read_only",
            "populate_separate_from_geometry": True,
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
    }
    payload["plan_sha256"] = _sha256(payload)
    return payload


__all__ = [
    "TODO4_QUALIFICATION_MANIFEST_SCHEMA",
    "TODO4_QUALIFICATION_PLAN_SCHEMA",
    "Todo4QualificationCampaign",
    "Todo4QualificationPlanEntry",
    "build_todo4_qualification_campaign",
    "fixed_outer_equivalence_policy",
    "geometry_equivalence_policy",
    "load_todo4_qualification_manifest",
    "qualification_plan_payload",
]
