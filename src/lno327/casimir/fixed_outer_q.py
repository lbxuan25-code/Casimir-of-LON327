"""Fixed-grid outer-Q planning and reduction for the production Casimir chain.

This module is intentionally non-adaptive.  It owns the already-qualified nested
composite radial-panel contract, exact node reuse, certified-point reduction, and
outer-ladder comparisons.  Validation may import this module; production code must
never import from :mod:`validation`.

The fixed contract is

    u = 2 Q d,
    d^2Q/(2 pi)^2 = u du dphi/(16 pi^2 d^2),

with the full angular interval ``[0, 2 pi)``, periodic equal-weight trapezoidal
nodes, no symmetry reduction, and no explicit ``Q=0`` node.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .compound_outer_quadrature import build_compound_outer_q_polar_grid
from .outer_quadrature import OuterQPolarGrid, free_energy_per_area_from_logdet


def _positive_unique_floats(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(dict.fromkeys(float(value) for value in values))
    if not result or any(not np.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    if tuple(sorted(result)) != result:
        raise ValueError(f"{name} must be strictly increasing and unique")
    return result


def _positive_unique_ints(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(value) for value in values))
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    if tuple(sorted(result)) != result:
        raise ValueError(f"{name} must be strictly increasing and unique")
    return result


def _offsets(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(dict.fromkeys(float(value) for value in values))
    if not result or any(
        not np.isfinite(value) or not 0.0 <= value < 1.0 for value in result
    ):
        raise ValueError("angular offsets must be unique finite values in [0, 1)")
    return result


def _token(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class OuterQGridSpec:
    """One fixed nested-composite outer-Q quadrature specification."""

    spec_id: str
    u_max: float
    radial_order: int
    angular_order: int
    angular_offset_fraction: float
    radial_panel_edges: tuple[float, ...]

    @property
    def radial_panel_order(self) -> int:
        return int(self.radial_order)

    @property
    def radial_panel_count(self) -> int:
        return len(self.radial_panel_edges) - 1


@dataclass(frozen=True)
class OuterQGridPlan:
    """Fixed cutoff/order/offset ladders sharing one reference grid."""

    specs: tuple[OuterQGridSpec, ...]
    ladders: Mapping[str, tuple[str, ...]]
    reference_spec_id: str
    reference_offset_fraction: float

    def __post_init__(self) -> None:
        ids = tuple(spec.spec_id for spec in self.specs)
        if len(set(ids)) != len(ids):
            raise ValueError("outer-q grid spec ids must be unique")
        known = set(ids)
        copied = {str(name): tuple(values) for name, values in self.ladders.items()}
        if any(not set(values).issubset(known) for values in copied.values()):
            raise ValueError("outer-q ladder references an unknown grid spec")
        if self.reference_spec_id not in known:
            raise ValueError("reference_spec_id is not present in specs")
        object.__setattr__(self, "ladders", MappingProxyType(copied))


def build_staged_grid_plan(
    *,
    u_max_values: Sequence[float],
    radial_orders: Sequence[int],
    angular_orders: Sequence[int],
    angular_offsets: Sequence[float],
) -> OuterQGridPlan:
    """Build fixed nested cutoff and one-at-a-time order/offset ladders.

    ``u_max_values`` are cumulative radial-panel boundaries.  For example,
    ``(6, 10, 14)`` produces ``[0,6]``, then adds ``[6,10]``, then adds
    ``[10,14]``.  ``radial_orders`` are Gauss-Legendre orders per panel.
    """

    u_values = _positive_unique_floats(u_max_values, "u_max_values")
    radial = _positive_unique_ints(radial_orders, "radial_orders")
    angular = _positive_unique_ints(angular_orders, "angular_orders")
    offsets = _offsets(angular_offsets)
    reference_offset = 0.5 if 0.5 in offsets else offsets[-1]

    registered: dict[tuple[tuple[float, ...], int, int, float], OuterQGridSpec] = {}

    def panel_edges_for(upper: float) -> tuple[float, ...]:
        selected = tuple(value for value in u_values if value <= float(upper))
        if not selected or selected[-1] != float(upper):
            raise ValueError("cutoff must be one of the registered panel boundaries")
        return (0.0, *selected)

    def register(upper: float, panel_order: int, nphi: int, offset: float) -> str:
        edges = panel_edges_for(upper)
        key = (edges, int(panel_order), int(nphi), float(offset))
        spec = registered.get(key)
        if spec is None:
            spec_id = (
                f"u{_token(upper)}_p{len(edges) - 1}_r{int(panel_order)}_"
                f"a{int(nphi)}_o{_token(offset)}"
            )
            spec = OuterQGridSpec(
                spec_id=spec_id,
                u_max=float(upper),
                radial_order=int(panel_order),
                angular_order=int(nphi),
                angular_offset_fraction=float(offset),
                radial_panel_edges=edges,
            )
            registered[key] = spec
        return spec.spec_id

    u_ref = u_values[-1]
    radial_ref = radial[-1]
    angular_ref = angular[-1]
    ladders = {
        "cutoff": tuple(
            register(value, radial_ref, angular_ref, reference_offset)
            for value in u_values
        ),
        "radial": tuple(
            register(u_ref, value, angular_ref, reference_offset)
            for value in radial
        ),
        "angular": tuple(
            register(u_ref, radial_ref, value, reference_offset)
            for value in angular
        ),
        "offset": tuple(
            register(u_ref, radial_ref, angular_ref, value) for value in offsets
        ),
    }
    reference = register(u_ref, radial_ref, angular_ref, reference_offset)
    return OuterQGridPlan(
        specs=tuple(registered.values()),
        ladders=ladders,
        reference_spec_id=reference,
        reference_offset_fraction=reference_offset,
    )


@dataclass(frozen=True)
class OuterQNodeManifest:
    """Deduplicated model-q nodes shared by every fixed-grid specification."""

    labels: tuple[str, ...]
    q_model: np.ndarray
    grids: Mapping[str, OuterQPolarGrid]
    labels_by_spec: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        q = np.array(self.q_model, dtype=float, copy=True)
        if q.shape != (len(self.labels), 2) or not np.isfinite(q).all():
            raise ValueError("q_model must have shape (node_count, 2)")
        q.setflags(write=False)
        object.__setattr__(self, "q_model", q)
        object.__setattr__(self, "grids", MappingProxyType(dict(self.grids)))
        object.__setattr__(
            self,
            "labels_by_spec",
            MappingProxyType(
                {key: tuple(value) for key, value in self.labels_by_spec.items()}
            ),
        )


def build_union_node_manifest(
    plan: OuterQGridPlan,
    *,
    separation_m: float,
    lattice_a_x_m: float,
    lattice_a_y_m: float,
) -> OuterQNodeManifest:
    """Build every fixed grid and deduplicate exactly repeated model-q nodes."""

    label_by_key: dict[tuple[str, str], str] = {}
    q_by_label: dict[str, tuple[float, float]] = {}
    grids: dict[str, OuterQPolarGrid] = {}
    labels_by_spec: dict[str, tuple[str, ...]] = {}

    for spec in plan.specs:
        grid = build_compound_outer_q_polar_grid(
            separation_m=float(separation_m),
            lattice_a_x_m=float(lattice_a_x_m),
            lattice_a_y_m=float(lattice_a_y_m),
            radial_panel_edges=spec.radial_panel_edges,
            radial_panel_order=int(spec.radial_order),
            angular_order=int(spec.angular_order),
            angular_offset_fraction=float(spec.angular_offset_fraction),
        )
        grids[spec.spec_id] = grid
        spec_labels: list[str] = []
        for qx, qy in grid.q_model:
            key = (float(qx).hex(), float(qy).hex())
            label = label_by_key.get(key)
            if label is None:
                label = f"outer_q_{len(label_by_key):06d}"
                label_by_key[key] = label
                q_by_label[label] = (float(qx), float(qy))
            spec_labels.append(label)
        labels_by_spec[spec.spec_id] = tuple(spec_labels)

    labels = tuple(q_by_label)
    q_model = np.asarray([q_by_label[label] for label in labels], dtype=float)
    return OuterQNodeManifest(
        labels=labels,
        q_model=q_model,
        grids=grids,
        labels_by_spec=labels_by_spec,
    )


def _primary_audit_state(point: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    sweet = point.get("sweet_spot", {})
    if sweet.get("status") != "established":
        raise ValueError("point sweet spot is not established")
    audit_n = int(sweet["audit_N"])
    row = next(
        (item for item in point.get("history", []) if int(item.get("N", -1)) == audit_n),
        None,
    )
    if row is None:
        raise ValueError("point history does not contain its audit_N")
    shifts = row.get("shifts", {})
    if not shifts:
        raise ValueError("audit_N history row has no shift states")
    primary_label = next(iter(shifts))
    primary = shifts[primary_label]
    value = float(primary["two_plate_logdet"])
    if not np.isfinite(value) or not bool(primary.get("hard_physical_passed")):
        raise ValueError("primary audit shift is not a finite hard-physical point")
    return value, {
        "working_N": int(sweet["working_N"]),
        "audit_N": audit_n,
        "establishment_mode": sweet.get("establishment_mode"),
        "primary_shift_label": primary_label,
        "cross_shift": row.get("two_plate_logdet_cross_shift"),
    }


def aggregate_certified_outer_q(
    *,
    sweet_spot_payload: Mapping[str, Any],
    plan: OuterQGridPlan,
    manifest: OuterQNodeManifest,
    pairings: Sequence[str],
    matsubara_indices: Sequence[int],
    temperature_K: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reduce certified primary-shift logdet values for every fixed grid."""

    points = {
        (str(row["pairing"]), str(row["q_label"]), int(row["n"])): row
        for row in sweet_spot_payload.get("point_results", [])
    }
    unresolved: list[dict[str, Any]] = []
    values: dict[tuple[str, str, int], float] = {}
    certifications: dict[tuple[str, str, int], dict[str, Any]] = {}

    for pairing in pairings:
        for label in manifest.labels:
            for n in matsubara_indices:
                key = (str(pairing), str(label), int(n))
                point = points.get(key)
                if point is None:
                    unresolved.append(
                        {
                            "pairing": pairing,
                            "q_label": label,
                            "n": int(n),
                            "reason": "missing",
                        }
                    )
                    continue
                try:
                    value, certification = _primary_audit_state(point)
                except (KeyError, TypeError, ValueError) as exc:
                    unresolved.append(
                        {
                            "pairing": pairing,
                            "q_label": label,
                            "n": int(n),
                            "reason": str(exc),
                        }
                    )
                    continue
                values[key] = value
                certifications[key] = certification

    results: dict[str, Any] = {}
    for spec in plan.specs:
        grid = manifest.grids[spec.spec_id]
        labels = manifest.labels_by_spec[spec.spec_id]
        pairing_results: dict[str, Any] = {}
        for pairing in pairings:
            missing = [
                (label, int(n))
                for n in matsubara_indices
                for label in labels
                if (str(pairing), str(label), int(n)) not in values
            ]
            if missing:
                pairing_results[str(pairing)] = {
                    "status": "unresolved",
                    "missing_count": len(missing),
                }
                continue
            matrix = np.asarray(
                [
                    [values[(str(pairing), label, int(n))] for label in labels]
                    for n in matsubara_indices
                ],
                dtype=float,
            )
            free = free_energy_per_area_from_logdet(
                matrix,
                matsubara_indices=matsubara_indices,
                temperature_K=float(temperature_K),
                grid=grid,
            )
            used_certifications = [
                certifications[(str(pairing), label, int(n))]
                for n in matsubara_indices
                for label in labels
            ]
            pairing_results[str(pairing)] = {
                "status": "integrated",
                "partial_free_energy_J_m2": float(free.total_J_m2),
                "contributions_J_m2": free.contributions_J_m2.tolist(),
                "outer_q_integrals_m_inv2": free.outer_q_integrals_m_inv2.tolist(),
                "matsubara_indices": [int(value) for value in matsubara_indices],
                "minimum_working_N": min(
                    item["working_N"] for item in used_certifications
                ),
                "maximum_working_N": max(
                    item["working_N"] for item in used_certifications
                ),
                "minimum_audit_N": min(
                    item["audit_N"] for item in used_certifications
                ),
                "maximum_audit_N": max(
                    item["audit_N"] for item in used_certifications
                ),
                "establishment_modes": sorted(
                    {str(item["establishment_mode"]) for item in used_certifications}
                ),
                "primary_shift_is_canonical_estimator": True,
                "other_shifts_are_convergence_audits": True,
            }
        results[spec.spec_id] = {
            "spec": {
                "u_max": spec.u_max,
                "radial_order": spec.radial_order,
                "angular_order": spec.angular_order,
                "angular_offset_fraction": spec.angular_offset_fraction,
                "node_count": grid.node_count,
                "q_max_m_inv": grid.q_max_m_inv,
                "radial_rule": "nested_composite_gauss_legendre",
                "radial_panel_edges": list(spec.radial_panel_edges),
                "radial_panel_count": spec.radial_panel_count,
                "radial_panel_order": spec.radial_panel_order,
                "total_radial_nodes": int(grid.radial_order),
                "nested_cutoff_node_reuse": True,
            },
            "pairings": pairing_results,
        }
    return results, unresolved


def absolute_then_relative(
    left: float,
    right: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Apply the established absolute-then-relative outer comparison rule."""

    a = float(left)
    b = float(right)
    if not np.isfinite(a) or not np.isfinite(b):
        return {
            "passed": False,
            "passed_by": "failed",
            "absolute": float("nan"),
            "relative": float("nan"),
        }
    absolute = abs(b - a)
    scale = max(abs(a), abs(b))
    relative = absolute / max(scale, np.finfo(float).tiny)
    absolute_passed = bool(absolute <= float(absolute_tolerance))
    relative_passed = bool(relative <= float(relative_tolerance))
    return {
        "left": a,
        "right": b,
        "absolute": absolute,
        "relative": relative,
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "absolute_passed": absolute_passed,
        "relative_passed": relative_passed,
        "passed_by": (
            "absolute"
            if absolute_passed
            else "relative"
            if relative_passed
            else "failed"
        ),
        "passed": bool(absolute_passed or relative_passed),
    }


def compare_ladders(
    *,
    plan: OuterQGridPlan,
    config_results: Mapping[str, Any],
    pairings: Sequence[str],
    absolute_tolerance_J_m2: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Compare fixed-grid partial free energies with the universal outer tolerance."""

    output: dict[str, Any] = {}
    by_id = {spec.spec_id: spec for spec in plan.specs}
    for ladder_name, spec_ids in plan.ladders.items():
        pairing_records: dict[str, Any] = {}
        for pairing in pairings:
            comparisons: list[dict[str, Any]] = []
            if ladder_name == "offset":
                pairs = [
                    (plan.reference_spec_id, spec_id)
                    for spec_id in spec_ids
                    if spec_id != plan.reference_spec_id
                ]
            else:
                pairs = list(zip(spec_ids[:-1], spec_ids[1:], strict=True))
            for left_id, right_id in pairs:
                left = config_results[left_id]["pairings"].get(str(pairing), {})
                right = config_results[right_id]["pairings"].get(str(pairing), {})
                if left.get("status") != "integrated" or right.get("status") != "integrated":
                    comparisons.append(
                        {
                            "left_spec_id": left_id,
                            "right_spec_id": right_id,
                            "passed": False,
                            "passed_by": "unresolved",
                        }
                    )
                    continue
                record = absolute_then_relative(
                    float(left["partial_free_energy_J_m2"]),
                    float(right["partial_free_energy_J_m2"]),
                    absolute_tolerance=float(absolute_tolerance_J_m2),
                    relative_tolerance=float(relative_tolerance),
                )
                record.update({"left_spec_id": left_id, "right_spec_id": right_id})
                if ladder_name == "cutoff":
                    left_spec = by_id[left_id]
                    right_spec = by_id[right_id]
                    nested = (
                        right_spec.radial_panel_edges[:-1]
                        == left_spec.radial_panel_edges
                    )
                    record["nested_panel_reuse"] = bool(nested)
                    record["added_radial_panel"] = [
                        float(left_spec.u_max),
                        float(right_spec.u_max),
                    ]
                    record["signed_tail_increment_J_m2"] = float(
                        right["partial_free_energy_J_m2"]
                        - left["partial_free_energy_J_m2"]
                    )
                comparisons.append(record)
            pairing_records[str(pairing)] = {
                "comparisons": comparisons,
                "all_passed": bool(
                    comparisons and all(item["passed"] for item in comparisons)
                ),
                "final_transition_passed": bool(
                    comparisons and comparisons[-1]["passed"]
                ),
            }
            if ladder_name == "cutoff":
                pairing_records[str(pairing)]["cutoff_comparison_semantics"] = (
                    "each transition adds exactly one new radial panel while retaining "
                    "all earlier nodes and weights"
                )
        output[ladder_name] = pairing_records
    return output


__all__ = [
    "OuterQGridPlan",
    "OuterQGridSpec",
    "OuterQNodeManifest",
    "absolute_then_relative",
    "aggregate_certified_outer_q",
    "build_staged_grid_plan",
    "build_union_node_manifest",
    "compare_ladders",
]
