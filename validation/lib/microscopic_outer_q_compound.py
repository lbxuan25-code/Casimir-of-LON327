"""Nested composite-panel planning for microscopic outer-q qualification."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.casimir.compound_outer_quadrature import (
    build_compound_outer_q_polar_grid,
)
from lno327.casimir.outer_quadrature import OuterQPolarGrid


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
    """Return nested cutoff and one-at-a-time order/offset ladders.

    ``u_max_values`` are cumulative panel boundaries.  For example ``6 10 14``
    generates cutoff grids on ``[0,6]``, ``[0,6]+[6,10]`` and
    ``[0,6]+[6,10]+[10,14]``.  ``radial_orders`` are per-panel Gauss orders.
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
            register(u_ref, radial_ref, angular_ref, value)
            for value in offsets
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
    """Build every staged grid and deduplicate exactly repeated model-q nodes."""

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


__all__ = [
    "OuterQGridPlan",
    "OuterQGridSpec",
    "OuterQNodeManifest",
    "build_staged_grid_plan",
    "build_union_node_manifest",
]
