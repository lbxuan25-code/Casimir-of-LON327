"""Physical-group scaling utilities shared by transverse quadrature backends."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def vector_norm(values: np.ndarray, norm: str) -> float:
    array = np.asarray(values, dtype=complex).reshape(-1)
    if array.size == 0:
        return 0.0
    if norm == "max":
        return float(np.max(np.abs(array)))
    if norm == "2":
        return float(np.linalg.norm(array))
    raise ValueError("norm must be 'max' or '2'")


def group_layout(
    width: int,
    *,
    component_group_ids: Sequence[int] | np.ndarray | None,
    group_names: Sequence[str] | None,
    group_control_weights: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Return contiguous component groups and their refinement weights.

    The canonical positive-d-wave payload is inferred when no explicit layout is
    supplied.  Ward RHS components share all quadrature nodes but have zero
    refinement weight; the final Ward check remains the authoritative gate.
    """

    width_value = int(width)
    if width_value <= 0:
        raise ValueError("width must be positive")
    if component_group_ids is None and width_value >= 43 and (width_value - 18) % 25 == 0:
        n_frequencies = (width_value - 18) // 25
        ids = np.empty(width_value, dtype=int)
        inferred_names: list[str] = []
        inferred_weights: list[float] = []
        group = 0
        ids[0:9] = group
        inferred_names.append("em_direct")
        inferred_weights.append(1.0)
        group += 1
        ids[9:15] = group
        inferred_names.append("collective_static")
        inferred_weights.append(1.0)
        group += 1
        ids[15:18] = group
        inferred_names.append("ward_rhs_monitor")
        inferred_weights.append(0.0)
        group += 1
        offset = 18
        for frequency in range(n_frequencies):
            ids[offset : offset + 9] = group
            inferred_names.append(f"xi_{frequency}_em")
            inferred_weights.append(1.0)
            group += 1
            offset += 9
            ids[offset : offset + 4] = group
            inferred_names.append(f"xi_{frequency}_collective")
            inferred_weights.append(1.0)
            group += 1
            offset += 4
            ids[offset : offset + 12] = group
            inferred_names.append(f"xi_{frequency}_mixed")
            inferred_weights.append(1.0)
            group += 1
            offset += 12
        if offset != width_value:
            raise RuntimeError("internal positive-d-wave block layout mismatch")
        if group_names is None:
            group_names = tuple(inferred_names)
        if group_control_weights is None:
            group_control_weights = np.asarray(inferred_weights, dtype=float)
    elif component_group_ids is None:
        ids = np.arange(width_value, dtype=int)
    else:
        ids = np.asarray(component_group_ids, dtype=int).reshape(-1)
        if ids.size != width_value:
            raise ValueError(
                f"component_group_ids has width {ids.size}; expected {width_value}"
            )
        if np.any(ids < 0):
            raise ValueError("component_group_ids must be non-negative")

    unique = np.unique(ids)
    if not np.array_equal(unique, np.arange(unique.size, dtype=int)):
        raise ValueError("component_group_ids must form contiguous ids 0..n_groups-1")
    n_groups = int(unique.size)

    if group_names is None:
        names = tuple(f"component_{index}" for index in range(n_groups))
    else:
        names = tuple(str(value) for value in group_names)
        if len(names) != n_groups:
            raise ValueError(f"group_names has length {len(names)}; expected {n_groups}")
        if len(set(names)) != len(names):
            raise ValueError("group_names must be unique")

    if group_control_weights is None:
        weights = np.ones(n_groups, dtype=float)
    else:
        weights = np.asarray(group_control_weights, dtype=float).reshape(-1)
        if weights.size != n_groups:
            raise ValueError(
                f"group_control_weights has length {weights.size}; expected {n_groups}"
            )
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("group_control_weights must be finite and non-negative")
    if not np.any(weights > 0.0):
        raise ValueError("at least one control group must have positive weight")
    return ids, names, weights


def group_point_scales(
    sampled_values: np.ndarray,
    group_ids: np.ndarray,
    *,
    norm: str,
    scale_floor_relative: float,
    scale_floor_absolute: float,
) -> np.ndarray:
    values = np.asarray(sampled_values, dtype=complex)
    if values.ndim != 2 or values.shape[1] != group_ids.size:
        raise ValueError("sampled_values and group_ids have incompatible shapes")
    n_groups = int(np.max(group_ids)) + 1
    raw = np.zeros(n_groups, dtype=float)
    for group in range(n_groups):
        mask = group_ids == group
        raw[group] = max(vector_norm(row[mask], norm) for row in values)
    global_scale = max(float(np.max(raw)), 1.0)
    floor = max(float(scale_floor_absolute), float(scale_floor_relative) * global_scale)
    return np.maximum(raw, floor)


def component_scales_from_groups(
    group_ids: np.ndarray, group_scales: np.ndarray
) -> np.ndarray:
    return np.asarray(group_scales[group_ids], dtype=float)


__all__ = [
    "component_scales_from_groups",
    "group_layout",
    "group_point_scales",
    "vector_norm",
]
