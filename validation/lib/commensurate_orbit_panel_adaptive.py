"""Deterministic nested panel-adaptive transverse integration.

Every sampled transverse coordinate evaluates one complete exact commensurate q
orbit through :class:`CompleteOrbitAggregateWorkspace`.  This module controls the
panel partition explicitly, uses nested Clenshaw-Curtis rules, computes errors per
physical primitive group, and checks the unique-node budget before every complete
p- or h-refinement operation.

No metric, Schur, sheet, reflection, or logdet operation is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import time
from typing import Callable, Sequence

import numpy as np

from validation.lib.commensurate_orbit_adaptive_gk21 import (
    CompleteOrbitAggregateWorkspace,
)
from validation.lib.commensurate_orbit_groups import group_layout, vector_norm

OrbitAggregateEvaluator = Callable[[np.ndarray, np.ndarray], np.ndarray]

_CC_LOW_ORDER = 9
_CC_BASE_ORDER = 17
_CC_HIGH_ORDER = 33
_CC_ORDERS = (_CC_LOW_ORDER, _CC_BASE_ORDER, _CC_HIGH_ORDER)
_INITIAL_PANEL_COUNT = 4
_MAX_PANEL_COUNT = 128
_MAX_PANEL_DEPTH = 20
_ERROR_SAFETY = 2.0
_SPECTRAL_HISTORY_FLOOR = 0.25


def _readonly(value: np.ndarray, *, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def clenshaw_curtis_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return nested Clenshaw-Curtis nodes and weights on ``[-1, 1]``.

    Only the production-candidate orders 9, 17, and 33 are accepted.  The
    implementation is the even-degree form of Trefethen's ``clencurt`` rule.
    """

    order_value = int(order)
    if order_value not in _CC_ORDERS:
        raise ValueError(f"Clenshaw-Curtis order must be one of {_CC_ORDERS}")
    degree = order_value - 1
    theta = np.pi * np.arange(order_value, dtype=float) / float(degree)
    nodes = np.cos(theta)
    weights = np.zeros(order_value, dtype=float)
    interior = np.arange(1, degree, dtype=int)
    values = np.ones(degree - 1, dtype=float)

    weights[0] = weights[-1] = 1.0 / float(degree * degree - 1)
    for harmonic in range(1, degree // 2):
        values -= (
            2.0
            * np.cos(2.0 * harmonic * theta[interior])
            / float(4 * harmonic * harmonic - 1)
        )
    values -= np.cos(degree * theta[interior]) / float(degree * degree - 1)
    weights[interior] = 2.0 * values / float(degree)
    return _readonly(nodes, dtype=float), _readonly(weights, dtype=float)


def _canonical_t_key(t_value: float) -> float:
    phase = ((float(t_value) + np.pi) / (2.0 * np.pi)) % 1.0
    return round(float(phase), 14)


def _mapped_panel_nodes(left: float, right: float, order: int) -> np.ndarray:
    nodes, _ = clenshaw_curtis_rule(order)
    midpoint = 0.5 * (float(left) + float(right))
    half_width = 0.5 * (float(right) - float(left))
    return midpoint + half_width * nodes


def _missing_node_count(
    workspace: CompleteOrbitAggregateWorkspace,
    t_values: np.ndarray,
) -> int:
    requested = {_canonical_t_key(value) for value in np.asarray(t_values).reshape(-1)}
    return len(requested.difference(workspace.cache))


def _group_norms(
    vector: np.ndarray,
    group_ids: np.ndarray,
    norm: str,
) -> np.ndarray:
    n_groups = int(np.max(group_ids)) + 1
    result = np.zeros(n_groups, dtype=float)
    for group in range(n_groups):
        result[group] = vector_norm(np.asarray(vector)[group_ids == group], norm)
    return result


def _group_point_scales(
    values: np.ndarray,
    group_ids: np.ndarray,
    norm: str,
) -> np.ndarray:
    n_groups = int(np.max(group_ids)) + 1
    result = np.zeros(n_groups, dtype=float)
    for group in range(n_groups):
        mask = group_ids == group
        result[group] = max(vector_norm(row[mask], norm) for row in values)
    return result


@dataclass(frozen=True)
class PanelState:
    left: float
    right: float
    depth: int
    order: int
    value: np.ndarray
    group_errors: np.ndarray
    group_point_scales: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.left) or not np.isfinite(self.right):
            raise ValueError("panel bounds must be finite")
        if not self.left < self.right:
            raise ValueError("panel left bound must be smaller than right bound")
        if self.order not in (_CC_BASE_ORDER, _CC_HIGH_ORDER):
            raise ValueError("panel order must be 17 or 33")
        object.__setattr__(self, "value", _readonly(self.value, dtype=complex))
        object.__setattr__(
            self,
            "group_errors",
            _readonly(self.group_errors, dtype=float),
        )
        object.__setattr__(
            self,
            "group_point_scales",
            _readonly(self.group_point_scales, dtype=float),
        )


@dataclass(frozen=True)
class PanelAdaptiveSnapshot:
    value: np.ndarray | None
    group_errors: np.ndarray
    group_tolerances: np.ndarray
    group_ratios: np.ndarray
    group_scales: np.ndarray
    success: bool
    tolerance_factor: float
    unique_evaluations: int
    cache_hits: int
    panel_count: int
    maximum_depth: int
    refinement_steps: int
    worst_group_index: int
    worst_group_name: str
    worst_panel: tuple[float, float]
    worst_panel_order: int
    worst_local_ratio: float
    message: str

    def __post_init__(self) -> None:
        if self.value is not None:
            object.__setattr__(self, "value", _readonly(self.value, dtype=complex))
        for name in (
            "group_errors",
            "group_tolerances",
            "group_ratios",
            "group_scales",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=float))

    @property
    def integral_error_ratio(self) -> float:
        finite = np.asarray(self.group_ratios, dtype=float)
        return float(np.max(finite)) if finite.size else float("inf")


@dataclass(frozen=True)
class PanelAdaptiveResult:
    primary: PanelAdaptiveSnapshot
    audit: PanelAdaptiveSnapshot | None
    q_model: np.ndarray
    primitive_direction: np.ndarray
    transverse_direction: np.ndarray
    orbit_shift_steps: int
    orbit_origins: tuple[float, ...]
    nk: int
    epsabs: float
    epsrel: float
    audit_tolerance_factor: float
    max_unique_transverse_evaluations: int
    transverse_evaluations: int
    cache_hits: int
    point_evaluations: int
    chunk_size: int
    wall_seconds: float
    geometry_wall_seconds: float
    evaluator_wall_seconds: float
    group_names: tuple[str, ...]
    control_group_names: tuple[str, ...]
    monitor_group_names: tuple[str, ...]
    primary_audit_group_ratios: tuple[float, ...]
    primitive_group_agreement_passed: bool
    success: bool
    status: int
    message: str
    failure_reason: str
    norm: str
    initial_panel_count: int = _INITIAL_PANEL_COUNT
    maximum_panel_count: int = _MAX_PANEL_COUNT
    maximum_panel_depth: int = _MAX_PANEL_DEPTH
    strategy: str = "deterministic_panel_adaptive"
    quadrature: str = "nested_clenshaw_curtis_9_17_33"
    pilot_order: int = _CC_BASE_ORDER
    summation_method: str = "complete_q_orbit_groupwise_panel_adaptive_shared_state_audit"

    def __post_init__(self) -> None:
        for name in ("q_model", "primitive_direction", "transverse_direction"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def value(self) -> np.ndarray | None:
        if self.audit is not None and self.audit.value is not None:
            return self.audit.value
        return self.primary.value

    @property
    def scaled_error_estimate(self) -> float:
        return self.primary.integral_error_ratio

    @property
    def limit(self) -> int:
        return self.maximum_panel_count

    @property
    def transverse_evaluations_unique(self) -> int:
        return self.transverse_evaluations

    @property
    def required_consecutive_levels(self) -> int:
        return 1

    @property
    def consecutive_converged_levels(self) -> int:
        return int(self.success)

    @property
    def final_transverse_order(self) -> int:
        return self.transverse_evaluations

    @property
    def max_transverse_order(self) -> int:
        return self.max_unique_transverse_evaluations

    @property
    def order_history(self) -> tuple[int, ...]:
        values = [self.primary.unique_evaluations]
        if self.audit is not None:
            values.append(self.audit.unique_evaluations)
        return tuple(values)

    @property
    def convergence_ratio_history(self) -> tuple[float, ...]:
        values = [self.primary.integral_error_ratio]
        if self.audit is not None:
            values.append(self.audit.integral_error_ratio)
        return tuple(values)

    @property
    def control_group_ratios(self) -> tuple[float, ...]:
        return self.primary_audit_group_ratios

    @property
    def monitor_group_ratios(self) -> tuple[float, ...]:
        return ()


@dataclass
class _ControllerState:
    workspace: CompleteOrbitAggregateWorkspace
    panels: list[PanelState]
    group_ids: np.ndarray
    group_names: tuple[str, ...]
    group_control_weights: np.ndarray
    norm: str
    epsabs: float
    epsrel: float
    scale_floor_relative: float
    scale_floor_absolute: float
    refinement_steps: int = 0


def _panel_state(
    workspace: CompleteOrbitAggregateWorkspace,
    *,
    left: float,
    right: float,
    depth: int,
    order: int,
    group_ids: np.ndarray,
    norm: str,
) -> PanelState:
    nodes, weights = clenshaw_curtis_rule(order)
    t_values = _mapped_panel_nodes(left, right, order)
    values = np.stack([workspace.evaluate_t(value) for value in t_values], axis=0)
    half_width = 0.5 * (float(right) - float(left))
    high = half_width * np.tensordot(weights, values, axes=(0, 0)) / (2.0 * np.pi)

    lower_order = _CC_LOW_ORDER if order == _CC_BASE_ORDER else _CC_BASE_ORDER
    _, lower_weights = clenshaw_curtis_rule(lower_order)
    lower_values = values[::2]
    low = (
        half_width
        * np.tensordot(lower_weights, lower_values, axes=(0, 0))
        / (2.0 * np.pi)
    )
    current_errors = _group_norms(high - low, group_ids, norm)

    if order == _CC_HIGH_ORDER:
        _, lowest_weights = clenshaw_curtis_rule(_CC_LOW_ORDER)
        lowest_values = values[::4]
        lowest = (
            half_width
            * np.tensordot(lowest_weights, lowest_values, axes=(0, 0))
            / (2.0 * np.pi)
        )
        history_errors = _group_norms(low - lowest, group_ids, norm)
        current_errors = np.maximum(
            current_errors,
            _SPECTRAL_HISTORY_FLOOR * history_errors,
        )

    return PanelState(
        left=float(left),
        right=float(right),
        depth=int(depth),
        order=int(order),
        value=high,
        group_errors=_ERROR_SAFETY * current_errors,
        group_point_scales=_group_point_scales(values, group_ids, norm),
    )


def _global_metrics(
    state: _ControllerState,
    tolerance_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = np.sum(np.stack([panel.value for panel in state.panels]), axis=0)
    errors = np.sum(np.stack([panel.group_errors for panel in state.panels]), axis=0)
    raw_scales = np.max(
        np.stack([panel.group_point_scales for panel in state.panels]),
        axis=0,
    )
    global_scale = max(float(np.max(raw_scales)), 1.0)
    floor = max(
        float(state.scale_floor_absolute),
        float(state.scale_floor_relative) * global_scale,
    )
    scales = np.maximum(raw_scales, floor)
    integral_norms = _group_norms(value, state.group_ids, state.norm)
    factor = float(tolerance_factor)
    tolerances = factor * (
        float(state.epsabs) + float(state.epsrel) * integral_norms / scales
    )
    ratios = np.zeros_like(errors)
    control = state.group_control_weights > 0.0
    ratios[control] = (
        state.group_control_weights[control]
        * errors[control]
        / scales[control]
        / np.maximum(tolerances[control], np.finfo(float).tiny)
    )
    return value, errors, tolerances, ratios, scales


def _snapshot(
    state: _ControllerState,
    *,
    tolerance_factor: float,
    success: bool,
    message: str,
) -> PanelAdaptiveSnapshot:
    value, errors, tolerances, ratios, scales = _global_metrics(
        state,
        tolerance_factor,
    )
    control = state.group_control_weights > 0.0
    control_indices = np.flatnonzero(control)
    if control_indices.size:
        worst_group = int(control_indices[np.argmax(ratios[control_indices])])
        local = np.asarray(
            [panel.group_errors[worst_group] for panel in state.panels],
            dtype=float,
        )
        worst_panel_index = int(np.argmax(local))
        worst_panel = state.panels[worst_panel_index]
        local_ratio = (
            state.group_control_weights[worst_group]
            * worst_panel.group_errors[worst_group]
            / scales[worst_group]
            / max(float(tolerances[worst_group]), np.finfo(float).tiny)
        )
    else:
        worst_group = -1
        worst_panel = state.panels[0]
        local_ratio = 0.0
    return PanelAdaptiveSnapshot(
        value=value,
        group_errors=errors,
        group_tolerances=tolerances,
        group_ratios=ratios,
        group_scales=scales,
        success=bool(success),
        tolerance_factor=float(tolerance_factor),
        unique_evaluations=state.workspace.transverse_evaluations_unique,
        cache_hits=state.workspace.cache_hits,
        panel_count=len(state.panels),
        maximum_depth=max(panel.depth for panel in state.panels),
        refinement_steps=state.refinement_steps,
        worst_group_index=worst_group,
        worst_group_name=(
            state.group_names[worst_group] if worst_group >= 0 else "none"
        ),
        worst_panel=(float(worst_panel.left), float(worst_panel.right)),
        worst_panel_order=int(worst_panel.order),
        worst_local_ratio=float(local_ratio),
        message=str(message),
    )


def _refinement_target(
    state: _ControllerState,
    tolerance_factor: float,
) -> int:
    _, _, tolerances, _, scales = _global_metrics(state, tolerance_factor)
    control = state.group_control_weights > 0.0
    scores = np.full(len(state.panels), -np.inf, dtype=float)
    for index, panel in enumerate(state.panels):
        local = np.zeros_like(panel.group_errors)
        local[control] = (
            state.group_control_weights[control]
            * panel.group_errors[control]
            / scales[control]
            / np.maximum(tolerances[control], np.finfo(float).tiny)
        )
        scores[index] = float(np.max(local[control]))
    return int(np.argmax(scores))


def _operation_nodes(panel: PanelState) -> np.ndarray:
    if panel.order == _CC_BASE_ORDER:
        return _mapped_panel_nodes(panel.left, panel.right, _CC_HIGH_ORDER)
    midpoint = 0.5 * (panel.left + panel.right)
    return np.concatenate(
        (
            _mapped_panel_nodes(panel.left, midpoint, _CC_BASE_ORDER),
            _mapped_panel_nodes(midpoint, panel.right, _CC_BASE_ORDER),
        )
    )


def _refine_once(
    state: _ControllerState,
    *,
    tolerance_factor: float,
) -> tuple[bool, str]:
    panel_index = _refinement_target(state, tolerance_factor)
    panel = state.panels[panel_index]
    if panel.order == _CC_HIGH_ORDER:
        if panel.depth >= _MAX_PANEL_DEPTH:
            return False, "maximum_panel_depth_reached"
        if len(state.panels) + 1 > _MAX_PANEL_COUNT:
            return False, "maximum_panel_count_reached"

    required_nodes = _operation_nodes(panel)
    missing = _missing_node_count(state.workspace, required_nodes)
    current = state.workspace.transverse_evaluations_unique
    maximum = state.workspace.max_unique_transverse_evaluations
    if current + missing > maximum:
        return False, (
            "panel_boundary_transverse_budget_exceeded: "
            f"current={current}, required_new={missing}, maximum={maximum}"
        )

    if panel.order == _CC_BASE_ORDER:
        replacement = _panel_state(
            state.workspace,
            left=panel.left,
            right=panel.right,
            depth=panel.depth,
            order=_CC_HIGH_ORDER,
            group_ids=state.group_ids,
            norm=state.norm,
        )
        state.panels[panel_index] = replacement
    else:
        midpoint = 0.5 * (panel.left + panel.right)
        left_child = _panel_state(
            state.workspace,
            left=panel.left,
            right=midpoint,
            depth=panel.depth + 1,
            order=_CC_BASE_ORDER,
            group_ids=state.group_ids,
            norm=state.norm,
        )
        right_child = _panel_state(
            state.workspace,
            left=midpoint,
            right=panel.right,
            depth=panel.depth + 1,
            order=_CC_BASE_ORDER,
            group_ids=state.group_ids,
            norm=state.norm,
        )
        state.panels[panel_index : panel_index + 1] = [left_child, right_child]
    state.refinement_steps += 1
    return True, ""


def _advance_to_tolerance(
    state: _ControllerState,
    *,
    tolerance_factor: float,
) -> tuple[PanelAdaptiveSnapshot, str]:
    while True:
        current = _snapshot(
            state,
            tolerance_factor=tolerance_factor,
            success=False,
            message="",
        )
        control = state.group_control_weights > 0.0
        if np.all(current.group_ratios[control] <= 1.0):
            return (
                _snapshot(
                    state,
                    tolerance_factor=tolerance_factor,
                    success=True,
                    message="groupwise panel error tolerance satisfied",
                ),
                "",
            )
        advanced, reason = _refine_once(
            state,
            tolerance_factor=tolerance_factor,
        )
        if not advanced:
            return (
                _snapshot(
                    state,
                    tolerance_factor=tolerance_factor,
                    success=False,
                    message=reason,
                ),
                reason,
            )


def _audit_group_ratios(
    primary: PanelAdaptiveSnapshot,
    audit: PanelAdaptiveSnapshot,
    *,
    group_ids: np.ndarray,
    control_weights: np.ndarray,
    epsabs: float,
    epsrel: float,
    norm: str,
) -> np.ndarray:
    if primary.value is None or audit.value is None:
        return np.full(control_weights.size, np.inf)
    scales = np.maximum(primary.group_scales, audit.group_scales)
    ratios = np.zeros(control_weights.size, dtype=float)
    for group in range(control_weights.size):
        if control_weights[group] <= 0.0:
            continue
        mask = group_ids == group
        left = vector_norm(primary.value[mask], norm) / scales[group]
        right = vector_norm(audit.value[mask], norm) / scales[group]
        delta = vector_norm(audit.value[mask] - primary.value[mask], norm) / scales[group]
        tolerance = float(epsabs) + float(epsrel) * max(left, right)
        ratios[group] = (
            control_weights[group]
            * delta
            / max(tolerance, np.finfo(float).tiny)
        )
    return ratios


def integrate_commensurate_orbit_panel_adaptive(
    evaluator: OrbitAggregateEvaluator,
    *,
    nk: int,
    mx: int,
    my: int,
    shift_s: float = 0.5,
    subgrid_average: str = "auto",
    max_unique_transverse_evaluations: int = 256,
    epsabs: float = 2e-5,
    epsrel: float = 2e-3,
    audit_tolerance_factor: float = 0.25,
    norm: str = "max",
    scale_floor_relative: float = 1e-8,
    scale_floor_absolute: float = 1e-14,
    component_group_ids: Sequence[int] | np.ndarray | None = None,
    group_names: Sequence[str] | None = None,
    group_control_weights: Sequence[float] | np.ndarray | None = None,
) -> PanelAdaptiveResult:
    """Integrate one complete-orbit primitive vector with deterministic panels."""

    for value, name in (
        (epsabs, "epsabs"),
        (epsrel, "epsrel"),
        (scale_floor_relative, "scale_floor_relative"),
        (scale_floor_absolute, "scale_floor_absolute"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0.0 < audit_tolerance_factor < 1.0:
        raise ValueError("audit_tolerance_factor must lie strictly between zero and one")
    if norm not in {"max", "2"}:
        raise ValueError("norm must be 'max' or '2'")

    started = time.perf_counter()
    workspace = CompleteOrbitAggregateWorkspace(
        evaluator=evaluator,
        nk=nk,
        mx=mx,
        my=my,
        shift_s=shift_s,
        subgrid_average=subgrid_average,
        max_unique_transverse_evaluations=max_unique_transverse_evaluations,
    )
    boundaries = np.linspace(-np.pi, np.pi, _INITIAL_PANEL_COUNT + 1)
    initial_nodes = np.concatenate(
        [
            _mapped_panel_nodes(boundaries[index], boundaries[index + 1], _CC_BASE_ORDER)
            for index in range(_INITIAL_PANEL_COUNT)
        ]
    )
    required = _missing_node_count(workspace, initial_nodes)
    if required > max_unique_transverse_evaluations:
        empty = np.empty(0, dtype=float)
        primary = PanelAdaptiveSnapshot(
            value=None,
            group_errors=empty,
            group_tolerances=empty,
            group_ratios=empty,
            group_scales=empty,
            success=False,
            tolerance_factor=1.0,
            unique_evaluations=0,
            cache_hits=0,
            panel_count=0,
            maximum_depth=0,
            refinement_steps=0,
            worst_group_index=-1,
            worst_group_name="none",
            worst_panel=(float("nan"), float("nan")),
            worst_panel_order=0,
            worst_local_ratio=float("inf"),
            message="initial_panel_budget_exceeded",
        )
        return PanelAdaptiveResult(
            primary=primary,
            audit=None,
            q_model=workspace.q_model,
            primitive_direction=workspace.primitive_direction,
            transverse_direction=workspace.transverse_direction,
            orbit_shift_steps=workspace.orbit_shift_steps,
            orbit_origins=workspace.orbit_origins,
            nk=workspace.nk,
            epsabs=epsabs,
            epsrel=epsrel,
            audit_tolerance_factor=audit_tolerance_factor,
            max_unique_transverse_evaluations=max_unique_transverse_evaluations,
            transverse_evaluations=0,
            cache_hits=0,
            point_evaluations=0,
            chunk_size=workspace.points_per_t,
            wall_seconds=time.perf_counter() - started,
            geometry_wall_seconds=0.0,
            evaluator_wall_seconds=0.0,
            group_names=(),
            control_group_names=(),
            monitor_group_names=(),
            primary_audit_group_ratios=(),
            primitive_group_agreement_passed=False,
            success=False,
            status=2,
            message="initial panel set exceeds transverse budget",
            failure_reason="initial_panel_budget_exceeded",
            norm=norm,
        )

    first_value = workspace.evaluate_t(initial_nodes[0])
    ids, names, control_weights = group_layout(
        int(first_value.size),
        component_group_ids=component_group_ids,
        group_names=group_names,
        group_control_weights=group_control_weights,
    )
    panels = [
        _panel_state(
            workspace,
            left=float(boundaries[index]),
            right=float(boundaries[index + 1]),
            depth=0,
            order=_CC_BASE_ORDER,
            group_ids=ids,
            norm=norm,
        )
        for index in range(_INITIAL_PANEL_COUNT)
    ]
    state = _ControllerState(
        workspace=workspace,
        panels=panels,
        group_ids=ids,
        group_names=names,
        group_control_weights=control_weights,
        norm=norm,
        epsabs=epsabs,
        epsrel=epsrel,
        scale_floor_relative=scale_floor_relative,
        scale_floor_absolute=scale_floor_absolute,
    )

    primary, primary_reason = _advance_to_tolerance(state, tolerance_factor=1.0)
    audit = None
    audit_reason = ""
    if primary.success:
        audit, audit_reason = _advance_to_tolerance(
            state,
            tolerance_factor=float(audit_tolerance_factor),
        )

    agreement_ratios = np.full(control_weights.size, np.inf)
    agreement = False
    if primary.success and audit is not None and audit.success:
        agreement_ratios = _audit_group_ratios(
            primary,
            audit,
            group_ids=ids,
            control_weights=control_weights,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=norm,
        )
        agreement = bool(np.all(agreement_ratios[control_weights > 0.0] <= 1.0))

    success = bool(primary.success and audit is not None and audit.success and agreement)
    if success:
        status = 0
        message = "primary and tightened panel-adaptive audit passed"
        failure_reason = ""
    elif not primary.success:
        status = 2 if "budget" in primary_reason else 1
        message = primary.message
        failure_reason = primary_reason or "primary_panel_error_not_converged"
    elif audit is None or not audit.success:
        status = 2 if "budget" in audit_reason else 1
        message = audit.message if audit is not None else "tightened audit unavailable"
        failure_reason = audit_reason or "tightened_panel_audit_not_converged"
    else:
        status = 1
        message = "primary/audit physical-group estimates disagree"
        failure_reason = "primitive_group_audit_disagreement"

    control_mask = control_weights > 0.0
    return PanelAdaptiveResult(
        primary=primary,
        audit=audit,
        q_model=workspace.q_model,
        primitive_direction=workspace.primitive_direction,
        transverse_direction=workspace.transverse_direction,
        orbit_shift_steps=workspace.orbit_shift_steps,
        orbit_origins=workspace.orbit_origins,
        nk=workspace.nk,
        epsabs=epsabs,
        epsrel=epsrel,
        audit_tolerance_factor=audit_tolerance_factor,
        max_unique_transverse_evaluations=max_unique_transverse_evaluations,
        transverse_evaluations=workspace.transverse_evaluations_unique,
        cache_hits=workspace.cache_hits,
        point_evaluations=workspace.point_evaluations,
        chunk_size=workspace.points_per_t,
        wall_seconds=time.perf_counter() - started,
        geometry_wall_seconds=workspace.geometry_wall_seconds,
        evaluator_wall_seconds=workspace.evaluator_wall_seconds,
        group_names=tuple(names),
        control_group_names=tuple(
            name for name, keep in zip(names, control_mask, strict=True) if keep
        ),
        monitor_group_names=tuple(
            name for name, keep in zip(names, ~control_mask, strict=True) if keep
        ),
        primary_audit_group_ratios=tuple(
            float(value) for value in agreement_ratios[control_mask]
        ),
        primitive_group_agreement_passed=agreement,
        success=success,
        status=status,
        message=message,
        failure_reason=failure_reason,
        norm=norm,
    )


__all__ = [
    "PanelAdaptiveResult",
    "PanelAdaptiveSnapshot",
    "PanelState",
    "clenshaw_curtis_rule",
    "integrate_commensurate_orbit_panel_adaptive",
]
