"""Deterministic nested panel-adaptive transverse integration.

Every sampled transverse coordinate evaluates one complete exact commensurate q
orbit through :class:`CompleteOrbitAggregateWorkspace`.  This module controls the
full-period panel partition explicitly, uses nested Clenshaw-Curtis rules, computes
errors per physical primitive group, and checks the unique-node budget before every
complete p- or h-refinement operation.

No metric, Schur, sheet, reflection, or logdet operation is performed here.  The
controller uses periodicity only to choose a numerically smooth cut for the full
``2*pi`` interval; it never assumes evenness, C4 symmetry, or q-direction equivalence.
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

_CC_ESTIMATE_ORDER = 5
_CC_LOW_ORDER = 9
_CC_BASE_ORDER = 17
_CC_HIGH_ORDER = 33
_CC_ORDERS = (
    _CC_ESTIMATE_ORDER,
    _CC_LOW_ORDER,
    _CC_BASE_ORDER,
    _CC_HIGH_ORDER,
)
_ACTIVE_CC_ORDERS = (_CC_LOW_ORDER, _CC_BASE_ORDER, _CC_HIGH_ORDER)
_INITIAL_PANEL_COUNT = 8
_PILOT_COUNT = 16
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

    Orders 9, 17, and 33 are active panel states.  Order 5 is accepted only as
    the nested lower-order estimator for a CC9 panel.  The implementation is the
    even-degree form of Trefethen's ``clencurt`` rule.
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
        if self.order not in _ACTIVE_CC_ORDERS:
            raise ValueError("panel order must be 9, 17, or 33")
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
class RefinementTraceEntry:
    step: int
    stage: str
    panel: tuple[float, float]
    old_order: int
    operation: str
    required_new_nodes: int
    unique_evaluations_after: int
    worst_group_before: str
    worst_group_after: str
    global_error_ratio_before: float
    global_error_ratio_after: float


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
    integration_start: float
    refinement_trace: tuple[RefinementTraceEntry, ...]
    initial_panel_count: int = _INITIAL_PANEL_COUNT
    pilot_count: int = _PILOT_COUNT
    maximum_panel_count: int = _MAX_PANEL_COUNT
    maximum_panel_depth: int = _MAX_PANEL_DEPTH
    strategy: str = "deterministic_panel_adaptive"
    quadrature: str = "nested_clenshaw_curtis_5_9_17_33"
    pilot_order: int = _CC_LOW_ORDER
    summation_method: str = (
        "complete_q_orbit_groupwise_budget_aware_panel_adaptive_shared_state_audit"
    )
    symmetry_reduction_applied: bool = False
    full_transverse_period_integrated: bool = True
    q_direction_special_case: bool = False

    def __post_init__(self) -> None:
        for name in ("q_model", "primitive_direction", "transverse_direction"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "refinement_trace", tuple(self.refinement_trace))

    @property
    def value(self) -> np.ndarray | None:
        if self.audit is not None and self.audit.success and self.audit.value is not None:
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


@dataclass(frozen=True)
class _OperationCandidate:
    panel_index: int
    operation: str
    new_order: int
    required_new_nodes: int
    local_score: float
    benefit_per_node: float


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
    integration_start: float
    refinement_steps: int = 0
    trace: list[RefinementTraceEntry] = field(default_factory=list)


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
    if order not in _ACTIVE_CC_ORDERS:
        raise ValueError("active panel order must be 9, 17, or 33")
    _, weights = clenshaw_curtis_rule(order)
    t_values = _mapped_panel_nodes(left, right, order)
    values = np.stack([workspace.evaluate_t(value) for value in t_values], axis=0)
    half_width = 0.5 * (float(right) - float(left))
    high = half_width * np.tensordot(weights, values, axes=(0, 0)) / (2.0 * np.pi)

    lower_order = {
        _CC_LOW_ORDER: _CC_ESTIMATE_ORDER,
        _CC_BASE_ORDER: _CC_LOW_ORDER,
        _CC_HIGH_ORDER: _CC_BASE_ORDER,
    }[order]
    _, lower_weights = clenshaw_curtis_rule(lower_order)
    lower_values = values[::2]
    low = (
        half_width
        * np.tensordot(lower_weights, lower_values, axes=(0, 0))
        / (2.0 * np.pi)
    )
    current_errors = _group_norms(high - low, group_ids, norm)

    if order in (_CC_BASE_ORDER, _CC_HIGH_ORDER):
        history_order = {
            _CC_BASE_ORDER: _CC_ESTIMATE_ORDER,
            _CC_HIGH_ORDER: _CC_LOW_ORDER,
        }[order]
        _, history_weights = clenshaw_curtis_rule(history_order)
        history_values = values[::4]
        history = (
            half_width
            * np.tensordot(history_weights, history_values, axes=(0, 0))
            / (2.0 * np.pi)
        )
        history_errors = _group_norms(low - history, group_ids, norm)
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


def _local_panel_score(
    state: _ControllerState,
    panel: PanelState,
    tolerance_factor: float,
) -> float:
    _, _, tolerances, _, scales = _global_metrics(state, tolerance_factor)
    control = state.group_control_weights > 0.0
    local = np.zeros_like(panel.group_errors)
    local[control] = (
        state.group_control_weights[control]
        * panel.group_errors[control]
        / scales[control]
        / np.maximum(tolerances[control], np.finfo(float).tiny)
    )
    return float(np.max(local[control]))


def _operation_nodes(panel: PanelState) -> tuple[str, int, np.ndarray]:
    if panel.order == _CC_LOW_ORDER:
        return (
            "p_refine",
            _CC_BASE_ORDER,
            _mapped_panel_nodes(panel.left, panel.right, _CC_BASE_ORDER),
        )
    if panel.order == _CC_BASE_ORDER:
        return (
            "p_refine",
            _CC_HIGH_ORDER,
            _mapped_panel_nodes(panel.left, panel.right, _CC_HIGH_ORDER),
        )
    midpoint = 0.5 * (panel.left + panel.right)
    return (
        "split",
        _CC_LOW_ORDER,
        np.concatenate(
            (
                _mapped_panel_nodes(panel.left, midpoint, _CC_LOW_ORDER),
                _mapped_panel_nodes(midpoint, panel.right, _CC_LOW_ORDER),
            )
        ),
    )


def _candidate_operations(
    state: _ControllerState,
    tolerance_factor: float,
) -> tuple[list[_OperationCandidate], str]:
    current = state.workspace.transverse_evaluations_unique
    maximum = state.workspace.max_unique_transverse_evaluations
    remaining = maximum - current
    feasible: list[_OperationCandidate] = []
    blocked_costs: list[int] = []
    structural_reasons: list[str] = []

    for index, panel in enumerate(state.panels):
        if panel.order == _CC_HIGH_ORDER:
            if panel.depth >= _MAX_PANEL_DEPTH:
                structural_reasons.append("maximum_panel_depth_reached")
                continue
            if len(state.panels) + 1 > _MAX_PANEL_COUNT:
                structural_reasons.append("maximum_panel_count_reached")
                continue
        operation, new_order, nodes = _operation_nodes(panel)
        missing = _missing_node_count(state.workspace, nodes)
        local_score = _local_panel_score(state, panel, tolerance_factor)
        candidate = _OperationCandidate(
            panel_index=index,
            operation=operation,
            new_order=new_order,
            required_new_nodes=missing,
            local_score=local_score,
            benefit_per_node=local_score / max(missing, 1),
        )
        if missing <= remaining:
            feasible.append(candidate)
        else:
            blocked_costs.append(missing)

    if feasible:
        return feasible, ""
    if blocked_costs:
        return [], (
            "panel_boundary_transverse_budget_exceeded: "
            f"current={current}, minimum_required_new={min(blocked_costs)}, "
            f"maximum={maximum}"
        )
    if structural_reasons:
        return [], sorted(structural_reasons)[0]
    return [], "no_refinement_operation_available"


def _refine_once(
    state: _ControllerState,
    *,
    tolerance_factor: float,
    stage: str,
) -> tuple[bool, str]:
    before = _snapshot(
        state,
        tolerance_factor=tolerance_factor,
        success=False,
        message="",
    )
    candidates, reason = _candidate_operations(state, tolerance_factor)
    if not candidates:
        return False, reason

    selected = max(
        candidates,
        key=lambda item: (
            item.benefit_per_node,
            item.local_score,
            -item.required_new_nodes,
            -item.panel_index,
        ),
    )
    panel = state.panels[selected.panel_index]
    panel_bounds = (float(panel.left), float(panel.right))
    old_order = int(panel.order)

    if selected.operation == "p_refine":
        replacement = _panel_state(
            state.workspace,
            left=panel.left,
            right=panel.right,
            depth=panel.depth,
            order=selected.new_order,
            group_ids=state.group_ids,
            norm=state.norm,
        )
        state.panels[selected.panel_index] = replacement
    else:
        midpoint = 0.5 * (panel.left + panel.right)
        left_child = _panel_state(
            state.workspace,
            left=panel.left,
            right=midpoint,
            depth=panel.depth + 1,
            order=_CC_LOW_ORDER,
            group_ids=state.group_ids,
            norm=state.norm,
        )
        right_child = _panel_state(
            state.workspace,
            left=midpoint,
            right=panel.right,
            depth=panel.depth + 1,
            order=_CC_LOW_ORDER,
            group_ids=state.group_ids,
            norm=state.norm,
        )
        state.panels[selected.panel_index : selected.panel_index + 1] = [
            left_child,
            right_child,
        ]

    state.refinement_steps += 1
    after = _snapshot(
        state,
        tolerance_factor=tolerance_factor,
        success=False,
        message="",
    )
    state.trace.append(
        RefinementTraceEntry(
            step=state.refinement_steps,
            stage=str(stage),
            panel=panel_bounds,
            old_order=old_order,
            operation=selected.operation,
            required_new_nodes=selected.required_new_nodes,
            unique_evaluations_after=state.workspace.transverse_evaluations_unique,
            worst_group_before=before.worst_group_name,
            worst_group_after=after.worst_group_name,
            global_error_ratio_before=before.integral_error_ratio,
            global_error_ratio_after=after.integral_error_ratio,
        )
    )
    return True, ""


def _advance_to_tolerance(
    state: _ControllerState,
    *,
    tolerance_factor: float,
    stage: str,
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
            stage=stage,
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


def _pilot_grid() -> np.ndarray:
    return -np.pi + 2.0 * np.pi * np.arange(_PILOT_COUNT, dtype=float) / _PILOT_COUNT


def _select_periodic_cut(
    pilot_t: np.ndarray,
    pilot_values: np.ndarray,
    norm: str,
    group_ids: np.ndarray,
    control_weights: np.ndarray,
) -> float:
    """Choose a smooth full-period cut using control groups only.

    The candidate cut points are the 16 pilot nodes that later reappear as the
    boundaries and midpoints of the eight initial CC9 panels.  Ward/monitor
    groups have zero weight and therefore cannot affect the cut.
    """

    group_scales = _group_point_scales(pilot_values, group_ids, norm)
    group_scales = np.maximum(group_scales, np.finfo(float).tiny)
    control = control_weights > 0.0
    scores = np.zeros(_PILOT_COUNT, dtype=float)
    for index in range(_PILOT_COUNT):
        previous = pilot_values[(index - 1) % _PILOT_COUNT]
        current = pilot_values[index]
        following = pilot_values[(index + 1) % _PILOT_COUNT]
        left = _group_norms(current - previous, group_ids, norm) / group_scales
        right = _group_norms(following - current, group_ids, norm) / group_scales
        weighted = control_weights * (left + right)
        scores[index] = float(np.max(weighted[control]))
    return float(pilot_t[int(np.argmin(scores))])


def _empty_snapshot(message: str) -> PanelAdaptiveSnapshot:
    empty = np.empty(0, dtype=float)
    return PanelAdaptiveSnapshot(
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
        message=message,
    )


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

    def early_failure(reason: str, message: str) -> PanelAdaptiveResult:
        primary = _empty_snapshot(reason)
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
            transverse_evaluations=workspace.transverse_evaluations_unique,
            cache_hits=workspace.cache_hits,
            point_evaluations=workspace.point_evaluations,
            chunk_size=workspace.points_per_t,
            wall_seconds=time.perf_counter() - started,
            geometry_wall_seconds=workspace.geometry_wall_seconds,
            evaluator_wall_seconds=workspace.evaluator_wall_seconds,
            group_names=(),
            control_group_names=(),
            monitor_group_names=(),
            primary_audit_group_ratios=(),
            primitive_group_agreement_passed=False,
            success=False,
            status=2,
            message=message,
            failure_reason=reason,
            norm=norm,
            integration_start=float("nan"),
            refinement_trace=(),
        )

    pilot_t = _pilot_grid()
    if _missing_node_count(workspace, pilot_t) > max_unique_transverse_evaluations:
        return early_failure(
            "pilot_budget_exceeded",
            "periodic cut pilot exceeds transverse budget",
        )
    pilot_values = np.stack([workspace.evaluate_t(value) for value in pilot_t], axis=0)
    ids, names, control_weights = group_layout(
        int(pilot_values.shape[1]),
        component_group_ids=component_group_ids,
        group_names=group_names,
        group_control_weights=group_control_weights,
    )
    integration_start = _select_periodic_cut(
        pilot_t,
        pilot_values,
        norm,
        ids,
        control_weights,
    )
    boundaries = integration_start + np.linspace(
        0.0,
        2.0 * np.pi,
        _INITIAL_PANEL_COUNT + 1,
    )
    initial_nodes = np.concatenate(
        [
            _mapped_panel_nodes(
                boundaries[index],
                boundaries[index + 1],
                _CC_LOW_ORDER,
            )
            for index in range(_INITIAL_PANEL_COUNT)
        ]
    )
    required = _missing_node_count(workspace, initial_nodes)
    if workspace.transverse_evaluations_unique + required > max_unique_transverse_evaluations:
        return early_failure(
            "initial_panel_budget_exceeded",
            "initial CC9 panel set exceeds transverse budget",
        )

    panels = [
        _panel_state(
            workspace,
            left=float(boundaries[index]),
            right=float(boundaries[index + 1]),
            depth=0,
            order=_CC_LOW_ORDER,
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
        integration_start=integration_start,
    )

    primary, primary_reason = _advance_to_tolerance(
        state,
        tolerance_factor=1.0,
        stage="primary",
    )
    audit = None
    audit_reason = ""
    if primary.success:
        audit, audit_reason = _advance_to_tolerance(
            state,
            tolerance_factor=float(audit_tolerance_factor),
            stage="audit",
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
        integration_start=integration_start,
        refinement_trace=tuple(state.trace),
    )


__all__ = [
    "PanelAdaptiveResult",
    "PanelAdaptiveSnapshot",
    "PanelState",
    "RefinementTraceEntry",
    "clenshaw_curtis_rule",
    "integrate_commensurate_orbit_panel_adaptive",
]
