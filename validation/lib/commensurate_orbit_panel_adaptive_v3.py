"""Split-history deterministic panel-adaptive transverse integration.

This v3 controller reuses the complete-orbit workspace, physical-group metrics,
periodic-cut selection, and budget-aware candidate enumeration from the v2 module.
It changes one operation contract only: splitting a converged CC33 parent into two
CC9 children must not discard the parent's high-order error and amplitude history.

The full transverse period is still integrated.  No even, C4, axis/diagonal, or
q-direction symmetry reduction is used.
"""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Callable, Sequence

import numpy as np

from validation.lib import commensurate_orbit_panel_adaptive as _base

OrbitAggregateEvaluator = Callable[[np.ndarray, np.ndarray], np.ndarray]

PanelAdaptiveResult = _base.PanelAdaptiveResult
PanelAdaptiveSnapshot = _base.PanelAdaptiveSnapshot
PanelState = _base.PanelState
RefinementTraceEntry = _base.RefinementTraceEntry
clenshaw_curtis_rule = _base.clenshaw_curtis_rule

# A newly split CC9 child has only a CC9-CC5 local estimate.  Retain a fraction
# of that raw child evidence, but never forget the complete CC33 parent estimate
# or the direct parent-versus-children integral discrepancy.
_SPLIT_RAW_CHILD_FLOOR = 0.25
_P_REFINEMENT_HISTORY_FLOOR = 0.25


def _replace_panel(
    panel: PanelState,
    *,
    group_errors: np.ndarray,
    group_point_scales: np.ndarray,
) -> PanelState:
    return replace(
        panel,
        group_errors=np.asarray(group_errors, dtype=float),
        group_point_scales=np.asarray(group_point_scales, dtype=float),
    )


def _split_children_with_parent_history(
    parent: PanelState,
    left_child: PanelState,
    right_child: PanelState,
    *,
    group_ids: np.ndarray,
    norm: str,
) -> tuple[PanelState, PanelState]:
    """Transfer one CC33 parent's complete error/scale history to CC9 children.

    The combined child error envelope is the maximum of:

    * the parent's accepted high-order group error;
    * twice the direct discrepancy between the sum of the two CC9 child
      integrals and the CC33 parent integral;
    * one quarter of the raw sum of the two CC9-CC5 child errors.

    The envelope is distributed between the children according to their raw local
    errors.  Consequently a split localizes the error without resetting the global
    estimate to two unrelated low-order histories.  Observed point scales from the
    parent are retained because those integrand amplitudes were actually sampled.
    """

    split_discrepancy = _base._group_norms(
        left_child.value + right_child.value - parent.value,
        group_ids,
        norm,
    )
    raw_child_sum = left_child.group_errors + right_child.group_errors
    combined_envelope = np.maximum.reduce(
        (
            parent.group_errors,
            _base._ERROR_SAFETY * split_discrepancy,
            _SPLIT_RAW_CHILD_FLOOR * raw_child_sum,
        )
    )

    tiny = np.finfo(float).tiny
    left_fraction = np.divide(
        left_child.group_errors,
        raw_child_sum,
        out=np.full_like(raw_child_sum, 0.5, dtype=float),
        where=raw_child_sum > tiny,
    )
    left_errors = combined_envelope * left_fraction
    right_errors = combined_envelope - left_errors

    inherited_scales = np.maximum.reduce(
        (
            parent.group_point_scales,
            left_child.group_point_scales,
            right_child.group_point_scales,
        )
    )
    return (
        _replace_panel(
            left_child,
            group_errors=left_errors,
            group_point_scales=inherited_scales,
        ),
        _replace_panel(
            right_child,
            group_errors=right_errors,
            group_point_scales=inherited_scales,
        ),
    )


def _p_refined_panel(
    state: _base._ControllerState,
    panel: PanelState,
    new_order: int,
) -> PanelState:
    replacement = _base._panel_state(
        state.workspace,
        left=panel.left,
        right=panel.right,
        depth=panel.depth,
        order=new_order,
        group_ids=state.group_ids,
        norm=state.norm,
    )
    # Nested p-refinement may reduce the inherited envelope, but it may not erase
    # it in one step.  This is identical in spirit to the existing spectral-history
    # floor and matters especially for children created by a split.
    errors = np.maximum(
        replacement.group_errors,
        _P_REFINEMENT_HISTORY_FLOOR * panel.group_errors,
    )
    scales = np.maximum(
        replacement.group_point_scales,
        panel.group_point_scales,
    )
    return _replace_panel(
        replacement,
        group_errors=errors,
        group_point_scales=scales,
    )


def _refine_once(
    state: _base._ControllerState,
    *,
    tolerance_factor: float,
    stage: str,
) -> tuple[bool, str]:
    before = _base._snapshot(
        state,
        tolerance_factor=tolerance_factor,
        success=False,
        message="",
    )
    candidates, reason = _base._candidate_operations(state, tolerance_factor)
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
    trace_operation = selected.operation

    if selected.operation == "p_refine":
        state.panels[selected.panel_index] = _p_refined_panel(
            state,
            panel,
            selected.new_order,
        )
    else:
        midpoint = 0.5 * (panel.left + panel.right)
        left_raw = _base._panel_state(
            state.workspace,
            left=panel.left,
            right=midpoint,
            depth=panel.depth + 1,
            order=_base._CC_LOW_ORDER,
            group_ids=state.group_ids,
            norm=state.norm,
        )
        right_raw = _base._panel_state(
            state.workspace,
            left=midpoint,
            right=panel.right,
            depth=panel.depth + 1,
            order=_base._CC_LOW_ORDER,
            group_ids=state.group_ids,
            norm=state.norm,
        )
        left_child, right_child = _split_children_with_parent_history(
            panel,
            left_raw,
            right_raw,
            group_ids=state.group_ids,
            norm=state.norm,
        )
        state.panels[selected.panel_index : selected.panel_index + 1] = [
            left_child,
            right_child,
        ]
        trace_operation = "split_history"

    state.refinement_steps += 1
    after = _base._snapshot(
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
            operation=trace_operation,
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
    state: _base._ControllerState,
    *,
    tolerance_factor: float,
    stage: str,
) -> tuple[PanelAdaptiveSnapshot, str]:
    while True:
        current = _base._snapshot(
            state,
            tolerance_factor=tolerance_factor,
            success=False,
            message="",
        )
        control = state.group_control_weights > 0.0
        if np.all(current.group_ratios[control] <= 1.0):
            return (
                _base._snapshot(
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
                _base._snapshot(
                    state,
                    tolerance_factor=tolerance_factor,
                    success=False,
                    message=reason,
                ),
                reason,
            )


def _empty_snapshot(message: str) -> PanelAdaptiveSnapshot:
    return _base._empty_snapshot(message)


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
    """Integrate one complete-orbit primitive vector with split-history panels."""

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
    workspace = _base.CompleteOrbitAggregateWorkspace(
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
            quadrature="nested_clenshaw_curtis_5_9_17_33_split_history",
            summation_method=(
                "complete_q_orbit_groupwise_budget_aware_split_history_"
                "panel_adaptive_shared_state_audit"
            ),
        )

    pilot_t = _base._pilot_grid()
    if _base._missing_node_count(workspace, pilot_t) > max_unique_transverse_evaluations:
        return early_failure(
            "pilot_budget_exceeded",
            "periodic cut pilot exceeds transverse budget",
        )
    pilot_values = np.stack([workspace.evaluate_t(value) for value in pilot_t], axis=0)
    ids, names, control_weights = _base.group_layout(
        int(pilot_values.shape[1]),
        component_group_ids=component_group_ids,
        group_names=group_names,
        group_control_weights=group_control_weights,
    )
    integration_start = _base._select_periodic_cut(
        pilot_t,
        pilot_values,
        norm,
        ids,
        control_weights,
    )
    boundaries = integration_start + np.linspace(
        0.0,
        2.0 * np.pi,
        _base._INITIAL_PANEL_COUNT + 1,
    )
    initial_nodes = np.concatenate(
        [
            _base._mapped_panel_nodes(
                boundaries[index],
                boundaries[index + 1],
                _base._CC_LOW_ORDER,
            )
            for index in range(_base._INITIAL_PANEL_COUNT)
        ]
    )
    required = _base._missing_node_count(workspace, initial_nodes)
    if workspace.transverse_evaluations_unique + required > max_unique_transverse_evaluations:
        return early_failure(
            "initial_panel_budget_exceeded",
            "initial CC9 panel set exceeds transverse budget",
        )

    panels = [
        _base._panel_state(
            workspace,
            left=float(boundaries[index]),
            right=float(boundaries[index + 1]),
            depth=0,
            order=_base._CC_LOW_ORDER,
            group_ids=ids,
            norm=norm,
        )
        for index in range(_base._INITIAL_PANEL_COUNT)
    ]
    state = _base._ControllerState(
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
        agreement_ratios = _base._audit_group_ratios(
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
        message = "primary and tightened split-history panel audit passed"
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
        quadrature="nested_clenshaw_curtis_5_9_17_33_split_history",
        summation_method=(
            "complete_q_orbit_groupwise_budget_aware_split_history_"
            "panel_adaptive_shared_state_audit"
        ),
    )


__all__ = [
    "PanelAdaptiveResult",
    "PanelAdaptiveSnapshot",
    "PanelState",
    "RefinementTraceEntry",
    "clenshaw_curtis_rule",
    "integrate_commensurate_orbit_panel_adaptive",
]
