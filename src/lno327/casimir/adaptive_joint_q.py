"""Fail-closed joint radial-angular outer-Q error control.

This module coordinates the existing adaptive radial estimator with a strict global
periodic angular-order ladder.  It allocates one finite-domain outer-Q error budget
between radial and angular components, compares their largest normalized channel
errors, and advances only the dominant direction.  The finite radial cutoff and
Matsubara set remain fixed; neither omitted tail is estimated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from .adaptive_outer_q import (
    AdaptiveRadialCasimirConfig,
    AdaptiveRadialCasimirResult,
    build_initial_adaptive_outer_q_model,
    run_adaptive_radial_casimir,
)
from .certified_point_provider import CertifiedOuterQProvider, CertifiedPointCacheError
from .fixed_chain import FixedCasimirExecutionError


class _PointProvider(Protocol):
    cached_point_count: int
    unique_q_count: int
    certification_batches: int
    requested_q_evaluations: int
    new_q_evaluations: int
    cache_hit_q_evaluations: int


class _RadialRunner(Protocol):
    def __call__(
        self,
        config: AdaptiveRadialCasimirConfig,
        *,
        provider: Any | None = None,
    ) -> AdaptiveRadialCasimirResult: ...


@dataclass(frozen=True)
class AdaptiveJointCasimirConfig:
    """Configuration for joint radial-angular finite-domain error control."""

    radial_config: AdaptiveRadialCasimirConfig = field(
        default_factory=AdaptiveRadialCasimirConfig
    )
    angular_orders: tuple[int, ...] = (4, 8, 16, 32)
    primary_offset_fraction: float = 0.5
    audit_offset_fraction: float = 0.0
    outer_rtol: float = 5e-2
    outer_atol_J_m2: float = 1e-10
    radial_budget_fraction: float = 0.75
    angular_budget_fraction: float = 0.25
    offset_rtol: float = 5e-2
    offset_atol_J_m2: float = 1e-10
    initial_radial_round_cap: int = 0
    radial_round_step: int = 1
    max_joint_iterations: int = 32
    max_total_microscopic_q_nodes: int = 100_000
    direction_tie_break: Literal["radial", "angular"] = "radial"

    def __post_init__(self) -> None:
        if not isinstance(self.radial_config, AdaptiveRadialCasimirConfig):
            raise TypeError("radial_config must be an AdaptiveRadialCasimirConfig")
        orders = tuple(int(value) for value in self.angular_orders)
        if len(orders) < 2 or any(value <= 0 for value in orders):
            raise ValueError("angular_orders must contain at least two positive values")
        if tuple(sorted(set(orders))) != orders:
            raise ValueError("angular_orders must be strictly increasing and unique")
        if any(
            right != 2 * left
            for left, right in zip(orders[:-1], orders[1:], strict=True)
        ):
            raise ValueError("angular_orders must form a strict doubling ladder")
        object.__setattr__(self, "angular_orders", orders)

        for name in ("primary_offset_fraction", "audit_offset_fraction"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
            object.__setattr__(self, name, value)
        if self.primary_offset_fraction == self.audit_offset_fraction:
            raise ValueError("primary and audit angular offsets must differ")

        for name in ("outer_rtol", "outer_atol_J_m2", "offset_rtol", "offset_atol_J_m2"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.outer_rtol == 0.0 and self.outer_atol_J_m2 == 0.0:
            raise ValueError("at least one outer tolerance must be positive")
        if self.offset_rtol == 0.0 and self.offset_atol_J_m2 == 0.0:
            raise ValueError("at least one offset tolerance must be positive")

        radial_fraction = float(self.radial_budget_fraction)
        angular_fraction = float(self.angular_budget_fraction)
        if (
            not np.isfinite(radial_fraction)
            or not np.isfinite(angular_fraction)
            or radial_fraction <= 0.0
            or angular_fraction <= 0.0
        ):
            raise ValueError("radial and angular budget fractions must be positive")
        if not np.isclose(radial_fraction + angular_fraction, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("radial and angular budget fractions must sum to one")
        object.__setattr__(self, "radial_budget_fraction", radial_fraction)
        object.__setattr__(self, "angular_budget_fraction", angular_fraction)

        initial = int(self.initial_radial_round_cap)
        if initial < 0 or initial > self.radial_config.max_refinement_rounds:
            raise ValueError(
                "initial_radial_round_cap must lie within the radial round ceiling"
            )
        object.__setattr__(self, "initial_radial_round_cap", initial)
        step = int(self.radial_round_step)
        if step <= 0:
            raise ValueError("radial_round_step must be positive")
        object.__setattr__(self, "radial_round_step", step)
        iterations = int(self.max_joint_iterations)
        if iterations <= 0:
            raise ValueError("max_joint_iterations must be positive")
        object.__setattr__(self, "max_joint_iterations", iterations)
        nodes = int(self.max_total_microscopic_q_nodes)
        if nodes <= 0:
            raise ValueError("max_total_microscopic_q_nodes must be positive")
        object.__setattr__(self, "max_total_microscopic_q_nodes", nodes)
        if self.direction_tie_break not in ("radial", "angular"):
            raise ValueError("direction_tie_break must be 'radial' or 'angular'")

    @property
    def per_run_radial_budget_fraction(self) -> float:
        """Half the radial allocation for each member of a comparison pair."""

        return 0.5 * self.radial_budget_fraction

    def as_dict(self) -> dict[str, Any]:
        return {
            "radial_config": self.radial_config.as_dict(),
            "angular_orders": list(self.angular_orders),
            "primary_offset_fraction": self.primary_offset_fraction,
            "audit_offset_fraction": self.audit_offset_fraction,
            "outer_rtol": self.outer_rtol,
            "outer_atol_J_m2": self.outer_atol_J_m2,
            "radial_budget_fraction": self.radial_budget_fraction,
            "angular_budget_fraction": self.angular_budget_fraction,
            "per_run_radial_budget_fraction": self.per_run_radial_budget_fraction,
            "offset_rtol": self.offset_rtol,
            "offset_atol_J_m2": self.offset_atol_J_m2,
            "initial_radial_round_cap": self.initial_radial_round_cap,
            "radial_round_step": self.radial_round_step,
            "max_joint_iterations": self.max_joint_iterations,
            "max_total_microscopic_q_nodes": self.max_total_microscopic_q_nodes,
            "direction_tie_break": self.direction_tie_break,
        }


@dataclass(frozen=True)
class AdaptiveJointCasimirResult:
    """Fail-closed finite Matsubara result with joint direction selection."""

    status: Literal["adaptive_finite_partial", "unresolved"]
    config: AdaptiveJointCasimirConfig
    joint_converged: bool
    radial_budget_passed: bool
    angular_budget_passed: bool
    offset_audit_passed: bool
    all_microscopic_nodes_certified: bool
    selected_angular_order: int | None
    selected_radial_round_cap: int | None
    pairing_results: Mapping[str, Any]
    direction_records: tuple[Mapping[str, Any], ...]
    radial_run_records: tuple[Mapping[str, Any], ...]
    offset_audit_record: Mapping[str, Any] | None
    termination_reason: str
    provider_statistics: Mapping[str, Any]

    @property
    def production_casimir_allowed(self) -> bool:
        return False

    @property
    def partial_sum_only(self) -> bool:
        return True

    @property
    def outer_tail_estimated(self) -> bool:
        return False

    @property
    def matsubara_tail_estimated(self) -> bool:
        return False

    @property
    def unique_microscopic_q_node_count(self) -> int:
        return int(self.provider_statistics.get("unique_q_count", 0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "adaptive-joint-casimir-result-v1",
            "status": self.status,
            "production_casimir_allowed": False,
            "partial_sum_only": True,
            "outer_cutoff_fixed": True,
            "outer_tail_estimated": False,
            "matsubara_tail_estimated": False,
            "joint_converged": self.joint_converged,
            "radial_budget_passed": self.radial_budget_passed,
            "angular_budget_passed": self.angular_budget_passed,
            "offset_audit_passed": self.offset_audit_passed,
            "all_microscopic_nodes_certified": self.all_microscopic_nodes_certified,
            "selected_angular_order": self.selected_angular_order,
            "selected_radial_round_cap": self.selected_radial_round_cap,
            "config": self.config.as_dict(),
            "pairing_results": dict(self.pairing_results),
            "direction_records": [dict(value) for value in self.direction_records],
            "radial_run_records": [dict(value) for value in self.radial_run_records],
            "offset_audit_record": (
                None
                if self.offset_audit_record is None
                else dict(self.offset_audit_record)
            ),
            "termination_reason": self.termination_reason,
            "provider_statistics": dict(self.provider_statistics),
            "unique_microscopic_q_node_count": self.unique_microscopic_q_node_count,
        }


def _provider_statistics(provider: Any) -> dict[str, Any]:
    performance_summary = getattr(provider, "performance_summary", None)
    if callable(performance_summary):
        return dict(performance_summary())
    names = (
        "cached_point_count",
        "unique_q_count",
        "certification_batches",
        "requested_q_evaluations",
        "new_q_evaluations",
        "cache_hit_q_evaluations",
    )
    return {name: int(getattr(provider, name, 0)) for name in names}


def _radial_run_record(
    result: AdaptiveRadialCasimirResult,
    *,
    angular_order: int,
    offset_fraction: float,
    radial_round_cap: int,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "angular_order": int(angular_order),
            "angular_offset_fraction": float(offset_fraction),
            "radial_round_cap": int(radial_round_cap),
            "status": str(result.status),
            "radial_converged": bool(result.radial_converged),
            "all_microscopic_nodes_certified": bool(
                result.all_microscopic_nodes_certified
            ),
            "termination_reason": str(result.termination_reason),
            "refinement_rounds": int(result.refinement_rounds),
            "unique_microscopic_q_node_count": int(
                result.unique_microscopic_q_node_count
            ),
            "pairing_results": dict(result.pairing_results),
        }
    )


def _usable_radial_estimate(result: AdaptiveRadialCasimirResult) -> tuple[bool, str]:
    if not bool(result.all_microscopic_nodes_certified):
        reason = str(result.termination_reason)
        details = [
            str(record.get("reason", ""))
            for record in result.unresolved_points
            if isinstance(record, Mapping) and record.get("reason")
        ]
        if details and details[0] not in reason:
            reason = f"{reason}: {details[0]}"
        return False, reason
    if result.status == "adaptive_finite_partial" and bool(result.radial_converged):
        return True, "radial_tolerance_met"
    if result.status == "unresolved" and result.termination_reason in {
        "maximum_refinement_rounds_reached",
        "maximum_panel_depth_reached",
    }:
        return True, str(result.termination_reason)
    return False, str(result.termination_reason)


def _channel_arrays(
    result: AdaptiveRadialCasimirResult,
    pairing: str,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    payload = result.pairing_results[pairing]
    contributions = np.asarray(payload["contributions_J_m2"], dtype=float)
    radial_errors = np.asarray(payload["estimated_radial_errors_J_m2"], dtype=float)
    if contributions.shape != (count,) or radial_errors.shape != (count,):
        raise ValueError("radial contribution/error arrays have incompatible shapes")
    if not np.isfinite(contributions).all() or not np.isfinite(radial_errors).all():
        raise ValueError("radial contribution/error arrays must be finite")
    if np.any(radial_errors < 0.0):
        raise ValueError("radial error estimates must be non-negative")
    return contributions, radial_errors


def _joint_comparison(
    previous: AdaptiveRadialCasimirResult,
    current: AdaptiveRadialCasimirResult,
    config: AdaptiveJointCasimirConfig,
) -> tuple[dict[str, Any], float, float, float, bool, bool, bool]:
    pairings = tuple(config.radial_config.point_config.pairings)
    indices = tuple(config.radial_config.point_config.matsubara_indices)
    if tuple(previous.config.point_config.pairings) != pairings:
        raise ValueError("previous radial run returned a different pairing order")
    if tuple(current.config.point_config.pairings) != pairings:
        raise ValueError("current radial run returned a different pairing order")
    if tuple(previous.config.point_config.matsubara_indices) != indices:
        raise ValueError("previous radial run returned different Matsubara indices")
    if tuple(current.config.point_config.matsubara_indices) != indices:
        raise ValueError("current radial run returned different Matsubara indices")

    output: dict[str, Any] = {}
    radial_score = 0.0
    angular_score = 0.0
    joint_score = 0.0
    radial_all = True
    angular_all = True
    joint_all = True
    for pairing in pairings:
        left, left_radial = _channel_arrays(previous, pairing, len(indices))
        right, right_radial = _channel_arrays(current, pairing, len(indices))
        scale = np.maximum(np.abs(left), np.abs(right))
        outer_tolerance = np.maximum(
            config.outer_atol_J_m2,
            config.outer_rtol * scale,
        )
        radial_tolerance = config.radial_budget_fraction * outer_tolerance
        angular_tolerance = config.angular_budget_fraction * outer_tolerance
        radial_error = left_radial + right_radial
        angular_error = np.abs(right - left)
        joint_error = radial_error + angular_error
        radial_passed = radial_error <= radial_tolerance
        angular_passed = angular_error <= angular_tolerance
        joint_passed = joint_error <= outer_tolerance
        radial_ratio = radial_error / np.maximum(
            radial_tolerance,
            np.finfo(float).tiny,
        )
        angular_ratio = angular_error / np.maximum(
            angular_tolerance,
            np.finfo(float).tiny,
        )
        joint_ratio = joint_error / np.maximum(
            outer_tolerance,
            np.finfo(float).tiny,
        )
        radial_score = max(radial_score, float(np.max(radial_ratio)))
        angular_score = max(angular_score, float(np.max(angular_ratio)))
        joint_score = max(joint_score, float(np.max(joint_ratio)))
        radial_all = radial_all and bool(np.all(radial_passed))
        angular_all = angular_all and bool(np.all(angular_passed))
        joint_all = joint_all and bool(np.all(joint_passed))
        output[pairing] = {
            "previous_contributions_J_m2": left.tolist(),
            "current_contributions_J_m2": right.tolist(),
            "previous_radial_errors_J_m2": left_radial.tolist(),
            "current_radial_errors_J_m2": right_radial.tolist(),
            "combined_radial_errors_J_m2": radial_error.tolist(),
            "angular_differences_J_m2": angular_error.tolist(),
            "estimated_joint_errors_J_m2": joint_error.tolist(),
            "outer_tolerances_J_m2": outer_tolerance.tolist(),
            "radial_budget_tolerances_J_m2": radial_tolerance.tolist(),
            "angular_budget_tolerances_J_m2": angular_tolerance.tolist(),
            "radial_normalized_errors": radial_ratio.tolist(),
            "angular_normalized_errors": angular_ratio.tolist(),
            "joint_normalized_errors": joint_ratio.tolist(),
            "radial_channel_passed": radial_passed.tolist(),
            "angular_channel_passed": angular_passed.tolist(),
            "joint_channel_passed": joint_passed.tolist(),
            "matsubara_indices": list(indices),
        }
    return (
        output,
        radial_score,
        angular_score,
        joint_score,
        radial_all,
        angular_all,
        joint_all,
    )


def _offset_comparison(
    primary: AdaptiveRadialCasimirResult,
    audit: AdaptiveRadialCasimirResult,
    config: AdaptiveJointCasimirConfig,
) -> tuple[dict[str, Any], float, float, bool, bool]:
    pairings = tuple(config.radial_config.point_config.pairings)
    indices = tuple(config.radial_config.point_config.matsubara_indices)
    output: dict[str, Any] = {}
    radial_score = 0.0
    offset_score = 0.0
    radial_all = True
    offset_all = True
    for pairing in pairings:
        primary_values, primary_radial = _channel_arrays(primary, pairing, len(indices))
        audit_values, audit_radial = _channel_arrays(audit, pairing, len(indices))
        scale = np.maximum(np.abs(primary_values), np.abs(audit_values))
        outer_tolerance = np.maximum(
            config.outer_atol_J_m2,
            config.outer_rtol * scale,
        )
        radial_tolerance = config.radial_budget_fraction * outer_tolerance
        radial_error = primary_radial + audit_radial
        offset_error = np.abs(audit_values - primary_values)
        offset_tolerance = np.maximum(
            config.offset_atol_J_m2,
            config.offset_rtol * scale,
        )
        radial_passed = radial_error <= radial_tolerance
        offset_passed = offset_error <= offset_tolerance
        radial_ratio = radial_error / np.maximum(
            radial_tolerance,
            np.finfo(float).tiny,
        )
        offset_ratio = offset_error / np.maximum(
            offset_tolerance,
            np.finfo(float).tiny,
        )
        radial_score = max(radial_score, float(np.max(radial_ratio)))
        offset_score = max(offset_score, float(np.max(offset_ratio)))
        radial_all = radial_all and bool(np.all(radial_passed))
        offset_all = offset_all and bool(np.all(offset_passed))
        output[pairing] = {
            "primary_contributions_J_m2": primary_values.tolist(),
            "audit_contributions_J_m2": audit_values.tolist(),
            "primary_radial_errors_J_m2": primary_radial.tolist(),
            "audit_radial_errors_J_m2": audit_radial.tolist(),
            "combined_radial_errors_J_m2": radial_error.tolist(),
            "radial_budget_tolerances_J_m2": radial_tolerance.tolist(),
            "offset_differences_J_m2": offset_error.tolist(),
            "offset_tolerances_J_m2": offset_tolerance.tolist(),
            "radial_normalized_errors": radial_ratio.tolist(),
            "offset_normalized_errors": offset_ratio.tolist(),
            "radial_channel_passed": radial_passed.tolist(),
            "offset_channel_passed": offset_passed.tolist(),
            "matsubara_indices": list(indices),
        }
    return output, radial_score, offset_score, radial_all, offset_all


def _choose_direction(
    radial_score: float,
    angular_score: float,
    *,
    tie_break: Literal["radial", "angular"],
) -> Literal["radial", "angular"]:
    if radial_score > angular_score:
        return "radial"
    if angular_score > radial_score:
        return "angular"
    return tie_break


def _final_pairing_results(
    primary: AdaptiveRadialCasimirResult,
    joint_metrics: Mapping[str, Any],
    offset_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pairing in primary.config.point_config.pairings:
        base = dict(primary.pairing_results[pairing])
        joint = dict(joint_metrics[pairing])
        offset = dict(offset_metrics[pairing])
        base.update(
            {
                "status": "integrated",
                "estimated_joint_errors_J_m2": joint["estimated_joint_errors_J_m2"],
                "outer_tolerances_J_m2": joint["outer_tolerances_J_m2"],
                "joint_channel_passed": joint["joint_channel_passed"],
                "estimated_angular_errors_J_m2": joint["angular_differences_J_m2"],
                "angular_budget_tolerances_J_m2": joint[
                    "angular_budget_tolerances_J_m2"
                ],
                "angular_channel_passed": joint["angular_channel_passed"],
                "combined_comparison_radial_errors_J_m2": joint[
                    "combined_radial_errors_J_m2"
                ],
                "radial_budget_tolerances_J_m2": joint[
                    "radial_budget_tolerances_J_m2"
                ],
                "radial_budget_channel_passed": joint["radial_channel_passed"],
                "estimated_offset_errors_J_m2": offset["offset_differences_J_m2"],
                "offset_tolerances_J_m2": offset["offset_tolerances_J_m2"],
                "offset_channel_passed": offset["offset_channel_passed"],
            }
        )
        output[pairing] = base
    return output


def _unresolved_result(
    config: AdaptiveJointCasimirConfig,
    *,
    direction_records: Sequence[Mapping[str, Any]],
    radial_run_records: Sequence[Mapping[str, Any]],
    offset_record: Mapping[str, Any] | None,
    selected_order: int | None,
    radial_round_cap: int | None,
    pairing_results: Mapping[str, Any],
    radial_passed: bool,
    angular_passed: bool,
    offset_passed: bool,
    all_certified: bool,
    reason: str,
    provider: Any,
) -> AdaptiveJointCasimirResult:
    return AdaptiveJointCasimirResult(
        status="unresolved",
        config=config,
        joint_converged=False,
        radial_budget_passed=radial_passed,
        angular_budget_passed=angular_passed,
        offset_audit_passed=offset_passed,
        all_microscopic_nodes_certified=all_certified,
        selected_angular_order=selected_order,
        selected_radial_round_cap=radial_round_cap,
        pairing_results=dict(pairing_results),
        direction_records=tuple(direction_records),
        radial_run_records=tuple(radial_run_records),
        offset_audit_record=offset_record,
        termination_reason=reason,
        provider_statistics=_provider_statistics(provider),
    )


def run_adaptive_joint_casimir(
    config: AdaptiveJointCasimirConfig,
    *,
    provider: _PointProvider | None = None,
    radial_runner: _RadialRunner = run_adaptive_radial_casimir,
) -> AdaptiveJointCasimirResult:
    """Coordinate radial refinement and angular-order increases by normalized error."""

    if not isinstance(config, AdaptiveJointCasimirConfig):
        raise TypeError("config must be an AdaptiveJointCasimirConfig")
    active_provider: Any = provider
    direction_records: list[Mapping[str, Any]] = []
    radial_run_records: list[Mapping[str, Any]] = []
    run_cache: dict[tuple[int, float, int], AdaptiveRadialCasimirResult] = {}
    previous_index = 0
    current_index = 1
    radial_round_cap = config.initial_radial_round_cap
    last_pairing_results: dict[str, Any] = {}
    last_radial_passed = False
    last_angular_passed = False
    last_offset_passed = False
    all_certified = True

    try:
        if active_provider is None:
            active_provider = CertifiedOuterQProvider(
                config.radial_config.point_config,
                cache_path=config.radial_config.point_cache_path,
            )

        def build_run_config(
            angular_order: int,
            offset_fraction: float,
            round_cap: int,
        ) -> AdaptiveRadialCasimirConfig:
            radial_fraction = config.per_run_radial_budget_fraction
            return replace(
                config.radial_config,
                angular_order=int(angular_order),
                angular_offset_fraction=float(offset_fraction),
                radial_rtol=config.outer_rtol * radial_fraction,
                radial_atol_J_m2=config.outer_atol_J_m2 * radial_fraction,
                max_refinement_rounds=int(round_cap),
            )


        def prefetch_comparison_runs(
            specifications: Sequence[tuple[int, float, int]],
        ) -> None:
            """Batch the known initial q grids for a comparison pair.

            This changes only orchestration. Each radial run subsequently requests
            the exact same IEEE-754 q coordinates and reads the certified points from
            the provider cache.
            """

            evaluate = getattr(active_provider, "evaluate", None)
            count_new_q = getattr(active_provider, "count_new_q", None)
            if not callable(evaluate) or not callable(count_new_q):
                return
            pending = [
                build_run_config(angular_order, offset_fraction, round_cap)
                for angular_order, offset_fraction, round_cap in specifications
                if (
                    int(angular_order),
                    float(offset_fraction),
                    int(round_cap),
                )
                not in run_cache
            ]
            if len(pending) < 2:
                return
            initial_arrays = [
                build_initial_adaptive_outer_q_model(radial_config)
                for radial_config in pending
            ]
            for radial_config, q_model in zip(
                pending,
                initial_arrays,
                strict=True,
            ):
                unique_count = len(
                    {
                        (float(q[0]).hex(), float(q[1]).hex())
                        for q in np.asarray(q_model, dtype=float)
                    }
                )
                if unique_count > radial_config.max_microscopic_q_nodes:
                    return
            combined = np.concatenate(initial_arrays, axis=0)
            new_q_count = int(count_new_q(combined))
            if new_q_count == 0:
                return
            current_q_count = int(getattr(active_provider, "unique_q_count", 0))
            if (
                current_q_count + new_q_count
                > config.max_total_microscopic_q_nodes
            ):
                return
            batch = evaluate(combined)
            if not batch.all_established:
                # Prefetch is only an optimization.  Let the real radial runs request
                # the same cached points so their detailed unresolved records propagate
                # through the normal fail-closed result path.
                return

        def get_run(
            angular_order: int,
            offset_fraction: float,
            round_cap: int,
        ) -> AdaptiveRadialCasimirResult:
            key = (int(angular_order), float(offset_fraction), int(round_cap))
            if key not in run_cache:
                radial_config = build_run_config(
                    angular_order,
                    offset_fraction,
                    round_cap,
                )
                result = radial_runner(radial_config, provider=active_provider)
                run_cache[key] = result
                radial_run_records.append(
                    _radial_run_record(
                        result,
                        angular_order=angular_order,
                        offset_fraction=offset_fraction,
                        radial_round_cap=round_cap,
                    )
                )
                if (
                    int(getattr(active_provider, "unique_q_count", 0))
                    > config.max_total_microscopic_q_nodes
                ):
                    raise RuntimeError("joint_microscopic_q_node_budget_exhausted")
            return run_cache[key]

        for iteration in range(config.max_joint_iterations):
            previous_order = config.angular_orders[previous_index]
            current_order = config.angular_orders[current_index]
            prefetch_comparison_runs(
                (
                    (
                        previous_order,
                        config.primary_offset_fraction,
                        radial_round_cap,
                    ),
                    (
                        current_order,
                        config.primary_offset_fraction,
                        radial_round_cap,
                    ),
                )
            )
            previous = get_run(
                previous_order,
                config.primary_offset_fraction,
                radial_round_cap,
            )
            current = get_run(
                current_order,
                config.primary_offset_fraction,
                radial_round_cap,
            )
            previous_usable, previous_reason = _usable_radial_estimate(previous)
            current_usable, current_reason = _usable_radial_estimate(current)
            all_certified = (
                all_certified
                and bool(previous.all_microscopic_nodes_certified)
                and bool(current.all_microscopic_nodes_certified)
            )
            last_pairing_results = dict(current.pairing_results)
            if not previous_usable or not current_usable:
                return _unresolved_result(
                    config,
                    direction_records=direction_records,
                    radial_run_records=radial_run_records,
                    offset_record=None,
                    selected_order=current_order,
                    radial_round_cap=radial_round_cap,
                    pairing_results=last_pairing_results,
                    radial_passed=False,
                    angular_passed=False,
                    offset_passed=False,
                    all_certified=all_certified,
                    reason=(
                        "radial_run_unresolved: "
                        f"previous={previous_reason}, current={current_reason}"
                    ),
                    provider=active_provider,
                )

            (
                joint_metrics,
                radial_score,
                angular_score,
                joint_score,
                radial_passed,
                angular_passed,
                joint_passed,
            ) = _joint_comparison(previous, current, config)
            last_radial_passed = radial_passed
            last_angular_passed = angular_passed
            decision_payload: dict[str, Any] = {
                "iteration": iteration,
                "previous_angular_order": previous_order,
                "current_angular_order": current_order,
                "radial_round_cap": radial_round_cap,
                "radial_score": radial_score,
                "angular_score": angular_score,
                "joint_score": joint_score,
                "radial_budget_passed": radial_passed,
                "angular_budget_passed": angular_passed,
                "joint_budget_passed": joint_passed,
                "pairings": joint_metrics,
                "selected_direction": None,
                "selection_reason": None,
            }

            if radial_passed and angular_passed and joint_passed:
                audit = get_run(
                    current_order,
                    config.audit_offset_fraction,
                    radial_round_cap,
                )
                audit_usable, audit_reason = _usable_radial_estimate(audit)
                all_certified = (
                    all_certified and bool(audit.all_microscopic_nodes_certified)
                )
                if not audit_usable:
                    decision_payload.update(
                        {
                            "selected_direction": "fail",
                            "selection_reason": (
                                f"offset_audit_radial_unresolved: {audit_reason}"
                            ),
                        }
                    )
                    direction_records.append(MappingProxyType(decision_payload))
                    return _unresolved_result(
                        config,
                        direction_records=direction_records,
                        radial_run_records=radial_run_records,
                        offset_record=_radial_run_record(
                            audit,
                            angular_order=current_order,
                            offset_fraction=config.audit_offset_fraction,
                            radial_round_cap=radial_round_cap,
                        ),
                        selected_order=current_order,
                        radial_round_cap=radial_round_cap,
                        pairing_results=last_pairing_results,
                        radial_passed=radial_passed,
                        angular_passed=angular_passed,
                        offset_passed=False,
                        all_certified=all_certified,
                        reason=f"offset_audit_radial_unresolved: {audit_reason}",
                        provider=active_provider,
                    )
                (
                    offset_metrics,
                    audit_radial_score,
                    offset_score,
                    audit_radial_passed,
                    offset_passed,
                ) = _offset_comparison(current, audit, config)
                last_offset_passed = offset_passed
                offset_record = MappingProxyType(
                    {
                        **dict(
                            _radial_run_record(
                                audit,
                                angular_order=current_order,
                                offset_fraction=config.audit_offset_fraction,
                                radial_round_cap=radial_round_cap,
                            )
                        ),
                        "comparison_to_primary_offset": offset_metrics,
                        "radial_score": audit_radial_score,
                        "offset_score": offset_score,
                        "radial_budget_passed": audit_radial_passed,
                        "offset_audit_passed": offset_passed,
                    }
                )
                if audit_radial_passed and offset_passed:
                    decision_payload.update(
                        {
                            "selected_direction": "accept",
                            "selection_reason": (
                                "joint radial-angular budget and offset audit passed"
                            ),
                            "offset_radial_score": audit_radial_score,
                            "offset_score": offset_score,
                        }
                    )
                    direction_records.append(MappingProxyType(decision_payload))
                    return AdaptiveJointCasimirResult(
                        status="adaptive_finite_partial",
                        config=config,
                        joint_converged=True,
                        radial_budget_passed=True,
                        angular_budget_passed=True,
                        offset_audit_passed=True,
                        all_microscopic_nodes_certified=True,
                        selected_angular_order=current_order,
                        selected_radial_round_cap=radial_round_cap,
                        pairing_results=_final_pairing_results(
                            current,
                            joint_metrics,
                            offset_metrics,
                        ),
                        direction_records=tuple(direction_records),
                        radial_run_records=tuple(radial_run_records),
                        offset_audit_record=offset_record,
                        termination_reason=(
                            "joint_radial_angular_budget_and_offset_tolerances_met"
                        ),
                        provider_statistics=_provider_statistics(active_provider),
                    )

                direction = _choose_direction(
                    audit_radial_score,
                    offset_score,
                    tie_break=config.direction_tie_break,
                )
                decision_payload.update(
                    {
                        "selected_direction": direction,
                        "selection_reason": (
                            "offset audit selected the larger normalized component"
                        ),
                        "offset_radial_score": audit_radial_score,
                        "offset_score": offset_score,
                    }
                )
                direction_records.append(MappingProxyType(decision_payload))
                if direction == "radial":
                    if previous_reason == "maximum_panel_depth_reached" or current_reason == "maximum_panel_depth_reached" or audit_reason == "maximum_panel_depth_reached":
                        return _unresolved_result(
                            config,
                            direction_records=direction_records,
                            radial_run_records=radial_run_records,
                            offset_record=offset_record,
                            selected_order=current_order,
                            radial_round_cap=radial_round_cap,
                            pairing_results=last_pairing_results,
                            radial_passed=False,
                            angular_passed=True,
                            offset_passed=offset_passed,
                            all_certified=all_certified,
                            reason="radial_panel_depth_exhausted",
                            provider=active_provider,
                        )
                    if radial_round_cap >= config.radial_config.max_refinement_rounds:
                        return _unresolved_result(
                            config,
                            direction_records=direction_records,
                            radial_run_records=radial_run_records,
                            offset_record=offset_record,
                            selected_order=current_order,
                            radial_round_cap=radial_round_cap,
                            pairing_results=last_pairing_results,
                            radial_passed=False,
                            angular_passed=True,
                            offset_passed=offset_passed,
                            all_certified=all_certified,
                            reason="joint_radial_round_budget_exhausted",
                            provider=active_provider,
                        )
                    radial_round_cap = min(
                        radial_round_cap + config.radial_round_step,
                        config.radial_config.max_refinement_rounds,
                    )
                    continue
                if current_index >= len(config.angular_orders) - 1:
                    return _unresolved_result(
                        config,
                        direction_records=direction_records,
                        radial_run_records=radial_run_records,
                        offset_record=offset_record,
                        selected_order=current_order,
                        radial_round_cap=radial_round_cap,
                        pairing_results=last_pairing_results,
                        radial_passed=True,
                        angular_passed=False,
                        offset_passed=False,
                        all_certified=all_certified,
                        reason="joint_offset_audit_failed_at_maximum_angular_order",
                        provider=active_provider,
                    )
                previous_index = current_index
                current_index += 1
                continue

            direction = _choose_direction(
                radial_score,
                angular_score,
                tie_break=config.direction_tie_break,
            )
            decision_payload.update(
                {
                    "selected_direction": direction,
                    "selection_reason": (
                        "selected the larger normalized allocated error component"
                    ),
                }
            )
            direction_records.append(MappingProxyType(decision_payload))
            if direction == "radial":
                if (
                    previous_reason == "maximum_panel_depth_reached"
                    or current_reason == "maximum_panel_depth_reached"
                ):
                    return _unresolved_result(
                        config,
                        direction_records=direction_records,
                        radial_run_records=radial_run_records,
                        offset_record=None,
                        selected_order=current_order,
                        radial_round_cap=radial_round_cap,
                        pairing_results=last_pairing_results,
                        radial_passed=False,
                        angular_passed=angular_passed,
                        offset_passed=False,
                        all_certified=all_certified,
                        reason="radial_panel_depth_exhausted",
                        provider=active_provider,
                    )
                if radial_round_cap >= config.radial_config.max_refinement_rounds:
                    return _unresolved_result(
                        config,
                        direction_records=direction_records,
                        radial_run_records=radial_run_records,
                        offset_record=None,
                        selected_order=current_order,
                        radial_round_cap=radial_round_cap,
                        pairing_results=last_pairing_results,
                        radial_passed=False,
                        angular_passed=angular_passed,
                        offset_passed=False,
                        all_certified=all_certified,
                        reason="joint_radial_round_budget_exhausted",
                        provider=active_provider,
                    )
                radial_round_cap = min(
                    radial_round_cap + config.radial_round_step,
                    config.radial_config.max_refinement_rounds,
                )
                continue

            if current_index >= len(config.angular_orders) - 1:
                return _unresolved_result(
                    config,
                    direction_records=direction_records,
                    radial_run_records=radial_run_records,
                    offset_record=None,
                    selected_order=current_order,
                    radial_round_cap=radial_round_cap,
                    pairing_results=last_pairing_results,
                    radial_passed=radial_passed,
                    angular_passed=False,
                    offset_passed=False,
                    all_certified=all_certified,
                    reason="joint_angular_order_ladder_exhausted",
                    provider=active_provider,
                )
            previous_index = current_index
            current_index += 1

        return _unresolved_result(
            config,
            direction_records=direction_records,
            radial_run_records=radial_run_records,
            offset_record=None,
            selected_order=config.angular_orders[current_index],
            radial_round_cap=radial_round_cap,
            pairing_results=last_pairing_results,
            radial_passed=last_radial_passed,
            angular_passed=last_angular_passed,
            offset_passed=last_offset_passed,
            all_certified=all_certified,
            reason="joint_iteration_budget_exhausted",
            provider=active_provider,
        )
    except (CertifiedPointCacheError, FixedCasimirExecutionError) as exc:
        return _unresolved_result(
            config,
            direction_records=direction_records,
            radial_run_records=radial_run_records,
            offset_record=None,
            selected_order=(
                config.angular_orders[current_index]
                if current_index < len(config.angular_orders)
                else None
            ),
            radial_round_cap=radial_round_cap,
            pairing_results=last_pairing_results,
            radial_passed=False,
            angular_passed=False,
            offset_passed=False,
            all_certified=False,
            reason=f"point_provider_failure: {exc}",
            provider=active_provider,
        )
    except RuntimeError as exc:
        reason = (
            "joint_microscopic_q_node_budget_exhausted"
            if str(exc) == "joint_microscopic_q_node_budget_exhausted"
            else f"joint_runtime_failure: {exc}"
        )
        return _unresolved_result(
            config,
            direction_records=direction_records,
            radial_run_records=radial_run_records,
            offset_record=None,
            selected_order=(
                config.angular_orders[current_index]
                if current_index < len(config.angular_orders)
                else None
            ),
            radial_round_cap=radial_round_cap,
            pairing_results=last_pairing_results,
            radial_passed=last_radial_passed,
            angular_passed=last_angular_passed,
            offset_passed=last_offset_passed,
            all_certified=all_certified,
            reason=reason,
            provider=active_provider,
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return _unresolved_result(
            config,
            direction_records=direction_records,
            radial_run_records=radial_run_records,
            offset_record=None,
            selected_order=(
                config.angular_orders[current_index]
                if current_index < len(config.angular_orders)
                else None
            ),
            radial_round_cap=radial_round_cap,
            pairing_results=last_pairing_results,
            radial_passed=False,
            angular_passed=False,
            offset_passed=False,
            all_certified=False,
            reason=f"radial_result_contract_failure: {exc}",
            provider=active_provider,
        )


__all__ = [
    "AdaptiveJointCasimirConfig",
    "AdaptiveJointCasimirResult",
    "run_adaptive_joint_casimir",
]
