"""Fail-closed adaptive outer-Q cutoff extension and tail envelope.

This controller wraps the existing joint radial-angular finite-domain controller.
It extends a cumulative ``u_max`` ladder, reuses one certified microscopic-point
provider, and establishes a channelwise geometric envelope for the omitted
``u > u_max`` contribution.  The Matsubara index set remains fixed and its tail
is not estimated here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from .adaptive_joint_q import (
    AdaptiveJointCasimirConfig,
    AdaptiveJointCasimirResult,
    run_adaptive_joint_casimir,
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


class _JointRunner(Protocol):
    def __call__(
        self,
        config: AdaptiveJointCasimirConfig,
        *,
        provider: Any | None = None,
    ) -> AdaptiveJointCasimirResult: ...


@dataclass(frozen=True)
class AdaptiveOuterTailCasimirConfig:
    """Configuration for cumulative cutoff extension and outer-Q tail control."""

    joint_config: AdaptiveJointCasimirConfig = field(
        default_factory=AdaptiveJointCasimirConfig
    )
    cutoff_u_values: tuple[float, ...] = (
        6.0,
        10.0,
        14.0,
        18.0,
        24.0,
        30.0,
        36.0,
        42.0,
    )
    total_outer_rtol: float = 5e-2
    total_outer_atol_J_m2: float = 1e-10
    finite_domain_budget_fraction: float = 0.7
    tail_budget_fraction: float = 0.3
    joint_budget_fraction_within_finite: float = 0.8
    offset_budget_fraction_within_finite: float = 0.2
    tail_start_u: float = 24.0
    tail_window_shells: int = 3
    tail_ratio_max: float = 0.8
    shell_width_rtol: float = 1e-12
    shell_width_atol: float = 1e-12
    max_total_microscopic_q_nodes: int = 250_000

    def __post_init__(self) -> None:
        if not isinstance(self.joint_config, AdaptiveJointCasimirConfig):
            raise TypeError("joint_config must be an AdaptiveJointCasimirConfig")
        cutoffs = tuple(float(value) for value in self.cutoff_u_values)
        if len(cutoffs) < 3 or not np.isfinite(cutoffs).all():
            raise ValueError("cutoff_u_values must contain at least three finite values")
        if cutoffs[0] <= 0.0 or any(
            right <= left for left, right in zip(cutoffs[:-1], cutoffs[1:], strict=True)
        ):
            raise ValueError("cutoff_u_values must be strictly increasing and positive")
        object.__setattr__(self, "cutoff_u_values", cutoffs)

        for name in ("total_outer_rtol", "total_outer_atol_J_m2"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.total_outer_rtol == 0.0 and self.total_outer_atol_J_m2 == 0.0:
            raise ValueError("at least one total outer tolerance must be positive")

        finite_fraction = float(self.finite_domain_budget_fraction)
        tail_fraction = float(self.tail_budget_fraction)
        if finite_fraction <= 0.0 or tail_fraction <= 0.0:
            raise ValueError("finite-domain and tail budget fractions must be positive")
        if not np.isclose(finite_fraction + tail_fraction, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("finite-domain and tail budget fractions must sum to one")
        object.__setattr__(self, "finite_domain_budget_fraction", finite_fraction)
        object.__setattr__(self, "tail_budget_fraction", tail_fraction)

        joint_fraction = float(self.joint_budget_fraction_within_finite)
        offset_fraction = float(self.offset_budget_fraction_within_finite)
        if joint_fraction <= 0.0 or offset_fraction <= 0.0:
            raise ValueError("joint and offset finite-budget fractions must be positive")
        if not np.isclose(joint_fraction + offset_fraction, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("joint and offset finite-budget fractions must sum to one")
        object.__setattr__(self, "joint_budget_fraction_within_finite", joint_fraction)
        object.__setattr__(self, "offset_budget_fraction_within_finite", offset_fraction)

        tail_start = float(self.tail_start_u)
        if not np.isfinite(tail_start) or tail_start < 0.0:
            raise ValueError("tail_start_u must be finite and non-negative")
        if tail_start >= cutoffs[-1]:
            raise ValueError("tail_start_u must lie below the maximum cutoff")
        object.__setattr__(self, "tail_start_u", tail_start)

        window = int(self.tail_window_shells)
        if window < 2:
            raise ValueError("tail_window_shells must be at least two")
        object.__setattr__(self, "tail_window_shells", window)
        ratio = float(self.tail_ratio_max)
        if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
            raise ValueError("tail_ratio_max must lie strictly between zero and one")
        object.__setattr__(self, "tail_ratio_max", ratio)

        for name in ("shell_width_rtol", "shell_width_atol"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        nodes = int(self.max_total_microscopic_q_nodes)
        if nodes <= 0:
            raise ValueError("max_total_microscopic_q_nodes must be positive")
        object.__setattr__(self, "max_total_microscopic_q_nodes", nodes)

        eligible = [
            (left, right)
            for left, right in zip((0.0, *cutoffs[:-1]), cutoffs, strict=True)
            if left >= tail_start
        ]
        if len(eligible) < window:
            raise ValueError(
                "cutoff ladder does not provide enough shells at or above tail_start_u"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "joint_config": self.joint_config.as_dict(),
            "cutoff_u_values": list(self.cutoff_u_values),
            "total_outer_rtol": self.total_outer_rtol,
            "total_outer_atol_J_m2": self.total_outer_atol_J_m2,
            "finite_domain_budget_fraction": self.finite_domain_budget_fraction,
            "tail_budget_fraction": self.tail_budget_fraction,
            "joint_budget_fraction_within_finite": (
                self.joint_budget_fraction_within_finite
            ),
            "offset_budget_fraction_within_finite": (
                self.offset_budget_fraction_within_finite
            ),
            "tail_start_u": self.tail_start_u,
            "tail_window_shells": self.tail_window_shells,
            "tail_ratio_max": self.tail_ratio_max,
            "shell_width_rtol": self.shell_width_rtol,
            "shell_width_atol": self.shell_width_atol,
            "max_total_microscopic_q_nodes": self.max_total_microscopic_q_nodes,
        }


@dataclass(frozen=True)
class AdaptiveOuterTailCasimirResult:
    """Finite Matsubara result with adaptive outer cutoff and tail evidence."""

    status: Literal["adaptive_finite_partial", "unresolved"]
    config: AdaptiveOuterTailCasimirConfig
    cutoff_converged: bool
    outer_tail_estimated_flag: bool
    all_finite_domain_runs_converged: bool
    all_microscopic_nodes_certified: bool
    selected_u_max: float | None
    pairing_results: Mapping[str, Any]
    cutoff_records: tuple[Mapping[str, Any], ...]
    shell_records: tuple[Mapping[str, Any], ...]
    termination_reason: str
    provider_statistics: Mapping[str, Any]

    @property
    def production_casimir_allowed(self) -> bool:
        return False

    @property
    def partial_sum_only(self) -> bool:
        return True

    @property
    def outer_cutoff_adaptive(self) -> bool:
        return True

    @property
    def outer_tail_estimated(self) -> bool:
        return bool(self.outer_tail_estimated_flag)

    @property
    def matsubara_tail_estimated(self) -> bool:
        return False

    @property
    def unique_microscopic_q_node_count(self) -> int:
        return int(self.provider_statistics.get("unique_q_count", 0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "adaptive-outer-tail-casimir-result-v1",
            "status": self.status,
            "production_casimir_allowed": False,
            "partial_sum_only": True,
            "outer_cutoff_fixed": False,
            "outer_cutoff_adaptive": True,
            "outer_tail_estimated": self.outer_tail_estimated,
            "matsubara_tail_estimated": False,
            "cutoff_converged": self.cutoff_converged,
            "all_finite_domain_runs_converged": (
                self.all_finite_domain_runs_converged
            ),
            "all_microscopic_nodes_certified": self.all_microscopic_nodes_certified,
            "selected_u_max": self.selected_u_max,
            "config": self.config.as_dict(),
            "pairing_results": dict(self.pairing_results),
            "cutoff_records": [dict(value) for value in self.cutoff_records],
            "shell_records": [dict(value) for value in self.shell_records],
            "termination_reason": self.termination_reason,
            "provider_statistics": dict(self.provider_statistics),
            "unique_microscopic_q_node_count": self.unique_microscopic_q_node_count,
        }


def _provider_statistics(provider: Any) -> dict[str, int]:
    names = (
        "cached_point_count",
        "unique_q_count",
        "certification_batches",
        "requested_q_evaluations",
        "new_q_evaluations",
        "cache_hit_q_evaluations",
    )
    return {name: int(getattr(provider, name, 0)) for name in names}


def _scaled_joint_config(
    config: AdaptiveOuterTailCasimirConfig,
    *,
    cutoff_index: int,
) -> AdaptiveJointCasimirConfig:
    finite = config.finite_domain_budget_fraction
    joint_share = config.joint_budget_fraction_within_finite
    offset_share = config.offset_budget_fraction_within_finite
    panel_edges = (0.0, *config.cutoff_u_values[: cutoff_index + 1])
    radial = replace(
        config.joint_config.radial_config,
        initial_panel_edges=panel_edges,
    )
    return replace(
        config.joint_config,
        radial_config=radial,
        outer_rtol=config.total_outer_rtol * finite * joint_share,
        outer_atol_J_m2=config.total_outer_atol_J_m2 * finite * joint_share,
        offset_rtol=config.total_outer_rtol * finite * offset_share,
        offset_atol_J_m2=config.total_outer_atol_J_m2 * finite * offset_share,
    )


def _joint_run_usable(result: AdaptiveJointCasimirResult) -> tuple[bool, str]:
    if not bool(result.all_microscopic_nodes_certified):
        return False, str(result.termination_reason)
    if result.status != "adaptive_finite_partial" or not bool(result.joint_converged):
        return False, str(result.termination_reason)
    if not bool(result.radial_budget_passed):
        return False, "radial_budget_unresolved"
    if not bool(result.angular_budget_passed):
        return False, "angular_budget_unresolved"
    if not bool(result.offset_audit_passed):
        return False, "offset_audit_unresolved"
    return True, "finite_domain_tolerances_met"


def _channel_arrays(
    result: AdaptiveJointCasimirResult,
    pairing: str,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    payload = result.pairing_results[pairing]
    values = np.asarray(payload["contributions_J_m2"], dtype=float)
    joint_error = np.asarray(payload["estimated_joint_errors_J_m2"], dtype=float)
    offset_error = np.asarray(payload["estimated_offset_errors_J_m2"], dtype=float)
    if values.shape != (count,) or joint_error.shape != (count,) or offset_error.shape != (count,):
        raise ValueError("joint contribution/error arrays have incompatible shapes")
    if not np.isfinite(values).all() or not np.isfinite(joint_error).all() or not np.isfinite(offset_error).all():
        raise ValueError("joint contribution/error arrays must be finite")
    if np.any(joint_error < 0.0) or np.any(offset_error < 0.0):
        raise ValueError("joint and offset errors must be non-negative")
    return values, joint_error + offset_error


def _cutoff_record(
    result: AdaptiveJointCasimirResult,
    *,
    u_max: float,
    finite_errors: Mapping[str, np.ndarray],
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "u_max": float(u_max),
            "status": str(result.status),
            "joint_converged": bool(result.joint_converged),
            "radial_budget_passed": bool(result.radial_budget_passed),
            "angular_budget_passed": bool(result.angular_budget_passed),
            "offset_audit_passed": bool(result.offset_audit_passed),
            "all_microscopic_nodes_certified": bool(
                result.all_microscopic_nodes_certified
            ),
            "termination_reason": str(result.termination_reason),
            "selected_angular_order": result.selected_angular_order,
            "selected_radial_round_cap": result.selected_radial_round_cap,
            "pairing_results": dict(result.pairing_results),
            "finite_domain_error_bounds_J_m2": {
                pairing: values.tolist() for pairing, values in finite_errors.items()
            },
            "provider_statistics": dict(result.provider_statistics),
        }
    )


def _shell_record(
    *,
    left_u: float,
    right_u: float,
    previous_values: Mapping[str, np.ndarray],
    current_values: Mapping[str, np.ndarray],
    previous_errors: Mapping[str, np.ndarray],
    current_errors: Mapping[str, np.ndarray],
    indices: Sequence[int],
) -> Mapping[str, Any]:
    pairings: dict[str, Any] = {}
    for pairing in current_values:
        delta = current_values[pairing] - previous_values[pairing]
        quadrature = current_errors[pairing] + previous_errors[pairing]
        amplitude = np.abs(delta) + quadrature
        pairings[pairing] = {
            "shell_contributions_J_m2": delta.tolist(),
            "shell_quadrature_error_bounds_J_m2": quadrature.tolist(),
            "shell_envelope_amplitudes_J_m2": amplitude.tolist(),
            "matsubara_indices": list(indices),
        }
    return MappingProxyType(
        {
            "left_u": float(left_u),
            "right_u": float(right_u),
            "width_u": float(right_u - left_u),
            "pairings": pairings,
        }
    )


def _tail_window_metrics(
    config: AdaptiveOuterTailCasimirConfig,
    *,
    shell_records: Sequence[Mapping[str, Any]],
    current_values: Mapping[str, np.ndarray],
    current_finite_errors: Mapping[str, np.ndarray],
    indices: Sequence[int],
) -> tuple[dict[str, Any] | None, bool, bool, bool]:
    eligible = [
        record for record in shell_records if float(record["left_u"]) >= config.tail_start_u
    ]
    if len(eligible) < config.tail_window_shells:
        return None, False, False, False
    window = eligible[-config.tail_window_shells :]
    widths = np.asarray([float(record["width_u"]) for record in window], dtype=float)
    equal_width = bool(
        np.allclose(
            widths,
            widths[-1],
            rtol=config.shell_width_rtol,
            atol=config.shell_width_atol,
        )
    )
    output: dict[str, Any] = {
        "window_left_u": float(window[0]["left_u"]),
        "window_right_u": float(window[-1]["right_u"]),
        "shell_widths_u": widths.tolist(),
        "equal_shell_widths": equal_width,
        "tail_ratio_max": config.tail_ratio_max,
        "pairings": {},
    }
    decay_all = equal_width
    tail_all = equal_width
    total_all = equal_width
    for pairing in current_values:
        amplitudes = np.asarray(
            [
                record["pairings"][pairing]["shell_envelope_amplitudes_J_m2"]
                for record in window
            ],
            dtype=float,
        )
        if amplitudes.shape != (config.tail_window_shells, len(indices)):
            raise ValueError("tail shell amplitude arrays have incompatible shapes")
        denominator = amplitudes[:-1]
        ratios = np.divide(
            amplitudes[1:],
            denominator,
            out=np.full_like(amplitudes[1:], np.inf),
            where=denominator > 0.0,
        )
        both_zero = (denominator == 0.0) & (amplitudes[1:] == 0.0)
        ratios[both_zero] = 0.0
        ratio_envelope = np.max(ratios, axis=0)
        decay_passed = ratio_envelope <= config.tail_ratio_max
        tail_bound = amplitudes[-1] * config.tail_ratio_max / (
            1.0 - config.tail_ratio_max
        )
        scale = np.abs(current_values[pairing])
        total_tolerance = np.maximum(
            config.total_outer_atol_J_m2,
            config.total_outer_rtol * scale,
        )
        finite_tolerance = config.finite_domain_budget_fraction * total_tolerance
        tail_tolerance = config.tail_budget_fraction * total_tolerance
        finite_passed = current_finite_errors[pairing] <= finite_tolerance
        tail_passed = tail_bound <= tail_tolerance
        total_error = current_finite_errors[pairing] + tail_bound
        total_passed = total_error <= total_tolerance
        channel_decay = bool(np.all(decay_passed))
        channel_tail = bool(np.all(tail_passed))
        channel_total = bool(np.all(total_passed))
        decay_all = decay_all and channel_decay
        tail_all = tail_all and channel_tail and bool(np.all(finite_passed))
        total_all = total_all and channel_total
        output["pairings"][pairing] = {
            "shell_envelope_amplitudes_J_m2": amplitudes.tolist(),
            "observed_shell_ratios": ratios.tolist(),
            "ratio_envelopes": ratio_envelope.tolist(),
            "decay_channel_passed": decay_passed.tolist(),
            "estimated_tail_bounds_J_m2": tail_bound.tolist(),
            "finite_domain_error_bounds_J_m2": current_finite_errors[pairing].tolist(),
            "estimated_total_outer_errors_J_m2": total_error.tolist(),
            "total_outer_tolerances_J_m2": total_tolerance.tolist(),
            "finite_domain_budget_tolerances_J_m2": finite_tolerance.tolist(),
            "tail_budget_tolerances_J_m2": tail_tolerance.tolist(),
            "finite_domain_channel_passed": finite_passed.tolist(),
            "tail_channel_passed": tail_passed.tolist(),
            "total_outer_channel_passed": total_passed.tolist(),
            "matsubara_indices": list(indices),
        }
    return output, decay_all, tail_all, total_all


def _final_pairing_results(
    result: AdaptiveJointCasimirResult,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pairing in result.config.radial_config.point_config.pairings:
        base = dict(result.pairing_results[pairing])
        tail = dict(metrics["pairings"][pairing])
        base.update(
            {
                "status": "integrated_with_outer_tail_bound",
                "finite_domain_error_bounds_J_m2": tail[
                    "finite_domain_error_bounds_J_m2"
                ],
                "estimated_outer_tail_bounds_J_m2": tail[
                    "estimated_tail_bounds_J_m2"
                ],
                "estimated_total_outer_errors_J_m2": tail[
                    "estimated_total_outer_errors_J_m2"
                ],
                "total_outer_tolerances_J_m2": tail[
                    "total_outer_tolerances_J_m2"
                ],
                "finite_domain_budget_tolerances_J_m2": tail[
                    "finite_domain_budget_tolerances_J_m2"
                ],
                "tail_budget_tolerances_J_m2": tail[
                    "tail_budget_tolerances_J_m2"
                ],
                "outer_tail_ratio_envelopes": tail["ratio_envelopes"],
                "outer_tail_decay_channel_passed": tail[
                    "decay_channel_passed"
                ],
                "outer_tail_channel_passed": tail["tail_channel_passed"],
                "total_outer_channel_passed": tail[
                    "total_outer_channel_passed"
                ],
            }
        )
        output[pairing] = base
    return output


def _unresolved_result(
    config: AdaptiveOuterTailCasimirConfig,
    *,
    cutoff_records: Sequence[Mapping[str, Any]],
    shell_records: Sequence[Mapping[str, Any]],
    pairing_results: Mapping[str, Any],
    selected_u_max: float | None,
    all_finite: bool,
    all_certified: bool,
    reason: str,
    provider: Any,
) -> AdaptiveOuterTailCasimirResult:
    return AdaptiveOuterTailCasimirResult(
        status="unresolved",
        config=config,
        cutoff_converged=False,
        outer_tail_estimated_flag=False,
        all_finite_domain_runs_converged=all_finite,
        all_microscopic_nodes_certified=all_certified,
        selected_u_max=selected_u_max,
        pairing_results=dict(pairing_results),
        cutoff_records=tuple(cutoff_records),
        shell_records=tuple(shell_records),
        termination_reason=reason,
        provider_statistics=_provider_statistics(provider),
    )


def run_adaptive_outer_tail_casimir(
    config: AdaptiveOuterTailCasimirConfig,
    *,
    provider: _PointProvider | None = None,
    joint_runner: _JointRunner = run_adaptive_joint_casimir,
) -> AdaptiveOuterTailCasimirResult:
    """Extend cumulative cutoffs until a channelwise tail envelope is established."""

    if not isinstance(config, AdaptiveOuterTailCasimirConfig):
        raise TypeError("config must be an AdaptiveOuterTailCasimirConfig")
    active_provider: Any = provider
    cutoff_records: list[Mapping[str, Any]] = []
    shell_records: list[Mapping[str, Any]] = []
    previous_values: dict[str, np.ndarray] | None = None
    previous_errors: dict[str, np.ndarray] | None = None
    previous_u = 0.0
    last_pairing_results: dict[str, Any] = {}
    all_finite = True
    all_certified = True
    pairings = tuple(config.joint_config.radial_config.point_config.pairings)
    indices = tuple(config.joint_config.radial_config.point_config.matsubara_indices)

    try:
        if active_provider is None:
            radial = config.joint_config.radial_config
            active_provider = CertifiedOuterQProvider(
                radial.point_config,
                cache_path=radial.point_cache_path,
            )
        for cutoff_index, u_max in enumerate(config.cutoff_u_values):
            run_config = _scaled_joint_config(config, cutoff_index=cutoff_index)
            result = joint_runner(run_config, provider=active_provider)
            usable, usable_reason = _joint_run_usable(result)
            all_finite = all_finite and usable
            all_certified = all_certified and bool(
                result.all_microscopic_nodes_certified
            )
            values: dict[str, np.ndarray] = {}
            finite_errors: dict[str, np.ndarray] = {}
            if usable:
                for pairing in pairings:
                    values[pairing], finite_errors[pairing] = _channel_arrays(
                        result,
                        pairing,
                        len(indices),
                    )
            cutoff_records.append(
                _cutoff_record(
                    result,
                    u_max=u_max,
                    finite_errors=finite_errors,
                )
            )
            last_pairing_results = dict(result.pairing_results)
            if not usable:
                return _unresolved_result(
                    config,
                    cutoff_records=cutoff_records,
                    shell_records=shell_records,
                    pairing_results=last_pairing_results,
                    selected_u_max=u_max,
                    all_finite=False,
                    all_certified=all_certified,
                    reason=f"finite_domain_run_unresolved: {usable_reason}",
                    provider=active_provider,
                )
            if int(getattr(active_provider, "unique_q_count", 0)) > config.max_total_microscopic_q_nodes:
                return _unresolved_result(
                    config,
                    cutoff_records=cutoff_records,
                    shell_records=shell_records,
                    pairing_results=last_pairing_results,
                    selected_u_max=u_max,
                    all_finite=all_finite,
                    all_certified=all_certified,
                    reason="outer_tail_microscopic_q_node_budget_exhausted",
                    provider=active_provider,
                )

            zero_values = {
                pairing: np.zeros(len(indices), dtype=float) for pairing in pairings
            }
            zero_errors = {
                pairing: np.zeros(len(indices), dtype=float) for pairing in pairings
            }
            shell_records.append(
                _shell_record(
                    left_u=previous_u,
                    right_u=u_max,
                    previous_values=(
                        zero_values if previous_values is None else previous_values
                    ),
                    current_values=values,
                    previous_errors=(
                        zero_errors if previous_errors is None else previous_errors
                    ),
                    current_errors=finite_errors,
                    indices=indices,
                )
            )
            metrics, decay_passed, tail_passed, total_passed = _tail_window_metrics(
                config,
                shell_records=shell_records,
                current_values=values,
                current_finite_errors=finite_errors,
                indices=indices,
            )
            if metrics is not None and decay_passed and tail_passed and total_passed:
                return AdaptiveOuterTailCasimirResult(
                    status="adaptive_finite_partial",
                    config=config,
                    cutoff_converged=True,
                    outer_tail_estimated_flag=True,
                    all_finite_domain_runs_converged=True,
                    all_microscopic_nodes_certified=True,
                    selected_u_max=u_max,
                    pairing_results=_final_pairing_results(result, metrics),
                    cutoff_records=tuple(cutoff_records),
                    shell_records=tuple(shell_records),
                    termination_reason="outer_cutoff_and_tail_tolerances_met",
                    provider_statistics=_provider_statistics(active_provider),
                )
            previous_values = values
            previous_errors = finite_errors
            previous_u = u_max

        final_metrics, decay_passed, tail_passed, total_passed = _tail_window_metrics(
            config,
            shell_records=shell_records,
            current_values=previous_values or {},
            current_finite_errors=previous_errors or {},
            indices=indices,
        )
        if final_metrics is None:
            reason = "outer_tail_window_not_established"
        elif not bool(final_metrics["equal_shell_widths"]):
            reason = "outer_tail_shell_width_contract_failed"
        elif not decay_passed:
            reason = "outer_tail_decay_ratio_not_established"
        elif not tail_passed:
            reason = "outer_tail_budget_not_met"
        elif not total_passed:
            reason = "total_outer_budget_not_met"
        else:
            reason = "outer_cutoff_ladder_exhausted"
        return _unresolved_result(
            config,
            cutoff_records=cutoff_records,
            shell_records=shell_records,
            pairing_results=last_pairing_results,
            selected_u_max=config.cutoff_u_values[-1],
            all_finite=all_finite,
            all_certified=all_certified,
            reason=reason,
            provider=active_provider,
        )
    except (CertifiedPointCacheError, FixedCasimirExecutionError) as exc:
        return _unresolved_result(
            config,
            cutoff_records=cutoff_records,
            shell_records=shell_records,
            pairing_results=last_pairing_results,
            selected_u_max=(
                None if not cutoff_records else float(cutoff_records[-1]["u_max"])
            ),
            all_finite=False,
            all_certified=False,
            reason=f"point_provider_failure: {exc}",
            provider=active_provider,
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return _unresolved_result(
            config,
            cutoff_records=cutoff_records,
            shell_records=shell_records,
            pairing_results=last_pairing_results,
            selected_u_max=(
                None if not cutoff_records else float(cutoff_records[-1]["u_max"])
            ),
            all_finite=False,
            all_certified=False,
            reason=f"outer_tail_result_contract_failure: {exc}",
            provider=active_provider,
        )


__all__ = [
    "AdaptiveOuterTailCasimirConfig",
    "AdaptiveOuterTailCasimirResult",
    "run_adaptive_outer_tail_casimir",
]
