"""Fail-closed global angular-order adaptivity over the radial controller.

This module keeps the finite radial domain, microscopic policy, Matsubara set and
full periodic angular interval fixed. It increases the global periodic angular
order on a strict doubling ladder, then performs an independent final-order offset
audit. Neither the omitted outer-Q tail nor the Matsubara tail is estimated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from .adaptive_outer_q import (
    AdaptiveRadialCasimirConfig,
    AdaptiveRadialCasimirResult,
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
class AdaptiveAngularCasimirConfig:
    """Configuration for global periodic angular-order adaptivity."""

    radial_config: AdaptiveRadialCasimirConfig = field(
        default_factory=AdaptiveRadialCasimirConfig
    )
    angular_orders: tuple[int, ...] = (4, 8, 16, 32)
    primary_offset_fraction: float = 0.5
    audit_offset_fraction: float = 0.0
    angular_rtol: float = 5e-2
    angular_atol_J_m2: float = 1e-10
    offset_rtol: float = 5e-2
    offset_atol_J_m2: float = 1e-10
    required_consecutive_passes: int = 1

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

        for name in (
            "angular_rtol",
            "angular_atol_J_m2",
            "offset_rtol",
            "offset_atol_J_m2",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.angular_rtol == 0.0 and self.angular_atol_J_m2 == 0.0:
            raise ValueError("at least one angular-order tolerance must be positive")
        if self.offset_rtol == 0.0 and self.offset_atol_J_m2 == 0.0:
            raise ValueError("at least one angular-offset tolerance must be positive")

        passes = int(self.required_consecutive_passes)
        if passes <= 0 or passes >= len(orders):
            raise ValueError(
                "required_consecutive_passes must be positive and smaller than "
                "the angular-order ladder"
            )
        object.__setattr__(self, "required_consecutive_passes", passes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "radial_config": self.radial_config.as_dict(),
            "angular_orders": list(self.angular_orders),
            "primary_offset_fraction": self.primary_offset_fraction,
            "audit_offset_fraction": self.audit_offset_fraction,
            "angular_rtol": self.angular_rtol,
            "angular_atol_J_m2": self.angular_atol_J_m2,
            "offset_rtol": self.offset_rtol,
            "offset_atol_J_m2": self.offset_atol_J_m2,
            "required_consecutive_passes": self.required_consecutive_passes,
        }


@dataclass(frozen=True)
class AdaptiveAngularCasimirResult:
    """Fail-closed finite Matsubara result after angular-order and offset gates."""

    status: Literal["adaptive_finite_partial", "unresolved"]
    config: AdaptiveAngularCasimirConfig
    angular_converged: bool
    offset_audit_passed: bool
    all_radial_runs_converged: bool
    all_microscopic_nodes_certified: bool
    selected_angular_order: int | None
    pairing_results: Mapping[str, Any]
    angular_order_records: tuple[Mapping[str, Any], ...]
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
            "schema": "adaptive-angular-casimir-result-v1",
            "status": self.status,
            "production_casimir_allowed": False,
            "partial_sum_only": True,
            "outer_cutoff_fixed": True,
            "outer_tail_estimated": False,
            "matsubara_tail_estimated": False,
            "angular_converged": self.angular_converged,
            "offset_audit_passed": self.offset_audit_passed,
            "all_radial_runs_converged": self.all_radial_runs_converged,
            "all_microscopic_nodes_certified": self.all_microscopic_nodes_certified,
            "selected_angular_order": self.selected_angular_order,
            "config": self.config.as_dict(),
            "pairing_results": dict(self.pairing_results),
            "angular_order_records": [
                dict(record) for record in self.angular_order_records
            ],
            "offset_audit_record": (
                None
                if self.offset_audit_record is None
                else dict(self.offset_audit_record)
            ),
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


def _radial_run_record(
    result: AdaptiveRadialCasimirResult,
    *,
    angular_order: int,
    offset_fraction: float,
    comparison_to_previous: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "angular_order": int(angular_order),
            "angular_offset_fraction": float(offset_fraction),
            "status": result.status,
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
            "comparison_to_previous": (
                None
                if comparison_to_previous is None
                else dict(comparison_to_previous)
            ),
        }
    )


def _channel_comparison(
    previous: AdaptiveRadialCasimirResult,
    current: AdaptiveRadialCasimirResult,
    *,
    rtol: float,
    atol_J_m2: float,
) -> tuple[dict[str, Any], bool]:
    previous_pairings = tuple(previous.config.point_config.pairings)
    current_pairings = tuple(current.config.point_config.pairings)
    if previous_pairings != current_pairings:
        raise ValueError("radial runs returned different pairing orders")
    previous_indices = tuple(previous.config.point_config.matsubara_indices)
    current_indices = tuple(current.config.point_config.matsubara_indices)
    if previous_indices != current_indices:
        raise ValueError("radial runs returned different Matsubara index sets")

    output: dict[str, Any] = {}
    all_passed = True
    for pairing in current_pairings:
        left = np.asarray(
            previous.pairing_results[pairing]["contributions_J_m2"],
            dtype=float,
        )
        right = np.asarray(
            current.pairing_results[pairing]["contributions_J_m2"],
            dtype=float,
        )
        if left.shape != right.shape or left.shape != (len(current_indices),):
            raise ValueError("radial contribution arrays have incompatible shapes")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("radial contribution arrays must be finite")
        absolute = np.abs(right - left)
        scale = np.maximum(np.abs(left), np.abs(right))
        relative = np.divide(
            absolute,
            np.maximum(scale, np.finfo(float).tiny),
        )
        tolerance = np.maximum(float(atol_J_m2), float(rtol) * scale)
        passed = absolute <= tolerance
        all_passed = all_passed and bool(np.all(passed))
        output[pairing] = {
            "previous_contributions_J_m2": left.tolist(),
            "current_contributions_J_m2": right.tolist(),
            "absolute_differences_J_m2": absolute.tolist(),
            "relative_differences": relative.tolist(),
            "tolerances_J_m2": tolerance.tolist(),
            "channel_passed": passed.tolist(),
            "all_passed": bool(np.all(passed)),
            "matsubara_indices": list(current_indices),
        }
    return output, all_passed


def _final_pairing_results(
    primary: AdaptiveRadialCasimirResult,
    *,
    angular_comparison: Mapping[str, Any],
    offset_comparison: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pairing in primary.config.point_config.pairings:
        base = dict(primary.pairing_results[pairing])
        angular = dict(angular_comparison[pairing])
        offset = (
            None
            if offset_comparison is None
            else dict(offset_comparison[pairing])
        )
        angular_passed = bool(angular["all_passed"])
        offset_passed = offset is not None and bool(offset["all_passed"])
        base.update(
            {
                "status": (
                    "integrated"
                    if angular_passed and offset_passed
                    else "angular_unresolved"
                ),
                "estimated_angular_errors_J_m2": angular[
                    "absolute_differences_J_m2"
                ],
                "angular_relative_differences": angular["relative_differences"],
                "angular_tolerances_J_m2": angular["tolerances_J_m2"],
                "angular_channel_passed": angular["channel_passed"],
                "estimated_offset_errors_J_m2": (
                    None
                    if offset is None
                    else offset["absolute_differences_J_m2"]
                ),
                "offset_relative_differences": (
                    None if offset is None else offset["relative_differences"]
                ),
                "offset_tolerances_J_m2": (
                    None if offset is None else offset["tolerances_J_m2"]
                ),
                "offset_channel_passed": (
                    None if offset is None else offset["channel_passed"]
                ),
            }
        )
        output[pairing] = base
    return output


def _unresolved_result(
    config: AdaptiveAngularCasimirConfig,
    *,
    order_records: Sequence[Mapping[str, Any]],
    offset_record: Mapping[str, Any] | None,
    selected_order: int | None,
    pairing_results: Mapping[str, Any],
    angular_converged: bool,
    offset_passed: bool,
    all_radial_converged: bool,
    all_certified: bool,
    reason: str,
    provider: Any,
) -> AdaptiveAngularCasimirResult:
    return AdaptiveAngularCasimirResult(
        status="unresolved",
        config=config,
        angular_converged=angular_converged,
        offset_audit_passed=offset_passed,
        all_radial_runs_converged=all_radial_converged,
        all_microscopic_nodes_certified=all_certified,
        selected_angular_order=selected_order,
        pairing_results=dict(pairing_results),
        angular_order_records=tuple(order_records),
        offset_audit_record=offset_record,
        termination_reason=reason,
        provider_statistics=_provider_statistics(provider),
    )


def run_adaptive_angular_casimir(
    config: AdaptiveAngularCasimirConfig,
    *,
    provider: _PointProvider | None = None,
    radial_runner: _RadialRunner = run_adaptive_radial_casimir,
) -> AdaptiveAngularCasimirResult:
    """Increase periodic angular order, then perform a final-order offset audit."""

    if not isinstance(config, AdaptiveAngularCasimirConfig):
        raise TypeError("config must be an AdaptiveAngularCasimirConfig")
    active_provider: Any = provider
    order_records: list[Mapping[str, Any]] = []
    previous: AdaptiveRadialCasimirResult | None = None
    selected: AdaptiveRadialCasimirResult | None = None
    selected_order: int | None = None
    selected_comparison: Mapping[str, Any] | None = None
    consecutive = 0
    all_radial_converged = True
    all_certified = True

    try:
        if active_provider is None:
            active_provider = CertifiedOuterQProvider(
                config.radial_config.point_config,
                cache_path=config.radial_config.point_cache_path,
            )
        for order in config.angular_orders:
            radial_config = replace(
                config.radial_config,
                angular_order=order,
                angular_offset_fraction=config.primary_offset_fraction,
            )
            current = radial_runner(radial_config, provider=active_provider)
            all_radial_converged = (
                all_radial_converged
                and current.status == "adaptive_finite_partial"
                and bool(current.radial_converged)
            )
            all_certified = (
                all_certified and bool(current.all_microscopic_nodes_certified)
            )
            if (
                current.status != "adaptive_finite_partial"
                or not current.radial_converged
                or not current.all_microscopic_nodes_certified
            ):
                order_records.append(
                    _radial_run_record(
                        current,
                        angular_order=order,
                        offset_fraction=config.primary_offset_fraction,
                    )
                )
                return _unresolved_result(
                    config,
                    order_records=order_records,
                    offset_record=None,
                    selected_order=None,
                    pairing_results=current.pairing_results,
                    angular_converged=False,
                    offset_passed=False,
                    all_radial_converged=False,
                    all_certified=all_certified,
                    reason="radial_run_unresolved",
                    provider=active_provider,
                )

            comparison = None
            comparison_passed = False
            if previous is not None:
                comparison, comparison_passed = _channel_comparison(
                    previous,
                    current,
                    rtol=config.angular_rtol,
                    atol_J_m2=config.angular_atol_J_m2,
                )
                consecutive = consecutive + 1 if comparison_passed else 0
            order_records.append(
                _radial_run_record(
                    current,
                    angular_order=order,
                    offset_fraction=config.primary_offset_fraction,
                    comparison_to_previous=comparison,
                )
            )
            if (
                previous is not None
                and consecutive >= config.required_consecutive_passes
            ):
                selected = current
                selected_order = order
                selected_comparison = comparison
                break
            previous = current

        if selected is None or selected_order is None or selected_comparison is None:
            pairing_results = {} if previous is None else dict(previous.pairing_results)
            return _unresolved_result(
                config,
                order_records=order_records,
                offset_record=None,
                selected_order=None,
                pairing_results=pairing_results,
                angular_converged=False,
                offset_passed=False,
                all_radial_converged=all_radial_converged,
                all_certified=all_certified,
                reason="angular_order_ladder_exhausted",
                provider=active_provider,
            )

        audit_config = replace(
            config.radial_config,
            angular_order=selected_order,
            angular_offset_fraction=config.audit_offset_fraction,
        )
        audit = radial_runner(audit_config, provider=active_provider)
        all_radial_converged = (
            all_radial_converged
            and audit.status == "adaptive_finite_partial"
            and bool(audit.radial_converged)
        )
        all_certified = all_certified and bool(audit.all_microscopic_nodes_certified)
        offset_record = _radial_run_record(
            audit,
            angular_order=selected_order,
            offset_fraction=config.audit_offset_fraction,
        )
        if (
            audit.status != "adaptive_finite_partial"
            or not audit.radial_converged
            or not audit.all_microscopic_nodes_certified
        ):
            pairing_results = _final_pairing_results(
                selected,
                angular_comparison=selected_comparison,
                offset_comparison=None,
            )
            return _unresolved_result(
                config,
                order_records=order_records,
                offset_record=offset_record,
                selected_order=selected_order,
                pairing_results=pairing_results,
                angular_converged=True,
                offset_passed=False,
                all_radial_converged=False,
                all_certified=all_certified,
                reason="offset_audit_radial_unresolved",
                provider=active_provider,
            )

        offset_comparison, offset_passed = _channel_comparison(
            selected,
            audit,
            rtol=config.offset_rtol,
            atol_J_m2=config.offset_atol_J_m2,
        )
        offset_record = MappingProxyType(
            {
                **dict(offset_record),
                "comparison_to_primary_offset": offset_comparison,
            }
        )
        pairing_results = _final_pairing_results(
            selected,
            angular_comparison=selected_comparison,
            offset_comparison=offset_comparison,
        )
        if not offset_passed:
            return _unresolved_result(
                config,
                order_records=order_records,
                offset_record=offset_record,
                selected_order=selected_order,
                pairing_results=pairing_results,
                angular_converged=True,
                offset_passed=False,
                all_radial_converged=all_radial_converged,
                all_certified=all_certified,
                reason="angular_offset_audit_failed",
                provider=active_provider,
            )

        return AdaptiveAngularCasimirResult(
            status="adaptive_finite_partial",
            config=config,
            angular_converged=True,
            offset_audit_passed=True,
            all_radial_runs_converged=True,
            all_microscopic_nodes_certified=True,
            selected_angular_order=selected_order,
            pairing_results=pairing_results,
            angular_order_records=tuple(order_records),
            offset_audit_record=offset_record,
            termination_reason="angular_order_and_offset_tolerances_met",
            provider_statistics=_provider_statistics(active_provider),
        )
    except (CertifiedPointCacheError, FixedCasimirExecutionError) as exc:
        return _unresolved_result(
            config,
            order_records=order_records,
            offset_record=None,
            selected_order=selected_order,
            pairing_results={},
            angular_converged=False,
            offset_passed=False,
            all_radial_converged=False,
            all_certified=False,
            reason=f"point_provider_failure: {exc}",
            provider=active_provider,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _unresolved_result(
            config,
            order_records=order_records,
            offset_record=None,
            selected_order=selected_order,
            pairing_results={},
            angular_converged=False,
            offset_passed=False,
            all_radial_converged=False,
            all_certified=False,
            reason=f"radial_result_contract_failure: {exc}",
            provider=active_provider,
        )


__all__ = [
    "AdaptiveAngularCasimirConfig",
    "AdaptiveAngularCasimirResult",
    "run_adaptive_angular_casimir",
]
