"""Fail-closed adaptive Matsubara cutoff and high-frequency tail envelope.

This controller wraps the adaptive outer-Q cutoff/tail controller.  It grows a
cumulative, contiguous Matsubara set ``n = 0, ..., N``, reuses a frequency-extendable
certified microscopic-point provider, and establishes a channelwise geometric bound
for the omitted ``n > N`` contribution.  The controller is diagnostic-only and never
authorizes a production Casimir result.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from .adaptive_outer_tail import (
    AdaptiveOuterTailCasimirConfig,
    AdaptiveOuterTailCasimirResult,
    run_adaptive_outer_tail_casimir,
)
from .certified_point_provider import (
    CertifiedPointCacheError,
    FrequencyExtendableCertifiedOuterQProvider,
)
from .fixed_chain import FixedCasimirConfig, FixedCasimirExecutionError


class _FrequencyProvider(Protocol):
    cached_point_count: int
    unique_q_count: int
    certification_batches: int
    requested_q_evaluations: int
    new_q_evaluations: int
    cache_hit_q_evaluations: int
    requested_point_evaluations: int
    new_point_evaluations: int
    cache_hit_point_evaluations: int

    def reconfigure(self, config: FixedCasimirConfig) -> None: ...


class _OuterTailRunner(Protocol):
    def __call__(
        self,
        config: AdaptiveOuterTailCasimirConfig,
        *,
        provider: Any | None = None,
    ) -> AdaptiveOuterTailCasimirResult: ...


@dataclass(frozen=True)
class AdaptiveMatsubaraCasimirConfig:
    """Configuration for cumulative Matsubara cutoff and tail control."""

    outer_tail_config: AdaptiveOuterTailCasimirConfig = field(
        default_factory=AdaptiveOuterTailCasimirConfig
    )
    matsubara_cutoff_values: tuple[int, ...] = (1, 3, 7, 11, 15, 23, 31)
    total_free_energy_rtol: float = 5e-2
    total_free_energy_atol_J_m2: float = 1e-10
    finite_matsubara_budget_fraction: float = 0.7
    matsubara_tail_budget_fraction: float = 0.3
    tail_start_n: int = 8
    tail_window_terms: int = 4
    tail_ratio_max: float = 0.8
    max_total_microscopic_point_entries: int = 1_000_000
    point_cache_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outer_tail_config, AdaptiveOuterTailCasimirConfig):
            raise TypeError(
                "outer_tail_config must be an AdaptiveOuterTailCasimirConfig"
            )
        cutoffs = tuple(int(value) for value in self.matsubara_cutoff_values)
        if len(cutoffs) < 2 or any(value < 1 for value in cutoffs):
            raise ValueError(
                "matsubara_cutoff_values must contain at least two positive maxima"
            )
        if tuple(sorted(set(cutoffs))) != cutoffs:
            raise ValueError(
                "matsubara_cutoff_values must be strictly increasing and unique"
            )
        object.__setattr__(self, "matsubara_cutoff_values", cutoffs)

        for name in ("total_free_energy_rtol", "total_free_energy_atol_J_m2"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if (
            self.total_free_energy_rtol == 0.0
            and self.total_free_energy_atol_J_m2 == 0.0
        ):
            raise ValueError("at least one total free-energy tolerance must be positive")

        finite_fraction = float(self.finite_matsubara_budget_fraction)
        tail_fraction = float(self.matsubara_tail_budget_fraction)
        if finite_fraction <= 0.0 or tail_fraction <= 0.0:
            raise ValueError(
                "finite-Matsubara and Matsubara-tail budget fractions must be positive"
            )
        if not np.isclose(
            finite_fraction + tail_fraction,
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "finite-Matsubara and Matsubara-tail budget fractions must sum to one"
            )
        object.__setattr__(
            self,
            "finite_matsubara_budget_fraction",
            finite_fraction,
        )
        object.__setattr__(
            self,
            "matsubara_tail_budget_fraction",
            tail_fraction,
        )

        start = int(self.tail_start_n)
        if start < 1 or start > cutoffs[-1]:
            raise ValueError("tail_start_n must lie between one and the maximum cutoff")
        object.__setattr__(self, "tail_start_n", start)
        window = int(self.tail_window_terms)
        if window < 2:
            raise ValueError("tail_window_terms must be at least two")
        object.__setattr__(self, "tail_window_terms", window)
        if cutoffs[-1] - start + 1 < window:
            raise ValueError(
                "the Matsubara ladder does not provide enough high-frequency terms"
            )
        ratio = float(self.tail_ratio_max)
        if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
            raise ValueError("tail_ratio_max must lie strictly between zero and one")
        object.__setattr__(self, "tail_ratio_max", ratio)

        entries = int(self.max_total_microscopic_point_entries)
        if entries <= 0:
            raise ValueError("max_total_microscopic_point_entries must be positive")
        object.__setattr__(self, "max_total_microscopic_point_entries", entries)
        if self.point_cache_path is not None:
            object.__setattr__(self, "point_cache_path", Path(self.point_cache_path))

    @property
    def maximum_term_count(self) -> int:
        return self.matsubara_cutoff_values[-1] + 1

    @property
    def per_term_outer_budget_fraction(self) -> float:
        return self.finite_matsubara_budget_fraction / self.maximum_term_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_tail_config": self.outer_tail_config.as_dict(),
            "matsubara_cutoff_values": list(self.matsubara_cutoff_values),
            "total_free_energy_rtol": self.total_free_energy_rtol,
            "total_free_energy_atol_J_m2": self.total_free_energy_atol_J_m2,
            "finite_matsubara_budget_fraction": (
                self.finite_matsubara_budget_fraction
            ),
            "matsubara_tail_budget_fraction": self.matsubara_tail_budget_fraction,
            "per_term_outer_budget_fraction": self.per_term_outer_budget_fraction,
            "tail_start_n": self.tail_start_n,
            "tail_window_terms": self.tail_window_terms,
            "tail_ratio_max": self.tail_ratio_max,
            "max_total_microscopic_point_entries": (
                self.max_total_microscopic_point_entries
            ),
            "point_cache_path": (
                None if self.point_cache_path is None else str(self.point_cache_path)
            ),
        }


@dataclass(frozen=True)
class AdaptiveMatsubaraCasimirResult:
    """Outer- and Matsubara-tail bounded diagnostic Casimir result."""

    status: Literal["adaptive_tail_bounded", "unresolved"]
    config: AdaptiveMatsubaraCasimirConfig
    matsubara_converged: bool
    matsubara_tail_estimated_flag: bool
    all_outer_tail_runs_converged: bool
    all_microscopic_nodes_certified: bool
    selected_matsubara_cutoff: int | None
    pairing_results: Mapping[str, Any]
    cutoff_records: tuple[Mapping[str, Any], ...]
    termination_reason: str
    provider_statistics: Mapping[str, Any]

    @property
    def production_casimir_allowed(self) -> bool:
        return False

    @property
    def partial_sum_only(self) -> bool:
        return not self.matsubara_tail_estimated

    @property
    def outer_cutoff_adaptive(self) -> bool:
        return True

    @property
    def outer_tail_estimated(self) -> bool:
        return bool(self.all_outer_tail_runs_converged and self.cutoff_records)

    @property
    def matsubara_cutoff_adaptive(self) -> bool:
        return True

    @property
    def matsubara_tail_estimated(self) -> bool:
        return bool(self.matsubara_tail_estimated_flag)

    @property
    def unique_microscopic_q_node_count(self) -> int:
        return int(self.provider_statistics.get("unique_q_count", 0))

    @property
    def cached_microscopic_point_count(self) -> int:
        return int(self.provider_statistics.get("cached_point_count", 0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "adaptive-matsubara-casimir-result-v1",
            "status": self.status,
            "production_casimir_allowed": False,
            "partial_sum_only": self.partial_sum_only,
            "outer_cutoff_fixed": False,
            "outer_cutoff_adaptive": True,
            "outer_tail_estimated": self.outer_tail_estimated,
            "matsubara_cutoff_adaptive": True,
            "matsubara_tail_estimated": self.matsubara_tail_estimated,
            "matsubara_converged": self.matsubara_converged,
            "all_outer_tail_runs_converged": self.all_outer_tail_runs_converged,
            "all_microscopic_nodes_certified": self.all_microscopic_nodes_certified,
            "selected_matsubara_cutoff": self.selected_matsubara_cutoff,
            "config": self.config.as_dict(),
            "pairing_results": dict(self.pairing_results),
            "cutoff_records": [dict(value) for value in self.cutoff_records],
            "termination_reason": self.termination_reason,
            "provider_statistics": dict(self.provider_statistics),
            "unique_microscopic_q_node_count": self.unique_microscopic_q_node_count,
            "cached_microscopic_point_count": self.cached_microscopic_point_count,
        }


def _provider_statistics(provider: Any) -> dict[str, int]:
    names = (
        "cached_point_count",
        "unique_q_count",
        "certification_batches",
        "requested_q_evaluations",
        "new_q_evaluations",
        "cache_hit_q_evaluations",
        "requested_point_evaluations",
        "new_point_evaluations",
        "cache_hit_point_evaluations",
    )
    return {name: int(getattr(provider, name, 0)) for name in names}


def _point_config(
    config: AdaptiveMatsubaraCasimirConfig,
    cutoff: int,
) -> FixedCasimirConfig:
    base = config.outer_tail_config.joint_config.radial_config.point_config
    return replace(base, matsubara_indices=tuple(range(int(cutoff) + 1)))


def _scaled_outer_tail_config(
    config: AdaptiveMatsubaraCasimirConfig,
    cutoff: int,
) -> AdaptiveOuterTailCasimirConfig:
    point = _point_config(config, cutoff)
    outer = config.outer_tail_config
    radial = replace(outer.joint_config.radial_config, point_config=point)
    joint = replace(outer.joint_config, radial_config=radial)
    # At each cumulative cutoff, divide the finite-Matsubara allocation across
    # the terms actually present.  Using the final 32-term count at cutoff=1
    # over-constrains the first two terms by a factor of sixteen without improving
    # the final summed error guarantee.
    share = config.finite_matsubara_budget_fraction / (int(cutoff) + 1)
    return replace(
        outer,
        joint_config=joint,
        total_outer_rtol=config.total_free_energy_rtol * share,
        total_outer_atol_J_m2=config.total_free_energy_atol_J_m2 * share,
    )


def _outer_run_usable(result: AdaptiveOuterTailCasimirResult) -> tuple[bool, str]:
    if not bool(result.all_microscopic_nodes_certified):
        return False, str(result.termination_reason)
    if result.status != "adaptive_finite_partial" or not bool(result.cutoff_converged):
        return False, str(result.termination_reason)
    if not bool(result.outer_tail_estimated):
        return False, "outer_tail_unresolved"
    return True, "outer_cutoff_and_tail_tolerances_met"


def _channel_arrays(
    result: AdaptiveOuterTailCasimirResult,
    pairing: str,
    indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    payload = result.pairing_results[pairing]
    returned_indices = tuple(int(value) for value in payload["matsubara_indices"])
    if returned_indices != tuple(indices):
        raise ValueError("outer-tail result returned a different Matsubara index set")
    contributions = np.asarray(payload["contributions_J_m2"], dtype=float)
    errors = np.asarray(payload["estimated_total_outer_errors_J_m2"], dtype=float)
    expected = (len(indices),)
    if contributions.shape != expected or errors.shape != expected:
        raise ValueError("Matsubara contribution/error arrays have incompatible shapes")
    if not np.isfinite(contributions).all() or not np.isfinite(errors).all():
        raise ValueError("Matsubara contribution/error arrays must be finite")
    if np.any(errors < 0.0):
        raise ValueError("outer-Q error bounds must be non-negative")
    return contributions, errors


def _tail_metrics(
    config: AdaptiveMatsubaraCasimirConfig,
    *,
    values: Mapping[str, np.ndarray],
    outer_errors: Mapping[str, np.ndarray],
    indices: Sequence[int],
) -> tuple[dict[str, Any] | None, bool, bool, bool, bool]:
    index_array = np.asarray(indices, dtype=int)
    eligible_positions = np.flatnonzero(index_array >= config.tail_start_n)
    if len(eligible_positions) < config.tail_window_terms:
        return None, False, False, False, False
    positions = eligible_positions[-config.tail_window_terms :]
    selected_indices = index_array[positions]
    if not np.array_equal(
        np.diff(selected_indices),
        np.ones(len(selected_indices) - 1, dtype=int),
    ):
        raise ValueError("Matsubara tail window must contain consecutive indices")

    output: dict[str, Any] = {
        "window_indices": selected_indices.tolist(),
        "tail_ratio_max": config.tail_ratio_max,
        "pairings": {},
    }
    decay_all = True
    finite_all = True
    tail_all = True
    total_all = True
    for pairing in values:
        contributions = values[pairing]
        errors = outer_errors[pairing]
        amplitudes = np.abs(contributions) + errors
        window = amplitudes[positions]
        denominator = window[:-1]
        ratios = np.divide(
            window[1:],
            denominator,
            out=np.full_like(window[1:], np.inf),
            where=denominator > 0.0,
        )
        both_zero = (denominator == 0.0) & (window[1:] == 0.0)
        ratios[both_zero] = 0.0
        ratio_envelope = float(np.max(ratios))
        decay_passed = bool(ratio_envelope <= config.tail_ratio_max)
        tail_bound = float(
            window[-1]
            * config.tail_ratio_max
            / (1.0 - config.tail_ratio_max)
        )
        partial = float(np.sum(contributions))
        finite_error = float(np.sum(errors))
        total_tolerance = float(
            max(
                config.total_free_energy_atol_J_m2,
                config.total_free_energy_rtol * abs(partial),
            )
        )
        finite_tolerance = (
            config.finite_matsubara_budget_fraction * total_tolerance
        )
        tail_tolerance = config.matsubara_tail_budget_fraction * total_tolerance
        total_error = finite_error + tail_bound
        finite_passed = bool(finite_error <= finite_tolerance)
        tail_passed = bool(tail_bound <= tail_tolerance)
        total_passed = bool(total_error <= total_tolerance)
        decay_all = decay_all and decay_passed
        finite_all = finite_all and finite_passed
        tail_all = tail_all and tail_passed
        total_all = total_all and total_passed
        output["pairings"][pairing] = {
            "matsubara_indices": list(indices),
            "contributions_J_m2": contributions.tolist(),
            "outer_error_bounds_J_m2": errors.tolist(),
            "term_envelope_amplitudes_J_m2": amplitudes.tolist(),
            "window_envelope_amplitudes_J_m2": window.tolist(),
            "observed_tail_ratios": ratios.tolist(),
            "tail_ratio_envelope": ratio_envelope,
            "tail_decay_passed": decay_passed,
            "finite_matsubara_partial_J_m2": partial,
            "finite_matsubara_outer_error_bound_J_m2": finite_error,
            "estimated_matsubara_tail_bound_J_m2": tail_bound,
            "estimated_total_error_J_m2": total_error,
            "total_free_energy_tolerance_J_m2": total_tolerance,
            "finite_matsubara_budget_tolerance_J_m2": finite_tolerance,
            "matsubara_tail_budget_tolerance_J_m2": tail_tolerance,
            "finite_matsubara_budget_passed": finite_passed,
            "matsubara_tail_budget_passed": tail_passed,
            "total_free_energy_budget_passed": total_passed,
        }
    return output, decay_all, finite_all, tail_all, total_all


def _cutoff_record(
    result: AdaptiveOuterTailCasimirResult,
    *,
    cutoff: int,
    metrics: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "matsubara_cutoff": int(cutoff),
            "matsubara_indices": list(range(int(cutoff) + 1)),
            "status": str(result.status),
            "cutoff_converged": bool(result.cutoff_converged),
            "outer_tail_estimated": bool(result.outer_tail_estimated),
            "all_microscopic_nodes_certified": bool(
                result.all_microscopic_nodes_certified
            ),
            "selected_u_max": result.selected_u_max,
            "termination_reason": str(result.termination_reason),
            "pairing_results": dict(result.pairing_results),
            "matsubara_tail_metrics": (
                None if metrics is None else dict(metrics)
            ),
            "provider_statistics": dict(result.provider_statistics),
        }
    )


def _final_pairing_results(
    result: AdaptiveOuterTailCasimirResult,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pairing in result.config.joint_config.radial_config.point_config.pairings:
        base = dict(result.pairing_results[pairing])
        tail = dict(metrics["pairings"][pairing])
        base.update(
            {
                "status": "integrated_with_outer_and_matsubara_tail_bounds",
                "finite_matsubara_partial_J_m2": tail[
                    "finite_matsubara_partial_J_m2"
                ],
                "finite_matsubara_outer_error_bound_J_m2": tail[
                    "finite_matsubara_outer_error_bound_J_m2"
                ],
                "estimated_matsubara_tail_bound_J_m2": tail[
                    "estimated_matsubara_tail_bound_J_m2"
                ],
                "estimated_total_error_J_m2": tail[
                    "estimated_total_error_J_m2"
                ],
                "total_free_energy_tolerance_J_m2": tail[
                    "total_free_energy_tolerance_J_m2"
                ],
                "finite_matsubara_budget_tolerance_J_m2": tail[
                    "finite_matsubara_budget_tolerance_J_m2"
                ],
                "matsubara_tail_budget_tolerance_J_m2": tail[
                    "matsubara_tail_budget_tolerance_J_m2"
                ],
                "matsubara_term_envelope_amplitudes_J_m2": tail[
                    "term_envelope_amplitudes_J_m2"
                ],
                "matsubara_tail_ratio_envelope": tail[
                    "tail_ratio_envelope"
                ],
                "matsubara_tail_decay_passed": tail["tail_decay_passed"],
                "finite_matsubara_budget_passed": tail[
                    "finite_matsubara_budget_passed"
                ],
                "matsubara_tail_budget_passed": tail[
                    "matsubara_tail_budget_passed"
                ],
                "total_free_energy_budget_passed": tail[
                    "total_free_energy_budget_passed"
                ],
            }
        )
        output[pairing] = base
    return output


def _unresolved_result(
    config: AdaptiveMatsubaraCasimirConfig,
    *,
    cutoff_records: Sequence[Mapping[str, Any]],
    pairing_results: Mapping[str, Any],
    selected_cutoff: int | None,
    all_outer: bool,
    all_certified: bool,
    reason: str,
    provider: Any,
) -> AdaptiveMatsubaraCasimirResult:
    return AdaptiveMatsubaraCasimirResult(
        status="unresolved",
        config=config,
        matsubara_converged=False,
        matsubara_tail_estimated_flag=False,
        all_outer_tail_runs_converged=all_outer,
        all_microscopic_nodes_certified=all_certified,
        selected_matsubara_cutoff=selected_cutoff,
        pairing_results=dict(pairing_results),
        cutoff_records=tuple(cutoff_records),
        termination_reason=reason,
        provider_statistics=_provider_statistics(provider),
    )


def run_adaptive_matsubara_casimir(
    config: AdaptiveMatsubaraCasimirConfig,
    *,
    provider: _FrequencyProvider | None = None,
    outer_tail_runner: _OuterTailRunner = run_adaptive_outer_tail_casimir,
) -> AdaptiveMatsubaraCasimirResult:
    """Grow the Matsubara cutoff until a channelwise tail bound is established."""

    if not isinstance(config, AdaptiveMatsubaraCasimirConfig):
        raise TypeError("config must be an AdaptiveMatsubaraCasimirConfig")
    active_provider: Any = provider
    cutoff_records: list[Mapping[str, Any]] = []
    last_pairing_results: dict[str, Any] = {}
    all_outer = True
    all_certified = True
    last_metrics: dict[str, Any] | None = None
    last_decay = False
    last_finite = False
    last_tail = False
    last_total = False

    try:
        if active_provider is None:
            first_point = _point_config(config, config.matsubara_cutoff_values[0])
            active_provider = FrequencyExtendableCertifiedOuterQProvider(
                first_point,
                cache_path=config.point_cache_path,
            )

        for cutoff in config.matsubara_cutoff_values:
            run_config = _scaled_outer_tail_config(config, cutoff)
            point = run_config.joint_config.radial_config.point_config
            active_provider.reconfigure(point)
            result = outer_tail_runner(run_config, provider=active_provider)
            usable, usable_reason = _outer_run_usable(result)
            all_outer = all_outer and usable
            all_certified = all_certified and bool(
                result.all_microscopic_nodes_certified
            )
            last_pairing_results = dict(result.pairing_results)
            metrics: dict[str, Any] | None = None
            if usable:
                indices = tuple(point.matsubara_indices)
                values: dict[str, np.ndarray] = {}
                errors: dict[str, np.ndarray] = {}
                for pairing in point.pairings:
                    values[pairing], errors[pairing] = _channel_arrays(
                        result,
                        pairing,
                        indices,
                    )
                (
                    metrics,
                    last_decay,
                    last_finite,
                    last_tail,
                    last_total,
                ) = _tail_metrics(
                    config,
                    values=values,
                    outer_errors=errors,
                    indices=indices,
                )
                last_metrics = metrics
            cutoff_records.append(
                _cutoff_record(
                    result,
                    cutoff=cutoff,
                    metrics=metrics,
                )
            )
            if not usable:
                return _unresolved_result(
                    config,
                    cutoff_records=cutoff_records,
                    pairing_results=last_pairing_results,
                    selected_cutoff=cutoff,
                    all_outer=False,
                    all_certified=all_certified,
                    reason=f"outer_tail_run_unresolved: {usable_reason}",
                    provider=active_provider,
                )
            if (
                int(getattr(active_provider, "cached_point_count", 0))
                > config.max_total_microscopic_point_entries
            ):
                return _unresolved_result(
                    config,
                    cutoff_records=cutoff_records,
                    pairing_results=last_pairing_results,
                    selected_cutoff=cutoff,
                    all_outer=all_outer,
                    all_certified=all_certified,
                    reason="matsubara_microscopic_point_entry_budget_exhausted",
                    provider=active_provider,
                )
            if (
                metrics is not None
                and last_decay
                and last_finite
                and last_tail
                and last_total
            ):
                return AdaptiveMatsubaraCasimirResult(
                    status="adaptive_tail_bounded",
                    config=config,
                    matsubara_converged=True,
                    matsubara_tail_estimated_flag=True,
                    all_outer_tail_runs_converged=True,
                    all_microscopic_nodes_certified=True,
                    selected_matsubara_cutoff=cutoff,
                    pairing_results=_final_pairing_results(result, metrics),
                    cutoff_records=tuple(cutoff_records),
                    termination_reason=(
                        "outer_and_matsubara_cutoff_tail_tolerances_met"
                    ),
                    provider_statistics=_provider_statistics(active_provider),
                )

        if last_metrics is None:
            reason = "matsubara_tail_window_not_established"
        elif not last_decay:
            reason = "matsubara_tail_decay_ratio_not_established"
        elif not last_finite:
            reason = "finite_matsubara_outer_budget_not_met"
        elif not last_tail:
            reason = "matsubara_tail_budget_not_met"
        elif not last_total:
            reason = "total_free_energy_budget_not_met"
        else:
            reason = "matsubara_cutoff_ladder_exhausted"
        return _unresolved_result(
            config,
            cutoff_records=cutoff_records,
            pairing_results=last_pairing_results,
            selected_cutoff=config.matsubara_cutoff_values[-1],
            all_outer=all_outer,
            all_certified=all_certified,
            reason=reason,
            provider=active_provider,
        )
    except (CertifiedPointCacheError, FixedCasimirExecutionError) as exc:
        return _unresolved_result(
            config,
            cutoff_records=cutoff_records,
            pairing_results=last_pairing_results,
            selected_cutoff=(
                None
                if not cutoff_records
                else int(cutoff_records[-1]["matsubara_cutoff"])
            ),
            all_outer=False,
            all_certified=False,
            reason=f"point_provider_failure: {exc}",
            provider=active_provider,
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return _unresolved_result(
            config,
            cutoff_records=cutoff_records,
            pairing_results=last_pairing_results,
            selected_cutoff=(
                None
                if not cutoff_records
                else int(cutoff_records[-1]["matsubara_cutoff"])
            ),
            all_outer=False,
            all_certified=False,
            reason=f"matsubara_result_contract_failure: {exc}",
            provider=active_provider,
        )


__all__ = [
    "AdaptiveMatsubaraCasimirConfig",
    "AdaptiveMatsubaraCasimirResult",
    "run_adaptive_matsubara_casimir",
]
