"""Production Matsubara cutoff control using dyadic block envelopes and holdout.

The legacy per-term geometric ratio is intentionally not used for formal acceptance.
Terms are grouped into cumulative dyadic blocks, an absolute block envelope prevents
sign cancellation, and the final block is an explicit holdout for the fixed future-block
ratio contract. Every already-computed Matsubara term must carry a certified outer-Q
error bound before the frequency tail can be considered.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .adaptive_matsubara_tail import (
    AdaptiveMatsubaraCasimirConfig,
    _channel_arrays,
    _cutoff_record,
    _outer_run_usable,
    _point_config,
    _provider_statistics,
    _scaled_outer_tail_config,
)
from .certified_point_provider import (
    CertifiedPointCacheError,
    FrequencyExtendableCertifiedOuterQProvider,
)
from .certified_tail import run_certified_outer_tail_casimir
from .fixed_chain import FixedCasimirExecutionError

MATSUBARA_TAIL_CERTIFICATE_CONTRACT = "dyadic-block-holdout-envelope-v1"
PRODUCTION_ERROR_BUDGET_CONTRACT = "full-casimir-error-budget-v1"


@dataclass(frozen=True)
class CertifiedMatsubaraCasimirResult:
    """Full result whose convergence flag already includes formal policy closure."""

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
    formal_policy_passed: bool
    error_budget_closed: bool
    certificate_contract_version: str = MATSUBARA_TAIL_CERTIFICATE_CONTRACT

    @property
    def production_casimir_allowed(self) -> bool:
        return bool(
            self.status == "adaptive_tail_bounded"
            and self.matsubara_converged
            and self.matsubara_tail_estimated_flag
            and self.all_outer_tail_runs_converged
            and self.all_microscopic_nodes_certified
            and self.formal_policy_passed
            and self.error_budget_closed
        )

    @property
    def partial_sum_only(self) -> bool:
        return not self.matsubara_tail_estimated_flag

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
            "production_casimir_allowed": self.production_casimir_allowed,
            "formal_policy_passed": self.formal_policy_passed,
            "error_budget_closed": self.error_budget_closed,
            "certificate_contract_version": self.certificate_contract_version,
            "production_error_budget_contract": PRODUCTION_ERROR_BUDGET_CONTRACT,
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


def validate_dyadic_matsubara_policy(
    cutoffs: Sequence[int],
    *,
    tail_start_n: int,
    tail_window_blocks: int,
) -> tuple[int, ...]:
    """Validate the formal dyadic block ladder ``1, 3, 7, 15, ...``."""

    values = tuple(int(value) for value in cutoffs)
    if len(values) < 3 or values[0] != 1:
        raise ValueError(
            "formal Matsubara cutoff ladder must start at 1 and contain at least three cutoffs"
        )
    if any(
        right != 2 * left + 1
        for left, right in zip(values[:-1], values[1:], strict=True)
    ):
        raise ValueError(
            "formal Matsubara cutoffs must follow the dyadic sequence 1,3,7,15,..."
        )
    starts = (0, *(value + 1 for value in values[:-1]))
    if int(tail_start_n) not in starts:
        raise ValueError(
            "matsubara tail_start_n must be the left edge of a dyadic block"
        )
    if int(tail_window_blocks) < 3:
        raise ValueError(
            "formal Matsubara tail window requires at least two training blocks and one holdout block"
        )
    eligible = [left for left in starts if left >= int(tail_start_n)]
    if len(eligible) < int(tail_window_blocks):
        raise ValueError("Matsubara ladder does not contain enough dyadic tail blocks")
    return values


def _block_ranges(cutoffs: Sequence[int]) -> tuple[tuple[int, int], ...]:
    previous = -1
    blocks: list[tuple[int, int]] = []
    for cutoff in cutoffs:
        right = int(cutoff)
        blocks.append((previous + 1, right))
        previous = right
    return tuple(blocks)


def _tail_metrics(
    config: AdaptiveMatsubaraCasimirConfig,
    *,
    values: Mapping[str, np.ndarray],
    outer_errors: Mapping[str, np.ndarray],
    indices: Sequence[int],
    completed_cutoffs: Sequence[int],
) -> tuple[dict[str, Any] | None, bool, bool, bool, bool]:
    blocks = _block_ranges(completed_cutoffs)
    eligible_positions = [
        position
        for position, (left, _right) in enumerate(blocks)
        if left >= int(config.tail_start_n)
    ]
    if len(eligible_positions) < int(config.tail_window_terms):
        return None, False, False, False, False
    positions = eligible_positions[-int(config.tail_window_terms) :]
    window_blocks = [blocks[position] for position in positions]
    output: dict[str, Any] = {
        "schema": "matsubara-dyadic-block-tail-metrics-v1",
        "certificate_path": "validated_dyadic_block_holdout",
        "certificate_contract": MATSUBARA_TAIL_CERTIFICATE_CONTRACT,
        "tail_ratio_max": float(config.tail_ratio_max),
        "tail_start_n": int(config.tail_start_n),
        "window_blocks": [[left, right] for left, right in window_blocks],
        "training_blocks": [[left, right] for left, right in window_blocks[:-1]],
        "holdout_block": list(window_blocks[-1]),
        "pairings": {},
    }
    decay_all = True
    finite_all = True
    tail_all = True
    total_all = True
    for pairing in values:
        contributions = np.asarray(values[pairing], dtype=float)
        errors = np.asarray(outer_errors[pairing], dtype=float)
        amplitudes = np.abs(contributions) + errors
        block_amplitudes = np.asarray(
            [float(np.sum(amplitudes[left : right + 1])) for left, right in blocks],
            dtype=float,
        )
        window = block_amplitudes[positions]
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
        holdout_ratio = float(ratios[-1])
        training_ratio_envelope = (
            float(np.max(ratios[:-1])) if len(ratios) > 1 else 0.0
        )
        decay_passed = bool(ratio_envelope <= config.tail_ratio_max)
        holdout_passed = bool(holdout_ratio <= config.tail_ratio_max)
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
        finite_tolerance = float(
            config.finite_matsubara_budget_fraction * total_tolerance
        )
        tail_tolerance = float(
            config.matsubara_tail_budget_fraction * total_tolerance
        )
        total_error = finite_error + tail_bound
        finite_passed = bool(finite_error <= finite_tolerance)
        tail_passed = bool(tail_bound <= tail_tolerance)
        total_passed = bool(total_error <= total_tolerance)
        pairing_decay = bool(decay_passed and holdout_passed)
        decay_all = decay_all and pairing_decay
        finite_all = finite_all and finite_passed
        tail_all = tail_all and tail_passed
        total_all = total_all and total_passed

        term_denominator = amplitudes[:-1]
        term_ratios = np.divide(
            amplitudes[1:],
            term_denominator,
            out=np.full_like(amplitudes[1:], np.inf),
            where=term_denominator > 0.0,
        )
        term_zero = (term_denominator == 0.0) & (amplitudes[1:] == 0.0)
        term_ratios[term_zero] = 0.0
        output["pairings"][pairing] = {
            "matsubara_indices": list(indices),
            "contributions_J_m2": contributions.tolist(),
            "outer_error_bounds_J_m2": errors.tolist(),
            "term_envelope_amplitudes_J_m2": amplitudes.tolist(),
            "term_ratios_diagnostic_only": term_ratios.tolist(),
            "blocks": [
                {
                    "left_n": left,
                    "right_n": right,
                    "width": right - left + 1,
                    "envelope_amplitude_J_m2": float(block_amplitudes[position]),
                }
                for position, (left, right) in enumerate(blocks)
            ],
            "window_block_envelope_amplitudes_J_m2": window.tolist(),
            "observed_block_ratios": ratios.tolist(),
            "training_block_ratio_envelope": training_ratio_envelope,
            "holdout_block_ratio": holdout_ratio,
            "block_ratio_envelope": ratio_envelope,
            "tail_decay_passed": decay_passed,
            "holdout_passed": holdout_passed,
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
            "certificate_passed": bool(
                pairing_decay and finite_passed and tail_passed and total_passed
            ),
        }
    output["certificate_passed"] = bool(
        decay_all and finite_all and tail_all and total_all
    )
    return output, decay_all, finite_all, tail_all, total_all


def _final_pairing_results(result: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
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
                "estimated_total_error_J_m2": tail["estimated_total_error_J_m2"],
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
                "matsubara_block_records": tail["blocks"],
                "matsubara_tail_block_ratio_envelope": tail[
                    "block_ratio_envelope"
                ],
                "matsubara_tail_holdout_ratio": tail["holdout_block_ratio"],
                "matsubara_tail_decay_passed": tail["tail_decay_passed"],
                "matsubara_tail_holdout_passed": tail["holdout_passed"],
                "finite_matsubara_budget_passed": tail[
                    "finite_matsubara_budget_passed"
                ],
                "matsubara_tail_budget_passed": tail[
                    "matsubara_tail_budget_passed"
                ],
                "total_free_energy_budget_passed": tail[
                    "total_free_energy_budget_passed"
                ],
                "matsubara_tail_certificate_path": "validated_dyadic_block_holdout",
                "matsubara_tail_certificate_contract": MATSUBARA_TAIL_CERTIFICATE_CONTRACT,
                "matsubara_tail_ratio_envelope": tail["block_ratio_envelope"],
            }
        )
        output[str(pairing)] = base
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
) -> CertifiedMatsubaraCasimirResult:
    return CertifiedMatsubaraCasimirResult(
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
        formal_policy_passed=False,
        error_budget_closed=False,
    )


def run_certified_matsubara_casimir(
    config: AdaptiveMatsubaraCasimirConfig,
    *,
    provider: Any | None = None,
    outer_tail_runner: Any = run_certified_outer_tail_casimir,
) -> CertifiedMatsubaraCasimirResult:
    """Grow a dyadic cutoff ladder until block holdout and all budgets pass."""

    if not isinstance(config, AdaptiveMatsubaraCasimirConfig):
        raise TypeError("config must be an AdaptiveMatsubaraCasimirConfig")
    try:
        policy_cutoffs = validate_dyadic_matsubara_policy(
            config.matsubara_cutoff_values,
            tail_start_n=config.tail_start_n,
            tail_window_blocks=config.tail_window_terms,
        )
    except ValueError as exc:
        return _unresolved_result(
            config,
            cutoff_records=(),
            pairing_results={},
            selected_cutoff=None,
            all_outer=False,
            all_certified=False,
            reason=f"matsubara_certificate_policy_invalid: {exc}",
            provider=provider,
        )

    active_provider = provider
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
            first_point = _point_config(config, policy_cutoffs[0])
            active_provider = FrequencyExtendableCertifiedOuterQProvider(
                first_point,
                cache_path=config.point_cache_path,
                certifier_q_batch_size=config.certifier_q_batch_size,
            )

        completed_cutoffs: list[int] = []
        for cutoff in policy_cutoffs:
            completed_cutoffs.append(int(cutoff))
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
                    completed_cutoffs=completed_cutoffs,
                )
                last_metrics = metrics
            record = dict(
                _cutoff_record(
                    result,
                    cutoff=cutoff,
                    metrics=metrics,
                )
            )
            record["matsubara_certificate_contract"] = (
                MATSUBARA_TAIL_CERTIFICATE_CONTRACT
            )
            record["formal_policy_passed"] = bool(
                metrics is not None
                and last_decay
                and last_finite
                and last_tail
                and last_total
            )
            cutoff_records.append(MappingProxyType(record))
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
                return CertifiedMatsubaraCasimirResult(
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
                        "outer_and_matsubara_certificates_and_total_budget_met"
                    ),
                    provider_statistics=_provider_statistics(active_provider),
                    formal_policy_passed=True,
                    error_budget_closed=True,
                )

        if last_metrics is None:
            reason = "matsubara_dyadic_tail_window_not_established"
        elif not last_decay:
            reason = "matsubara_block_decay_or_holdout_not_established"
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
            selected_cutoff=policy_cutoffs[-1],
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
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
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
            reason=f"matsubara_certificate_contract_failure: {exc}",
            provider=active_provider,
        )


__all__ = [
    "CertifiedMatsubaraCasimirResult",
    "MATSUBARA_TAIL_CERTIFICATE_CONTRACT",
    "PRODUCTION_ERROR_BUDGET_CONTRACT",
    "run_certified_matsubara_casimir",
    "validate_dyadic_matsubara_policy",
]
