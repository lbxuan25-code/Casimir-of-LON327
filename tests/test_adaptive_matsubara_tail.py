from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from lno327.casimir import (
    AdaptiveJointCasimirConfig,
    AdaptiveMatsubaraCasimirConfig,
    AdaptiveOuterTailCasimirConfig,
    AdaptiveOuterTailCasimirResult,
    AdaptiveRadialCasimirConfig,
    FixedCasimirConfig,
    run_adaptive_matsubara_casimir,
)


class _FrequencyProvider:
    def __init__(self) -> None:
        self.configs: list[tuple[int, ...]] = []
        self.cached_point_count = 0
        self.unique_q_count = 0
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0
        self.requested_point_evaluations = 0
        self.new_point_evaluations = 0
        self.cache_hit_point_evaluations = 0

    def reconfigure(self, config) -> None:
        self.configs.append(tuple(config.matsubara_indices))


class _ScriptedOuterRunner:
    def __init__(
        self,
        term_function,
        *,
        error_function=lambda n: 1e-15,
        unresolved_cutoff: int | None = None,
    ) -> None:
        self.term_function = term_function
        self.error_function = error_function
        self.unresolved_cutoff = unresolved_cutoff
        self.provider_ids: list[int] = []

    def __call__(self, config, *, provider=None):
        self.provider_ids.append(id(provider))
        point = config.joint_config.radial_config.point_config
        indices = tuple(point.matsubara_indices)
        cutoff = indices[-1]
        unresolved = self.unresolved_cutoff == cutoff
        values = np.asarray([self.term_function(n) for n in indices], dtype=float)
        errors = np.asarray([self.error_function(n) for n in indices], dtype=float)
        provider.certification_batches += 1
        provider.requested_q_evaluations += 1
        provider.requested_point_evaluations += len(indices)
        provider.cached_point_count = max(provider.cached_point_count, len(indices))
        payload = {
            pairing: {
                "status": "integrated_with_outer_tail_bound",
                "partial_free_energy_J_m2": float(np.sum(values)),
                "contributions_J_m2": values.tolist(),
                "estimated_total_outer_errors_J_m2": errors.tolist(),
                "matsubara_indices": list(indices),
            }
            for pairing in point.pairings
        }
        return AdaptiveOuterTailCasimirResult(
            status="unresolved" if unresolved else "adaptive_finite_partial",
            config=config,
            cutoff_converged=not unresolved,
            outer_tail_estimated_flag=not unresolved,
            all_finite_domain_runs_converged=not unresolved,
            all_microscopic_nodes_certified=not unresolved,
            selected_u_max=config.cutoff_u_values[-1],
            pairing_results=MappingProxyType(payload),
            cutoff_records=(),
            shell_records=(),
            termination_reason=(
                "microscopic_point_unresolved"
                if unresolved
                else "outer_cutoff_and_tail_tolerances_met"
            ),
            provider_statistics=MappingProxyType(
                {
                    "cached_point_count": provider.cached_point_count,
                    "unique_q_count": provider.unique_q_count,
                    "certification_batches": provider.certification_batches,
                    "requested_q_evaluations": provider.requested_q_evaluations,
                    "new_q_evaluations": provider.new_q_evaluations,
                    "cache_hit_q_evaluations": provider.cache_hit_q_evaluations,
                }
            ),
        )


def _outer_config() -> AdaptiveOuterTailCasimirConfig:
    point = FixedCasimirConfig(
        matsubara_indices=(0, 1),
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
    )
    radial = AdaptiveRadialCasimirConfig(
        point_config=point,
        initial_panel_edges=(0.0, 1.0),
        radial_order=1,
        angular_order=2,
        max_refinement_rounds=1,
        max_panel_depth=2,
        max_microscopic_q_nodes=1000,
    )
    joint = AdaptiveJointCasimirConfig(
        radial_config=radial,
        angular_orders=(2, 4),
        primary_offset_fraction=0.0,
        audit_offset_fraction=0.5,
        initial_radial_round_cap=0,
        max_joint_iterations=4,
        max_total_microscopic_q_nodes=1000,
    )
    return AdaptiveOuterTailCasimirConfig(
        joint_config=joint,
        cutoff_u_values=(1.0, 2.0, 3.0, 4.0),
        total_outer_rtol=0.1,
        total_outer_atol_J_m2=1e-12,
        finite_domain_budget_fraction=0.6,
        tail_budget_fraction=0.4,
        joint_budget_fraction_within_finite=0.75,
        offset_budget_fraction_within_finite=0.25,
        tail_start_u=1.0,
        tail_window_shells=3,
        tail_ratio_max=0.8,
        max_total_microscopic_q_nodes=10000,
    )


def _config(**changes) -> AdaptiveMatsubaraCasimirConfig:
    base = AdaptiveMatsubaraCasimirConfig(
        outer_tail_config=_outer_config(),
        matsubara_cutoff_values=(1, 3, 5),
        total_free_energy_rtol=0.0,
        total_free_energy_atol_J_m2=1e-8,
        finite_matsubara_budget_fraction=0.6,
        matsubara_tail_budget_fraction=0.4,
        tail_start_n=2,
        tail_window_terms=3,
        tail_ratio_max=0.6,
        max_total_microscopic_point_entries=10000,
    )
    return replace(base, **changes)


def test_config_requires_budget_partition_and_valid_tail_window() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        _config(
            finite_matsubara_budget_fraction=0.7,
            matsubara_tail_budget_fraction=0.4,
        )
    with pytest.raises(ValueError, match="enough high-frequency"):
        _config(tail_start_n=5, tail_window_terms=3)


def test_geometric_matsubara_decay_establishes_tail_and_reuses_provider() -> None:
    provider = _FrequencyProvider()
    runner = _ScriptedOuterRunner(lambda n: 1e-9 * 0.25**n)
    result = run_adaptive_matsubara_casimir(
        _config(),
        provider=provider,
        outer_tail_runner=runner,
    )

    assert result.status == "adaptive_tail_bounded"
    assert result.matsubara_converged
    assert result.matsubara_tail_estimated
    assert result.outer_tail_estimated
    assert result.selected_matsubara_cutoff == 5
    assert provider.configs == [(0, 1), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5)]
    assert len(set(runner.provider_ids)) == 1
    channel = result.pairing_results["spm"]
    assert channel["matsubara_tail_decay_passed"] is True
    assert channel["finite_matsubara_budget_passed"] is True
    assert channel["matsubara_tail_budget_passed"] is True
    assert channel["total_free_energy_budget_passed"] is True
    assert result.production_casimir_allowed is False


def test_sign_cancellation_cannot_hide_nondecaying_matsubara_terms() -> None:
    runner = _ScriptedOuterRunner(lambda n: 1e-9 if n % 2 == 0 else -1e-9)
    result = run_adaptive_matsubara_casimir(
        _config(tail_ratio_max=0.8),
        provider=_FrequencyProvider(),
        outer_tail_runner=runner,
    )

    assert result.status == "unresolved"
    assert result.termination_reason == "matsubara_tail_decay_ratio_not_established"
    final = result.cutoff_records[-1]["matsubara_tail_metrics"]["pairings"]["spm"]
    assert abs(final["finite_matsubara_partial_J_m2"]) < 1.1e-9
    assert final["tail_decay_passed"] is False


def test_accumulated_outer_errors_must_fit_finite_matsubara_budget() -> None:
    runner = _ScriptedOuterRunner(
        lambda n: 1e-9 * 0.25**n,
        error_function=lambda n: 2e-10 * 0.25**n,
    )
    result = run_adaptive_matsubara_casimir(
        _config(
            total_free_energy_atol_J_m2=1e-8,
            finite_matsubara_budget_fraction=0.01,
            matsubara_tail_budget_fraction=0.99,
        ),
        provider=_FrequencyProvider(),
        outer_tail_runner=runner,
    )

    assert result.status == "unresolved"
    assert result.termination_reason == "finite_matsubara_outer_budget_not_met"


def test_unresolved_outer_tail_run_stops_matsubara_extension() -> None:
    provider = _FrequencyProvider()
    runner = _ScriptedOuterRunner(
        lambda n: 1e-9 * 0.25**n,
        unresolved_cutoff=3,
    )
    result = run_adaptive_matsubara_casimir(
        _config(),
        provider=provider,
        outer_tail_runner=runner,
    )

    assert result.status == "unresolved"
    assert result.selected_matsubara_cutoff == 3
    assert result.termination_reason == (
        "outer_tail_run_unresolved: microscopic_point_unresolved"
    )
    assert provider.configs == [(0, 1), (0, 1, 2, 3)]
    assert len(result.cutoff_records) == 2
