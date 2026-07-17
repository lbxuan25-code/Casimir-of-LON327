from __future__ import annotations

from dataclasses import replace
import math
from types import MappingProxyType

import numpy as np
import pytest

from lno327.casimir.adaptive_joint_q import (
    AdaptiveJointCasimirConfig,
    AdaptiveJointCasimirResult,
)
from lno327.casimir.adaptive_outer_q import AdaptiveRadialCasimirConfig
from lno327.casimir.adaptive_outer_tail import (
    AdaptiveOuterTailCasimirConfig,
    run_adaptive_outer_tail_casimir,
)
from lno327.casimir.certified_point_provider import CertifiedPointBatch
from lno327.casimir.fixed_chain import FixedCasimirConfig
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE


class _CountingProvider:
    def __init__(self) -> None:
        self.cached_point_count = 0
        self.unique_q_count = 0
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0


class _ScriptedJointRunner:
    def __init__(
        self,
        cumulative_by_u: dict[float, tuple[float, ...]],
        *,
        finite_error: float = 1e-15,
        unresolved_u: float | None = None,
    ) -> None:
        self.cumulative_by_u = cumulative_by_u
        self.finite_error = float(finite_error)
        self.unresolved_u = unresolved_u
        self.provider_ids: list[int] = []
        self.panel_edges: list[tuple[float, ...]] = []

    def __call__(self, config, *, provider=None):
        self.provider_ids.append(id(provider))
        edges = tuple(config.radial_config.initial_panel_edges)
        self.panel_edges.append(edges)
        u_max = float(edges[-1])
        provider.certification_batches += 1
        provider.requested_q_evaluations += 1
        provider.new_q_evaluations += 1
        provider.unique_q_count += 1
        provider.cached_point_count += 1
        values = np.asarray(self.cumulative_by_u[u_max], dtype=float)
        count = len(values)
        pairings = tuple(config.radial_config.point_config.pairings)
        payload = {
            pairing: {
                "status": "integrated",
                "partial_free_energy_J_m2": float(np.sum(values)),
                "contributions_J_m2": values.tolist(),
                "estimated_joint_errors_J_m2": [self.finite_error] * count,
                "estimated_offset_errors_J_m2": [self.finite_error] * count,
                "matsubara_indices": list(
                    config.radial_config.point_config.matsubara_indices
                ),
            }
            for pairing in pairings
        }
        unresolved = self.unresolved_u is not None and u_max == self.unresolved_u
        return AdaptiveJointCasimirResult(
            status="unresolved" if unresolved else "adaptive_finite_partial",
            config=config,
            joint_converged=not unresolved,
            radial_budget_passed=not unresolved,
            angular_budget_passed=not unresolved,
            offset_audit_passed=not unresolved,
            all_microscopic_nodes_certified=not unresolved,
            selected_angular_order=config.angular_orders[-1],
            selected_radial_round_cap=config.initial_radial_round_cap,
            pairing_results=MappingProxyType(payload),
            direction_records=(),
            radial_run_records=(),
            offset_audit_record=None,
            termination_reason=(
                "microscopic_point_unresolved"
                if unresolved
                else "joint_radial_angular_budget_and_offset_tolerances_met"
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


def _joint_config(*, matsubara_indices=(0,)) -> AdaptiveJointCasimirConfig:
    point = FixedCasimirConfig(
        pairings=("spm",),
        matsubara_indices=tuple(matsubara_indices),
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
        separation_nm=1e9,
    )
    radial = AdaptiveRadialCasimirConfig(
        point_config=point,
        initial_panel_edges=(0.0, 1.0),
        radial_order=2,
        angular_order=2,
        radial_rtol=1e-2,
        radial_atol_J_m2=1e-14,
        max_refinement_rounds=4,
        max_panel_depth=6,
        refine_panels_per_round=2,
        max_microscopic_q_nodes=10000,
    )
    return AdaptiveJointCasimirConfig(
        radial_config=radial,
        angular_orders=(2, 4),
        primary_offset_fraction=0.0,
        audit_offset_fraction=0.5,
        outer_rtol=1e-2,
        outer_atol_J_m2=1e-14,
        radial_budget_fraction=0.5,
        angular_budget_fraction=0.5,
        offset_rtol=1e-2,
        offset_atol_J_m2=1e-14,
        initial_radial_round_cap=0,
        radial_round_step=1,
        max_joint_iterations=8,
        max_total_microscopic_q_nodes=10000,
    )


def _tail_config(
    *,
    matsubara_indices=(0,),
    cutoffs=(1.0, 2.0, 3.0, 4.0),
    tail_start_u=1.0,
    tail_ratio_max=0.5,
) -> AdaptiveOuterTailCasimirConfig:
    return AdaptiveOuterTailCasimirConfig(
        joint_config=_joint_config(matsubara_indices=matsubara_indices),
        cutoff_u_values=tuple(cutoffs),
        total_outer_rtol=0.0,
        total_outer_atol_J_m2=1e-10,
        finite_domain_budget_fraction=0.6,
        tail_budget_fraction=0.4,
        joint_budget_fraction_within_finite=0.75,
        offset_budget_fraction_within_finite=0.25,
        tail_start_u=tail_start_u,
        tail_window_shells=3,
        tail_ratio_max=tail_ratio_max,
        max_total_microscopic_q_nodes=1000,
    )


def test_config_requires_two_budget_partitions_to_sum_to_one() -> None:
    with pytest.raises(ValueError, match="finite-domain and tail"):
        replace(
            _tail_config(),
            finite_domain_budget_fraction=0.7,
            tail_budget_fraction=0.4,
        )
    with pytest.raises(ValueError, match="joint and offset"):
        replace(
            _tail_config(),
            joint_budget_fraction_within_finite=0.8,
            offset_budget_fraction_within_finite=0.3,
        )


def test_geometric_shell_decay_establishes_tail_and_reuses_one_provider() -> None:
    config = _tail_config()
    runner = _ScriptedJointRunner(
        {
            1.0: (1.0e-9,),
            2.0: (1.2e-9,),
            3.0: (1.24e-9,),
            4.0: (1.248e-9,),
        }
    )
    provider = _CountingProvider()

    result = run_adaptive_outer_tail_casimir(
        config,
        provider=provider,
        joint_runner=runner,
    )

    assert result.status == "adaptive_finite_partial"
    assert result.cutoff_converged
    assert result.outer_tail_estimated
    assert result.selected_u_max == 4.0
    assert len(set(runner.provider_ids)) == 1
    assert runner.panel_edges == [
        (0.0, 1.0),
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 2.0, 3.0),
        (0.0, 1.0, 2.0, 3.0, 4.0),
    ]
    channel = result.pairing_results["spm"]
    assert channel["outer_tail_decay_channel_passed"] == [True]
    assert channel["outer_tail_channel_passed"] == [True]
    assert channel["total_outer_channel_passed"] == [True]
    assert result.production_casimir_allowed is False
    assert result.matsubara_tail_estimated is False


def test_noncontracting_shells_exhaust_cutoff_ladder_fail_closed() -> None:
    config = _tail_config(tail_ratio_max=0.8)
    runner = _ScriptedJointRunner(
        {
            1.0: (1.0e-9,),
            2.0: (1.2e-9,),
            3.0: (1.38e-9,),
            4.0: (1.55e-9,),
        }
    )
    result = run_adaptive_outer_tail_casimir(
        config,
        provider=_CountingProvider(),
        joint_runner=runner,
    )

    assert result.status == "unresolved"
    assert not result.outer_tail_estimated
    assert result.termination_reason == "outer_tail_decay_ratio_not_established"


def test_matsubara_cancellation_cannot_hide_nondecaying_tail() -> None:
    config = _tail_config(matsubara_indices=(0, 1), tail_ratio_max=0.8)
    runner = _ScriptedJointRunner(
        {
            1.0: (1.0e-9, -1.0e-9),
            2.0: (1.2e-9, -1.2e-9),
            3.0: (1.38e-9, -1.38e-9),
            4.0: (1.55e-9, -1.55e-9),
        }
    )
    result = run_adaptive_outer_tail_casimir(
        config,
        provider=_CountingProvider(),
        joint_runner=runner,
    )

    assert result.status == "unresolved"
    assert all(
        math.isclose(sum(record["pairings"]["spm"]["shell_contributions_J_m2"]), 0.0, abs_tol=1e-24)
        for record in result.shell_records
    )
    assert result.termination_reason == "outer_tail_decay_ratio_not_established"


def test_tail_window_requires_equal_shell_widths() -> None:
    config = _tail_config(
        cutoffs=(1.0, 2.0, 4.0, 6.0),
        tail_start_u=1.0,
        tail_ratio_max=0.8,
    )
    runner = _ScriptedJointRunner(
        {
            1.0: (1.0e-9,),
            2.0: (1.2e-9,),
            4.0: (1.24e-9,),
            6.0: (1.248e-9,),
        }
    )
    result = run_adaptive_outer_tail_casimir(
        config,
        provider=_CountingProvider(),
        joint_runner=runner,
    )

    assert result.status == "unresolved"
    assert result.termination_reason == "outer_tail_shell_width_contract_failed"


def test_unresolved_joint_run_stops_before_tail_inference() -> None:
    config = _tail_config()
    runner = _ScriptedJointRunner(
        {
            1.0: (1.0e-9,),
            2.0: (1.2e-9,),
            3.0: (1.24e-9,),
            4.0: (1.248e-9,),
        },
        unresolved_u=2.0,
    )
    result = run_adaptive_outer_tail_casimir(
        config,
        provider=_CountingProvider(),
        joint_runner=runner,
    )

    assert result.status == "unresolved"
    assert result.selected_u_max == 2.0
    assert result.termination_reason.startswith("finite_domain_run_unresolved")
    assert len(result.shell_records) == 1


class _AnalyticProvider:
    def __init__(self, point_config) -> None:
        self.point_config = point_config
        self._points: dict[tuple[str, str], np.ndarray] = {}
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0

    @property
    def unique_q_count(self):
        return len(self._points)

    @property
    def cached_point_count(self):
        return len(self._points) * len(self.point_config.pairings) * len(
            self.point_config.matsubara_indices
        )

    def evaluate(self, q_model):
        array = np.asarray(q_model, dtype=float)
        unique = {
            (float(q[0]).hex(), float(q[1]).hex()): q for q in array
        }
        new = [key for key in unique if key not in self._points]
        self.requested_q_evaluations += len(unique)
        self.new_q_evaluations += len(new)
        self.cache_hit_q_evaluations += len(unique) - len(new)
        self.certification_batches += int(bool(new))
        self._points.update(unique)
        return CertifiedPointBatch(
            point_results=(),
            unresolved_points=(),
            requested_q_count=len(unique),
            new_q_count=len(new),
            cache_hit_q_count=len(unique) - len(new),
            certification_batches=int(bool(new)),
        )

    def primary_logdet(self, pairing, n, q):
        material = LNO327_THIN_FILM_SLAO_IN_PLANE
        qx = float(q[0]) / material.lattice_a_x_m
        qy = float(q[1]) / material.lattice_a_y_m
        u = 2.0 * self.point_config.separation_m * math.hypot(qx, qy)
        return float(math.exp(-2.0 * u) * (1.0 + 0.05 * math.cos(2.0 * math.atan2(qy, qx))))


def test_real_joint_controller_composes_with_analytic_decaying_provider() -> None:
    joint = _joint_config()
    joint = replace(
        joint,
        radial_config=replace(
            joint.radial_config,
            radial_order=2,
            max_refinement_rounds=5,
            max_panel_depth=7,
        ),
        angular_orders=(4, 8),
        outer_rtol=0.2,
        outer_atol_J_m2=1e-16,
        offset_rtol=0.2,
        offset_atol_J_m2=1e-16,
        initial_radial_round_cap=1,
        max_joint_iterations=12,
    )
    config = AdaptiveOuterTailCasimirConfig(
        joint_config=joint,
        cutoff_u_values=(1.0, 2.0, 3.0, 4.0, 5.0),
        total_outer_rtol=0.5,
        total_outer_atol_J_m2=1e-16,
        finite_domain_budget_fraction=0.7,
        tail_budget_fraction=0.3,
        joint_budget_fraction_within_finite=0.8,
        offset_budget_fraction_within_finite=0.2,
        tail_start_u=2.0,
        tail_window_shells=3,
        tail_ratio_max=0.8,
        max_total_microscopic_q_nodes=50000,
    )
    provider = _AnalyticProvider(joint.radial_config.point_config)

    result = run_adaptive_outer_tail_casimir(config, provider=provider)

    assert result.status == "adaptive_finite_partial"
    assert result.outer_tail_estimated
    assert result.selected_u_max in config.cutoff_u_values
    assert result.unique_microscopic_q_node_count > 0
    assert result.provider_statistics["cache_hit_q_evaluations"] > 0
