from __future__ import annotations

import math
import numpy as np

from lno327.casimir.adaptive_joint_q import AdaptiveJointCasimirConfig
from lno327.casimir.adaptive_matsubara_tail import (
    AdaptiveMatsubaraCasimirConfig,
    run_adaptive_matsubara_casimir,
)
from lno327.casimir.adaptive_outer_q import AdaptiveRadialCasimirConfig
from lno327.casimir.adaptive_outer_tail import AdaptiveOuterTailCasimirConfig
from lno327.casimir.certified_point_provider import CertifiedPointBatch
from lno327.casimir.fixed_chain import FixedCasimirConfig
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE


class _AnalyticFrequencyProvider:
    def __init__(self, config: FixedCasimirConfig) -> None:
        self.config = config
        self._entries: set[tuple[str, int, str, str]] = set()
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0
        self.requested_point_evaluations = 0
        self.new_point_evaluations = 0
        self.cache_hit_point_evaluations = 0

    def reconfigure(self, config: FixedCasimirConfig) -> None:
        self.config = config

    @property
    def cached_point_count(self) -> int:
        return len(self._entries)

    @property
    def unique_q_count(self) -> int:
        return len({(entry[2], entry[3]) for entry in self._entries})

    def evaluate(self, q_model) -> CertifiedPointBatch:
        array = np.asarray(q_model, dtype=float)
        unique = {(float(q[0]).hex(), float(q[1]).hex()): q for q in array}
        requested = {
            (pairing, int(n), key[0], key[1])
            for key in unique
            for pairing in self.config.pairings
            for n in self.config.matsubara_indices
        }
        new_entries = requested - self._entries
        incomplete_q = {(entry[2], entry[3]) for entry in new_entries}
        self.requested_q_evaluations += len(unique)
        self.new_q_evaluations += len(incomplete_q)
        self.cache_hit_q_evaluations += len(unique) - len(incomplete_q)
        self.requested_point_evaluations += len(requested)
        self.new_point_evaluations += len(new_entries)
        self.cache_hit_point_evaluations += len(requested) - len(new_entries)
        self.certification_batches += int(bool(new_entries))
        self._entries.update(new_entries)
        return CertifiedPointBatch(
            point_results=(),
            unresolved_points=(),
            requested_q_count=len(unique),
            new_q_count=len(incomplete_q),
            cache_hit_q_count=len(unique) - len(incomplete_q),
            certification_batches=int(bool(new_entries)),
            requested_point_count=len(requested),
            new_point_count=len(new_entries),
            cache_hit_point_count=len(requested) - len(new_entries),
        )

    def primary_logdet(self, pairing, n, q) -> float:
        material = LNO327_THIN_FILM_SLAO_IN_PLANE
        qx = float(q[0]) / material.lattice_a_x_m
        qy = float(q[1]) / material.lattice_a_y_m
        u = 2.0 * self.config.separation_m * math.hypot(qx, qy)
        phi = math.atan2(qy, qx)
        return float(
            math.exp(-2.0 * u)
            * 0.2 ** int(n)
            * (1.0 + 0.05 * math.cos(2.0 * phi))
        )


def _config() -> AdaptiveMatsubaraCasimirConfig:
    point = FixedCasimirConfig(
        pairings=("spm",),
        matsubara_indices=(0, 1),
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(4, 8),
        angular_offsets=(0.0, 0.5),
        separation_nm=1e9,
    )
    radial = AdaptiveRadialCasimirConfig(
        point_config=point,
        initial_panel_edges=(0.0, 1.0),
        radial_order=2,
        angular_order=4,
        radial_rtol=0.2,
        radial_atol_J_m2=1e-16,
        max_refinement_rounds=5,
        max_panel_depth=7,
        refine_panels_per_round=2,
        max_microscopic_q_nodes=100000,
    )
    joint = AdaptiveJointCasimirConfig(
        radial_config=radial,
        angular_orders=(4, 8),
        primary_offset_fraction=0.0,
        audit_offset_fraction=0.5,
        outer_rtol=0.2,
        outer_atol_J_m2=1e-16,
        radial_budget_fraction=0.5,
        angular_budget_fraction=0.5,
        offset_rtol=0.2,
        offset_atol_J_m2=1e-16,
        initial_radial_round_cap=1,
        max_joint_iterations=12,
        max_total_microscopic_q_nodes=100000,
    )
    outer = AdaptiveOuterTailCasimirConfig(
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
        max_total_microscopic_q_nodes=100000,
    )
    return AdaptiveMatsubaraCasimirConfig(
        outer_tail_config=outer,
        matsubara_cutoff_values=(1, 3, 5),
        total_free_energy_rtol=0.5,
        total_free_energy_atol_J_m2=1e-16,
        finite_matsubara_budget_fraction=0.7,
        matsubara_tail_budget_fraction=0.3,
        tail_start_n=2,
        tail_window_terms=3,
        tail_ratio_max=0.5,
        max_total_microscopic_point_entries=500000,
    )


def test_full_adaptive_stack_composes_with_frequency_extension() -> None:
    config = _config()
    provider = _AnalyticFrequencyProvider(
        config.outer_tail_config.joint_config.radial_config.point_config
    )
    result = run_adaptive_matsubara_casimir(config, provider=provider)
    evidence = {
        "termination_reason": result.termination_reason,
        "selected_matsubara_cutoff": result.selected_matsubara_cutoff,
        "cutoffs": [
            {
                "n": row["matsubara_cutoff"],
                "status": row["status"],
                "outer_reason": row["termination_reason"],
                "selected_u_max": row["selected_u_max"],
                "has_tail_metrics": row["matsubara_tail_metrics"] is not None,
            }
            for row in result.cutoff_records
        ],
    }
    assert result.status == "adaptive_tail_bounded", evidence
    assert result.matsubara_tail_estimated
    assert result.outer_tail_estimated
    assert result.selected_matsubara_cutoff == 5
    assert result.unique_microscopic_q_node_count > 0
    assert result.cached_microscopic_point_count > result.unique_microscopic_q_node_count
    assert result.provider_statistics["new_point_evaluations"] > 0
    assert result.provider_statistics["cache_hit_point_evaluations"] > 0
