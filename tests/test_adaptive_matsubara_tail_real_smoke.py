from __future__ import annotations

from pathlib import Path

from lno327.casimir import (
    AdaptiveJointCasimirConfig,
    AdaptiveMatsubaraCasimirConfig,
    AdaptiveOuterTailCasimirConfig,
    AdaptiveRadialCasimirConfig,
    FixedCasimirConfig,
    run_adaptive_matsubara_casimir,
)


def test_real_certifier_stops_before_matsubara_tail_inference(
    tmp_path: Path,
) -> None:
    """Tiny unresolved microscopic ladders must block frequency extrapolation."""

    point = FixedCasimirConfig(
        pairings=("spm",),
        matsubara_indices=(0, 1),
        u_max_values=(1.5, 3.0),
        radial_orders=(1, 2),
        angular_orders=(1, 2),
        angular_offsets=(0.125, 0.25),
        N_candidates=(2, 4, 6),
        shifts=((0.5, 0.5), (0.25, 0.75)),
        required_consecutive_passes=1,
        workers=1,
        parallel_mode="serial",
        canonical_block=4,
        runtime_chunk=4,
        ward_tolerance=1e6,
        ward_absolute_tolerance=1e6,
        condition_max=1e30,
        static_reality_tolerance=1e6,
        static_longitudinal_tolerance=1e6,
        static_mixing_tolerance=1e6,
        static_passivity_tolerance=1e6,
        logdet_rtol=1e6,
        logdet_atol=1e6,
    )
    radial = AdaptiveRadialCasimirConfig(
        point_config=point,
        initial_panel_edges=(0.0, 3.0),
        radial_order=1,
        angular_order=1,
        angular_offset_fraction=0.125,
        radial_rtol=1e6,
        radial_atol_J_m2=1e6,
        max_refinement_rounds=0,
        max_panel_depth=0,
        refine_panels_per_round=1,
        max_microscopic_q_nodes=12,
    )
    joint = AdaptiveJointCasimirConfig(
        radial_config=radial,
        angular_orders=(1, 2),
        primary_offset_fraction=0.125,
        audit_offset_fraction=0.25,
        outer_rtol=1e6,
        outer_atol_J_m2=1e6,
        radial_budget_fraction=0.5,
        angular_budget_fraction=0.5,
        offset_rtol=1e6,
        offset_atol_J_m2=1e6,
        initial_radial_round_cap=0,
        max_joint_iterations=4,
        max_total_microscopic_q_nodes=30,
    )
    outer = AdaptiveOuterTailCasimirConfig(
        joint_config=joint,
        cutoff_u_values=(3.0, 6.0, 9.0, 12.0),
        total_outer_rtol=1e6,
        total_outer_atol_J_m2=1e6,
        finite_domain_budget_fraction=0.6,
        tail_budget_fraction=0.4,
        joint_budget_fraction_within_finite=0.75,
        offset_budget_fraction_within_finite=0.25,
        tail_start_u=3.0,
        tail_window_shells=3,
        tail_ratio_max=0.8,
        max_total_microscopic_q_nodes=100,
    )
    cache = tmp_path / "real-matsubara-points-v2.json"
    config = AdaptiveMatsubaraCasimirConfig(
        outer_tail_config=outer,
        matsubara_cutoff_values=(1, 2),
        total_free_energy_rtol=1e6,
        total_free_energy_atol_J_m2=1e6,
        finite_matsubara_budget_fraction=0.6,
        matsubara_tail_budget_fraction=0.4,
        tail_start_n=1,
        tail_window_terms=2,
        tail_ratio_max=0.8,
        max_total_microscopic_point_entries=1000,
        point_cache_path=cache,
    )

    result = run_adaptive_matsubara_casimir(config)

    assert result.status == "unresolved"
    assert result.selected_matsubara_cutoff == 1
    assert result.termination_reason.startswith("outer_tail_run_unresolved: ")
    assert result.matsubara_converged is False
    assert result.matsubara_tail_estimated is False
    assert result.all_microscopic_nodes_certified is False
    assert len(result.cutoff_records) == 1
    assert result.cutoff_records[0]["matsubara_indices"] == [0, 1]
    assert result.cutoff_records[0]["matsubara_tail_metrics"] is None
    assert result.provider_statistics["certification_batches"] >= 1
    assert result.provider_statistics["new_point_evaluations"] > 0
    assert result.production_casimir_allowed is False
