from __future__ import annotations

from pathlib import Path

from lno327.casimir import (
    AdaptiveRadialCasimirConfig,
    FixedCasimirConfig,
    run_adaptive_radial_casimir,
)


def test_real_production_certifier_connects_to_adaptive_radial_controller(
    tmp_path: Path,
) -> None:
    point_config = FixedCasimirConfig(
        pairings=("spm",),
        matsubara_indices=(0,),
        u_max_values=(0.01, 0.02),
        radial_orders=(1, 2),
        angular_orders=(1, 2),
        angular_offsets=(0.0, 0.5),
        N_candidates=(4, 6, 8),
        required_consecutive_passes=1,
        workers=1,
        parallel_mode="serial",
        canonical_block=16,
        runtime_chunk=16,
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
    cache = tmp_path / "real-adaptive-points.json"
    config = AdaptiveRadialCasimirConfig(
        point_config=point_config,
        initial_panel_edges=(0.0, 0.02),
        radial_order=1,
        angular_order=1,
        angular_offset_fraction=0.5,
        radial_rtol=1e6,
        radial_atol_J_m2=1e6,
        max_refinement_rounds=0,
        max_panel_depth=0,
        refine_panels_per_round=1,
        max_microscopic_q_nodes=3,
        point_cache_path=cache,
    )

    result = run_adaptive_radial_casimir(config)

    assert result.status in {"adaptive_finite_partial", "unresolved"}
    assert result.termination_reason in {
        "radial_tolerance_met",
        "microscopic_point_unresolved",
    }
    assert result.provider_statistics["certification_batches"] == 1
    assert result.provider_statistics["new_q_evaluations"] == 3
    assert result.unique_microscopic_q_node_count == 3
    assert cache.is_file()
    assert result.production_casimir_allowed is False
    assert result.outer_tail_estimated is False
    assert result.matsubara_tail_estimated is False
