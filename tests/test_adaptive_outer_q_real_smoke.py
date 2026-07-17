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
    """Exercise the adaptive/provider boundary with the established tiny command."""

    point_config = FixedCasimirConfig(
        pairings=("spm",),
        matsubara_indices=(1,),
        u_max_values=(1.5, 3.0),
        radial_orders=(1, 2),
        angular_orders=(1, 2),
        angular_offsets=(0.0, 0.5),
        N_candidates=(2, 4, 6),
        shifts=((0.5, 0.5), (0.25, 0.75)),
        required_consecutive_passes=1,
        workers=1,
        parallel_mode="serial",
        canonical_block=4,
        runtime_chunk=4,
    )
    cache = tmp_path / "real-adaptive-points.json"
    config = AdaptiveRadialCasimirConfig(
        point_config=point_config,
        initial_panel_edges=(0.0, 3.0),
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
