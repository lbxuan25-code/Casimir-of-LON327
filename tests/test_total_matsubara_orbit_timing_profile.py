from __future__ import annotations

from validation.commands.matsubara.orbit_gauss_timing_profile import (
    _pairing_record,
    _parse_args,
)


def _small_args():
    return _parse_args(
        [
            "--pairings",
            "spm",
            "--nk",
            "4",
            "--mx",
            "1",
            "--my",
            "0",
            "--matsubara-indices",
            "0",
            "1",
            "--transverse-order",
            "4",
            "--panel-count",
            "1",
            "--transverse-workers",
            "2",
            "--transverse-task-size",
            "2",
            "--minimum-speedup",
            "0",
            "--minimum-cpu-wall-ratio",
            "0",
        ]
    )


def test_total_timing_profile_uses_combined_batched_fork_backend() -> None:
    record = _pairing_record(_small_args(), "spm")
    parallel = record["parallel"]

    assert record["serial_parallel_exact_equal"] is True
    assert record["optimization_sufficient"] is True
    assert parallel["material_workspace_implementation"] == "batched_model_capability"
    assert parallel["q_workspace_implementation"] == "batched_model_capability"
    assert parallel["execution_strategy"] == (
        "fork_process_transverse_nodes_ordered_parent_reduction"
    )
    assert parallel["callbacks"] == 4
    assert parallel["frequency_count"] == 2
    assert parallel["callbacks_not_multiplied_by_frequency_count"] is True
    assert parallel["full_transverse_period_integrated"] is True
    assert parallel["symmetry_reduction_applied"] is False
    assert set(parallel["stage_worker_seconds"]) == {
        "material_workspace",
        "q_workspace",
        "kubo_factors",
        "kubo_contraction",
        "primitive_packing",
    }
    assert all(value >= 0.0 for value in parallel["stage_worker_seconds"].values())
