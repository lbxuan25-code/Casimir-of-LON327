from __future__ import annotations

from validation.commands.matsubara.orbit_gauss_preflight import (
    _legacy_static_record,
    _parse_args,
    _serial_parallel_record,
)


def _small_args():
    return _parse_args(
        [
            "--pairings",
            "spm",
            "dwave",
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
            "--legacy-static-nk",
            "4",
            "--legacy-static-order",
            "4",
            "--minimum-speedup",
            "0",
            "--minimum-parallel-cpu-wall-ratio",
            "0",
            "--comparison-rtol",
            "1e-8",
            "--comparison-atol",
            "1e-9",
            "--no-require-physical",
        ]
    )


def test_preflight_uses_combined_zero_positive_batched_process_path() -> None:
    args = _small_args()
    record = _serial_parallel_record(args, "spm")

    assert record["correctness_passed"] is True
    assert record["optimization_passed"] is True
    assert record["material_workspace_implementation"] == "batched_model_capability"
    assert record["q_workspace_implementation"] == "batched_model_capability"
    assert record["execution_strategy"] == (
        "fork_process_transverse_nodes_ordered_parent_reduction"
    )
    assert record["frequency_count"] == 2
    assert record["callbacks_not_multiplied_by_frequency_count"] is True
    assert record["evaluator_callbacks"] == 4
    assert record["full_transverse_period_integrated"] is True
    assert record["symmetry_reduction_applied"] is False


def test_preflight_zero_matches_independent_exact_static_dwave_path() -> None:
    args = _small_args()
    record = _legacy_static_record(args)

    assert record["passed"] is True, record
    assert record["maximum_relative"] < 1e-7, record
