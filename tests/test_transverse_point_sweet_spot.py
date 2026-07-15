from __future__ import annotations

import json
from pathlib import Path

from validation.__main__ import resolve_command
from validation.commands.matsubara.transverse_point_sweet_spot import (
    assess_frequency_level,
    main,
)


def _state(logdet: float, passed: bool = True) -> dict[str, object]:
    return {
        "two_plate_logdet": float(logdet),
        "hard_physical_passed": bool(passed),
    }


def _tiny_command(output: Path) -> list[str]:
    return [
        "--q-point",
        "generic",
        "0.03",
        "0.02",
        "--pairings",
        "spm",
        "--matsubara-indices",
        "1",
        "--N-candidates",
        "2",
        "4",
        "6",
        "--shift",
        "0.5",
        "0.5",
        "--shift",
        "0.25",
        "0.75",
        "--required-consecutive-passes",
        "1",
        "--canonical-block",
        "4",
        "--runtime-chunk",
        "4",
        "--output",
        str(output),
    ]


def test_frequency_level_requires_closure_N_and_shift_convergence() -> None:
    previous = {
        "primary": _state(-0.02),
        "audit": _state(-0.020001),
    }
    current = {
        "primary": _state(-0.020002),
        "audit": _state(-0.020003),
    }
    result = assess_frequency_level(
        current_by_shift=current,
        previous_by_shift=previous,
        rtol=1e-3,
        atol=1e-14,
    )
    assert result["hard_physical_closure_across_shifts"] is True
    assert result["two_plate_logdet_cross_shift"]["passed"] is True
    assert result["adjacent_N_all_shifts_passed"] is True
    assert result["accepted_transition"] is True


def test_frequency_level_rejects_failed_physical_gate_even_if_logdet_is_stable() -> None:
    previous = {
        "primary": _state(-0.02),
        "audit": _state(-0.02),
    }
    current = {
        "primary": _state(-0.02, passed=False),
        "audit": _state(-0.02),
    }
    result = assess_frequency_level(
        current_by_shift=current,
        previous_by_shift=previous,
        rtol=1e-3,
        atol=1e-14,
    )
    assert result["two_plate_logdet_cross_shift"]["passed"] is True
    assert result["hard_physical_closure_across_shifts"] is False
    assert result["accepted_transition"] is False


def test_unified_point_command_writes_point_specific_history_and_parallel_plan(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sweet_spot.json"
    main(
        [
            *_tiny_command(output),
            "--workers",
            "1",
            "--parallel-mode",
            "serial",
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "transverse-point-sweet-spot-v2"
    assert payload["single_public_point_convergence_script"] is True
    assert payload["point_specific_early_stop"] is True
    assert payload["hard_gate_policy"]["static_longitudinal"] is False
    assert payload["run_complete"] is True
    assert payload["cpu_parallel_policy"]["nested_process_pools"] is False
    assert payload["cpu_parallel_policy"]["one_process_parallel_layer_only"] is True
    assert payload["execution_levels"]
    first_plan = payload["execution_levels"][0]["parallel_plan"]
    assert first_plan["strategy"] == "serial"
    assert first_plan["total_worker_budget"] == 1
    assert len(payload["point_results"]) == 1
    point = payload["point_results"][0]
    assert point["pairing"] == "spm"
    assert point["q_label"] == "generic"
    assert point["n"] == 1
    assert [row["N"] for row in point["history"]]
    first_shift = next(iter(point["history"][0]["shifts"].values()))
    assert "two_plate_logdet" in first_shift
    assert "plate_1" in first_shift
    assert "plate_2" in first_shift


def test_context_parallel_command_uses_spawn_workers_without_nested_q_pool(
    tmp_path: Path,
) -> None:
    output = tmp_path / "context_parallel.json"
    main(
        [
            *_tiny_command(output),
            "--workers",
            "2",
            "--parallel-mode",
            "context",
            "--memory-budget-gb",
            "4",
            "--max-context-workers",
            "2",
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    first_level = payload["execution_levels"][0]
    plan = first_level["parallel_plan"]
    assert plan["strategy"] == "context"
    assert plan["context_workers"] == 2
    assert plan["q_workers"] == 1
    records = first_level["pairings"]["spm"]
    assert len(records) == 2
    for record in records:
        assert record["context_worker_actual_threadpool_passed"] is True
        assert all(group["workers"] == 1 for group in record["groups"])


def test_unified_point_command_is_the_only_public_point_convergence_route() -> None:
    assert resolve_command("diagnostic", "transverse-point-sweet-spot") == (
        "validation.commands.matsubara.transverse_point_sweet_spot"
    )


def test_superseded_point_scripts_are_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (
        root
        / "validation"
        / "commands"
        / "matsubara"
        / "arbitrary_q_uniform_refinement_diagnostic.py"
    ).exists()
    assert not (
        root / "validation" / "commands" / "matsubara" / "positive_point.py"
    ).exists()
    assert not (
        root / "validation" / "commands" / "static" / "nk_scan.py"
    ).exists()


def test_retained_arbitrary_q_workflow_surface_is_allowlisted() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow_dir = root / "src" / "lno327" / "workflows"
    actual = {path.name for path in workflow_dir.glob("arbitrary_q_*.py")}
    assert actual == {
        "arbitrary_q_matsubara.py",
        "arbitrary_q_parallel.py",
    }


def test_retained_arbitrary_q_validation_surface_is_allowlisted() -> None:
    root = Path(__file__).resolve().parents[1]
    command_dir = root / "validation" / "commands" / "matsubara"
    actual = {path.name for path in command_dir.glob("arbitrary_q_*.py")}
    assert actual == {
        "arbitrary_q_performance_preflight.py",
        "arbitrary_q_performance_smoke.py",
        "arbitrary_q_periodic_bz_qualification.py",
        "arbitrary_q_periodic_bz_qualification_gate.py",
        "arbitrary_q_physics_smoke.py",
    }
