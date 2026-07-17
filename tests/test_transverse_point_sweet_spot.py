from __future__ import annotations

import json
from pathlib import Path

from lno327.casimir import fixed_transverse_point_certification as certification


def _state(logdet: float, passed: bool = True) -> dict[str, object]:
    return {
        "two_plate_logdet": float(logdet),
        "hard_physical_passed": bool(passed),
    }


def _history_row(N: int, values: tuple[float, float, float]) -> dict[str, object]:
    shifts = {
        f"shift_{index}": _state(value)
        for index, value in enumerate(values)
    }
    assessment = certification.assess_frequency_level(
        current_by_shift=shifts,
        previous_by_shift=None,
        rtol=1e-3,
        atol=1e-6,
    )
    return {
        "N": int(N),
        "shifts": shifts,
        **assessment,
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
        "--workers",
        "1",
        "--parallel-mode",
        "serial",
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
    result = certification.assess_frequency_level(
        current_by_shift=current,
        previous_by_shift=previous,
        rtol=1e-3,
        atol=1e-14,
    )
    assert result["hard_physical_closure_across_shifts"] is True
    assert result["two_plate_logdet_cross_shift"]["passed"] is True
    assert result["adjacent_N_all_shifts_passed"] is True
    assert result["accepted_transition"] is True


def test_absolute_tolerance_is_checked_before_relative_tolerance() -> None:
    previous = {
        "primary": _state(0.0),
        "audit": _state(0.0),
    }
    current = {
        "primary": _state(5e-7),
        "audit": _state(5e-7),
    }
    result = certification.assess_frequency_level(
        current_by_shift=current,
        previous_by_shift=previous,
        rtol=1e-3,
        atol=1e-6,
    )
    row = result["adjacent_N_by_shift"]["primary"]
    assert row["absolute_passed"] is True
    assert row["relative_passed"] is False
    assert row["passed_by"] == "absolute"
    assert result["accepted_transition"] is True


def test_relative_tolerance_is_fallback_when_absolute_fails() -> None:
    previous = {
        "primary": _state(-0.010000),
        "audit": _state(-0.010000),
    }
    current = {
        "primary": _state(-0.010005),
        "audit": _state(-0.010005),
    }
    result = certification.assess_frequency_level(
        current_by_shift=current,
        previous_by_shift=previous,
        rtol=1e-3,
        atol=1e-6,
    )
    row = result["adjacent_N_by_shift"]["primary"]
    assert row["absolute_passed"] is False
    assert row["relative_passed"] is True
    assert row["passed_by"] == "relative"
    assert result["accepted_transition"] is True


def test_frequency_level_rejects_failed_physical_gate() -> None:
    previous = {
        "primary": _state(-0.02),
        "audit": _state(-0.02),
    }
    current = {
        "primary": _state(-0.02, passed=False),
        "audit": _state(-0.02),
    }
    result = certification.assess_frequency_level(
        current_by_shift=current,
        previous_by_shift=previous,
        rtol=1e-3,
        atol=1e-6,
    )
    assert result["two_plate_logdet_cross_shift"]["passed"] is True
    assert result["hard_physical_closure_across_shifts"] is False
    assert result["accepted_transition"] is False


def test_three_level_envelope_accepts_universal_absolute_near_zero() -> None:
    history = [
        _history_row(128, (0.0, 1e-7, -1e-7)),
        _history_row(192, (4e-7, 3e-7, 2e-7)),
        _history_row(256, (-2e-7, -1e-7, 0.0)),
    ]
    result = certification.assess_oscillatory_envelope(
        history,
        rtol=1e-3,
        atol=1e-6,
    )
    envelope = result["joint_logdet_envelope"]
    assert result["passed"] is True
    assert result["N_window"] == [128, 192, 256]
    assert envelope["absolute_passed"] is True
    assert envelope["relative_passed"] is False
    assert envelope["passed_by"] == "absolute"


def test_production_point_command_writes_v4_policy(tmp_path: Path) -> None:
    output = tmp_path / "sweet_spot.json"
    certification.main(_tiny_command(output))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "transverse-point-sweet-spot-v4"
    assert payload["single_public_point_convergence_script"] is True
    assert payload["point_specific_early_stop"] is True
    assert payload["run_complete"] is True
    assert payload["logdet_atol"] == 1e-6
    policy = payload["convergence_policy"]
    assert policy["q_or_frequency_specific_exceptions"] is False
    assert policy["comparison_order"] == "absolute_first_then_relative_fallback"
    assert policy["oscillatory_envelope_path"]["levels"] == 3
    first_plan = payload["execution_levels"][0]["parallel_plan"]
    assert first_plan["strategy"] == "serial"
    assert first_plan["total_worker_budget"] == 1
