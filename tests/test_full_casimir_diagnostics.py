from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.full_casimir.diagnostics import diagnose_run, main
from scripts.full_casimir.outer_tail_diagnostics import outer_tail_metrics
from scripts.full_casimir.point_diagnostics import point_cache_diagnostics


def _comparison(*, absolute: float, relative: float, passed: bool) -> dict:
    return {
        "finite": True,
        "absolute": absolute,
        "relative": relative,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1.5e-3,
        "absolute_passed": absolute <= 1e-6,
        "relative_passed": relative <= 1.5e-3,
        "passed_by": "absolute" if absolute <= 1e-6 else "relative" if passed else "failed",
        "passed": passed,
    }


def _unresolved_cache() -> dict:
    qx = float(-0.04767068844022)
    qy = float(0.04767068844022)
    shift_labels = (
        "shift_0:(0.5, 0.5)",
        "shift_1:(0.25, 0.75)",
        "shift_2:(0.75, 0.25)",
    )
    first_shifts = {
        label: {
            "two_plate_logdet": -7.491077681745207e-4,
            "hard_physical_passed": True,
        }
        for label in shift_labels
    }
    second_values = (
        -7.502748328325166e-4,
        -7.493162435741662e-4,
        -7.493162435764952e-4,
    )
    second_shifts = {
        label: {
            "two_plate_logdet": value,
            "hard_physical_passed": True,
        }
        for label, value in zip(shift_labels, second_values, strict=True)
    }
    adjacent = {
        shift_labels[0]: _comparison(
            absolute=1.167064658e-6,
            relative=1.55552e-3,
            passed=False,
        ),
        shift_labels[1]: _comparison(
            absolute=2.084754e-7,
            relative=2.78e-4,
            passed=True,
        ),
        shift_labels[2]: _comparison(
            absolute=2.084754e-7,
            relative=2.78e-4,
            passed=True,
        ),
    }
    history = [
        {
            "N": 1152,
            "shifts": first_shifts,
            "hard_physical_closure_across_shifts": True,
            "two_plate_logdet_cross_shift": _comparison(
                absolute=8e-9,
                relative=1e-5,
                passed=True,
            ),
            "adjacent_N_by_shift": None,
            "adjacent_N_all_shifts_passed": False,
            "accepted_transition": False,
            "consecutive_accepted_transitions": 0,
            "oscillatory_envelope": {"available": False, "passed": False},
        },
        {
            "N": 1280,
            "shifts": second_shifts,
            "hard_physical_closure_across_shifts": True,
            "two_plate_logdet_cross_shift": _comparison(
                absolute=9.58e-7,
                relative=1.27e-3,
                passed=True,
            ),
            "adjacent_N_by_shift": adjacent,
            "adjacent_N_all_shifts_passed": False,
            "accepted_transition": False,
            "consecutive_accepted_transitions": 0,
            "oscillatory_envelope": {
                "available": True,
                "passed": False,
                "N_window": [1024, 1152, 1280],
            },
        },
    ]
    return {
        "schema": "certified-outer-q-point-cache-v2-matsubara-extendable",
        "point_policy": {"required_consecutive_passes": 2},
        "entries": [
            {
                "pairing": "dwave",
                "n": 1,
                "qx_hex": qx.hex(),
                "qy_hex": qy.hex(),
                "point_result": {
                    "sweet_spot": {
                        "status": "not_established",
                        "working_N": None,
                        "audit_N": None,
                    },
                    "history": history,
                },
            }
        ],
    }


def test_point_diagnostics_recover_exact_q_and_latest_failed_gate() -> None:
    result = point_cache_diagnostics(_unresolved_cache())

    assert result["entry_count"] == 1
    assert result["unresolved_count"] == 1
    point = result["unresolved_points"][0]
    assert point["q_model"] == [-0.04767068844022, 0.04767068844022]
    assert point["evaluated_N"] == [1152, 1280]
    blocker = point["latest_blocker"]
    assert blocker["classification"] == "adjacent_N_stability_failed"
    assert blocker["cross_shift_passed"] is True
    assert blocker["adjacent_N_all_shifts_passed"] is False
    assert list(blocker["adjacent_N_failures"]) == ["shift_0:(0.5, 0.5)"]


def test_outer_tail_metrics_reconstruct_ratio_and_noise_floor_evidence() -> None:
    payload = {
        "config": {
            "tail_window_shells": 3,
            "tail_start_u": 24.0,
            "tail_ratio_max": 0.8,
            "shell_width_rtol": 1e-12,
            "shell_width_atol": 1e-12,
            "total_outer_rtol": 0.0,
            "total_outer_atol_J_m2": 1e-10,
            "finite_domain_budget_fraction": 0.7,
            "tail_budget_fraction": 0.3,
        },
        "shell_records": [
            {
                "left_u": 24.0,
                "right_u": 30.0,
                "width_u": 6.0,
                "pairings": {
                    "spm": {"shell_envelope_amplitudes_J_m2": [2e-11, 4e-12]}
                },
            },
            {
                "left_u": 30.0,
                "right_u": 36.0,
                "width_u": 6.0,
                "pairings": {
                    "spm": {"shell_envelope_amplitudes_J_m2": [4e-12, 8e-13]}
                },
            },
            {
                "left_u": 36.0,
                "right_u": 42.0,
                "width_u": 6.0,
                "pairings": {
                    "spm": {"shell_envelope_amplitudes_J_m2": [8e-13, 1.6e-13]}
                },
            },
        ],
        "cutoff_records": [
            {
                "u_max": 42.0,
                "pairing_results": {
                    "spm": {
                        "matsubara_indices": [0, 1],
                        "contributions_J_m2": [-1e-9, -2e-10],
                    }
                },
                "finite_domain_error_bounds_J_m2": {"spm": [1e-13, 1e-13]},
            }
        ],
    }

    result = outer_tail_metrics(payload)

    assert result["window_available"] is True
    assert result["equal_shell_widths"] is True
    assert result["dominant_failure"] == "outer_cutoff_and_tail_tolerances_met"
    channel = result["pairings"]["spm"]
    assert channel["ratio_envelopes"] == [0.2, 0.2]
    assert channel["decay_channel_passed"] == [True, True]
    assert channel["latest_shell_to_finite_error_ratio"] == pytest.approx([8.0, 1.6])


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_diagnose_run_writes_report_without_mutating_cache(tmp_path: Path) -> None:
    run = tmp_path / "case"
    _write(run / "summary.json", {"status": "unresolved", "termination_reason": "x"})
    _write(run / "manifest.json", {"status": "unresolved", "git_commit": "abc"})
    _write(run / "result.json", {"status": "unresolved", "cutoff_records": []})
    _write(run / "config.json", {})
    cache_path = run / "cache" / "certified_points.json"
    _write(cache_path, _unresolved_cache())
    before = cache_path.read_bytes()

    report, output = diagnose_run(run)

    assert output == run / "reports" / "diagnostics.json"
    assert output.is_file()
    assert report["point_cache"]["unresolved_count"] == 1
    assert cache_path.read_bytes() == before


def test_cli_returns_failure_when_outer_replay_is_requested_for_unresolved_cache(
    tmp_path: Path,
) -> None:
    run = tmp_path / "case"
    _write(run / "summary.json", {"status": "unresolved", "termination_reason": "x"})
    _write(run / "manifest.json", {"status": "unresolved"})
    _write(run / "result.json", {"status": "unresolved", "cutoff_records": []})
    _write(run / "config.json", {})
    _write(run / "cache" / "certified_points.json", _unresolved_cache())

    status = main(["--run-dir", str(run), "--replay-outer-tail", "--quiet"])

    assert status == 2
