from __future__ import annotations

from pathlib import Path
import json
from time import perf_counter
from types import SimpleNamespace

from lno327.casimir.progress import (
    emit_progress,
    progress_context,
    wrap_certifier_runner,
)
from scripts.full_casimir.progress import (
    CampaignProgressReporter,
    PROGRESS_EVENT_SCHEMA,
    PROGRESS_SCHEMA,
    main as status_main,
)


def _plan() -> dict[str, object]:
    return {
        "campaign_id": "campaign-progress-test",
        "campaign_sha256": "a" * 64,
        "scientific_policy_sha256": "b" * 64,
        "plan_sha256": "c" * 64,
        "cases": [
            {
                "case": "spm_T10K_d20nm_theta_p000deg",
                "case_identity": {
                    "pairing": "spm",
                    "temperature_K": 10.0,
                    "separation_nm": 20.0,
                    "plate_angles_deg": [0.0, 0.0],
                },
            },
            {
                "case": "dwave_T10K_d20nm_theta_p000deg",
                "case_identity": {
                    "pairing": "dwave",
                    "temperature_K": 10.0,
                    "separation_nm": 20.0,
                    "plate_angles_deg": [0.0, 0.0],
                },
            },
        ],
    }


def test_core_progress_observer_is_non_authoritative() -> None:
    events: list[dict[str, object]] = []
    with progress_context(events.append):
        emit_progress("example", value=7)
    assert events == [{"event": "example", "value": 7}]

    def broken_sink(_payload):
        raise OSError("reporting path unavailable")

    with progress_context(broken_sink):
        emit_progress("ignored_reporter_failure", value=9)


def test_certifier_wrapper_observes_exactly_one_existing_call(tmp_path: Path) -> None:
    calls = 0

    def fake_runner(config, manifest, output):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            payload={
                "point_results": [
                    {
                        "sweet_spot": {
                            "status": "established",
                            "audit_N": 256,
                        }
                    },
                    {
                        "sweet_spot": {
                            "status": "not_established",
                            "reason": "cross_shift_not_closed",
                        }
                    },
                ]
            },
            stdout="",
            stderr="",
        )

    config = SimpleNamespace(
        pairings=("spm",),
        matsubara_indices=(0, 1),
        N_candidates=(128, 192, 256),
    )
    manifest = SimpleNamespace(labels=("q0",))
    events: list[dict[str, object]] = []
    with progress_context(events.append):
        wrapped = wrap_certifier_runner(fake_runner)
        wrapped(config, manifest, tmp_path / "certification.json")

    assert calls == 1
    assert [event["event"] for event in events] == [
        "microscopic_batch_started",
        "microscopic_batch_completed",
    ]
    completed = events[-1]
    assert completed["selected_N_distribution"] == {"256": 1}
    assert completed["unresolved_reason_counts"] == {
        "cross_shift_not_closed": 1
    }


def test_campaign_progress_persists_activity_counts_and_reasons(tmp_path: Path) -> None:
    reporter = CampaignProgressReporter(
        campaign_dir=tmp_path,
        plan=_plan(),
        stream=None,
        snapshot_interval_seconds=0.0,
        heartbeat_interval_seconds=0.0,
    )
    reporter.campaign_started(mode="fresh", resources={"workers": 4})
    case = "spm_T10K_d20nm_theta_p000deg"
    reporter.case_started(case, action="start")
    reporter.emit(
        {
            "event": "matsubara_block_started",
            "block_index": 2,
            "block_count": 6,
            "left_n": 2,
            "right_n": 3,
        }
    )
    reporter.emit(
        {
            "event": "outer_cutoff_started",
            "cutoff_index": 3,
            "cutoff_count": 11,
            "u_max": 14.0,
        }
    )
    reporter.emit(
        {
            "event": "radial_run_started",
            "angular_order": 16,
            "angular_offset_fraction": 0.5,
            "radial_order": 8,
            "radial_round_cap": 2,
            "initial_panel_count": 3,
        }
    )
    reporter.emit(
        {
            "event": "microscopic_batch_started",
            "requested_q_count": 64,
            "requested_point_count": 128,
            "matsubara_indices": [2, 3],
            "N_candidates": [128, 192, 256],
        }
    )

    case_snapshot = json.loads(
        (tmp_path / "runs" / case / "progress.json").read_text(encoding="utf-8")
    )
    assert case_snapshot["schema"] == PROGRESS_SCHEMA
    assert [row["label"] for row in case_snapshot["activity_stack"]] == [
        "Matsubara 2-3",
        "outer u=14",
        "angular order=16, radial cap=2",
        "microscopic batch q=64, points=128",
    ]

    reporter.emit(
        {
            "event": "microscopic_batch_completed",
            "selected_N_distribution": {"192": 96, "256": 31},
            "unresolved_reason_counts": {"cross_shift_not_closed": 1},
            "provider_statistics": {
                "requested_point_evaluations": 128,
                "new_point_evaluations": 127,
                "cache_hit_point_evaluations": 1,
                "certification_batches": 1,
            },
        }
    )
    reporter.emit(
        {
            "event": "matsubara_controller_completed",
            "budget_ratios": {
                "spm": {
                    "matsubara_finite": 0.4,
                    "matsubara_tail": 1.2,
                    "total": 0.7,
                }
            },
        }
    )
    reporter.case_finished(
        case,
        status="production_authorized",
        termination_reason="formal_contract_met",
        action="start",
    )
    reporter.case_finished(
        "dwave_T10K_d20nm_theta_p000deg",
        status="numerically_unresolved",
        termination_reason="matsubara_tail_budget_not_met",
        action="start",
    )
    reporter.campaign_finished(exit_code=2)

    campaign = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert campaign["schema"] == PROGRESS_SCHEMA
    assert campaign["lifecycle_status"] == "completed_with_unresolved_cases"
    assert campaign["case_counts"]["production_authorized"] == 1
    assert campaign["case_counts"]["numerically_unresolved"] == 1
    assert campaign["pairing_counts"]["spm"]["production_authorized"] == 1
    assert campaign["pairing_counts"]["dwave"]["numerically_unresolved"] == 1
    assert campaign["cases"][case]["selected_N_distribution"] == {
        "192": 96,
        "256": 31,
    }
    assert campaign["cases"][case]["unresolved_reason_counts"] == {
        "cross_shift_not_closed": 1
    }
    assert campaign["cases"][case]["activity_stack"] == []

    events = [
        json.loads(line)
        for line in (tmp_path / "progress.events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events
    assert all(event["schema"] == PROGRESS_EVENT_SCHEMA for event in events)
    assert [event["state_sequence"] for event in events] == sorted(
        event["state_sequence"] for event in events
    )


def test_status_command_reads_snapshot_without_starting_work(
    tmp_path: Path,
    capsys,
) -> None:
    reporter = CampaignProgressReporter(
        campaign_dir=tmp_path,
        plan=_plan(),
        stream=None,
        heartbeat_interval_seconds=0.0,
    )
    reporter.campaign_started(mode="fresh")
    reporter.campaign_finished(exit_code=0)

    before = (tmp_path / "progress.events.jsonl").read_bytes()
    assert status_main(["--campaign-dir", str(tmp_path)]) == 0
    after = (tmp_path / "progress.events.jsonl").read_bytes()
    assert before == after
    assert "Campaign campaign-progress-test" in capsys.readouterr().out


def test_progress_event_overhead_is_below_one_percent_of_short_batch_model(
    tmp_path: Path,
) -> None:
    reporter = CampaignProgressReporter(
        campaign_dir=tmp_path,
        plan=_plan(),
        stream=None,
        snapshot_interval_seconds=3600.0,
        heartbeat_interval_seconds=0.0,
    )
    reporter.campaign_started(mode="fresh")
    reporter.case_started("spm_T10K_d20nm_theta_p000deg", action="start")

    event_count = 64
    representative_scientific_seconds_per_event = 5.0
    started = perf_counter()
    for index in range(event_count):
        reporter.emit(
            {
                "event": "microscopic_counter_sample",
                "provider_statistics": {
                    "requested_point_evaluations": index + 1,
                    "new_point_evaluations": index + 1,
                },
            }
        )
    progress_seconds = perf_counter() - started
    modeled_scientific_seconds = (
        event_count * representative_scientific_seconds_per_event
    )
    assert progress_seconds / modeled_scientific_seconds < 0.01
