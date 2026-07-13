from __future__ import annotations

from pathlib import Path
import threading
import time

from validation.commands.matsubara import dwave_orbit_certification_scan as base
from validation.commands.matsubara import dwave_orbit_certification_scan_parallel as parallel
from validation.__main__ import resolve_command


def _task(tmp_path: Path, label: str) -> base.TaskSpec:
    case = base.CaseSpec(label=label, mx=1, my=0, level="screen")
    return base.TaskSpec(
        case=case,
        stage="periodic_screen",
        output_csv=tmp_path / label / "result.csv",
        command=("python", "-c", "pass"),
        fingerprint=f"fingerprint-{label}",
    )


def test_certification_cli_routes_to_parallel_orchestrator():
    assert resolve_command(
        "matsubara",
        "dwave-orbit-certification-scan",
    ).endswith("dwave_orbit_certification_scan_parallel")


def test_child_environment_forces_explicit_thread_budget(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    environment = parallel._child_environment(2)
    for name in parallel._THREAD_ENVIRONMENT_VARIABLES:
        assert environment[name] == "2"
    assert environment["OMP_DYNAMIC"] == "FALSE"
    assert environment["MKL_DYNAMIC"] == "FALSE"
    assert environment["PYTHONUNBUFFERED"] == "1"


def test_default_single_child_thread_preserves_resume_fingerprint(tmp_path):
    task = _task(tmp_path, "axis")
    assert parallel._runtime_task(task, child_threads=1) is task
    threaded = parallel._runtime_task(task, child_threads=2)
    assert threaded.fingerprint != task.fingerprint
    assert threaded.output_csv == task.output_csv


def test_parallel_batch_respects_worker_cap_and_preserves_input_order(
    tmp_path,
    monkeypatch,
):
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def fake_run_task(task, *, resume, dry_run, child_threads):
        nonlocal active, maximum_active
        assert resume is False
        assert dry_run is False
        assert child_threads == 1
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return base.TaskResult(task, "completed", 0, "completed", 0.05)

    monkeypatch.setattr(parallel, "_run_task", fake_run_task)
    tasks = [_task(tmp_path, label) for label in ("a", "b", "c", "d")]
    results = parallel._run_task_batch(
        tasks,
        workers=2,
        resume=False,
        dry_run=False,
        child_threads=1,
        stop_on_error=False,
    )

    assert maximum_active == 2
    assert [result.task.case.label for result in results] == ["a", "b", "c", "d"]
    assert all(result.state == "completed" for result in results)
