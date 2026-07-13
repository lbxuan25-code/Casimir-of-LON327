"""Parallel orchestration for the positive-Matsubara d-wave certification scan.

Independent q cases are executed concurrently at each certification stage.  Stage
boundaries remain strict: all screen tasks finish before difficult cases are selected
for tight integration, and all tight tasks finish before fixed-Gauss references are
selected.  Each child process receives an explicit small BLAS/OpenMP thread budget so
process-level parallelism does not become nested oversubscription.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Iterable, Sequence

from validation.commands.matsubara import dwave_orbit_certification_scan as base

_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_PRINT_LOCK = threading.Lock()


def _emit(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _parse_args() -> argparse.Namespace:
    runtime_parser = argparse.ArgumentParser(add_help=False)
    runtime_parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="maximum independent q-case child processes per stage (default: 2)",
    )
    runtime_parser.add_argument(
        "--child-threads",
        type=int,
        default=1,
        help="BLAS/OpenMP threads assigned to each child process (default: 1)",
    )
    runtime_args, remaining = runtime_parser.parse_known_args(sys.argv[1:])

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = base._parse_args()
    finally:
        sys.argv = original_argv

    if runtime_args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if runtime_args.child_threads <= 0:
        raise SystemExit("--child-threads must be positive")
    args.workers = int(runtime_args.workers)
    args.child_threads = int(runtime_args.child_threads)
    return args


def _child_environment(child_threads: int) -> dict[str, str]:
    threads = str(int(child_threads))
    environment = os.environ.copy()
    for name in _THREAD_ENVIRONMENT_VARIABLES:
        environment[name] = threads
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["MKL_DYNAMIC"] = "FALSE"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _runtime_task(task: base.TaskSpec, *, child_threads: int) -> base.TaskSpec:
    """Include non-default child threading in the resume fingerprint.

    A one-thread child preserves the fingerprints produced by the original serial
    scanner, so already-completed validation outputs remain resumable after this
    parallel orchestration layer is introduced.
    """

    if int(child_threads) == 1:
        return task
    fingerprint = base._fingerprint(
        {
            "schema": "certification_runtime_task_v1",
            "base_fingerprint": task.fingerprint,
            "child_threads": int(child_threads),
        }
    )
    return replace(task, fingerprint=fingerprint)


def _run_task(
    task: base.TaskSpec,
    *,
    resume: bool,
    dry_run: bool,
    child_threads: int,
) -> base.TaskResult:
    tag = f"{task.case.label}/{task.stage}"
    if resume and base._resume_match(task):
        _emit(f"[skip] {tag}: exact completed fingerprint")
        return base.TaskResult(
            task,
            "skipped",
            0,
            "exact completed fingerprint",
            0.0,
        )

    _emit(f"[run]  {tag}")
    _emit("       " + " ".join(task.command))
    if dry_run:
        return base.TaskResult(task, "planned", None, "dry run", 0.0)

    task.output_csv.parent.mkdir(parents=True, exist_ok=True)
    base._atomic_write_json(
        task.manifest_path,
        base._task_payload(
            task,
            state="running",
            returncode=None,
            message="",
            wall=0.0,
        ),
    )
    started = time.perf_counter()
    returncode: int | None = None
    message = ""
    try:
        with task.log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(task.command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=_child_environment(child_threads),
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                _emit(f"[{tag}] {line.rstrip()}")
            returncode = int(process.wait())
    except OSError as exc:
        returncode = -1
        message = f"failed to start child process: {exc}"

    wall = float(time.perf_counter() - started)
    if returncode == 0 and task.output_csv.is_file() and task.output_json.is_file():
        state = "completed"
        message = "completed"
    else:
        state = "failed"
        if not message:
            missing = [
                str(path)
                for path in (task.output_csv, task.output_json)
                if not path.is_file()
            ]
            detail = f"; missing outputs={missing}" if missing else ""
            message = f"child return code {returncode}{detail}"

    base._atomic_write_json(
        task.manifest_path,
        base._task_payload(
            task,
            state=state,
            returncode=returncode,
            message=message,
            wall=wall,
        ),
    )
    _emit(f"[{state}] {tag}: {message}; wall={wall:.3f}s")
    return base.TaskResult(task, state, returncode, message, wall)


def _failed_result(task: base.TaskSpec, exc: BaseException) -> base.TaskResult:
    message = f"parallel worker raised {type(exc).__name__}: {exc}"
    try:
        base._atomic_write_json(
            task.manifest_path,
            base._task_payload(
                task,
                state="failed",
                returncode=-1,
                message=message,
                wall=0.0,
            ),
        )
    except OSError:
        pass
    _emit(f"[failed] {task.case.label}/{task.stage}: {message}")
    return base.TaskResult(task, "failed", -1, message, 0.0)


def _run_task_batch(
    tasks: Iterable[base.TaskSpec],
    *,
    workers: int,
    resume: bool,
    dry_run: bool,
    child_threads: int,
    stop_on_error: bool,
) -> list[base.TaskResult]:
    task_values = [
        _runtime_task(task, child_threads=child_threads)
        for task in tasks
    ]
    if not task_values:
        return []

    effective_workers = min(int(workers), len(task_values))
    _emit(
        f"stage dispatch: tasks={len(task_values)}, workers={effective_workers}, "
        f"child_threads={child_threads}"
    )

    if dry_run or effective_workers == 1:
        results = [
            _run_task(
                task,
                resume=resume,
                dry_run=dry_run,
                child_threads=child_threads,
            )
            for task in task_values
        ]
    else:
        ordered: list[base.TaskResult | None] = [None] * len(task_values)
        with ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="dwave-certification",
        ) as executor:
            futures = {
                executor.submit(
                    _run_task,
                    task,
                    resume=resume,
                    dry_run=False,
                    child_threads=child_threads,
                ): index
                for index, task in enumerate(task_values)
            }
            for future in as_completed(futures):
                index = futures[future]
                task = task_values[index]
                try:
                    ordered[index] = future.result()
                except BaseException as exc:  # fail closed while retaining other cases
                    ordered[index] = _failed_result(task, exc)
        results = [result for result in ordered if result is not None]

    if stop_on_error:
        failed = next((result for result in results if result.state == "failed"), None)
        if failed is not None:
            raise SystemExit(
                f"{failed.task.case.label}/{failed.task.stage}: {failed.message}"
            )
    return results


def _print_plan(args: argparse.Namespace, cases: Sequence[base.CaseSpec]) -> None:
    print("positive d-wave q-orbit certification plan")
    for case in cases:
        print(f"  {case.label:20s} q=({case.mx},{case.my}) level={case.level}")
    cpu_count = os.cpu_count()
    cpu_text = "unknown" if cpu_count is None else str(cpu_count)
    print(
        f"parallel runtime: workers={args.workers}; "
        f"child_threads={args.child_threads}; visible_cpus={cpu_text}"
    )
    print(
        f"dynamic additions: tight top-k={args.t512_top_k}; "
        f"Gauss top-k={args.gauss_top_k}",
        flush=True,
    )
    if cpu_count is not None and args.workers * args.child_threads > cpu_count:
        print(
            "warning: workers * child_threads exceeds visible CPU count; "
            "nested oversubscription may reduce performance",
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    cases: tuple[base.CaseSpec, ...] = args.cases
    _print_plan(args, cases)

    task_results: list[base.TaskResult] = []
    screen_batch = _run_task_batch(
        (base._make_periodic_task(args, case, tight=False) for case in cases),
        workers=args.workers,
        resume=args.resume,
        dry_run=args.dry_run,
        child_threads=args.child_threads,
        stop_on_error=args.stop_on_error,
    )
    task_results.extend(screen_batch)
    screen_results = {result.task.case.label: result for result in screen_batch}

    if args.dry_run:
        mandatory_tight = tuple(
            case.label for case in cases if case.mandatory_tight
        )
        mandatory_gauss = tuple(
            case.label for case in cases if case.mandatory_gauss
        )
        print(f"mandatory tight cases = {mandatory_tight}")
        print(f"mandatory Gauss cases = {mandatory_gauss}")
        print("dynamic top-k selections are resolved after screen results")
        return

    metrics_by_label = {
        label: base._metrics(result)
        for label, result in screen_results.items()
    }
    tight_labels: tuple[str, ...] = ()
    gauss_labels: tuple[str, ...] = ()
    tight_results: dict[str, base.TaskResult] = {}

    if not args.screen_only:
        tight_labels = base._select_labels(
            cases,
            metrics_by_label,
            mandatory_attribute="mandatory_tight",
            top_k=args.t512_top_k,
            include_failures=True,
        )
        _emit(f"selected tight cases: {tight_labels}")
        case_by_label = {case.label: case for case in cases}
        tight_batch = _run_task_batch(
            (
                base._make_periodic_task(
                    args,
                    case_by_label[label],
                    tight=True,
                )
                for label in tight_labels
            ),
            workers=args.workers,
            resume=args.resume,
            dry_run=False,
            child_threads=args.child_threads,
            stop_on_error=args.stop_on_error,
        )
        task_results.extend(tight_batch)
        tight_results = {
            result.task.case.label: result
            for result in tight_batch
        }

        if not args.no_gauss:
            tight_metrics = {
                label: base._metrics(result)
                for label, result in tight_results.items()
            }
            eligible_cases = tuple(
                case
                for case in cases
                if case.label in tight_metrics
                and tight_metrics[case.label].task_usable
                and tight_metrics[case.label].all_point_pipelines_passed
            )
            eligible_metrics = {
                case.label: tight_metrics[case.label]
                for case in eligible_cases
            }
            gauss_labels = base._select_labels(
                eligible_cases,
                eligible_metrics,
                mandatory_attribute="mandatory_gauss",
                top_k=args.gauss_top_k,
                include_failures=False,
            )
            _emit(f"selected fixed-Gauss cases: {gauss_labels}")
            gauss_batch = _run_task_batch(
                (
                    base._make_gauss_task(
                        args,
                        case_by_label[label],
                        tight_results[label].task.output_csv,
                    )
                    for label in gauss_labels
                ),
                workers=args.workers,
                resume=args.resume,
                dry_run=False,
                child_threads=args.child_threads,
                stop_on_error=args.stop_on_error,
            )
            task_results.extend(gauss_batch)

    base._write_scan_outputs(
        args,
        cases,
        tight_labels,
        gauss_labels,
        task_results,
    )


if __name__ == "__main__":
    main()
