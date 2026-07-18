from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import csv
import json
import traceback

from lno327.casimir.cli import execute_case

from .config import (
    DEFAULT_ATOL_J_M2,
    DEFAULT_LOG_ROOT,
    DEFAULT_MATSUBARA_CUTOFFS,
    DEFAULT_MAX_CONTEXT_WORKERS,
    DEFAULT_MEMORY_BUDGET_GB,
    DEFAULT_N_CANDIDATES,
    DEFAULT_OUTER_CUTOFFS_U,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RTOL,
    DEFAULT_SEPARATION_NM,
    DEFAULT_TEMPERATURE_K,
    RuntimeResources,
    apply_cpu_affinity,
    apply_single_thread_environment,
    case_name,
)


@dataclass(frozen=True)
class EnergyRunOptions:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    log_root: Path = DEFAULT_LOG_ROOT
    temperature_K: float = DEFAULT_TEMPERATURE_K
    separation_nm: float = DEFAULT_SEPARATION_NM
    N_candidates: tuple[int, ...] = DEFAULT_N_CANDIDATES
    matsubara_cutoffs: tuple[int, ...] = DEFAULT_MATSUBARA_CUTOFFS
    outer_cutoffs_u: tuple[float, ...] = DEFAULT_OUTER_CUTOFFS_U
    rtol: float = DEFAULT_RTOL
    atol_J_m2: float = DEFAULT_ATOL_J_M2
    memory_budget_gb: float = DEFAULT_MEMORY_BUDGET_GB
    max_context_workers: int = DEFAULT_MAX_CONTEXT_WORKERS
    parallel_mode: str = "q"
    required_consecutive_passes: int = 2
    retry_unresolved: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _case_state(run_dir: Path) -> str:
    summary = _read_json(run_dir / "summary.json")
    if summary:
        return str(summary.get("status", "result_present"))
    manifest = _read_json(run_dir / "manifest.json")
    if manifest:
        return str(manifest.get("status", "directory_present"))
    return "missing"


def _append_status(log_root: Path, row: dict[str, Any]) -> None:
    path = log_root / "energy_cases.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "timestamp_utc",
        "pairing",
        "angle_deg",
        "case",
        "action",
        "status",
        "termination_reason",
        "selected_matsubara_cutoff",
        "selected_u_max",
        "wall_seconds",
        "error_type",
        "error",
    )
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def _summary_row(
    *,
    pairing: str,
    angle_deg: int,
    case: str,
    action: str,
    run_dir: Path,
    wall_seconds: float,
    error: BaseException | None = None,
) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    manifest = _read_json(run_dir / "manifest.json")
    return {
        "timestamp_utc": _utc_now(),
        "pairing": pairing,
        "angle_deg": angle_deg,
        "case": case,
        "action": action,
        "status": summary.get("status", manifest.get("status", "unknown")),
        "termination_reason": summary.get(
            "termination_reason",
            manifest.get("termination_reason", ""),
        ),
        "selected_matsubara_cutoff": summary.get("selected_matsubara_cutoff", ""),
        "selected_u_max": summary.get("selected_u_max", ""),
        "wall_seconds": wall_seconds,
        "error_type": "" if error is None else type(error).__name__,
        "error": "" if error is None else str(error),
    }


def run_energy_cases(
    *,
    pairings: Sequence[str],
    angles_deg: Sequence[int],
    resources: RuntimeResources,
    options: EnergyRunOptions,
    profile: str,
) -> int:
    import time

    apply_single_thread_environment()
    apply_cpu_affinity(resources)
    options.output_root.mkdir(parents=True, exist_ok=True)
    options.log_root.mkdir(parents=True, exist_ok=True)

    print(f"visible CPUs: {resources.visible_cpus}", flush=True)
    print(f"selected CPUs: {resources.selected_cpus}", flush=True)
    print(f"reserved CPUs: {resources.reserved_cpus}", flush=True)
    print(f"workers: {resources.workers}", flush=True)
    print(f"profile: {profile}", flush=True)

    engineering_failures = 0

    for pairing in pairings:
        for angle_deg in angles_deg:
            case = case_name(pairing, angle_deg, profile=profile)
            run_dir = options.output_root / case
            state = _case_state(run_dir)

            if state == "completed":
                print(f"SKIP completed: {case}", flush=True)
                _append_status(
                    options.log_root,
                    _summary_row(
                        pairing=pairing,
                        angle_deg=angle_deg,
                        case=case,
                        action="skip_completed",
                        run_dir=run_dir,
                        wall_seconds=0.0,
                    ),
                )
                continue

            if state == "unresolved" and not options.retry_unresolved:
                print(f"SKIP unresolved result present: {case}", flush=True)
                _append_status(
                    options.log_root,
                    _summary_row(
                        pairing=pairing,
                        angle_deg=angle_deg,
                        case=case,
                        action="skip_unresolved",
                        run_dir=run_dir,
                        wall_seconds=0.0,
                    ),
                )
                continue

            resume = run_dir.exists()
            action = "resume" if resume else "start"
            print(f"{action.upper()}: {case}", flush=True)
            started = time.perf_counter()
            error: BaseException | None = None

            try:
                result = execute_case(
                    case=case,
                    output_root=options.output_root,
                    resume=resume,
                    pairings=(pairing,),
                    temperature_K=options.temperature_K,
                    separation_nm=options.separation_nm,
                    plate_angles_deg=(0.0, float(angle_deg)),
                    N_candidates=options.N_candidates,
                    required_consecutive_passes=options.required_consecutive_passes,
                    workers=resources.workers,
                    parallel_mode=options.parallel_mode,
                    memory_budget_gb=options.memory_budget_gb,
                    max_context_workers=options.max_context_workers,
                    matsubara_cutoff_values=options.matsubara_cutoffs,
                    cutoff_u_values=options.outer_cutoffs_u,
                    total_free_energy_rtol=options.rtol,
                    total_free_energy_atol_J_m2=options.atol_J_m2,
                )
                if result.matsubara_converged:
                    print(f"CONVERGED: {case}", flush=True)
                else:
                    print(
                        f"UNRESOLVED: {case}: {result.termination_reason}",
                        flush=True,
                    )
            except BaseException as exc:
                error = exc
                engineering_failures += 1
                print(
                    f"ENGINEERING FAILURE: {case}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                traceback.print_exc()

            wall_seconds = time.perf_counter() - started
            _append_status(
                options.log_root,
                _summary_row(
                    pairing=pairing,
                    angle_deg=angle_deg,
                    case=case,
                    action=action,
                    run_dir=run_dir,
                    wall_seconds=wall_seconds,
                    error=error,
                ),
            )

            if isinstance(error, KeyboardInterrupt):
                raise error

    return 1 if engineering_failures else 0
