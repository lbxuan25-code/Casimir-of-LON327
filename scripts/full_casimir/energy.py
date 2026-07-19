from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import csv
import json
import traceback

from .config import (
    DEFAULT_ATOL_J_M2,
    DEFAULT_CERTIFIER_Q_BATCH_SIZE,
    DEFAULT_LOGDET_ATOL,
    DEFAULT_LOGDET_RTOL,
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

# These variables must be fixed before importing NumPy/BLAS through lno327.
apply_single_thread_environment()

from lno327.casimir.cli import execute_case  # noqa: E402


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
    logdet_rtol: float = DEFAULT_LOGDET_RTOL
    logdet_atol: float = DEFAULT_LOGDET_ATOL
    certifier_q_batch_size: int = DEFAULT_CERTIFIER_Q_BATCH_SIZE
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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _case_state(run_dir: Path) -> str:
    """Classify one artifact directory without trusting a stale summary first."""

    if not run_dir.exists():
        return "missing"
    manifest = _read_json(run_dir / "manifest.json")
    summary = _read_json(run_dir / "summary.json")
    manifest_status = str(manifest.get("status", "incomplete"))

    if manifest_status == "completed":
        return "completed" if bool(summary.get("matsubara_converged")) else "inconsistent"
    if manifest_status == "unresolved":
        return "unresolved"
    if manifest_status == "failed":
        return "failed"
    if manifest_status in {"running", "interrupted"}:
        return "interrupted"
    return "incomplete"


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
        handle.flush()


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
        "status": manifest.get("status", summary.get("status", "unknown")),
        "termination_reason": summary.get(
            "termination_reason",
            manifest.get("termination_reason", manifest.get("error", "")),
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
    print(f"logdet tolerance: rtol={options.logdet_rtol}, atol={options.logdet_atol}", flush=True)
    print(f"certifier q batch size: {options.certifier_q_batch_size}", flush=True)

    engineering_failures = 0
    unresolved_cases = 0

    for pairing in pairings:
        for angle_deg in angles_deg:
            case = case_name(
                pairing,
                angle_deg,
                temperature_K=options.temperature_K,
                separation_nm=options.separation_nm,
                profile=profile,
            )
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
                unresolved_cases += 1
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
            print(f"{action.upper()} ({state}): {case}", flush=True)
            started = time.perf_counter()
            error: BaseException | None = None
            converged = False

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
                    logdet_rtol=options.logdet_rtol,
                    logdet_atol=options.logdet_atol,
                    certifier_q_batch_size=options.certifier_q_batch_size,
                )
                converged = bool(result.matsubara_converged)
                if converged:
                    print(f"CONVERGED: {case}", flush=True)
                else:
                    unresolved_cases += 1
                    print(
                        f"UNRESOLVED: {case}: {result.termination_reason}",
                        flush=True,
                    )
            except KeyboardInterrupt as exc:
                error = exc
                print(f"INTERRUPTED: {case}", flush=True)
            except Exception as exc:
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
            if error is None and not converged:
                continue

    if engineering_failures:
        return 1
    if unresolved_cases:
        return 2
    return 0
