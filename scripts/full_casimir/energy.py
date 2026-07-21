from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json
import traceback

from lno327.casimir.cli import execute_case
from lno327.casimir.production import build_full_casimir_config

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
    continue_on_engineering_failure: bool = False


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


def _summary_matches_result(
    summary: Mapping[str, Any],
    result: Mapping[str, Any],
) -> bool:
    summary_pairings = summary.get("pairings")
    result_pairings = result.get("pairing_results")
    if not isinstance(summary_pairings, Mapping) or not isinstance(
        result_pairings, Mapping
    ):
        return False
    if set(summary_pairings) != set(result_pairings):
        return False
    for pairing, summary_record in summary_pairings.items():
        result_record = result_pairings.get(pairing)
        if not isinstance(summary_record, Mapping) or not isinstance(
            result_record, Mapping
        ):
            return False
        if any(result_record.get(name) != value for name, value in summary_record.items()):
            return False
    return bool(
        summary.get("selected_matsubara_cutoff")
        == result.get("selected_matsubara_cutoff")
        and summary.get("production_casimir_allowed")
        == result.get("production_casimir_allowed")
        and summary.get("provider_statistics") == result.get("provider_statistics")
    )


def _artifacts_consistent(
    run_dir: Path,
    *,
    manifest_status: str,
    converged: bool,
    expected_config: Mapping[str, Any] | None = None,
) -> bool:
    summary = _read_json(run_dir / "summary.json")
    manifest = _read_json(run_dir / "manifest.json")
    result = _read_json(run_dir / "result.json")
    expected_result_status = "adaptive_tail_bounded" if converged else "unresolved"
    return bool(
        summary.get("schema") == "full-casimir-run-summary"
        and manifest.get("schema") == "full-casimir-run-manifest"
        and result.get("schema") == "adaptive-matsubara-casimir-result-v1"
        and summary.get("case") == run_dir.name
        and manifest.get("case") == run_dir.name
        and manifest.get("status") == manifest_status
        and bool(summary.get("matsubara_converged")) is converged
        and bool(result.get("matsubara_converged")) is converged
        and summary.get("status") == expected_result_status
        and result.get("status") == expected_result_status
        and summary.get("termination_reason") == result.get("termination_reason")
        and manifest.get("termination_reason") == result.get("termination_reason")
        and _summary_matches_result(summary, result)
        and (
            expected_config is None
            or result.get("config") == dict(expected_config)
        )
    )


def _case_state(
    run_dir: Path,
    *,
    expected_config: Mapping[str, Any] | None = None,
) -> str:
    if expected_config is not None and run_dir.exists():
        config_path = run_dir / "config.json"
        if config_path.exists():
            stored_config = _read_json(config_path)
            if not stored_config or stored_config != dict(expected_config):
                return "configuration_mismatch"
        elif any(
            (run_dir / name).exists()
            for name in ("manifest.json", "summary.json", "result.json")
        ):
            # A cache-only target created by the v2->v3 migration is a valid seed.
            # Missing configuration beside actual run artifacts is not.
            return "configuration_mismatch"
    manifest = _read_json(run_dir / "manifest.json")
    summary = _read_json(run_dir / "summary.json")
    if manifest.get("status") == "completed":
        return (
            "completed"
            if _artifacts_consistent(
                run_dir,
                manifest_status="completed",
                converged=True,
                expected_config=expected_config,
            )
            else "artifact_inconsistent"
        )
    if manifest.get("status") == "unresolved" or summary.get("status") == "unresolved":
        return (
            "unresolved"
            if _artifacts_consistent(
                run_dir,
                manifest_status="unresolved",
                converged=False,
                expected_config=expected_config,
            )
            else "artifact_inconsistent"
        )
    if manifest.get("status") == "failed":
        return "failed"
    if manifest.get("status") == "running":
        return "interrupted"
    if (run_dir / "cache" / "certified_points.json").is_file():
        return "cache_seeded"
    return "directory_present" if run_dir.exists() else "missing"


def _requested_config_payload(
    *,
    pairing: str,
    angle_deg: float,
    separation_nm: float,
    run_dir: Path,
    resources: RuntimeResources,
    options: EnergyRunOptions,
) -> dict[str, Any]:
    return build_full_casimir_config(
        point_cache_path=run_dir / "cache" / "certified_points.json",
        pairings=(pairing,),
        temperature_K=options.temperature_K,
        separation_nm=separation_nm,
        plate_angles_deg=(0.0, float(angle_deg)),
        N_candidates=options.N_candidates,
        required_consecutive_passes=options.required_consecutive_passes,
        logdet_rtol=options.logdet_rtol,
        logdet_atol=options.logdet_atol,
        certifier_q_batch_size=options.certifier_q_batch_size,
        workers=resources.workers,
        parallel_mode=options.parallel_mode,
        memory_budget_gb=options.memory_budget_gb,
        max_context_workers=options.max_context_workers,
        matsubara_cutoff_values=options.matsubara_cutoffs,
        cutoff_u_values=options.outer_cutoffs_u,
        total_free_energy_rtol=options.rtol,
        total_free_energy_atol_J_m2=options.atol_J_m2,
    ).as_dict()


def _append_status(log_root: Path, row: dict[str, Any]) -> None:
    path = log_root / "energy_cases.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "timestamp_utc",
        "pairing",
        "temperature_K",
        "separation_nm",
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
    temperature_K: float,
    separation_nm: float,
    angle_deg: float,
    case: str,
    action: str,
    run_dir: Path,
    wall_seconds: float,
    error: Exception | None = None,
) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    manifest = _read_json(run_dir / "manifest.json")
    return {
        "timestamp_utc": _utc_now(),
        "pairing": pairing,
        "temperature_K": temperature_K,
        "separation_nm": separation_nm,
        "angle_deg": angle_deg,
        "case": case,
        "action": action,
        "status": summary.get("status", manifest.get("status", "unknown")),
        "termination_reason": summary.get(
            "termination_reason", manifest.get("termination_reason", "")
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
    angles_deg: Sequence[int | float],
    resources: RuntimeResources,
    options: EnergyRunOptions,
    profile: str,
    distances_nm: Sequence[int | float] | None = None,
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
    distances = (
        tuple(float(value) for value in distances_nm)
        if distances_nm is not None
        else (float(options.separation_nm),)
    )
    angles = tuple(float(value) for value in angles_deg)
    if not distances:
        raise ValueError("at least one distance is required")
    if not angles:
        raise ValueError("at least one angle is required")
    engineering_failures = 0
    unresolved_results = 0
    for pairing in pairings:
        for separation_nm in distances:
            for angle_deg in angles:
                case = case_name(
                    pairing,
                    angle_deg,
                    temperature_K=options.temperature_K,
                    separation_nm=separation_nm,
                    profile=profile,
                )
                run_dir = options.output_root / case
                expected_config = _requested_config_payload(
                    pairing=pairing,
                    angle_deg=angle_deg,
                    separation_nm=separation_nm,
                    run_dir=run_dir,
                    resources=resources,
                    options=options,
                )
                state = _case_state(run_dir, expected_config=expected_config)
                common = {
                    "pairing": pairing,
                    "temperature_K": float(options.temperature_K),
                    "separation_nm": separation_nm,
                    "angle_deg": angle_deg,
                    "case": case,
                    "run_dir": run_dir,
                }
                if state == "configuration_mismatch":
                    error = ValueError(
                        "existing case configuration differs from the requested run; "
                        "choose a new --profile or restore the original options"
                    )
                    engineering_failures += 1
                    print(f"CONFIGURATION MISMATCH: {case}: {error}", flush=True)
                    row = _summary_row(
                        **common,
                        action="reject_configuration_mismatch",
                        wall_seconds=0.0,
                        error=error,
                    )
                    row["status"] = "configuration_mismatch"
                    row["termination_reason"] = (
                        "requested_configuration_does_not_match_existing_case"
                    )
                    _append_status(options.log_root, row)
                    if not options.continue_on_engineering_failure:
                        return 1
                    continue
                if state == "completed":
                    print(f"SKIP completed: {case}", flush=True)
                    _append_status(
                        options.log_root,
                        _summary_row(
                            **common,
                            action="skip_completed",
                            wall_seconds=0.0,
                        ),
                    )
                    continue
                if state == "unresolved" and not options.retry_unresolved:
                    unresolved_results += 1
                    print(f"SKIP unresolved result present: {case}", flush=True)
                    _append_status(
                        options.log_root,
                        _summary_row(
                            **common,
                            action="skip_unresolved",
                            wall_seconds=0.0,
                        ),
                    )
                    continue
                if state == "artifact_inconsistent":
                    print(f"RESUME inconsistent run artifacts: {case}", flush=True)
                resume = run_dir.exists()
                action = "resume" if resume else "start"
                print(f"{action.upper()}: {case}", flush=True)
                started = time.perf_counter()
                error: Exception | None = None
                try:
                    result = execute_case(
                        case=case,
                        output_root=options.output_root,
                        resume=resume,
                        pairings=(pairing,),
                        temperature_K=options.temperature_K,
                        separation_nm=separation_nm,
                        plate_angles_deg=(0.0, angle_deg),
                        N_candidates=options.N_candidates,
                        required_consecutive_passes=options.required_consecutive_passes,
                        logdet_rtol=options.logdet_rtol,
                        logdet_atol=options.logdet_atol,
                        certifier_q_batch_size=options.certifier_q_batch_size,
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
                        unresolved_results += 1
                        print(
                            f"UNRESOLVED: {case}: {result.termination_reason}",
                            flush=True,
                        )
                except Exception as exc:
                    error = exc
                    engineering_failures += 1
                    print(
                        f"ENGINEERING FAILURE: {case}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                wall_seconds = time.perf_counter() - started
                _append_status(
                    options.log_root,
                    _summary_row(
                        **common,
                        action=action,
                        wall_seconds=wall_seconds,
                        error=error,
                    ),
                )
                if error is not None and not options.continue_on_engineering_failure:
                    return 1
    if engineering_failures:
        return 1
    return 2 if unresolved_results else 0
