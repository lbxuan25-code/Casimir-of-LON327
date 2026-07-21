from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import csv
import json
import traceback

from lno327.casimir.cli import execute_case
from lno327.casimir.production import build_full_casimir_config
from lno327.casimir.run_identity import scientific_config_payload

from .config import (
    DEFAULT_CERTIFIER_Q_BATCH_SIZE,
    DEFAULT_MAX_CONTEXT_WORKERS,
    DEFAULT_MEMORY_BUDGET_GB,
    DEFAULT_PRODUCTION_ROOT,
    RuntimeResources,
    apply_cpu_affinity,
    apply_single_thread_environment,
)
from .identity import case_sidecars, prepare_campaign, read_json_object


@dataclass(frozen=True)
class ProductionRunOptions:
    """Execution-only settings that may change between resume attempts."""

    campaign_root: Path = DEFAULT_PRODUCTION_ROOT
    certifier_q_batch_size: int = DEFAULT_CERTIFIER_Q_BATCH_SIZE
    memory_budget_gb: float = DEFAULT_MEMORY_BUDGET_GB
    max_context_workers: int = DEFAULT_MAX_CONTEXT_WORKERS
    parallel_mode: str = "q"
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


def _scientific_configs_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    return scientific_config_payload(left) == scientific_config_payload(right)


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
    result_config = result.get("config")
    expected_result_status = "adaptive_tail_bounded" if converged else "unresolved"
    authorization_consistent = bool(
        not converged
        or (
            summary.get("production_casimir_allowed") is True
            and result.get("production_casimir_allowed") is True
        )
    )
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
        and authorization_consistent
        and _summary_matches_result(summary, result)
        and (
            expected_config is None
            or (
                isinstance(result_config, Mapping)
                and _scientific_configs_match(result_config, expected_config)
            )
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
            if not stored_config or not _scientific_configs_match(
                stored_config,
                expected_config,
            ):
                return "configuration_mismatch"
        elif any(
            (run_dir / name).exists()
            for name in ("manifest.json", "summary.json", "result.json")
        ):
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


def _append_status(report_root: Path, row: dict[str, Any]) -> None:
    path = report_root / "energy_cases.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "timestamp_utc",
        "campaign_id",
        "plan_sha256",
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
    campaign_id: str = "",
    plan_sha256: str = "",
) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    manifest = _read_json(run_dir / "manifest.json")
    return {
        "timestamp_utc": _utc_now(),
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "pairing": pairing,
        "temperature_K": temperature_K,
        "separation_nm": separation_nm,
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


def _production_config_kwargs(
    *,
    plan: Mapping[str, Any],
    case_identity: Mapping[str, Any],
    resources: RuntimeResources,
    options: ProductionRunOptions,
) -> dict[str, Any]:
    policy = plan["scientific_policy"]
    model = policy["model"]
    microscopic = policy["microscopic"]
    outer = policy["outer_integration"]
    matsubara = policy["matsubara"]
    total = policy["total_free_energy"]
    return {
        "pairings": (str(case_identity["pairing"]),),
        "temperature_K": float(case_identity["temperature_K"]),
        "separation_nm": float(case_identity["separation_nm"]),
        "plate_angles_deg": tuple(
            float(value) for value in case_identity["plate_angles_deg"]
        ),
        "delta0_eV": float(model["delta0_eV"]),
        "eta_eV": float(model["eta_eV"]),
        "degeneracy": float(model["degeneracy"]),
        "N_candidates": tuple(int(value) for value in microscopic["N_candidates"]),
        "required_consecutive_passes": int(
            microscopic["required_consecutive_passes"]
        ),
        "logdet_rtol": float(microscopic["logdet_rtol"]),
        "logdet_atol": float(microscopic["logdet_atol"]),
        "workers": resources.workers,
        "parallel_mode": options.parallel_mode,
        "memory_budget_gb": options.memory_budget_gb,
        "max_context_workers": options.max_context_workers,
        "cutoff_u_values": tuple(float(value) for value in outer["cutoff_u_values"]),
        "outer_tail_start_u": float(outer["tail_start_u"]),
        "outer_tail_window_shells": int(outer["tail_window_shells"]),
        "outer_tail_ratio_max": float(outer["tail_ratio_max"]),
        "radial_budget_fraction": float(outer["radial_budget_fraction"]),
        "max_total_microscopic_q_nodes": int(
            outer["max_total_microscopic_q_nodes"]
        ),
        "matsubara_cutoff_values": tuple(
            int(value) for value in matsubara["cutoff_values"]
        ),
        "matsubara_tail_start_n": int(matsubara["tail_start_n"]),
        "matsubara_tail_window_terms": int(matsubara["tail_window_terms"]),
        "matsubara_tail_ratio_max": float(matsubara["tail_ratio_max"]),
        "max_total_microscopic_point_entries": int(
            matsubara["max_total_microscopic_point_entries"]
        ),
        "total_free_energy_rtol": float(total["rtol"]),
        "total_free_energy_atol_J_m2": float(total["atol_J_m2"]),
        "certifier_q_batch_size": options.certifier_q_batch_size,
    }


def _verify_formal_case_sidecars(
    run_dir: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_cache_identity: Mapping[str, Any],
) -> None:
    identity_path = run_dir / "identity.json"
    cache_identity_path = run_dir / "cache" / "identity.json"
    if not identity_path.is_file() or not cache_identity_path.is_file():
        raise ValueError(
            "formal resume refuses a directory without production identity sidecars: "
            f"{run_dir}"
        )
    if read_json_object(identity_path) != dict(expected_identity):
        raise ValueError(f"physical case identity mismatch: {run_dir}")
    if read_json_object(cache_identity_path) != dict(expected_cache_identity):
        raise ValueError(f"certified cache identity mismatch: {run_dir}")


def run_production_plan(
    *,
    plan: Mapping[str, Any],
    mode: str,
    resources: RuntimeResources,
    options: ProductionRunOptions,
) -> int:
    """Execute formal cases from an immutable top-level production plan."""

    import time

    apply_single_thread_environment()
    apply_cpu_affinity(resources)
    campaign_dir = prepare_campaign(
        campaign_root=options.campaign_root,
        plan=plan,
        mode=mode,
    )
    run_root = campaign_dir / "runs"
    report_root = campaign_dir / "reports"
    print(f"campaign directory: {campaign_dir}", flush=True)
    print(f"visible CPUs: {resources.visible_cpus}", flush=True)
    print(f"selected CPUs: {resources.selected_cpus}", flush=True)
    print(f"reserved CPUs: {resources.reserved_cpus}", flush=True)
    print(f"workers: {resources.workers}", flush=True)

    engineering_failures = 0
    unresolved_results = 0
    for row in plan["cases"]:
        case = str(row["case"])
        case_identity = dict(row["case_identity"])
        pairing = str(case_identity["pairing"])
        temperature_K = float(case_identity["temperature_K"])
        separation_nm = float(case_identity["separation_nm"])
        angle_deg = float(case_identity["plate_angles_deg"][1])
        run_dir = run_root / case
        identity_payload, cache_identity_payload = case_sidecars(
            case_identity=case_identity,
            campaign_sha256=str(plan["campaign_sha256"]),
            scientific_policy_sha256=str(plan["scientific_policy_sha256"]),
            git_commit=str(plan["code_identity"]["git_commit"]),
        )
        existed = run_dir.exists()
        if existed:
            _verify_formal_case_sidecars(
                run_dir,
                expected_identity=identity_payload,
                expected_cache_identity=cache_identity_payload,
            )
        config_kwargs = _production_config_kwargs(
            plan=plan,
            case_identity=case_identity,
            resources=resources,
            options=options,
        )
        expected_config = build_full_casimir_config(
            point_cache_path=run_dir / "cache" / "certified_points.json",
            **config_kwargs,
        ).as_dict()
        state = _case_state(run_dir, expected_config=expected_config)
        common = {
            "campaign_id": str(plan["campaign_id"]),
            "plan_sha256": str(plan["plan_sha256"]),
            "pairing": pairing,
            "temperature_K": temperature_K,
            "separation_nm": separation_nm,
            "angle_deg": angle_deg,
            "case": case,
            "run_dir": run_dir,
        }
        if state == "configuration_mismatch":
            error = ValueError(
                "existing case scientific configuration differs from the frozen plan"
            )
            engineering_failures += 1
            print(f"CONFIGURATION MISMATCH: {case}: {error}", flush=True)
            status_row = _summary_row(
                **common,
                action="reject_configuration_mismatch",
                wall_seconds=0.0,
                error=error,
            )
            status_row["status"] = "configuration_mismatch"
            status_row["termination_reason"] = "scientific_identity_mismatch"
            _append_status(report_root, status_row)
            if not options.continue_on_engineering_failure:
                return 1
            continue
        if state == "completed":
            print(f"SKIP completed: {case}", flush=True)
            _append_status(
                report_root,
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
                report_root,
                _summary_row(
                    **common,
                    action="skip_unresolved",
                    wall_seconds=0.0,
                ),
            )
            continue
        action = "resume" if existed else "start"
        print(f"{action.upper()}: {case}", flush=True)
        started = time.perf_counter()
        error: Exception | None = None
        try:
            result = execute_case(
                case=case,
                output_root=run_root,
                resume=existed,
                identity_payload=identity_payload,
                cache_identity_payload=cache_identity_payload,
                **config_kwargs,
            )
            if result.production_casimir_allowed:
                print(f"AUTHORIZED: {case}", flush=True)
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
                f"ENGINEERING FAILURE: {case}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
        wall_seconds = time.perf_counter() - started
        _append_status(
            report_root,
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


__all__ = ["ProductionRunOptions", "run_production_plan"]
