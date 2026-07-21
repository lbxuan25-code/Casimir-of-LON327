"""Command-line entry for one named full adaptive Casimir run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .production import (
    FullCasimirConfig,
    FullCasimirResult,
    build_full_casimir_config,
    run_full_casimir,
)
from .run_identity import scientific_config_payload, scientific_config_sha256

_CASE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_commit() -> str | None:
    """Return this checkout's commit without consulting the caller's cwd."""

    repository_root = Path(__file__).resolve().parents[3]
    if not (repository_root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _summary(case: str, result: FullCasimirResult) -> dict[str, Any]:
    pairings: dict[str, Any] = {}
    for pairing, payload in result.pairing_results.items():
        record = dict(payload)
        pairings[pairing] = {
            name: record.get(name)
            for name in (
                "status",
                "finite_matsubara_partial_J_m2",
                "finite_matsubara_outer_error_bound_J_m2",
                "estimated_matsubara_tail_bound_J_m2",
                "estimated_total_error_J_m2",
                "total_free_energy_tolerance_J_m2",
                "matsubara_tail_ratio_envelope",
                "matsubara_tail_decay_passed",
                "finite_matsubara_budget_passed",
                "matsubara_tail_budget_passed",
                "total_free_energy_budget_passed",
            )
        }
    selected_u_max = None
    if result.cutoff_records:
        selected_u_max = result.cutoff_records[-1].get("selected_u_max")
    return {
        "schema": "full-casimir-run-summary",
        "case": case,
        "status": result.status,
        "termination_reason": result.termination_reason,
        "matsubara_converged": result.matsubara_converged,
        "outer_tail_estimated": result.outer_tail_estimated,
        "matsubara_tail_estimated": result.matsubara_tail_estimated,
        "production_casimir_allowed": result.production_casimir_allowed,
        "selected_matsubara_cutoff": result.selected_matsubara_cutoff,
        "selected_u_max": selected_u_max,
        "pairings": pairings,
        "provider_statistics": dict(result.provider_statistics),
    }


def _write_identity_sidecar(
    path: Path,
    payload: Mapping[str, Any] | None,
    *,
    resume: bool,
    label: str,
) -> None:
    if payload is None:
        return
    expected = dict(payload)
    if path.exists():
        actual = _read_json_object(path)
        if actual != expected:
            raise ValueError(f"{label} does not match the existing run: {path}")
    elif resume:
        raise ValueError(f"{label} is missing from the requested resume run: {path}")
    _atomic_json(path, expected)


def execute_case(
    *,
    case: str,
    output_root: Path,
    resume: bool,
    config_builder: Callable[..., FullCasimirConfig] = build_full_casimir_config,
    runner: Callable[[FullCasimirConfig], FullCasimirResult] = run_full_casimir,
    identity_payload: Mapping[str, Any] | None = None,
    cache_identity_payload: Mapping[str, Any] | None = None,
    **config_kwargs: Any,
) -> FullCasimirResult:
    """Execute one named run and maintain its deterministic artifact directory."""

    if not _CASE_PATTERN.fullmatch(case):
        raise ValueError(
            "case must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    run_dir = Path(output_root) / case
    result_path = run_dir / "result.json"
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"run directory already exists: {run_dir}; use --resume to reuse it"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_identity_sidecar(
        run_dir / "identity.json",
        identity_payload,
        resume=resume,
        label="physical case identity",
    )
    _write_identity_sidecar(
        run_dir / "cache" / "identity.json",
        cache_identity_payload,
        resume=resume,
        label="certified cache identity",
    )
    cache_path = run_dir / "cache" / "certified_points.json"
    config = config_builder(point_cache_path=cache_path, **config_kwargs)
    config_payload = config.as_dict()
    config_path = run_dir / "config.json"
    if resume and config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "--resume cannot verify the existing run configuration because "
                f"config.json is unreadable: {exc}"
            ) from exc
        if not isinstance(existing, Mapping) or scientific_config_payload(
            existing
        ) != scientific_config_payload(config_payload):
            raise ValueError(
                "--resume requires the exact existing scientific configuration; "
                "execution-only worker, scheduling, memory and batch settings may change"
            )
    _atomic_json(config_path, config_payload)

    previous_manifest: dict[str, Any] = {}
    manifest_path = run_dir / "manifest.json"
    if resume and manifest_path.exists():
        previous_manifest = _read_json_object(manifest_path)
    now = _utc_now()
    started_at = previous_manifest.get("started_at_utc")
    if not isinstance(started_at, str) or not started_at:
        started_at = now
    try:
        previous_attempts = int(previous_manifest.get("attempt_count", 0))
    except (TypeError, ValueError, OverflowError):
        previous_attempts = 0
    previous_attempts = max(previous_attempts, 0)
    paths = {
        "config": "config.json",
        "summary": "summary.json",
        "result": "result.json",
        "point_cache": "cache/certified_points.json",
    }
    if identity_payload is not None:
        paths["identity"] = "identity.json"
    if cache_identity_payload is not None:
        paths["cache_identity"] = "cache/identity.json"
    manifest = {
        "schema": "full-casimir-run-manifest",
        "case": case,
        "status": "running",
        "started_at_utc": started_at,
        "last_started_at_utc": now,
        "attempt_count": previous_attempts + 1,
        "git_commit": _git_commit(),
        "scientific_config_sha256": scientific_config_sha256(config_payload),
        "paths": paths,
        "resume_requested": bool(resume),
    }
    _atomic_json(run_dir / "manifest.json", manifest)

    try:
        result = runner(config)
        _atomic_json(result_path, result.as_dict())
        _atomic_json(run_dir / "summary.json", _summary(case, result))
        manifest.update(
            {
                "status": "completed" if result.matsubara_converged else "unresolved",
                "finished_at_utc": _utc_now(),
                "termination_reason": result.termination_reason,
            }
        )
        _atomic_json(run_dir / "manifest.json", manifest)
        return result
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _atomic_json(run_dir / "manifest.json", manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lno327-casimir",
        description="Run one named full adaptive LNO327 Casimir calculation.",
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/casimir/runs"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pairings", nargs="+", choices=("spm", "dwave"), default=("spm",))
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--plate-angles-deg", nargs=2, type=float, default=(0.0, 17.0))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--logdet-rtol", type=float, default=1.5e-3)
    parser.add_argument("--logdet-atol", type=float, default=1e-6)
    parser.add_argument("--certifier-q-batch-size", type=int, default=512)
    parser.add_argument(
        "--parallel-mode",
        choices=("auto", "serial", "q", "context", "wave"),
        default="auto",
    )
    parser.add_argument("--memory-budget-gb", type=float, default=0.0)
    parser.add_argument("--max-context-workers", type=int, default=0)
    parser.add_argument(
        "--N-candidates",
        nargs="+",
        type=int,
        default=(128, 192, 256),
    )
    parser.add_argument(
        "--matsubara-cutoffs",
        nargs="+",
        type=int,
        default=(1, 3, 7, 11, 15, 23, 31),
    )
    parser.add_argument(
        "--outer-cutoffs-u",
        nargs="+",
        type=float,
        default=(6.0, 10.0, 14.0, 18.0, 24.0, 30.0, 36.0, 42.0),
    )
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol-J-m2", type=float, default=1e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = execute_case(
        case=args.case,
        output_root=args.output_root,
        resume=args.resume,
        pairings=tuple(args.pairings),
        temperature_K=args.temperature_K,
        separation_nm=args.separation_nm,
        plate_angles_deg=tuple(args.plate_angles_deg),
        workers=args.workers,
        parallel_mode=args.parallel_mode,
        memory_budget_gb=args.memory_budget_gb,
        max_context_workers=args.max_context_workers,
        N_candidates=tuple(args.N_candidates),
        logdet_rtol=args.logdet_rtol,
        logdet_atol=args.logdet_atol,
        certifier_q_batch_size=args.certifier_q_batch_size,
        matsubara_cutoff_values=tuple(args.matsubara_cutoffs),
        cutoff_u_values=tuple(args.outer_cutoffs_u),
        total_free_energy_rtol=args.rtol,
        total_free_energy_atol_J_m2=args.atol_J_m2,
    )
    print(json.dumps(_summary(args.case, result), sort_keys=True, indent=2))
    return 0 if result.matsubara_converged else 2


__all__ = ["execute_case", "main"]
