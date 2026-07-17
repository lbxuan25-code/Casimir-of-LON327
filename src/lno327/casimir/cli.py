"""Command-line entry for one named full adaptive Casimir run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Sequence

from .production import (
    FullCasimirConfig,
    FullCasimirResult,
    build_full_casimir_config,
    run_full_casimir,
)

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


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
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


def execute_case(
    *,
    case: str,
    output_root: Path,
    resume: bool,
    config_builder: Callable[..., FullCasimirConfig] = build_full_casimir_config,
    runner: Callable[[FullCasimirConfig], FullCasimirResult] = run_full_casimir,
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
    cache_path = run_dir / "cache" / "certified_points.json"
    config = config_builder(point_cache_path=cache_path, **config_kwargs)
    config_payload = config.as_dict()
    config_path = run_dir / "config.json"
    if resume and config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config_payload:
            raise ValueError(
                "--resume requires the exact existing run configuration; "
                "use a new case name for changed physical or numerical inputs"
            )
    _atomic_json(config_path, config_payload)

    manifest = {
        "schema": "full-casimir-run-manifest",
        "case": case,
        "status": "running",
        "started_at_utc": _utc_now(),
        "git_commit": _git_commit(),
        "paths": {
            "config": "config.json",
            "summary": "summary.json",
            "result": "result.json",
            "point_cache": "cache/certified_points.json",
        },
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
    parser.add_argument(
        "--parallel-mode",
        choices=("auto", "serial", "q", "context", "wave"),
        default="auto",
    )
    parser.add_argument("--memory-budget-gb", type=float, default=0.0)
    parser.add_argument("--max-context-workers", type=int, default=0)
    parser.add_argument("--matsubara-cutoffs", nargs="+", type=int, default=(1, 3, 7, 15, 31))
    parser.add_argument(
        "--outer-cutoffs-u",
        nargs="+",
        type=float,
        default=(6.0, 10.0, 14.0, 18.0, 24.0, 30.0, 36.0, 42.0),
    )
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol-J-m2", type=float, default=1e-10)
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
        matsubara_cutoff_values=tuple(args.matsubara_cutoffs),
        cutoff_u_values=tuple(args.outer_cutoffs_u),
        total_free_energy_rtol=args.rtol,
        total_free_energy_atol_J_m2=args.atol_J_m2,
    )
    print(json.dumps(_summary(args.case, result), sort_keys=True, indent=2))
    return 0 if result.matsubara_converged else 2


__all__ = ["execute_case", "main"]
