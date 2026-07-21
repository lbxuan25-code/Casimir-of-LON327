"""Internal artifact writer used by the unified production campaign runner."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping

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
                "outer_tail_certificate_path",
                "matsubara_tail_certificate_path",
                "matsubara_tail_ratio_envelope",
                "matsubara_tail_holdout_ratio",
                "matsubara_tail_decay_passed",
                "matsubara_tail_holdout_passed",
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
        "formal_policy_passed": bool(
            getattr(result, "formal_policy_passed", False)
        ),
        "error_budget_closed": bool(
            getattr(result, "error_budget_closed", False)
        ),
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
    """Execute one plan-owned case and maintain its deterministic artifacts."""

    if not _CASE_PATTERN.fullmatch(case):
        raise ValueError(
            "case must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    run_dir = Path(output_root) / case
    result_path = run_dir / "result.json"
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"run directory already exists: {run_dir}; use formal --resume to reuse it"
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
                "formal resume cannot verify config.json because it is unreadable: "
                f"{exc}"
            ) from exc
        if not isinstance(existing, Mapping) or scientific_config_payload(
            existing
        ) != scientific_config_payload(config_payload):
            raise ValueError(
                "formal resume requires the exact existing scientific configuration; "
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
    _atomic_json(manifest_path, manifest)

    try:
        result = runner(config)
        _atomic_json(result_path, result.as_dict())
        _atomic_json(run_dir / "summary.json", _summary(case, result))
        authorized = bool(result.production_casimir_allowed)
        numerically_converged = bool(result.matsubara_converged)
        manifest_status = (
            "completed"
            if authorized
            else "diagnostic_only"
            if numerically_converged
            else "unresolved"
        )
        manifest.update(
            {
                "status": manifest_status,
                "finished_at_utc": _utc_now(),
                "termination_reason": result.termination_reason,
                "numerically_converged": numerically_converged,
                "formal_policy_passed": bool(
                    getattr(result, "formal_policy_passed", False)
                ),
                "error_budget_closed": bool(
                    getattr(result, "error_budget_closed", False)
                ),
                "production_casimir_allowed": authorized,
            }
        )
        _atomic_json(manifest_path, manifest)
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
        _atomic_json(manifest_path, manifest)
        raise


__all__ = ["execute_case"]
