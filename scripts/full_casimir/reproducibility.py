"""Reproducibility, source identity and scientific artifact proof records."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from lno327.casimir.run_identity import sha256_json

from .config import DEFAULT_PRODUCTION_ROOT, REPO_ROOT, RuntimeResources
from .identity import atomic_json, campaign_directory, read_json_object, verify_plan_payload


SOURCE_PROOF_SCHEMA = "full-casimir-source-proof-v1"
ARTIFACT_MANIFEST_SCHEMA = "full-casimir-artifact-manifest-v1"
REPRODUCIBILITY_SCHEMA = "full-casimir-reproducibility-record-v1"
VERIFICATION_SCHEMA = "full-casimir-proof-verification-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"cannot execute Git for source proof: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Git source-proof command failed: {' '.join(args)}: {detail}")
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _signed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    return {**unsigned, "payload_sha256": sha256_json(unsigned)}


def _verify_signed_payload(payload: Mapping[str, Any]) -> bool:
    stored = payload.get("payload_sha256")
    if not isinstance(stored, str) or not stored:
        return False
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    return stored == sha256_json(unsigned)


def build_source_proof(*, repository_root: Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    commit = _run_git(root, "rev-parse", "HEAD")
    tree = _run_git(root, "rev-parse", "HEAD^{tree}")
    status = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError("source proof requires a clean tracked worktree")
    raw_files = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if raw_files.returncode != 0:
        raise RuntimeError("cannot enumerate tracked source files")
    paths = [
        value.decode("utf-8", errors="surrogateescape")
        for value in raw_files.stdout.split(b"\0")
        if value
    ]
    records: list[dict[str, Any]] = []
    for relative in sorted(paths):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"tracked source file is missing: {relative}")
        records.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    remote = ""
    try:
        remote = _run_git(root, "remote", "get-url", "origin")
    except RuntimeError:
        remote = ""
    file_set_sha = sha256_json(records)
    return _signed_payload(
        {
            "schema": SOURCE_PROOF_SCHEMA,
            "generated_at_utc": _utc_now(),
            "git_commit": commit,
            "git_tree": tree,
            "tracked_worktree_clean": True,
            "remote_origin": remote,
            "tracked_file_count": len(records),
            "tracked_file_set_sha256": file_set_sha,
            "tracked_files": records,
        }
    )


def _scientific_artifact_paths(campaign_dir: Path) -> tuple[Path, ...]:
    root = Path(campaign_dir)
    candidates: list[Path] = []
    for fixed in ("campaign.json", "policy.json"):
        path = root / fixed
        if path.is_file():
            candidates.append(path)
    candidates.extend(path for path in (root / "plans").glob("*.json") if path.is_file())
    for run_dir in sorted((root / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        for relative in (
            "identity.json",
            "config.json",
            "manifest.json",
            "summary.json",
            "result.json",
            "cache/identity.json",
            "cache/certified_points.json",
            "cache/certified_points.telemetry.json",
        ):
            path = run_dir / relative
            if path.is_file():
                candidates.append(path)
    report = root / "reports" / "energy_cases.csv"
    if report.is_file():
        candidates.append(report)
    return tuple(sorted(set(candidates), key=lambda path: str(path.relative_to(root))))


def build_artifact_manifest(*, campaign_dir: Path) -> dict[str, Any]:
    root = Path(campaign_dir).resolve()
    records = [
        {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in _scientific_artifact_paths(root)
    ]
    return _signed_payload(
        {
            "schema": ARTIFACT_MANIFEST_SCHEMA,
            "generated_at_utc": _utc_now(),
            "scope": "authoritative scientific and identity artifacts",
            "progress_and_lock_files_excluded": True,
            "artifact_count": len(records),
            "artifact_set_sha256": sha256_json(records),
            "artifacts": records,
        }
    )


def _package_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for name in ("numpy", "scipy", "matplotlib", "pytest"):
        try:
            output[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return output


def _environment_payload(resources: RuntimeResources) -> dict[str, Any]:
    thread_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_DYNAMIC",
        "MKL_DYNAMIC",
    )
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "package_versions": _package_versions(),
        "thread_environment": {
            name: os.environ.get(name) for name in thread_names
        },
        "resources": {
            "visible_cpus": list(resources.visible_cpus),
            "selected_cpus": list(resources.selected_cpus),
            "reserved_cpus": list(resources.reserved_cpus),
            "workers": resources.workers,
        },
    }


def write_reproducibility_bundle(
    *,
    campaign_dir: Path,
    plan: Mapping[str, Any],
    resources: RuntimeResources,
    execution_options: Mapping[str, Any],
    final_exit_code: int,
    attempt_count: int,
    repository_root: Path = REPO_ROOT,
) -> tuple[Path, Path, Path]:
    root = Path(campaign_dir)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    source = build_source_proof(repository_root=repository_root)
    source_path = reports / "source_proof.json"
    atomic_json(source_path, source)
    artifacts = build_artifact_manifest(campaign_dir=root)
    artifacts_path = reports / "artifact_manifest.json"
    atomic_json(artifacts_path, artifacts)
    record = _signed_payload(
        {
            "schema": REPRODUCIBILITY_SCHEMA,
            "generated_at_utc": _utc_now(),
            "campaign_id": str(plan["campaign_id"]),
            "campaign_sha256": str(plan["campaign_sha256"]),
            "scientific_policy_sha256": str(plan["scientific_policy_sha256"]),
            "plan_sha256": str(plan["plan_sha256"]),
            "planned_git_commit": str(plan["code_identity"]["git_commit"]),
            "source_proof_sha256": str(source["payload_sha256"]),
            "source_file_set_sha256": str(source["tracked_file_set_sha256"]),
            "artifact_manifest_sha256": str(artifacts["payload_sha256"]),
            "artifact_set_sha256": str(artifacts["artifact_set_sha256"]),
            "environment": _environment_payload(resources),
            "execution_options": dict(execution_options),
            "final_exit_code": int(final_exit_code),
            "attempt_count": int(attempt_count),
            "scientific_identity_separate_from_execution_environment": True,
            "verification_command": (
                "python -m scripts.full_casimir proof --campaign "
                + str(plan["campaign_id"])
            ),
        }
    )
    record_path = reports / "reproducibility.json"
    atomic_json(record_path, record)
    return source_path, artifacts_path, record_path


def verify_reproducibility_bundle(
    *,
    campaign_dir: Path,
    repository_root: Path = REPO_ROOT,
    check_current_source: bool = True,
) -> dict[str, Any]:
    root = Path(campaign_dir)
    reports = root / "reports"
    source = read_json_object(reports / "source_proof.json")
    artifacts = read_json_object(reports / "artifact_manifest.json")
    record = read_json_object(reports / "reproducibility.json")
    failures: list[str] = []
    for label, payload, schema in (
        ("source_proof", source, SOURCE_PROOF_SCHEMA),
        ("artifact_manifest", artifacts, ARTIFACT_MANIFEST_SCHEMA),
        ("reproducibility", record, REPRODUCIBILITY_SCHEMA),
    ):
        if payload.get("schema") != schema:
            failures.append(f"{label}: schema mismatch")
        if not _verify_signed_payload(payload):
            failures.append(f"{label}: payload digest mismatch")
    plan_sha = record.get("plan_sha256")
    plan_path = root / "plans" / f"{plan_sha}.json"
    try:
        plan = read_json_object(plan_path)
        verify_plan_payload(plan, expected_sha256=str(plan_sha))
    except (OSError, TypeError, ValueError) as exc:
        failures.append(f"plan: {exc}")
        plan = {}
    if plan:
        for name in (
            "campaign_id",
            "campaign_sha256",
            "scientific_policy_sha256",
        ):
            if record.get(name) != plan.get(name):
                failures.append(f"reproducibility: {name} differs from registered plan")
        if record.get("planned_git_commit") != plan.get("code_identity", {}).get(
            "git_commit"
        ):
            failures.append("reproducibility: planned Git commit differs from plan")
    if record.get("source_proof_sha256") != source.get("payload_sha256"):
        failures.append("reproducibility: source proof link mismatch")
    if record.get("artifact_manifest_sha256") != artifacts.get("payload_sha256"):
        failures.append("reproducibility: artifact manifest link mismatch")
    for row in artifacts.get("artifacts", ()):  # type: ignore[assignment]
        if not isinstance(row, Mapping):
            failures.append("artifact_manifest: malformed artifact row")
            continue
        relative = str(row.get("path", ""))
        path = root / relative
        if not path.is_file():
            failures.append(f"artifact missing: {relative}")
            continue
        if _sha256_file(path) != row.get("sha256"):
            failures.append(f"artifact digest mismatch: {relative}")
    current_source: dict[str, Any] | None = None
    if check_current_source:
        try:
            current_source = build_source_proof(repository_root=repository_root)
        except RuntimeError as exc:
            failures.append(f"current source: {exc}")
        else:
            if current_source.get("git_commit") != source.get("git_commit"):
                failures.append("current source Git commit differs from recorded source")
            if current_source.get("git_tree") != source.get("git_tree"):
                failures.append("current source Git tree differs from recorded source")
            if current_source.get("tracked_file_set_sha256") != source.get(
                "tracked_file_set_sha256"
            ):
                failures.append("current tracked source file set differs from record")
    return {
        "schema": VERIFICATION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "campaign_id": record.get("campaign_id", root.name),
        "passed": not failures,
        "current_source_checked": bool(check_current_source),
        "artifact_count": int(artifacts.get("artifact_count", 0) or 0),
        "failures": failures,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir proof",
        description="Verify a campaign's source and authoritative artifact proof bundle.",
    )
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--skip-current-source-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign_dir = campaign_directory(args.campaign_root, args.campaign)
    try:
        result = verify_reproducibility_bundle(
            campaign_dir=campaign_dir,
            check_current_source=not args.skip_current_source_check,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"PROOF FAILED: {type(exc).__name__}: {exc}")
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        print(f"campaign: {result['campaign_id']}")
        print(f"passed: {result['passed']}")
        print(f"artifact_count: {result['artifact_count']}")
        for failure in result["failures"]:
            print(f"FAIL: {failure}")
    return 0 if result["passed"] else 1


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA",
    "REPRODUCIBILITY_SCHEMA",
    "SOURCE_PROOF_SCHEMA",
    "VERIFICATION_SCHEMA",
    "build_artifact_manifest",
    "build_source_proof",
    "main",
    "verify_reproducibility_bundle",
    "write_reproducibility_bundle",
]
