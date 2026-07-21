"""Fail-closed production campaign, plan and case identity contracts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
from typing import Any, Mapping

from lno327.casimir.run_identity import sha256_json

from .config import REPO_ROOT

CAMPAIGN_SCHEMA = "full-casimir-production-campaign-v1"
PLAN_SCHEMA = "full-casimir-production-plan-v1"
POLICY_SCHEMA = "full-casimir-scientific-policy-v1"
CASE_IDENTITY_SCHEMA = "full-casimir-physical-case-identity-v1"
CACHE_IDENTITY_SCHEMA = "full-casimir-certified-cache-identity-v1"
CONTRACT_VERSION = "full-casimir-production-contract-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def git_code_identity(
    *,
    repository_root: Path = REPO_ROOT,
    require_clean: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root)
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"cannot inspect Git identity: {exc}") from exc
    commit_sha = commit.stdout.strip()
    if commit.returncode != 0 or not commit_sha:
        raise RuntimeError("production planning requires a Git checkout with a resolvable HEAD")
    if status.returncode != 0:
        raise RuntimeError("cannot inspect tracked Git worktree state")
    dirty_lines = [line for line in status.stdout.splitlines() if line.strip()]
    clean = not dirty_lines
    if require_clean and not clean:
        raise RuntimeError(
            "production planning requires a clean tracked worktree; "
            f"modified paths: {dirty_lines}"
        )
    return {
        "git_commit": commit_sha,
        "tracked_worktree_clean": clean,
    }


def build_campaign_identity(
    *,
    scientific_policy: Mapping[str, Any],
    code_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if scientific_policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("scientific policy schema mismatch")
    commit = code_identity.get("git_commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError("code identity requires git_commit")
    policy_sha = sha256_json(scientific_policy)
    identity_payload = {
        "contract_version": CONTRACT_VERSION,
        "scientific_policy_sha256": policy_sha,
        "git_commit": commit,
    }
    full_sha = sha256_json(identity_payload)
    return {
        "campaign_id": f"campaign-{full_sha[:12]}",
        "campaign_sha256": full_sha,
        "scientific_policy_sha256": policy_sha,
        "identity_payload": identity_payload,
    }


def build_case_identity(
    *,
    campaign_id: str,
    pairing: str,
    temperature_K: float,
    separation_nm: float,
    plate_angles_deg: tuple[float, float],
) -> dict[str, Any]:
    payload = {
        "schema": CASE_IDENTITY_SCHEMA,
        "campaign_id": str(campaign_id),
        "pairing": str(pairing),
        "temperature_K": float(temperature_K),
        "separation_nm": float(separation_nm),
        "plate_angles_deg": [float(value) for value in plate_angles_deg],
    }
    return {
        **payload,
        "case_identity_sha256": sha256_json(payload),
    }


def finalize_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(payload)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("plan schema mismatch")
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = sha256_json(plan)
    return plan


def verify_plan_payload(plan: Mapping[str, Any], *, expected_sha256: str) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported production plan schema")
    stored = plan.get("plan_sha256")
    if not isinstance(stored, str) or not stored:
        raise ValueError("production plan is missing plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    calculated = sha256_json(unsigned)
    if stored != calculated:
        raise ValueError("production plan self-digest does not verify")
    if str(expected_sha256) != stored:
        raise ValueError(
            "confirmed plan SHA does not match the production plan: "
            f"expected {expected_sha256}, stored {stored}"
        )


def campaign_directory(campaign_root: Path, campaign_id: str) -> Path:
    return Path(campaign_root) / str(campaign_id)


def _campaign_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CAMPAIGN_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "campaign_id": plan["campaign_id"],
        "campaign_sha256": plan["campaign_sha256"],
        "scientific_policy_sha256": plan["scientific_policy_sha256"],
        "git_commit": plan["code_identity"]["git_commit"],
        "created_at_utc": utc_now(),
        "paths": {
            "policy": "policy.json",
            "plans": "plans",
            "runs": "runs",
            "reports": "reports",
        },
    }


def _assert_exact_json(path: Path, expected: Mapping[str, Any], *, label: str) -> None:
    actual = read_json_object(path)
    if actual != dict(expected):
        raise ValueError(f"{label} does not match the requested production identity: {path}")


def prepare_campaign(
    *,
    campaign_root: Path,
    plan: Mapping[str, Any],
    mode: str,
) -> Path:
    if mode not in {"fresh", "resume"}:
        raise ValueError("mode must be 'fresh' or 'resume'")
    campaign_dir = campaign_directory(campaign_root, str(plan["campaign_id"]))
    campaign_path = campaign_dir / "campaign.json"
    policy_path = campaign_dir / "policy.json"
    plan_path = campaign_dir / "plans" / f"{plan['plan_sha256']}.json"

    if mode == "fresh":
        if campaign_dir.exists():
            raise FileExistsError(
                f"fresh production campaign already exists: {campaign_dir}; use --resume"
            )
        campaign_dir.mkdir(parents=True)
        atomic_json(campaign_path, _campaign_payload(plan))
        atomic_json(policy_path, plan["scientific_policy"])
        atomic_json(plan_path, plan)
        (campaign_dir / "runs").mkdir()
        (campaign_dir / "reports").mkdir()
        return campaign_dir

    if not campaign_dir.is_dir():
        raise FileNotFoundError(
            f"resume requires an existing production campaign: {campaign_dir}"
        )
    campaign = read_json_object(campaign_path)
    expected_fields = {
        "schema": CAMPAIGN_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "campaign_id": plan["campaign_id"],
        "campaign_sha256": plan["campaign_sha256"],
        "scientific_policy_sha256": plan["scientific_policy_sha256"],
        "git_commit": plan["code_identity"]["git_commit"],
    }
    for name, expected in expected_fields.items():
        if campaign.get(name) != expected:
            raise ValueError(
                f"campaign identity mismatch for {name}: "
                f"stored={campaign.get(name)!r}, requested={expected!r}"
            )
    _assert_exact_json(policy_path, plan["scientific_policy"], label="scientific policy")
    if plan_path.exists():
        _assert_exact_json(plan_path, plan, label="registered production plan")
    else:
        atomic_json(plan_path, plan)
    (campaign_dir / "runs").mkdir(exist_ok=True)
    (campaign_dir / "reports").mkdir(exist_ok=True)
    return campaign_dir


def case_sidecars(
    *,
    case_identity: Mapping[str, Any],
    campaign_sha256: str,
    scientific_policy_sha256: str,
    git_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = dict(case_identity)
    cache_identity = {
        "schema": CACHE_IDENTITY_SCHEMA,
        "campaign_id": case_identity["campaign_id"],
        "campaign_sha256": str(campaign_sha256),
        "case_identity_sha256": case_identity["case_identity_sha256"],
        "scientific_policy_sha256": str(scientific_policy_sha256),
        "git_commit": str(git_commit),
        "frequency_extendable": True,
        "point_cache_path": "certified_points.json",
    }
    return identity, cache_identity


__all__ = [
    "CACHE_IDENTITY_SCHEMA",
    "CAMPAIGN_SCHEMA",
    "CASE_IDENTITY_SCHEMA",
    "CONTRACT_VERSION",
    "PLAN_SCHEMA",
    "POLICY_SCHEMA",
    "atomic_json",
    "build_campaign_identity",
    "build_case_identity",
    "campaign_directory",
    "case_sidecars",
    "finalize_plan",
    "git_code_identity",
    "prepare_campaign",
    "read_json_object",
    "verify_plan_payload",
]
