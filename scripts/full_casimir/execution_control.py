"""Campaign ownership, stale-lock takeover and checkpoint recovery reports."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Mapping
from uuid import uuid4

from .identity import atomic_json, case_sidecars


LOCK_SCHEMA = "full-casimir-campaign-lock-v1"
HEARTBEAT_SCHEMA = "full-casimir-campaign-lock-heartbeat-v1"
RECOVERY_SCHEMA = "full-casimir-recovery-report-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(payload), sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class CampaignRunLock:
    """One immutable owner record plus a token-scoped heartbeat sidecar.

    The owner record is never rewritten.  This prevents an old process from
    overwriting a replacement owner's lock after an explicit stale takeover.
    """

    def __init__(
        self,
        *,
        campaign_root: Path,
        campaign_id: str,
        plan_sha256: str,
        mode: str,
        stale_after_seconds: float = 300.0,
        heartbeat_interval_seconds: float = 30.0,
        take_over_stale: bool = False,
    ) -> None:
        if mode not in {"fresh", "resume"}:
            raise ValueError("lock mode must be fresh or resume")
        stale = float(stale_after_seconds)
        interval = float(heartbeat_interval_seconds)
        if stale <= 0.0 or interval <= 0.0:
            raise ValueError("lock stale and heartbeat intervals must be positive")
        if take_over_stale and mode != "resume":
            raise ValueError("stale lock takeover is allowed only with --resume")
        self.campaign_root = Path(campaign_root)
        self.campaign_id = str(campaign_id)
        self.plan_sha256 = str(plan_sha256)
        self.mode = mode
        self.stale_after_seconds = stale
        self.heartbeat_interval_seconds = interval
        self.take_over_stale = bool(take_over_stale)
        self.token = uuid4().hex
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.lock_root = self.campaign_root / ".locks"
        self.owner_path = self.lock_root / f"{self.campaign_id}.lock.json"
        self.heartbeat_path = self.lock_root / (
            f"{self.campaign_id}.{self.token}.heartbeat.json"
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._acquired = False

    def _owner_payload(self) -> dict[str, Any]:
        return {
            "schema": LOCK_SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "token": self.token,
            "hostname": self.hostname,
            "pid": self.pid,
            "mode": self.mode,
            "started_at_utc": _utc_now(),
            "heartbeat_path": self.heartbeat_path.name,
        }

    def _heartbeat_payload(self) -> dict[str, Any]:
        return {
            "schema": HEARTBEAT_SCHEMA,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "token": self.token,
            "hostname": self.hostname,
            "pid": self.pid,
            "heartbeat_at_utc": _utc_now(),
        }

    def _existing_state(self) -> tuple[dict[str, Any], dict[str, Any], float, bool]:
        owner = _read_json(self.owner_path)
        heartbeat_name = owner.get("heartbeat_path")
        heartbeat_path = (
            self.lock_root / str(heartbeat_name)
            if isinstance(heartbeat_name, str) and heartbeat_name
            else Path("__missing__")
        )
        heartbeat = _read_json(heartbeat_path)
        timestamp = _parse_utc(heartbeat.get("heartbeat_at_utc"))
        if timestamp is None:
            timestamp = _parse_utc(owner.get("started_at_utc"))
        age = float("inf")
        if timestamp is not None:
            age = max(
                0.0,
                (datetime.now(timezone.utc) - timestamp).total_seconds(),
            )
        same_host_live = bool(
            owner.get("hostname") == self.hostname
            and _pid_alive(int(owner.get("pid", -1)))
        )
        return owner, heartbeat, age, same_host_live

    def _archive_and_remove_stale_owner(
        self,
        owner: Mapping[str, Any],
        heartbeat: Mapping[str, Any],
        *,
        age_seconds: float,
    ) -> None:
        current = _read_json(self.owner_path)
        if current.get("token") != owner.get("token"):
            raise RuntimeError("campaign lock owner changed during stale takeover")
        history = self.lock_root / "history"
        history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = history / f"{stamp}-{self.campaign_id}.json"
        atomic_json(
            archive,
            {
                "schema": "full-casimir-stale-lock-takeover-v1",
                "taken_over_at_utc": _utc_now(),
                "stale_after_seconds": self.stale_after_seconds,
                "observed_heartbeat_age_seconds": age_seconds,
                "previous_owner": dict(owner),
                "previous_heartbeat": dict(heartbeat),
                "replacement": {
                    "token": self.token,
                    "hostname": self.hostname,
                    "pid": self.pid,
                    "plan_sha256": self.plan_sha256,
                },
            },
        )
        try:
            self.owner_path.unlink()
        except FileNotFoundError as exc:
            raise RuntimeError("campaign lock disappeared during stale takeover") from exc

    def acquire(self) -> "CampaignRunLock":
        self.lock_root.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                _exclusive_json(self.owner_path, self._owner_payload())
                break
            except FileExistsError:
                owner, heartbeat, age, same_host_live = self._existing_state()
                stale = age > self.stale_after_seconds and not same_host_live
                description = (
                    f"host={owner.get('hostname')!r}, pid={owner.get('pid')!r}, "
                    f"mode={owner.get('mode')!r}, heartbeat_age={age:.1f}s"
                )
                if same_host_live:
                    raise RuntimeError(
                        "campaign already has a live local owner: " + description
                    )
                if not stale:
                    raise RuntimeError(
                        "campaign lock is not stale; duplicate execution refused: "
                        + description
                    )
                if not self.take_over_stale:
                    raise RuntimeError(
                        "campaign lock is stale but takeover was not authorized; rerun "
                        "with --resume --take-over-stale-lock after checking the owner: "
                        + description
                    )
                self._archive_and_remove_stale_owner(
                    owner,
                    heartbeat,
                    age_seconds=age,
                )
        atomic_json(self.heartbeat_path, self._heartbeat_payload())
        self._acquired = True
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"casimir-lock-heartbeat-{self.campaign_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_seconds):
            owner = _read_json(self.owner_path)
            if owner.get("token") != self.token:
                return
            try:
                atomic_json(self.heartbeat_path, self._heartbeat_payload())
            except OSError:
                # The scientific process remains authoritative.  A failed lock heartbeat
                # makes future takeover conservative rather than changing its result.
                continue

    def release(self) -> None:
        if not self._acquired:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.heartbeat_interval_seconds + 1.0))
        owner = _read_json(self.owner_path)
        if owner.get("token") == self.token:
            try:
                self.owner_path.unlink()
            except FileNotFoundError:
                pass
        try:
            self.heartbeat_path.unlink()
        except FileNotFoundError:
            pass
        self._acquired = False

    def __enter__(self) -> "CampaignRunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def build_recovery_report(
    *,
    campaign_dir: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe resume checkpoints without mutating scientific artifacts."""

    root = Path(campaign_dir)
    rows: list[dict[str, Any]] = []
    resumable = 0
    for plan_row in plan.get("cases", ()):  # type: ignore[assignment]
        if not isinstance(plan_row, Mapping):
            continue
        case = str(plan_row.get("case", "unknown"))
        identity = plan_row.get("case_identity", {})
        if not isinstance(identity, Mapping):
            identity = {}
        run_dir = root / "runs" / case
        manifest = _read_json(run_dir / "manifest.json")
        expected_identity, expected_cache_identity = case_sidecars(
            case_identity=identity,
            campaign_sha256=str(plan["campaign_sha256"]),
            scientific_policy_sha256=str(plan["scientific_policy_sha256"]),
            git_commit=str(plan["code_identity"]["git_commit"]),
        )
        identity_matches = _read_json(run_dir / "identity.json") == expected_identity
        cache_identity_matches = (
            _read_json(run_dir / "cache" / "identity.json")
            == expected_cache_identity
        )
        cache = run_dir / "cache" / "certified_points.json"
        try:
            cache_bytes = cache.stat().st_size
        except OSError:
            cache_bytes = 0
        status = str(manifest.get("status", "missing"))
        checkpoint_usable = bool(
            run_dir.is_dir()
            and identity_matches
            and cache_identity_matches
            and (cache_bytes > 0 or status in {"running", "failed", "unresolved"})
        )
        if checkpoint_usable and status not in {"completed", "diagnostic_only"}:
            resumable += 1
        rows.append(
            {
                "case": case,
                "manifest_status": status,
                "attempt_count": int(manifest.get("attempt_count", 0) or 0),
                "identity_matches": identity_matches,
                "cache_identity_matches": cache_identity_matches,
                "cache_present": cache.is_file(),
                "cache_bytes": cache_bytes,
                "checkpoint_usable": checkpoint_usable,
                "orphan_temporary_files": sorted(
                    str(path.relative_to(run_dir))
                    for path in run_dir.rglob("*.tmp")
                    if path.is_file()
                )
                if run_dir.is_dir()
                else [],
            }
        )
    return {
        "schema": RECOVERY_SCHEMA,
        "generated_at_utc": _utc_now(),
        "campaign_id": str(plan["campaign_id"]),
        "plan_sha256": str(plan["plan_sha256"]),
        "read_only_preflight": True,
        "automatic_checkpoint_source": "last_atomic_certified_point_cache",
        "case_count": len(rows),
        "resumable_case_count": resumable,
        "cases": rows,
    }


def write_recovery_report(*, campaign_dir: Path, plan: Mapping[str, Any]) -> Path:
    path = Path(campaign_dir) / "reports" / "recovery.json"
    atomic_json(path, build_recovery_report(campaign_dir=campaign_dir, plan=plan))
    return path


__all__ = [
    "CampaignRunLock",
    "HEARTBEAT_SCHEMA",
    "LOCK_SCHEMA",
    "RECOVERY_SCHEMA",
    "build_recovery_report",
    "write_recovery_report",
]
