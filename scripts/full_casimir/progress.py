"""Persistent runtime progress reporting and read-only status display."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping, Sequence, TextIO

from .config import DEFAULT_PRODUCTION_ROOT


PROGRESS_SCHEMA = "full-casimir-progress-v1"
PROGRESS_EVENT_SCHEMA = "full-casimir-progress-event-v1"
_TERMINAL_CASE_STATES = {
    "production_authorized",
    "numerically_unresolved",
    "engineering_failed",
    "diagnostic_only",
}
_ACTIVITY_ORDER = (
    "matsubara_block",
    "outer_cutoff",
    "joint_controller",
    "radial_run",
    "microscopic_request",
    "microscopic_batch",
)
_IMMEDIATE_SNAPSHOT_EVENTS = {
    "campaign_started",
    "campaign_finished",
    "case_started",
    "case_finished",
    "matsubara_block_started",
    "matsubara_block_completed",
    "matsubara_block_failed",
    "outer_cutoff_started",
    "outer_cutoff_completed",
    "outer_cutoff_failed",
    "radial_run_started",
    "radial_run_completed",
    "radial_run_failed",
    "microscopic_batch_started",
    "microscopic_batch_completed",
    "microscopic_batch_failed",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def _case_from_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = row.get("case_identity", {})
    if not isinstance(identity, Mapping):
        identity = {}
    plate_angles = identity.get("plate_angles_deg", (0.0, 0.0))
    try:
        angle = float(plate_angles[1])
    except (TypeError, ValueError, IndexError, OverflowError):
        angle = 0.0
    return {
        "case": str(row.get("case", "unknown")),
        "pairing": str(identity.get("pairing", "unknown")),
        "temperature_K": float(identity.get("temperature_K", 0.0)),
        "separation_nm": float(identity.get("separation_nm", 0.0)),
        "angle_deg": angle,
        "lifecycle_status": "queued",
        "action": "queued",
        "termination_reason": None,
        "started_at_utc": None,
        "finished_at_utc": None,
        "last_progress_at_utc": None,
        "last_heartbeat_at_utc": None,
        "state_sequence": 0,
        "current_phase": "queued",
        "activity": {},
        "activity_stack": [],
        "provider_statistics": {},
        "provider_counter_delta": {},
        "selected_N_distribution": {},
        "unresolved_reason_counts": {},
        "budgets": {},
        "latest_event": None,
    }


def _activity_entry(event: str, payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if event == "matsubara_block_started":
        left = int(payload.get("left_n", 0))
        right = int(payload.get("right_n", 0))
        return "matsubara_block", {
            "label": f"Matsubara {left}-{right}",
            "left_n": left,
            "right_n": right,
            "index": int(payload.get("block_index", 0)),
            "count": int(payload.get("block_count", 0)),
        }
    if event == "outer_cutoff_started":
        u_max = float(payload.get("u_max", 0.0))
        return "outer_cutoff", {
            "label": f"outer u={u_max:g}",
            "u_max": u_max,
            "index": int(payload.get("cutoff_index", 0)),
            "count": int(payload.get("cutoff_count", 0)),
        }
    if event == "joint_controller_started":
        return "joint_controller", {
            "label": "joint radial-angular controller",
            "angular_orders": list(payload.get("angular_orders", ())),
            "u_max": float(payload.get("u_max", 0.0)),
        }
    if event == "radial_run_started":
        order = int(payload.get("angular_order", 0))
        cap = int(payload.get("radial_round_cap", 0))
        return "radial_run", {
            "label": f"angular order={order}, radial cap={cap}",
            "angular_order": order,
            "angular_offset_fraction": float(
                payload.get("angular_offset_fraction", 0.0)
            ),
            "radial_order": int(payload.get("radial_order", 0)),
            "radial_round_cap": cap,
            "initial_panel_count": int(payload.get("initial_panel_count", 0)),
        }
    if event == "microscopic_request_started":
        requested = int(payload.get("requested_q_nodes", 0))
        return "microscopic_request", {
            "label": f"microscopic request q={requested}",
            "requested_q_nodes": requested,
        }
    if event == "microscopic_batch_started":
        q_count = int(payload.get("requested_q_count", 0))
        point_count = int(payload.get("requested_point_count", 0))
        indices = [int(value) for value in payload.get("matsubara_indices", ())]
        return "microscopic_batch", {
            "label": f"microscopic batch q={q_count}, points={point_count}",
            "requested_q_count": q_count,
            "requested_point_count": point_count,
            "matsubara_indices": indices,
            "N_candidates": [int(value) for value in payload.get("N_candidates", ())],
        }
    return None


def _clear_activity(case: dict[str, Any], level: str) -> None:
    try:
        position = _ACTIVITY_ORDER.index(level)
    except ValueError:
        return
    for name in _ACTIVITY_ORDER[position:]:
        case["activity"].pop(name, None)


def _refresh_activity_stack(case: dict[str, Any]) -> None:
    stack = [
        deepcopy(case["activity"][name])
        for name in _ACTIVITY_ORDER
        if name in case["activity"]
    ]
    case["activity_stack"] = stack
    case["current_phase"] = (
        next(
            (
                name
                for name in reversed(_ACTIVITY_ORDER)
                if name in case["activity"]
            ),
            case["lifecycle_status"],
        )
    )


def _update_case_from_scientific_event(case: dict[str, Any], payload: Mapping[str, Any]) -> None:
    event = str(payload.get("event", "unknown"))
    started = _activity_entry(event, payload)
    if started is not None:
        level, entry = started
        _clear_activity(case, level)
        case["activity"][level] = entry
    completed_levels = {
        "matsubara_block_completed": "matsubara_block",
        "matsubara_block_failed": "matsubara_block",
        "outer_cutoff_completed": "outer_cutoff",
        "outer_cutoff_failed": "outer_cutoff",
        "joint_controller_completed": "joint_controller",
        "joint_controller_failed": "joint_controller",
        "radial_run_completed": "radial_run",
        "radial_run_failed": "radial_run",
        "microscopic_request_completed": "microscopic_request",
        "microscopic_request_failed": "microscopic_request",
        "microscopic_batch_completed": "microscopic_batch",
        "microscopic_batch_failed": "microscopic_batch",
    }
    level = completed_levels.get(event)
    if level is not None:
        _clear_activity(case, level)

    provider = payload.get("provider_statistics")
    if isinstance(provider, Mapping):
        case["provider_statistics"] = dict(provider)
    delta = payload.get("provider_counter_delta")
    if isinstance(delta, Mapping):
        case["provider_counter_delta"] = dict(delta)
    selected = payload.get("selected_N_distribution")
    if isinstance(selected, Mapping):
        case["selected_N_distribution"] = {
            str(name): int(value) for name, value in selected.items()
        }
    reasons = payload.get("unresolved_reason_counts")
    if isinstance(reasons, Mapping):
        case["unresolved_reason_counts"] = {
            str(name): int(value) for name, value in reasons.items()
        }
    budgets = payload.get("budget_ratios")
    if isinstance(budgets, Mapping):
        if event.startswith("matsubara"):
            case["budgets"]["matsubara"] = dict(budgets)
        elif event.startswith("outer"):
            case["budgets"]["outer"] = dict(budgets)
        elif event.startswith("joint") or event.startswith("radial"):
            case["budgets"]["finite_domain"] = dict(budgets)
    case["latest_event"] = event
    _refresh_activity_stack(case)


def _format_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_progress_snapshot(snapshot: Mapping[str, Any]) -> str:
    """Render a compact terminal-safe campaign summary."""

    campaign = str(snapshot.get("campaign_id", "unknown"))
    elapsed = _format_duration(float(snapshot.get("elapsed_seconds", 0.0)))
    counts = snapshot.get("case_counts", {})
    if not isinstance(counts, Mapping):
        counts = {}
    lines = [
        f"Campaign {campaign} | {snapshot.get('lifecycle_status', 'unknown')} | elapsed {elapsed}",
        (
            "Cases: "
            f"{counts.get('production_authorized', 0)}/{counts.get('total', 0)} authorized"
            f" | {counts.get('running', 0)} running"
            f" | {counts.get('numerically_unresolved', 0)} unresolved"
            f" | {counts.get('diagnostic_only', 0)} diagnostic-only"
            f" | {counts.get('engineering_failed', 0)} engineering-failed"
        ),
    ]
    cases = snapshot.get("cases", {})
    if not isinstance(cases, Mapping):
        cases = {}
    active = [
        row
        for row in cases.values()
        if isinstance(row, Mapping) and row.get("lifecycle_status") == "running"
    ]
    for row in active:
        lines.append("")
        lines.append(
            f"{row.get('case')} [{row.get('pairing')}] "
            f"T={row.get('temperature_K'):g} K d={row.get('separation_nm'):g} nm "
            f"theta={row.get('angle_deg'):g} deg"
        )
        stack = row.get("activity_stack", ())
        if isinstance(stack, Sequence):
            labels = [
                str(item.get("label"))
                for item in stack
                if isinstance(item, Mapping) and item.get("label")
            ]
            if labels:
                lines.append("  " + " -> ".join(labels))
        stats = row.get("provider_statistics", {})
        if isinstance(stats, Mapping) and stats:
            requested = int(stats.get("requested_point_evaluations", 0))
            new = int(stats.get("new_point_evaluations", 0))
            hits = int(stats.get("cache_hit_point_evaluations", 0))
            batches = int(stats.get("certification_batches", 0))
            lines.append(
                f"  microscopic points: requested={requested} new={new} "
                f"cache-hit={hits} batches={batches}"
            )
        reasons = row.get("unresolved_reason_counts", {})
        if isinstance(reasons, Mapping) and reasons:
            top = list(reasons.items())[:3]
            lines.append(
                "  unresolved: "
                + ", ".join(f"{name}={count}" for name, count in top)
            )
        budgets = row.get("budgets", {})
        if isinstance(budgets, Mapping) and budgets:
            compact: list[str] = []
            for layer, pairing_rows in budgets.items():
                if not isinstance(pairing_rows, Mapping):
                    continue
                values: list[float] = []
                for pairing_row in pairing_rows.values():
                    if not isinstance(pairing_row, Mapping):
                        continue
                    for name, value in pairing_row.items():
                        if name.endswith("path") or isinstance(value, bool):
                            continue
                        try:
                            values.append(float(value))
                        except (TypeError, ValueError, OverflowError):
                            continue
                if values:
                    compact.append(f"{layer} max={max(values):.3g}x budget")
            if compact:
                lines.append("  " + " | ".join(compact))
        lines.append(
            f"  last progress: {row.get('last_progress_at_utc') or 'not recorded'}"
        )
    return "\n".join(lines)


class CampaignProgressReporter:
    """Persist campaign/case state while observing structured scientific events."""

    def __init__(
        self,
        *,
        campaign_dir: Path,
        plan: Mapping[str, Any],
        stream: TextIO | None = sys.stdout,
        snapshot_interval_seconds: float = 10.0,
        render_interval_seconds: float = 2.0,
        heartbeat_interval_seconds: float = 30.0,
        non_tty_log_interval_seconds: float = 60.0,
    ) -> None:
        self.campaign_dir = Path(campaign_dir)
        self.plan = dict(plan)
        self.stream = stream
        self.snapshot_interval_seconds = max(float(snapshot_interval_seconds), 0.0)
        self.render_interval_seconds = max(float(render_interval_seconds), 0.0)
        self.heartbeat_interval_seconds = max(float(heartbeat_interval_seconds), 0.0)
        self.non_tty_log_interval_seconds = max(
            float(non_tty_log_interval_seconds),
            0.0,
        )
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._active_case: str | None = None
        self._sequence = 0
        self._started_monotonic = time.monotonic()
        self._started_at_utc = _utc_now()
        self._last_snapshot_monotonic = 0.0
        self._last_render_monotonic = 0.0
        self._last_non_tty_log_monotonic = 0.0
        self.snapshot_write_count = 0
        rows = self.plan.get("cases", ())
        self._cases = {
            case["case"]: case
            for case in (
                _case_from_plan_row(row)
                for row in rows
                if isinstance(row, Mapping)
            )
        }
        self._campaign_status = "prepared"
        self._last_progress_at_utc: str | None = None
        self._last_heartbeat_at_utc: str | None = None
        self.campaign_dir.mkdir(parents=True, exist_ok=True)

    @property
    def progress_path(self) -> Path:
        return self.campaign_dir / "progress.json"

    @property
    def events_path(self) -> Path:
        return self.campaign_dir / "progress.events.jsonl"

    def _case_dir(self, case: str) -> Path:
        return self.campaign_dir / "runs" / case

    def _counts(self) -> dict[str, int]:
        counts = {
            "total": len(self._cases),
            "queued": 0,
            "running": 0,
            "production_authorized": 0,
            "numerically_unresolved": 0,
            "engineering_failed": 0,
            "diagnostic_only": 0,
        }
        for row in self._cases.values():
            status = str(row["lifecycle_status"])
            if status in counts:
                counts[status] += 1
        return counts

    def _pairing_counts(self) -> dict[str, dict[str, int]]:
        output: dict[str, dict[str, int]] = {}
        for row in self._cases.values():
            pairing = str(row["pairing"])
            bucket = output.setdefault(
                pairing,
                {
                    "total": 0,
                    "queued": 0,
                    "running": 0,
                    "production_authorized": 0,
                    "numerically_unresolved": 0,
                    "engineering_failed": 0,
                    "diagnostic_only": 0,
                },
            )
            bucket["total"] += 1
            status = str(row["lifecycle_status"])
            if status in bucket:
                bucket[status] += 1
        return output

    def _snapshot_payload(self) -> dict[str, Any]:
        generated = _utc_now()
        return {
            "schema": PROGRESS_SCHEMA,
            "scope": "campaign",
            "campaign_id": str(self.plan.get("campaign_id", self.campaign_dir.name)),
            "plan_sha256": str(self.plan.get("plan_sha256", "")),
            "state_sequence": self._sequence,
            "generated_at_utc": generated,
            "started_at_utc": self._started_at_utc,
            "last_progress_at_utc": self._last_progress_at_utc,
            "last_heartbeat_at_utc": self._last_heartbeat_at_utc,
            "elapsed_seconds": float(time.monotonic() - self._started_monotonic),
            "lifecycle_status": self._campaign_status,
            "case_counts": self._counts(),
            "pairing_counts": self._pairing_counts(),
            "events_path": "progress.events.jsonl",
            "cases": deepcopy(self._cases),
        }

    def _case_snapshot_payload(self, case: str) -> dict[str, Any]:
        row = deepcopy(self._cases[case])
        return {
            "schema": PROGRESS_SCHEMA,
            "scope": "case",
            "campaign_id": str(self.plan.get("campaign_id", self.campaign_dir.name)),
            "plan_sha256": str(self.plan.get("plan_sha256", "")),
            "generated_at_utc": _utc_now(),
            **row,
            "events_path": "progress.events.jsonl",
        }

    def _write_snapshots(self, case: str | None) -> None:
        _atomic_json(self.progress_path, self._snapshot_payload())
        self.snapshot_write_count += 1
        if case is not None and case in self._cases:
            run_dir = self._case_dir(case)
            _atomic_json(run_dir / "progress.json", self._case_snapshot_payload(case))
            self.snapshot_write_count += 1
        self._last_snapshot_monotonic = time.monotonic()

    def _render(self, *, force: bool = False) -> None:
        if self.stream is None:
            return
        now = time.monotonic()
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        interval = (
            self.render_interval_seconds
            if is_tty
            else self.non_tty_log_interval_seconds
        )
        last = (
            self._last_render_monotonic
            if is_tty
            else self._last_non_tty_log_monotonic
        )
        if not force and interval > 0.0 and now - last < interval:
            return
        text = format_progress_snapshot(self._snapshot_payload())
        if is_tty:
            self.stream.write("\x1b[2J\x1b[H")
            self.stream.write(text + "\n")
            self._last_render_monotonic = now
        else:
            self.stream.write(text + "\n")
            self._last_non_tty_log_monotonic = now
        self.stream.flush()

    def _record(
        self,
        event: str,
        *,
        case: str | None,
        payload: Mapping[str, Any] | None = None,
        heartbeat: bool = False,
        force_snapshot: bool = False,
        force_render: bool = False,
    ) -> None:
        with self._lock:
            self._sequence += 1
            now_utc = _utc_now()
            body = {} if payload is None else dict(payload)
            record = {
                "schema": PROGRESS_EVENT_SCHEMA,
                "state_sequence": self._sequence,
                "timestamp_utc": now_utc,
                "campaign_id": str(
                    self.plan.get("campaign_id", self.campaign_dir.name)
                ),
                "case": case,
                "event": event,
                **body,
            }
            _append_jsonl(self.events_path, record)
            if case is not None:
                _append_jsonl(
                    self._case_dir(case) / "progress.events.jsonl",
                    record,
                )
            if heartbeat:
                self._last_heartbeat_at_utc = now_utc
                if case is not None and case in self._cases:
                    self._cases[case]["last_heartbeat_at_utc"] = now_utc
            else:
                self._last_progress_at_utc = now_utc
                if case is not None and case in self._cases:
                    self._cases[case]["last_progress_at_utc"] = now_utc
            if case is not None and case in self._cases:
                self._cases[case]["state_sequence"] = self._sequence
            now = time.monotonic()
            should_snapshot = bool(
                force_snapshot
                or event in _IMMEDIATE_SNAPSHOT_EVENTS
                or self.snapshot_interval_seconds == 0.0
                or now - self._last_snapshot_monotonic
                >= self.snapshot_interval_seconds
            )
            if should_snapshot:
                self._write_snapshots(case)
            self._render(force=force_render)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_seconds):
            with self._lock:
                active = self._active_case
            self._record(
                "heartbeat",
                case=active,
                heartbeat=True,
                force_snapshot=True,
            )

    def campaign_started(
        self,
        *,
        mode: str,
        resources: Mapping[str, Any] | None = None,
    ) -> None:
        self._campaign_status = "running"
        self._record(
            "campaign_started",
            case=None,
            payload={
                "mode": str(mode),
                "resources": {} if resources is None else dict(resources),
            },
            force_snapshot=True,
            force_render=True,
        )
        if self.heartbeat_interval_seconds > 0.0 and self._heartbeat_thread is None:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="full-casimir-progress-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def case_started(self, case: str, *, action: str) -> None:
        row = self._cases[case]
        row["lifecycle_status"] = "running"
        row["action"] = str(action)
        row["termination_reason"] = None
        row["started_at_utc"] = row["started_at_utc"] or _utc_now()
        row["finished_at_utc"] = None
        row["activity"] = {}
        _refresh_activity_stack(row)
        self._active_case = case
        self._record(
            "case_started",
            case=case,
            payload={"action": str(action)},
            force_snapshot=True,
            force_render=True,
        )

    def emit(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            return
        event = str(payload.get("event", "scientific_progress"))
        with self._lock:
            case = self._active_case
            if case is None or case not in self._cases:
                return
            _update_case_from_scientific_event(self._cases[case], payload)
        body = {name: value for name, value in payload.items() if name != "event"}
        self._record(event, case=case, payload=body)

    def case_finished(
        self,
        case: str,
        *,
        status: str,
        termination_reason: str | None,
        action: str,
    ) -> None:
        if status not in _TERMINAL_CASE_STATES:
            raise ValueError(f"unknown terminal case progress status: {status}")
        row = self._cases[case]
        row["lifecycle_status"] = status
        row["action"] = str(action)
        row["termination_reason"] = termination_reason
        row["finished_at_utc"] = _utc_now()
        row["activity"] = {}
        _refresh_activity_stack(row)
        self._record(
            "case_finished",
            case=case,
            payload={
                "status": status,
                "termination_reason": termination_reason,
                "action": str(action),
            },
            force_snapshot=True,
            force_render=True,
        )
        if self._active_case == case:
            self._active_case = None

    def campaign_finished(self, *, exit_code: int) -> None:
        self._stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(self.heartbeat_interval_seconds, 0.1) + 0.1)
        counts = self._counts()
        if counts["engineering_failed"]:
            status = "engineering_failed"
        elif counts["numerically_unresolved"] or counts["diagnostic_only"]:
            status = "completed_with_unresolved_cases"
        else:
            status = "completed"
        self._campaign_status = status
        self._record(
            "campaign_finished",
            case=None,
            payload={"exit_code": int(exit_code), "status": status},
            force_snapshot=True,
            force_render=True,
        )


def _resolve_campaign_dir(args: argparse.Namespace) -> Path:
    if args.campaign_dir is not None:
        return Path(args.campaign_dir)
    return Path(args.campaign_root) / str(args.campaign)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir status",
        description="Read persisted full-Casimir progress without starting work.",
    )
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--campaign", help="Campaign directory name under --campaign-root.")
    location.add_argument("--campaign-dir", type=Path)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_PRODUCTION_ROOT,
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign_dir = _resolve_campaign_dir(args)
    path = campaign_dir / "progress.json"
    interval = max(float(args.interval), 0.1)
    try:
        while True:
            payload = _read_json(path)
            if not payload:
                print(f"STATUS FAILED: progress snapshot is missing or unreadable: {path}")
                return 2
            if args.json:
                text = json.dumps(payload, sort_keys=True, indent=2)
            else:
                text = format_progress_snapshot(payload)
            if args.watch and sys.stdout.isatty():
                print("\x1b[2J\x1b[H", end="")
            print(text, flush=True)
            if not args.watch:
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        return 130


__all__ = [
    "CampaignProgressReporter",
    "PROGRESS_EVENT_SCHEMA",
    "PROGRESS_SCHEMA",
    "format_progress_snapshot",
    "main",
]
