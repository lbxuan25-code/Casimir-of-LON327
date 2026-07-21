"""Guarded formal run command with ownership, recovery and proof generation."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from . import scan
from .config import RuntimeResources
from .energy import ProductionRunOptions, run_production_plan
from .execution_control import CampaignRunLock, write_recovery_report
from .identity import campaign_directory, read_json_object, verify_plan_payload
from .reproducibility import write_reproducibility_bundle


Runner = Callable[..., int]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir run",
        description=(
            "Execute a SHA-confirmed formal campaign under one campaign lock, with "
            "bounded engineering retries, checkpoint recovery and reproducibility proof."
        ),
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--confirm-plan-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    scan._add_execution_args(parser)
    parser.add_argument(
        "--max-engineering-retries",
        type=int,
        default=0,
        help="Bounded whole-plan retries after exit code 1; completed cases are skipped.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--stale-lock-seconds",
        type=float,
        default=300.0,
        help="Heartbeat age required before an explicit stale-lock takeover is eligible.",
    )
    parser.add_argument(
        "--lock-heartbeat-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--take-over-stale-lock",
        action="store_true",
        help="With --resume only, archive and replace a stale non-live campaign owner.",
    )
    return parser


def _execution_record(
    *,
    resources: RuntimeResources,
    options: ProductionRunOptions,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "parallel_mode": options.parallel_mode,
        "certifier_q_batch_size": options.certifier_q_batch_size,
        "memory_budget_gb": options.memory_budget_gb,
        "max_context_workers": options.max_context_workers,
        "retry_unresolved": options.retry_unresolved,
        "continue_on_engineering_failure": options.continue_on_engineering_failure,
        "max_engineering_retries": int(args.max_engineering_retries),
        "retry_delay_seconds": float(args.retry_delay_seconds),
        "stale_lock_seconds": float(args.stale_lock_seconds),
        "lock_heartbeat_seconds": float(args.lock_heartbeat_seconds),
        "take_over_stale_lock": bool(args.take_over_stale_lock),
        "selected_cpus": list(resources.selected_cpus),
        "reserved_cpus": list(resources.reserved_cpus),
        "workers": resources.workers,
    }


def execute_guarded_plan(
    *,
    plan: Mapping[str, Any],
    mode: str,
    resources: RuntimeResources,
    options: ProductionRunOptions,
    max_engineering_retries: int,
    retry_delay_seconds: float,
    stale_lock_seconds: float,
    lock_heartbeat_seconds: float,
    take_over_stale_lock: bool,
    execution_record: Mapping[str, Any],
    runner: Runner = run_production_plan,
) -> int:
    retries = int(max_engineering_retries)
    delay = float(retry_delay_seconds)
    if retries < 0:
        raise ValueError("max_engineering_retries must be non-negative")
    if delay < 0.0:
        raise ValueError("retry_delay_seconds must be non-negative")
    campaign_id = str(plan["campaign_id"])
    campaign_dir = campaign_directory(options.campaign_root, campaign_id)
    current_mode = str(mode)
    attempts = 0
    with CampaignRunLock(
        campaign_root=options.campaign_root,
        campaign_id=campaign_id,
        plan_sha256=str(plan["plan_sha256"]),
        mode=current_mode,
        stale_after_seconds=float(stale_lock_seconds),
        heartbeat_interval_seconds=float(lock_heartbeat_seconds),
        take_over_stale=bool(take_over_stale_lock),
    ):
        while True:
            attempts += 1
            if current_mode == "resume":
                write_recovery_report(campaign_dir=campaign_dir, plan=plan)
            status = int(
                runner(
                    plan=plan,
                    mode=current_mode,
                    resources=resources,
                    options=options,
                )
            )
            if campaign_dir.is_dir():
                write_reproducibility_bundle(
                    campaign_dir=campaign_dir,
                    plan=plan,
                    resources=resources,
                    execution_options=execution_record,
                    final_exit_code=status,
                    attempt_count=attempts,
                )
            if status != 1 or attempts > retries:
                return status
            print(
                "ENGINEERING RETRY: "
                f"attempt {attempts}/{retries + 1} failed; resuming from atomic caches",
                flush=True,
            )
            if delay:
                time.sleep(delay)
            current_mode = "resume"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = read_json_object(args.plan)
        verify_plan_payload(plan, expected_sha256=str(args.confirm_plan_sha256))
        mode = "fresh" if args.fresh else "resume"
        if args.take_over_stale_lock and mode != "resume":
            raise ValueError("--take-over-stale-lock requires --resume")
        resources = scan._resources(args)
        options = scan._execution_options(args)
        record = _execution_record(resources=resources, options=options, args=args)
        print(f"campaign_id: {plan['campaign_id']}", flush=True)
        print(f"plan_sha256: {plan['plan_sha256']}", flush=True)
        print(f"mode: {mode}", flush=True)
        return execute_guarded_plan(
            plan=plan,
            mode=mode,
            resources=resources,
            options=options,
            max_engineering_retries=int(args.max_engineering_retries),
            retry_delay_seconds=float(args.retry_delay_seconds),
            stale_lock_seconds=float(args.stale_lock_seconds),
            lock_heartbeat_seconds=float(args.lock_heartbeat_seconds),
            take_over_stale_lock=bool(args.take_over_stale_lock),
            execution_record=record,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"RUN FAILED: {type(exc).__name__}: {exc}")
        return 2


__all__ = ["execute_guarded_plan", "main"]
