from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import DEFAULT_OUTPUT_ROOT
from .data_management import _digest, _read, _write
from .qualification import (
    HOLDOUT_EXECUTION_SCHEMA,
    HOLDOUT_PLAN_SCHEMA,
    _utc_now,
    _verify_bound_inputs,
)
from .qualification_holdout_group import build_groups, run_group

CHECKPOINT_SCHEMA = "zero-degree-qualification-holdout-checkpoint-v2"
EXECUTOR_SCHEMA = "zero-degree-qualification-holdout-executor-v2"


def build_policy(
    *,
    max_concurrent_groups: int,
    workers_per_group: int,
    parallel_mode_per_group: str,
    memory_budget_gb_per_group: float,
    max_context_workers_per_group: int,
    total_worker_budget: int,
    total_memory_budget_gb: float,
) -> dict[str, Any]:
    groups = int(max_concurrent_groups)
    workers = int(workers_per_group)
    contexts = int(max_context_workers_per_group)
    total_workers = int(total_worker_budget)
    per_memory = float(memory_budget_gb_per_group)
    total_memory = float(total_memory_budget_gb)
    mode = str(parallel_mode_per_group)
    if min(groups, workers, contexts, total_workers) <= 0:
        raise ValueError("all holdout worker controls must be positive")
    if mode not in {"auto", "serial", "q", "context", "wave"}:
        raise ValueError("invalid per-group parallel mode")
    if per_memory <= 0.0 or total_memory <= 0.0:
        raise ValueError("holdout memory budgets must be positive")
    if groups * workers > total_workers:
        raise ValueError("group worker product exceeds total_worker_budget")
    if groups * per_memory > total_memory + 1e-12:
        raise ValueError("group memory product exceeds total_memory_budget_gb")
    if contexts > workers:
        raise ValueError("max_context_workers_per_group exceeds workers_per_group")
    return {
        "schema": EXECUTOR_SCHEMA,
        "max_concurrent_groups": groups,
        "workers_per_group": workers,
        "parallel_mode_per_group": mode,
        "memory_budget_gb_per_group": per_memory,
        "max_context_workers_per_group": contexts,
        "total_worker_budget": total_workers,
        "total_memory_budget_gb": total_memory,
        "blas_threads_per_process": 1,
        "checkpoint_after_every_group": True,
    }


def load_checkpoint(
    path: Path,
    *,
    plan_sha256: str,
    policy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = _read(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"checkpoint must use schema {CHECKPOINT_SCHEMA}")
    unsigned = dict(payload)
    stored = unsigned.pop("checkpoint_sha256", None)
    if stored != _digest(unsigned):
        raise ValueError("checkpoint self digest does not match")
    if payload.get("plan_sha256") != plan_sha256:
        raise ValueError("checkpoint belongs to a different holdout plan")
    if payload.get("executor_policy") != dict(policy):
        raise ValueError("checkpoint uses a different executor policy")
    completed = payload.get("completed_groups")
    if not isinstance(completed, Mapping):
        raise ValueError("checkpoint completed_groups is malformed")
    return {str(key): dict(value) for key, value in completed.items()}


def write_checkpoint(
    path: Path,
    *,
    plan_sha256: str,
    policy: Mapping[str, Any],
    total_groups: int,
    completed: Mapping[str, Mapping[str, Any]],
) -> None:
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "updated_at_utc": _utc_now(),
        "plan_sha256": plan_sha256,
        "executor_policy": dict(policy),
        "total_group_count": int(total_groups),
        "completed_group_count": len(completed),
        "completed_groups": {
            str(key): dict(value) for key, value in sorted(completed.items())
        },
    }
    payload["checkpoint_sha256"] = _digest(payload)
    _write(path, payload)


def execute(
    *,
    plan_path: Path,
    confirm_plan_sha256: str,
    output_root: Path,
    output_path: Path,
    checkpoint_path: Path,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _read(plan_path)
    if not isinstance(plan, Mapping) or plan.get("schema") != HOLDOUT_PLAN_SCHEMA:
        raise ValueError(f"holdout plan must use schema {HOLDOUT_PLAN_SCHEMA}")
    if str(plan.get("plan_sha256")) != str(confirm_plan_sha256):
        raise ValueError("holdout plan confirmation SHA-256 does not match")
    _verify_bound_inputs(plan, output_root)
    groups = build_groups(plan)
    completed = load_checkpoint(
        checkpoint_path,
        plan_sha256=str(plan["plan_sha256"]),
        policy=policy,
    )
    known = {str(group["group_id"]) for group in groups}
    if set(completed) - known:
        raise ValueError("checkpoint contains groups outside the current plan")
    pending = [group for group in groups if str(group["group_id"]) not in completed]
    write_checkpoint(
        checkpoint_path,
        plan_sha256=str(plan["plan_sha256"]),
        policy=policy,
        total_groups=len(groups),
        completed=completed,
    )
    print(
        f"holdout groups: total={len(groups)} completed={len(completed)} "
        f"pending={len(pending)} concurrent={policy['max_concurrent_groups']}",
        flush=True,
    )
    with ThreadPoolExecutor(
        max_workers=int(policy["max_concurrent_groups"]),
        thread_name_prefix="qualification-holdout",
    ) as pool:
        futures = {
            pool.submit(
                run_group,
                group,
                output_root=output_root,
                profile=str(plan["profile"]),
                policy=policy,
            ): group
            for group in pending
        }
        for future in as_completed(futures):
            group = futures[future]
            record = future.result()
            completed[str(group["group_id"])] = record
            write_checkpoint(
                checkpoint_path,
                plan_sha256=str(plan["plan_sha256"]),
                policy=policy,
                total_groups=len(groups),
                completed=completed,
            )
            print(
                f"completed {len(completed)}/{len(groups)} group={record['group']} "
                f"points={record['point_count']} wall={record['wall_seconds']:.1f}s "
                f"passed={record['all_points_passed']}",
                flush=True,
            )
    _verify_bound_inputs(plan, output_root)
    ordered = [completed[str(group["group_id"])] for group in groups]
    results = [row for record in ordered for row in record["results"]]
    report: dict[str, Any] = {
        "schema": HOLDOUT_EXECUTION_SCHEMA,
        "created_at_utc": _utc_now(),
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "profile": plan["profile"],
        "result_count": len(results),
        "results": results,
        "group_reports": [
            {key: value for key, value in record.items() if key != "results"}
            for record in ordered
        ],
        "all_points_passed": bool(results) and all(bool(x["passed"]) for x in results),
        "source_v4_unchanged": True,
        "target_seed_caches_unchanged": True,
        "real_microscopic_work_executed": True,
        "candidate_retuning_forbidden": True,
        "parallel_executor": dict(policy),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_group_count": len(completed),
        "resumable_execution": True,
    }
    report["execution_sha256"] = _digest(report)
    _write(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir.qualification_holdout",
        description="Parallel, bounded and resumable v5 high-N holdout executor.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--confirm-plan-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/casimir/reports/0deg_qualification_v5_holdout.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/casimir/workflow_logs/0deg_qualification_v5_holdout.checkpoint.json"
        ),
    )
    parser.add_argument("--max-concurrent-groups", type=int, default=6)
    parser.add_argument("--workers-per-group", type=int, default=3)
    parser.add_argument(
        "--parallel-mode-per-group",
        choices=("auto", "serial", "q", "context", "wave"),
        default="context",
    )
    parser.add_argument("--memory-budget-gb-per-group", type=float, default=3.0)
    parser.add_argument("--max-context-workers-per-group", type=int, default=3)
    parser.add_argument("--total-worker-budget", type=int, default=18)
    parser.add_argument("--total-memory-budget-gb", type=float, default=18.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = build_policy(
            max_concurrent_groups=args.max_concurrent_groups,
            workers_per_group=args.workers_per_group,
            parallel_mode_per_group=args.parallel_mode_per_group,
            memory_budget_gb_per_group=args.memory_budget_gb_per_group,
            max_context_workers_per_group=args.max_context_workers_per_group,
            total_worker_budget=args.total_worker_budget,
            total_memory_budget_gb=args.total_memory_budget_gb,
        )
        report = execute(
            plan_path=Path(args.plan),
            confirm_plan_sha256=str(args.confirm_plan_sha256),
            output_root=Path(args.output_root),
            output_path=Path(args.output),
            checkpoint_path=Path(args.checkpoint),
            policy=policy,
        )
        print(f"written: {Path(args.output).resolve()}")
        print(f"holdout points: {report['result_count']}")
        print(f"all_points_passed: {report['all_points_passed']}")
        print(f"execution_sha256: {report['execution_sha256']}")
        return 0 if report["all_points_passed"] else 2
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"QUALIFICATION HOLDOUT FAILED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
