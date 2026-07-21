from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_OUTPUT_ROOT, REPO_ROOT
from .output_layout import write_output_layout_audit
from .output_layout_migration import (
    LAYOUT_FINALIZE_CONFIRMATION,
    build_layout_finalize_plan,
    build_layout_migration_plan,
    execute_layout_finalize_plan,
    stage_layout_migration,
    write_layout_finalize_execution,
    write_layout_finalize_plan,
    write_layout_migration_plan,
    write_layout_stage_execution,
)
from .output_layout_review import build_reviewed_output_layout_audit


def _root() -> Path:
    return DEFAULT_OUTPUT_ROOT.parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir.layout",
        description="Audit and normalize the local Casimir output-root layout.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="Build a read-only output-root audit.")
    audit.add_argument("--casimir-root", type=Path, default=_root())
    audit.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    audit.add_argument("--json-path", type=Path, default=None)
    audit.add_argument("--tsv-path", type=Path, default=None)

    plan = commands.add_parser("plan", help="Build a staged legacy-layout migration plan.")
    plan.add_argument("--audit-path", type=Path, default=None)
    plan.add_argument("--plan-path", type=Path, default=None)

    stage = commands.add_parser(
        "stage",
        help="Create and verify canonical legacy destinations without removing sources.",
    )
    stage.add_argument("--plan-path", type=Path, required=True)
    stage.add_argument("--confirm-plan-sha256", required=True)
    stage.add_argument("--execution-report", type=Path, default=None)

    finalize_plan = commands.add_parser(
        "finalize-plan",
        help="Plan source removal after every legacy destination has been staged and verified.",
    )
    finalize_plan.add_argument("--migration-plan-path", type=Path, required=True)
    finalize_plan.add_argument("--stage-execution-path", type=Path, required=True)
    finalize_plan.add_argument("--plan-path", type=Path, default=None)

    finalize = commands.add_parser(
        "finalize",
        help="Remove exact staged legacy root entries from an approved finalize plan.",
    )
    finalize.add_argument("--plan-path", type=Path, required=True)
    finalize.add_argument("--confirm-plan-sha256", required=True)
    finalize.add_argument("--confirm-delete", required=True)
    finalize.add_argument("--execution-report", type=Path, default=None)
    return parser


def _run_audit(args: argparse.Namespace) -> int:
    root = Path(args.casimir_root).resolve()
    catalog = root / "catalog"
    json_path = Path(args.json_path).resolve() if args.json_path else catalog / "output_layout_audit.json"
    tsv_path = Path(args.tsv_path).resolve() if args.tsv_path else catalog / "output_layout_audit.tsv"
    audit = build_reviewed_output_layout_audit(root, repo_root=Path(args.repo_root))
    write_output_layout_audit(audit, json_path=json_path, tsv_path=tsv_path)
    print(f"written: {json_path}")
    print(f"written: {tsv_path}")
    print(f"legacy entries: {audit['legacy_entry_count']}")
    print(f"review required: {audit['review_required_count']}")
    print(f"migration blockers: {len(audit['migration_blockers'])}")
    print(f"audit_sha256: {audit['audit_sha256']}")
    print("No output entry was moved or deleted.")
    return 0


def _run_plan(args: argparse.Namespace) -> int:
    audit_path = (
        Path(args.audit_path).resolve()
        if args.audit_path
        else _root() / "catalog" / "output_layout_audit.json"
    )
    plan_path = (
        Path(args.plan_path).resolve()
        if args.plan_path
        else _root() / "catalog" / "output_layout_migration_plan.json"
    )
    plan = build_layout_migration_plan(audit_path)
    write_layout_migration_plan(plan, plan_path)
    print(f"written: {plan_path}")
    print(f"migration items: {plan['item_count']}")
    print(f"source bytes: {plan['source_total_bytes']}")
    print(f"plan_sha256: {plan['plan_sha256']}")
    print("No destination was created and no source was modified.")
    return 0


def _run_stage(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan_path).resolve()
    report = stage_layout_migration(
        plan_path,
        confirm_plan_sha256=str(args.confirm_plan_sha256),
    )
    report_path = (
        Path(args.execution_report).resolve()
        if args.execution_report
        else plan_path.with_name("output_layout_stage_execution.json")
    )
    write_layout_stage_execution(report, report_path)
    print(f"written: {report_path}")
    print(f"staged and verified: {report['result_count']}")
    print(f"stage_sha256: {report['stage_sha256']}")
    print("All source root entries remain present.")
    return 0


def _run_finalize_plan(args: argparse.Namespace) -> int:
    migration_path = Path(args.migration_plan_path).resolve()
    stage_path = Path(args.stage_execution_path).resolve()
    plan_path = (
        Path(args.plan_path).resolve()
        if args.plan_path
        else migration_path.with_name("output_layout_finalize_plan.json")
    )
    plan = build_layout_finalize_plan(migration_path, stage_path)
    write_layout_finalize_plan(plan, plan_path)
    print(f"written: {plan_path}")
    print(f"finalize items: {plan['item_count']}")
    print(f"releasable bytes: {plan['source_total_bytes']}")
    print(f"plan_sha256: {plan['plan_sha256']}")
    print(f"required deletion phrase: {LAYOUT_FINALIZE_CONFIRMATION}")
    print("No source root entry was removed.")
    return 0


def _run_finalize(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan_path).resolve()
    report = execute_layout_finalize_plan(
        plan_path,
        confirm_plan_sha256=str(args.confirm_plan_sha256),
        confirm_delete=str(args.confirm_delete),
    )
    report_path = (
        Path(args.execution_report).resolve()
        if args.execution_report
        else plan_path.with_name("output_layout_finalize_execution.json")
    )
    write_layout_finalize_execution(report, report_path)
    print(f"written: {report_path}")
    print(f"removed root entries: {report['removed_entry_count']}")
    print(f"released bytes: {report['released_bytes']}")
    print("All staged destinations and manifests remain present.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "audit":
            return _run_audit(args)
        if args.command == "plan":
            return _run_plan(args)
        if args.command == "stage":
            return _run_stage(args)
        if args.command == "finalize-plan":
            return _run_finalize_plan(args)
        if args.command == "finalize":
            return _run_finalize(args)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"OUTPUT LAYOUT FAILED: {type(exc).__name__}: {exc}")
        return 2
    raise AssertionError(f"unhandled layout command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
