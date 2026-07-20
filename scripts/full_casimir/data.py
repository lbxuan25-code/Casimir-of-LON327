from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_OUTPUT_ROOT
from .data_management import (
    build_archive_plan,
    build_data_catalog,
    execute_archive_plan,
    write_archive_execution,
    write_archive_plan,
    write_data_catalog,
    write_registry_template,
)


def _default_casimir_root() -> Path:
    return DEFAULT_OUTPUT_ROOT.parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir.data",
        description=(
            "Catalog and archive local Casimir run data without deleting source artifacts."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser(
        "catalog",
        help="Build a read-only catalog of runs and global output artifacts.",
    )
    catalog.add_argument("--casimir-root", type=Path, default=_default_casimir_root())
    catalog.add_argument("--catalog-root", type=Path, default=None)
    catalog.add_argument("--registry", type=Path, default=None)
    catalog.add_argument("--write-registry-template", action="store_true")

    plan = subparsers.add_parser(
        "plan",
        help="Build an archive plan from explicit local registry actions.",
    )
    plan.add_argument("--casimir-root", type=Path, default=_default_casimir_root())
    plan.add_argument("--catalog-root", type=Path, default=None)
    plan.add_argument("--registry", type=Path, default=None)
    plan.add_argument("--archive-root", type=Path, default=None)
    plan.add_argument("--plan-path", type=Path, default=None)

    archive = subparsers.add_parser(
        "archive",
        help="Create and verify compressed copies from an approved archive plan.",
    )
    archive.add_argument("--plan-path", type=Path, required=True)
    archive.add_argument("--confirm-plan-sha256", required=True)
    archive.add_argument("--run", action="append", default=[])
    archive.add_argument("--execution-report", type=Path, default=None)
    return parser


def _catalog_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    casimir_root = Path(args.casimir_root).resolve()
    catalog_root = (
        Path(args.catalog_root).resolve()
        if args.catalog_root is not None
        else casimir_root / "catalog"
    )
    registry = (
        Path(args.registry).resolve()
        if args.registry is not None
        else catalog_root / "registry.json"
    )
    registry_path = registry if registry.is_file() else None
    return casimir_root, catalog_root, registry_path


def _run_catalog(args: argparse.Namespace) -> int:
    casimir_root, catalog_root, registry_path = _catalog_paths(args)
    catalog = build_data_catalog(casimir_root, registry_path=registry_path)
    json_path, tsv_path = write_data_catalog(catalog, catalog_root=catalog_root)
    print(f"written: {json_path}")
    print(f"written: {tsv_path}")
    print(
        "runs: "
        f"{catalog['run_count']}, "
        f"run_bytes: {catalog['total_run_bytes']}, "
        f"global_artifacts: {catalog['global_artifact_count']}"
    )
    if args.write_registry_template:
        template = catalog_root / "registry.template.json"
        write_registry_template(catalog, template)
        print(f"written: {template}")
        if registry_path is None:
            print(
                "No active registry was applied. Copy and edit registry.template.json "
                "as registry.json before building an archive plan."
            )
    return 0


def _run_plan(args: argparse.Namespace) -> int:
    casimir_root = Path(args.casimir_root).resolve()
    catalog_root = (
        Path(args.catalog_root).resolve()
        if args.catalog_root is not None
        else casimir_root / "catalog"
    )
    registry = (
        Path(args.registry).resolve()
        if args.registry is not None
        else catalog_root / "registry.json"
    )
    if not registry.is_file():
        raise FileNotFoundError(
            f"archive planning requires an explicit registry: {registry}"
        )
    archive_root = (
        Path(args.archive_root).resolve()
        if args.archive_root is not None
        else casimir_root / "archive"
    )
    plan_path = (
        Path(args.plan_path).resolve()
        if args.plan_path is not None
        else catalog_root / "archive_plan.json"
    )
    catalog = build_data_catalog(casimir_root, registry_path=registry)
    write_data_catalog(catalog, catalog_root=catalog_root)
    plan = build_archive_plan(catalog, archive_root=archive_root)
    write_archive_plan(plan, plan_path)
    print(f"written: {plan_path}")
    print(f"archive items: {plan['item_count']}")
    print(f"plan_sha256: {plan['plan_sha256']}")
    print("No source directory was modified.")
    return 0


def _run_archive(args: argparse.Namespace) -> int:
    report = execute_archive_plan(
        Path(args.plan_path),
        confirm_plan_sha256=str(args.confirm_plan_sha256),
        selected_runs=tuple(args.run),
    )
    report_path = (
        Path(args.execution_report).resolve()
        if args.execution_report is not None
        else Path(args.plan_path).resolve().with_name("archive_execution.json")
    )
    write_archive_execution(report, report_path)
    print(f"written: {report_path}")
    for result in report["results"]:
        print(
            f"archived: {result['run_name']} -> {result['archive_path']} "
            f"({result['archive_bytes']} bytes)"
        )
    print("Source directories remain present; this command never deletes them.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "catalog":
            return _run_catalog(args)
        if args.command == "plan":
            return _run_plan(args)
        if args.command == "archive":
            return _run_archive(args)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"DATA MANAGEMENT FAILED: {type(exc).__name__}: {exc}")
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
