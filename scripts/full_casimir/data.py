from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .archive_catalog import augment_catalog_with_archives
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
from .data_retention import (
    PRUNE_CONFIRMATION,
    build_prune_plan,
    execute_prune_plan,
    pack_json_report,
    verify_archive_plan,
    write_archive_verification,
    write_prune_execution,
    write_prune_plan,
)
from .report_retention import (
    REPORT_PRUNE_CONFIRMATION,
    build_report_prune_plan,
    execute_report_prune_plan,
    write_report_prune_execution,
    write_report_prune_plan,
)


def _default_casimir_root() -> Path:
    return DEFAULT_OUTPUT_ROOT.parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir.data",
        description=(
            "Catalog, archive, restore-verify, and explicitly prune local Casimir data."
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

    verify = subparsers.add_parser(
        "verify",
        help="Restore archives to temporary storage and compare every file to the manifest.",
    )
    verify.add_argument("--plan-path", type=Path, required=True)
    verify.add_argument("--run", action="append", default=[])
    verify.add_argument("--verification-report", type=Path, default=None)

    prune_plan = subparsers.add_parser(
        "prune-plan",
        help="Plan source removal only for explicitly selected, restored-and-verified runs.",
    )
    prune_plan.add_argument("--casimir-root", type=Path, default=_default_casimir_root())
    prune_plan.add_argument("--catalog-root", type=Path, default=None)
    prune_plan.add_argument("--registry", type=Path, default=None)
    prune_plan.add_argument("--verification-report", type=Path, required=True)
    prune_plan.add_argument("--run", action="append", required=True)
    prune_plan.add_argument("--plan-path", type=Path, default=None)

    prune = subparsers.add_parser(
        "prune",
        help="Delete source run directories from an exact approved prune plan.",
    )
    prune.add_argument("--plan-path", type=Path, required=True)
    prune.add_argument("--confirm-plan-sha256", required=True)
    prune.add_argument("--confirm-delete", required=True)
    prune.add_argument("--execution-report", type=Path, default=None)

    pack = subparsers.add_parser(
        "pack-report",
        help="Externalize large JSON lists into verified compressed sidecars.",
    )
    pack.add_argument("--report-path", type=Path, required=True)
    pack.add_argument("--compact-path", type=Path, default=None)
    pack.add_argument("--pack-root", type=Path, default=None)
    pack.add_argument("--manifest-path", type=Path, default=None)
    pack.add_argument("--threshold-mib", type=float, default=1.0)

    report_prune_plan = subparsers.add_parser(
        "report-prune-plan",
        help="Plan removal of an original JSON report after verified reconstruction.",
    )
    report_prune_plan.add_argument("--manifest-path", type=Path, required=True)
    report_prune_plan.add_argument("--plan-path", type=Path, default=None)

    report_prune = subparsers.add_parser(
        "report-prune",
        help="Delete an original report from an exact verified report-prune plan.",
    )
    report_prune.add_argument("--plan-path", type=Path, required=True)
    report_prune.add_argument("--confirm-plan-sha256", required=True)
    report_prune.add_argument("--confirm-delete", required=True)
    report_prune.add_argument("--execution-report", type=Path, default=None)
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


def _catalog(casimir_root: Path, registry_path: Path | None) -> dict:
    return augment_catalog_with_archives(
        build_data_catalog(casimir_root, registry_path=registry_path)
    )


def _run_catalog(args: argparse.Namespace) -> int:
    casimir_root, catalog_root, registry_path = _catalog_paths(args)
    catalog = _catalog(casimir_root, registry_path)
    json_path, tsv_path = write_data_catalog(catalog, catalog_root=catalog_root)
    print(f"written: {json_path}")
    print(f"written: {tsv_path}")
    print(
        "runs: "
        f"{catalog['run_count']}, "
        f"run_bytes: {catalog['total_run_bytes']}, "
        f"cold_archives: {catalog['archived_run_count']}, "
        f"archive_bytes: {catalog['total_archive_bytes']}, "
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
    catalog = _catalog(casimir_root, registry)
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


def _run_verify(args: argparse.Namespace) -> int:
    report = verify_archive_plan(
        Path(args.plan_path),
        selected_runs=tuple(args.run),
    )
    report_path = (
        Path(args.verification_report).resolve()
        if args.verification_report is not None
        else Path(args.plan_path).resolve().with_name("archive_verification.json")
    )
    write_archive_verification(report, report_path)
    print(f"written: {report_path}")
    print(f"restored and verified: {report['result_count']}")
    print(f"verification_sha256: {report['verification_sha256']}")
    print("All temporary restored copies were removed; archives and sources remain present.")
    return 0


def _run_prune_plan(args: argparse.Namespace) -> int:
    casimir_root, catalog_root, registry_path = _catalog_paths(args)
    if registry_path is None:
        raise FileNotFoundError("prune planning requires an explicit registry")
    catalog = _catalog(casimir_root, registry_path)
    write_data_catalog(catalog, catalog_root=catalog_root)
    verification = json.loads(
        Path(args.verification_report).read_text(encoding="utf-8")
    )
    plan = build_prune_plan(
        catalog,
        verification,
        selected_runs=tuple(args.run),
    )
    plan_path = (
        Path(args.plan_path).resolve()
        if args.plan_path is not None
        else catalog_root / "prune_plan.json"
    )
    write_prune_plan(plan, plan_path)
    print(f"written: {plan_path}")
    print(f"prune items: {plan['item_count']}")
    print(f"releasable bytes: {plan['source_total_bytes']}")
    print(f"plan_sha256: {plan['plan_sha256']}")
    print(f"required deletion phrase: {PRUNE_CONFIRMATION}")
    print("No source directory was modified.")
    return 0


def _run_prune(args: argparse.Namespace) -> int:
    report = execute_prune_plan(
        Path(args.plan_path),
        confirm_plan_sha256=str(args.confirm_plan_sha256),
        confirm_delete=str(args.confirm_delete),
    )
    report_path = (
        Path(args.execution_report).resolve()
        if args.execution_report is not None
        else Path(args.plan_path).resolve().with_name("prune_execution.json")
    )
    write_prune_execution(report, report_path)
    print(f"written: {report_path}")
    print(f"removed runs: {report['removed_run_count']}")
    print(f"released bytes: {report['released_bytes']}")
    print("Verified archive copies remain present.")
    return 0


def _run_pack_report(args: argparse.Namespace) -> int:
    source = Path(args.report_path).resolve()
    compact = (
        Path(args.compact_path).resolve()
        if args.compact_path is not None
        else source.with_name(f"{source.stem}.compact.json")
    )
    pack_root = (
        Path(args.pack_root).resolve()
        if args.pack_root is not None
        else source.with_name(f"{source.stem}.pack")
    )
    manifest = (
        Path(args.manifest_path).resolve()
        if args.manifest_path is not None
        else source.with_name(f"{source.stem}.pack_manifest.json")
    )
    threshold = int(float(args.threshold_mib) * 1024 * 1024)
    report = pack_json_report(
        source,
        compact_path=compact,
        pack_root=pack_root,
        manifest_path=manifest,
        threshold_bytes=threshold,
    )
    print(f"written: {compact}")
    print(f"written: {manifest}")
    print(f"sidecars: {report['sidecar_count']}")
    print(f"source bytes: {report['source_bytes']}")
    print(f"compact bytes: {report['compact_bytes']}")
    print(f"sidecar bytes: {report['sidecar_total_bytes']}")
    print("Reconstruction verified; the original report remains present.")
    return 0


def _run_report_prune_plan(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest_path).resolve()
    plan_path = (
        Path(args.plan_path).resolve()
        if args.plan_path is not None
        else manifest.with_name(f"{manifest.stem}.prune_plan.json")
    )
    plan = build_report_prune_plan(manifest)
    write_report_prune_plan(plan, plan_path)
    print(f"written: {plan_path}")
    print(f"releasable bytes: {plan['source_bytes']}")
    print(f"plan_sha256: {plan['plan_sha256']}")
    print(f"required deletion phrase: {REPORT_PRUNE_CONFIRMATION}")
    print("The original report remains present.")
    return 0


def _run_report_prune(args: argparse.Namespace) -> int:
    report = execute_report_prune_plan(
        Path(args.plan_path),
        confirm_plan_sha256=str(args.confirm_plan_sha256),
        confirm_delete=str(args.confirm_delete),
    )
    report_path = (
        Path(args.execution_report).resolve()
        if args.execution_report is not None
        else Path(args.plan_path).resolve().with_name("report_prune_execution.json")
    )
    write_report_prune_execution(report, report_path)
    print(f"written: {report_path}")
    print(f"released bytes: {report['released_bytes']}")
    print("Compact report and verified sidecars remain present.")
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
        if args.command == "verify":
            return _run_verify(args)
        if args.command == "prune-plan":
            return _run_prune_plan(args)
        if args.command == "prune":
            return _run_prune(args)
        if args.command == "pack-report":
            return _run_pack_report(args)
        if args.command == "report-prune-plan":
            return _run_report_prune_plan(args)
        if args.command == "report-prune":
            return _run_report_prune(args)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"DATA MANAGEMENT FAILED: {type(exc).__name__}: {exc}")
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
