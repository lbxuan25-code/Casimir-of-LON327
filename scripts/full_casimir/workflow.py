from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lno327.casimir.production import build_full_casimir_config

from .cache_migration import migrate_pilot_caches
from .cleanup_legacy_root import cleanup_legacy_root_scripts
from .config import (
    DEFAULT_ATOL_J_M2, DEFAULT_CERTIFIER_Q_BATCH_SIZE, DEFAULT_LOGDET_ATOL,
    DEFAULT_LOGDET_RTOL, DEFAULT_LOG_ROOT, DEFAULT_MATSUBARA_CUTOFFS,
    DEFAULT_MAX_CONTEXT_WORKERS, DEFAULT_MEMORY_BUDGET_GB, DEFAULT_N_CANDIDATES,
    DEFAULT_OUTER_CUTOFFS_U, DEFAULT_OUTPUT_ROOT, DEFAULT_POSTPROCESS_ROOT,
    DEFAULT_RESERVED_LOGICAL_CPUS, DEFAULT_RTOL, DEFAULT_SCAN_MAX_DEG,
    DEFAULT_SCAN_MIN_DEG, DEFAULT_SCAN_STEP_DEG, DEFAULT_WORKER_CAP,
    LEGACY_PILOT_PROFILE, PILOT_PROFILE, PROFILE_NAME, inclusive_integer_grid,
    select_runtime_resources, validate_pairings,
)
from .energy import EnergyRunOptions, run_energy_cases
from .plotting import plot_results
from .postprocess import postprocess_torque


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir.workflow",
        description="Run LNO327 Casimir energies, torque post-processing, and plots.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    resources = subparsers.add_parser("resources")
    _add_resource_args(resources)
    pilots = subparsers.add_parser("pilots")
    _add_energy_args(pilots)
    pilots.set_defaults(profile=PILOT_PROFILE)
    migrate = subparsers.add_parser("migrate-pilots")
    _add_energy_args(migrate)
    migrate.set_defaults(profile=PILOT_PROFILE)
    scan = subparsers.add_parser("scan")
    _add_energy_args(scan)
    scan.add_argument("--angle-min", type=int, default=DEFAULT_SCAN_MIN_DEG)
    scan.add_argument("--angle-max", type=int, default=DEFAULT_SCAN_MAX_DEG)
    scan.add_argument("--angle-step", type=int, default=DEFAULT_SCAN_STEP_DEG)
    scan.set_defaults(profile=PROFILE_NAME)
    torque = subparsers.add_parser("torque")
    _add_postprocess_args(torque)
    plot = subparsers.add_parser("plot")
    _add_postprocess_args(plot)
    all_command = subparsers.add_parser("all")
    _add_energy_args(all_command)
    all_command.add_argument("--angle-min", type=int, default=DEFAULT_SCAN_MIN_DEG)
    all_command.add_argument("--angle-max", type=int, default=DEFAULT_SCAN_MAX_DEG)
    all_command.add_argument("--angle-step", type=int, default=DEFAULT_SCAN_STEP_DEG)
    all_command.add_argument("--postprocess-root", type=Path, default=DEFAULT_POSTPROCESS_ROOT)
    all_command.set_defaults(profile=PROFILE_NAME)
    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--quiet", action="store_true")
    return parser


def _add_resource_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reserve-cpus", type=int, default=DEFAULT_RESERVED_LOGICAL_CPUS)
    parser.add_argument("--worker-cap", type=int, default=DEFAULT_WORKER_CAP)


def _add_energy_args(parser: argparse.ArgumentParser) -> None:
    _add_resource_args(parser)
    parser.add_argument("--pairings", nargs="+", default=("spm", "dwave"), choices=("spm", "dwave"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--N-candidates", nargs="+", type=int, default=DEFAULT_N_CANDIDATES)
    parser.add_argument("--matsubara-cutoffs", nargs="+", type=int, default=DEFAULT_MATSUBARA_CUTOFFS)
    parser.add_argument("--outer-cutoffs-u", nargs="+", type=float, default=DEFAULT_OUTER_CUTOFFS_U)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol-J-m2", type=float, default=DEFAULT_ATOL_J_M2)
    parser.add_argument("--logdet-rtol", type=float, default=DEFAULT_LOGDET_RTOL)
    parser.add_argument("--logdet-atol", type=float, default=DEFAULT_LOGDET_ATOL)
    parser.add_argument("--certifier-q-batch-size", type=int, default=DEFAULT_CERTIFIER_Q_BATCH_SIZE)
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
    parser.add_argument("--memory-budget-gb", type=float, default=DEFAULT_MEMORY_BUDGET_GB)
    parser.add_argument("--max-context-workers", type=int, default=DEFAULT_MAX_CONTEXT_WORKERS)
    parser.add_argument("--parallel-mode", choices=("auto", "serial", "q", "context", "wave"), default="q")
    parser.add_argument("--retry-unresolved", action="store_true")
    parser.add_argument("--continue-on-engineering-failure", action="store_true")
    parser.add_argument("--no-migrate-v2-cache", action="store_true")


def _add_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--postprocess-root", type=Path, default=DEFAULT_POSTPROCESS_ROOT)
    parser.add_argument("--profile", default=PROFILE_NAME)
    parser.add_argument("--angle-step", type=int, default=DEFAULT_SCAN_STEP_DEG)


def _energy_options(args: argparse.Namespace) -> EnergyRunOptions:
    return EnergyRunOptions(
        output_root=args.output_root, log_root=args.log_root,
        N_candidates=tuple(args.N_candidates), matsubara_cutoffs=tuple(args.matsubara_cutoffs),
        outer_cutoffs_u=tuple(args.outer_cutoffs_u), rtol=float(args.rtol),
        atol_J_m2=float(args.atol_J_m2), logdet_rtol=float(args.logdet_rtol),
        logdet_atol=float(args.logdet_atol),
        certifier_q_batch_size=int(args.certifier_q_batch_size),
        memory_budget_gb=float(args.memory_budget_gb),
        max_context_workers=int(args.max_context_workers),
        parallel_mode=str(args.parallel_mode),
        required_consecutive_passes=int(args.required_consecutive_passes),
        retry_unresolved=bool(args.retry_unresolved),
        continue_on_engineering_failure=bool(args.continue_on_engineering_failure),
    )


def _resources(args: argparse.Namespace):
    return select_runtime_resources(
        reserve_logical_cpus=int(args.reserve_cpus), worker_cap=int(args.worker_cap))


def _target_point_configs(pairings, resources, options):
    output = {}
    for pairing in pairings:
        full = build_full_casimir_config(
            pairings=(pairing,), temperature_K=options.temperature_K,
            separation_nm=options.separation_nm, plate_angles_deg=(0.0, 0.0),
            N_candidates=options.N_candidates,
            required_consecutive_passes=options.required_consecutive_passes,
            logdet_rtol=options.logdet_rtol, logdet_atol=options.logdet_atol,
            certifier_q_batch_size=options.certifier_q_batch_size,
            workers=resources.workers, parallel_mode=options.parallel_mode,
            memory_budget_gb=options.memory_budget_gb,
            max_context_workers=options.max_context_workers,
            matsubara_cutoff_values=options.matsubara_cutoffs,
            cutoff_u_values=options.outer_cutoffs_u,
            total_free_energy_rtol=options.rtol,
            total_free_energy_atol_J_m2=options.atol_J_m2,
        )
        output[pairing] = full.outer_tail_config.joint_config.radial_config.point_config
    return output


def _migrate(args, pairings, resources, options) -> None:
    reports = migrate_pilot_caches(
        pairings=pairings, output_root=options.output_root,
        source_profile=LEGACY_PILOT_PROFILE, target_profile=PILOT_PROFILE,
        target_configs=_target_point_configs(pairings, resources, options),
    )
    for report in reports:
        print(
            f"cache migration {report.pairing}: entries={report.target_entry_count}, "
            f"established={report.established_after}, newly_established={report.newly_established}, "
            f"skipped={report.skipped}", flush=True)


def _run_postprocess(args: argparse.Namespace) -> int:
    energy_csv, torque_csv, metadata, complete = postprocess_torque(
        run_root=args.run_root, output_root=args.postprocess_root,
        profile=args.profile, step_deg=args.angle_step)
    for path in (energy_csv, torque_csv, metadata):
        print(f"written: {path}")
    return 0 if complete else 2


def _run_plot(args: argparse.Namespace) -> int:
    outputs = plot_results(output_root=args.postprocess_root, profile=args.profile)
    for path in outputs:
        print(f"written: {path}")
    return 0 if outputs else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "cleanup":
        removed = cleanup_legacy_root_scripts()
        if not args.quiet:
            print("repository root is already clean" if not removed else "\n".join(f"removed: {p}" for p in removed))
        return 0
    if args.command == "resources":
        resources = _resources(args)
        print(f"visible CPUs ({len(resources.visible_cpus)}): {resources.visible_cpus}")
        print(f"selected CPUs ({resources.workers}): {resources.selected_cpus}")
        print(f"reserved CPUs ({len(resources.reserved_cpus)}): {resources.reserved_cpus}")
        return 0
    if args.command == "torque":
        return _run_postprocess(args)
    if args.command == "plot":
        return _run_plot(args)
    for path in cleanup_legacy_root_scripts():
        print(f"removed obsolete root-level script: {path.name}")
    resources = _resources(args)
    pairings = validate_pairings(args.pairings)
    options = _energy_options(args)
    profile = args.profile or (PILOT_PROFILE if args.command in ("pilots", "migrate-pilots") else PROFILE_NAME)
    if args.command in ("pilots", "migrate-pilots") and not args.no_migrate_v2_cache:
        _migrate(args, pairings, resources, options)
    if args.command == "migrate-pilots":
        return 0
    if args.command == "pilots":
        return run_energy_cases(pairings=pairings, angles_deg=(0,), resources=resources, options=options, profile=profile)
    angles = inclusive_integer_grid(args.angle_min, args.angle_max, args.angle_step)
    energy_status = run_energy_cases(
        pairings=pairings, angles_deg=angles, resources=resources,
        options=options, profile=profile)
    if args.command == "scan" or energy_status != 0:
        return energy_status
    postprocess_args = argparse.Namespace(
        run_root=args.output_root, postprocess_root=args.postprocess_root,
        profile=profile, angle_step=args.angle_step)
    torque_status = _run_postprocess(postprocess_args)
    if torque_status != 0:
        return torque_status
    return _run_plot(postprocess_args)


if __name__ == "__main__":
    raise SystemExit(main())
