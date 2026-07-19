from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .cleanup_legacy_root import cleanup_legacy_root_scripts
from .config import (
    DEFAULT_ATOL_J_M2,
    DEFAULT_CERTIFIER_Q_BATCH_SIZE,
    DEFAULT_LOGDET_ATOL,
    DEFAULT_LOGDET_ROOT if False else DEFAULT_LOG_ROOT,
    DEFAULT_LOGDET_RTOL,
    DEFAULT_MATSUBARA_CUTOFFS,
    DEFAULT_MAX_CONTEXT_WORKERS,
    DEFAULT_MEMORY_BUDGET_GB,
    DEFAULT_N_CANDIDATES,
    DEFAULT_OUTER_CUTOFFS_U,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POSTPROCESS_ROOT,
    DEFAULT_RESERVED_LOGICAL_CPUS,
    DEFAULT_RTOL,
    DEFAULT_SCAN_MAX_DEG,
    DEFAULT_SCAN_MIN_DEG,
    DEFAULT_SCAN_STEP_DEG,
    DEFAULT_SEPARATION_NM,
    DEFAULT_TEMPERATURE_K,
    DEFAULT_WORKER_CAP,
    PILOT_PROFILE,
    PROFILE_NAME,
    apply_single_thread_environment,
    inclusive_integer_grid,
    select_runtime_resources,
    validate_pairings,
)

# Set BLAS/OpenMP policy before importing the numerical package through energy.py.
apply_single_thread_environment()

from .energy import EnergyRunOptions, run_energy_cases  # noqa: E402
from .plotting import plot_results  # noqa: E402
from .postprocess import postprocess_torque  # noqa: E402


# The odd-looking compatibility alias above is intentionally avoided at runtime.
DEFAULT_LOG_ROOT = globals().get("DEFAULT_LOG_ROOT")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir.workflow",
        description=(
            "Run LNO327 Casimir energy calculations, torque post-processing, "
            "and plotting from the organized scripts package."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    resources = subparsers.add_parser(
        "resources",
        description="Show CPU reservation without starting a calculation.",
    )
    _add_resource_args(resources)

    pilots = subparsers.add_parser(
        "pilots",
        description="Run SPM and d-wave 0-degree pilot cases sequentially.",
    )
    _add_energy_args(pilots)
    pilots.set_defaults(profile=PILOT_PROFILE)

    scan = subparsers.add_parser(
        "scan",
        description="Run the padded angle scan used for torque extraction.",
    )
    _add_energy_args(scan)
    scan.add_argument("--angle-min", type=int, default=DEFAULT_SCAN_MIN_DEG)
    scan.add_argument("--angle-max", type=int, default=DEFAULT_SCAN_MAX_DEG)
    scan.add_argument("--angle-step", type=int, default=DEFAULT_SCAN_STEP_DEG)
    scan.set_defaults(profile=PROFILE_NAME)

    torque = subparsers.add_parser(
        "torque",
        description="Extract free-energy tables and torque diagnostics.",
    )
    _add_postprocess_args(torque)

    plot = subparsers.add_parser(
        "plot",
        description="Plot free-energy and torque CSV files.",
    )
    _add_postprocess_args(plot)

    all_command = subparsers.add_parser(
        "all",
        description="Run the full angle scan, then torque extraction and plotting.",
    )
    _add_energy_args(all_command)
    all_command.add_argument("--angle-min", type=int, default=DEFAULT_SCAN_MIN_DEG)
    all_command.add_argument("--angle-max", type=int, default=DEFAULT_SCAN_MAX_DEG)
    all_command.add_argument("--angle-step", type=int, default=DEFAULT_SCAN_STEP_DEG)
    all_command.add_argument(
        "--postprocess-root",
        type=Path,
        default=DEFAULT_POSTPROCESS_ROOT,
    )
    all_command.set_defaults(profile=PROFILE_NAME)

    cleanup = subparsers.add_parser(
        "cleanup",
        description="Remove obsolete root-level helper scripts left by older workflows.",
    )
    cleanup.add_argument("--quiet", action="store_true")

    return parser


def _add_resource_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reserve-cpus",
        type=int,
        default=DEFAULT_RESERVED_LOGICAL_CPUS,
        help="Logical CPUs kept free for desktop responsiveness (default: 6).",
    )
    parser.add_argument(
        "--worker-cap",
        type=int,
        default=DEFAULT_WORKER_CAP,
        help="Maximum worker processes (default: 26).",
    )


def _add_energy_args(parser: argparse.ArgumentParser) -> None:
    _add_resource_args(parser)
    parser.add_argument(
        "--pairings",
        nargs="+",
        default=("spm", "dwave"),
        choices=("spm", "dwave"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--temperature-K", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--separation-nm", type=float, default=DEFAULT_SEPARATION_NM)
    parser.add_argument(
        "--N-candidates",
        nargs="+",
        type=int,
        default=DEFAULT_N_CANDIDATES,
    )
    parser.add_argument(
        "--matsubara-cutoffs",
        nargs="+",
        type=int,
        default=DEFAULT_MATSUBARA_CUTOFFS,
    )
    parser.add_argument(
        "--outer-cutoffs-u",
        nargs="+",
        type=float,
        default=DEFAULT_OUTER_CUTOFFS_U,
    )
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol-J-m2", type=float, default=DEFAULT_ATOL_J_M2)
    parser.add_argument("--logdet-rtol", type=float, default=DEFAULT_LOGDET_RTOL)
    parser.add_argument("--logdet-atol", type=float, default=DEFAULT_LOGDET_ATOL)
    parser.add_argument(
        "--certifier-q-batch-size",
        type=int,
        default=DEFAULT_CERTIFIER_Q_BATCH_SIZE,
    )
    parser.add_argument(
        "--required-consecutive-passes",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=DEFAULT_MEMORY_BUDGET_GB,
    )
    parser.add_argument(
        "--max-context-workers",
        type=int,
        default=DEFAULT_MAX_CONTEXT_WORKERS,
    )
    parser.add_argument(
        "--parallel-mode",
        choices=("auto", "serial", "q", "context", "wave"),
        default="q",
    )
    parser.add_argument("--retry-unresolved", action="store_true")


def _add_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--postprocess-root",
        type=Path,
        default=DEFAULT_POSTPROCESS_ROOT,
    )
    parser.add_argument("--profile", default=PROFILE_NAME)
    parser.add_argument("--angle-step", type=int, default=DEFAULT_SCAN_STEP_DEG)


def _energy_options(args: argparse.Namespace) -> EnergyRunOptions:
    return EnergyRunOptions(
        output_root=args.output_root,
        log_root=args.log_root,
        temperature_K=float(args.temperature_K),
        separation_nm=float(args.separation_nm),
        N_candidates=tuple(args.N_candidates),
        matsubara_cutoffs=tuple(args.matsubara_cutoffs),
        outer_cutoffs_u=tuple(args.outer_cutoffs_u),
        rtol=float(args.rtol),
        atol_J_m2=float(args.atol_J_m2),
        logdet_rtol=float(args.logdet_rtol),
        logdet_atol=float(args.logdet_atol),
        certifier_q_batch_size=int(args.certifier_q_batch_size),
        memory_budget_gb=float(args.memory_budget_gb),
        max_context_workers=int(args.max_context_workers),
        parallel_mode=str(args.parallel_mode),
        required_consecutive_passes=int(args.required_consecutive_passes),
        retry_unresolved=bool(args.retry_unresolved),
    )


def _resources(args: argparse.Namespace):
    return select_runtime_resources(
        reserve_logical_cpus=int(args.reserve_cpus),
        worker_cap=int(args.worker_cap),
    )


def _run_postprocess(args: argparse.Namespace) -> int:
    energy_csv, torque_csv, metadata, complete = postprocess_torque(
        run_root=args.run_root,
        output_root=args.postprocess_root,
        profile=args.profile,
        step_deg=args.angle_step,
    )
    print(f"written: {energy_csv}")
    print(f"written: {torque_csv}")
    print(f"written: {metadata}")
    if not complete:
        print(
            "torque table contains missing angles because some energies are "
            "absent or unresolved"
        )
    return 0 if complete else 2


def _run_plot(args: argparse.Namespace) -> int:
    outputs = plot_results(
        output_root=args.postprocess_root,
        profile=args.profile,
    )
    if not outputs:
        print("no usable energy or torque rows were available to plot")
        return 2
    for path in outputs:
        print(f"written: {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "cleanup":
        removed = cleanup_legacy_root_scripts()
        if not args.quiet:
            if removed:
                for path in removed:
                    print(f"removed: {path}")
            else:
                print("repository root is already clean")
        return 0

    if args.command == "resources":
        resources = _resources(args)
        print(f"visible CPUs ({len(resources.visible_cpus)}): {resources.visible_cpus}")
        print(f"selected CPUs ({resources.workers}): {resources.selected_cpus}")
        print(
            f"reserved CPUs ({len(resources.reserved_cpus)}): "
            f"{resources.reserved_cpus}"
        )
        return 0

    if args.command == "torque":
        return _run_postprocess(args)

    if args.command == "plot":
        return _run_plot(args)

    removed = cleanup_legacy_root_scripts()
    for path in removed:
        print(f"removed obsolete root-level script: {path.name}")

    resources = _resources(args)
    pairings = validate_pairings(args.pairings)
    options = _energy_options(args)
    profile = args.profile or (
        PILOT_PROFILE if args.command == "pilots" else PROFILE_NAME
    )

    if args.command == "pilots":
        return run_energy_cases(
            pairings=pairings,
            angles_deg=(0,),
            resources=resources,
            options=options,
            profile=profile,
        )

    angles = inclusive_integer_grid(
        args.angle_min,
        args.angle_max,
        args.angle_step,
    )
    energy_status = run_energy_cases(
        pairings=pairings,
        angles_deg=angles,
        resources=resources,
        options=options,
        profile=profile,
    )
    if args.command == "scan" or energy_status == 1:
        return energy_status

    postprocess_args = argparse.Namespace(
        run_root=args.output_root,
        postprocess_root=args.postprocess_root,
        profile=profile,
        angle_step=args.angle_step,
    )
    torque_status = _run_postprocess(postprocess_args)
    plot_status = _run_plot(postprocess_args)
    return max(energy_status, torque_status, plot_status)


if __name__ == "__main__":
    raise SystemExit(main())
