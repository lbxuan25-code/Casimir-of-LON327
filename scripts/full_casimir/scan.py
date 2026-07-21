from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .config import (
    DEFAULT_ATOL_J_M2,
    DEFAULT_CERTIFIER_Q_BATCH_SIZE,
    DEFAULT_LOGDET_ATOL,
    DEFAULT_LOGDET_RTOL,
    DEFAULT_LOG_ROOT,
    DEFAULT_MATSUBARA_CUTOFFS,
    DEFAULT_MAX_CONTEXT_WORKERS,
    DEFAULT_MEMORY_BUDGET_GB,
    DEFAULT_N_CANDIDATES,
    DEFAULT_OUTER_CUTOFFS_U,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RESERVED_LOGICAL_CPUS,
    DEFAULT_RTOL,
    DEFAULT_SEPARATION_NM,
    DEFAULT_TEMPERATURE_K,
    DEFAULT_WORKER_CAP,
    PROFILE_NAME,
    case_name,
    inclusive_float_grid,
    select_runtime_resources,
    validate_pairings,
)
from .energy import EnergyRunOptions, run_energy_cases

_PLAN_SCHEMA = "full-casimir-physical-scan-plan-v1"


def _add_resource_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reserve-cpus",
        type=int,
        default=DEFAULT_RESERVED_LOGICAL_CPUS,
    )
    parser.add_argument("--worker-cap", type=int, default=DEFAULT_WORKER_CAP)


def _add_grid_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--angles-deg",
        nargs="+",
        type=float,
        help="Explicit plate-2 angles in degrees.",
    )
    parser.add_argument("--angle-min-deg", type=float)
    parser.add_argument("--angle-max-deg", type=float)
    parser.add_argument("--angle-step-deg", type=float)
    parser.add_argument(
        "--distances-nm",
        nargs="+",
        type=float,
        help="Explicit separations in nm.",
    )
    parser.add_argument("--distance-min-nm", type=float)
    parser.add_argument("--distance-max-nm", type=float)
    parser.add_argument("--distance-step-nm", type=float)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    _add_resource_args(parser)
    _add_grid_args(parser)
    parser.add_argument(
        "--pairings",
        nargs="+",
        default=("spm", "dwave"),
        choices=("spm", "dwave"),
    )
    parser.add_argument("--temperature-K", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--profile", default=PROFILE_NAME)
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
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
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
    parser.add_argument("--continue-on-engineering-failure", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir",
        description=(
            "Plan or run one or more LNO327 Casimir physical cases through a single "
            "multi-angle, multi-distance interface."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser(
        "plan",
        help="Resolve and display the physical case matrix without running it.",
    )
    _add_run_args(plan)
    plan.add_argument("--plan-output", type=Path)

    run = commands.add_parser(
        "run",
        help="Run a single case or a multi-angle/multi-distance case matrix.",
    )
    _add_run_args(run)

    resources = commands.add_parser(
        "resources",
        help="Display the CPU selection that would be used by a run.",
    )
    _add_resource_args(resources)
    return parser


def _resolve_axis(
    *,
    explicit: Sequence[float] | None,
    start: float | None,
    stop: float | None,
    step: float | None,
    default: float,
    label: str,
) -> tuple[float, ...]:
    range_values = (start, stop, step)
    range_requested = any(value is not None for value in range_values)
    if explicit is not None and range_requested:
        raise ValueError(
            f"{label}: explicit values cannot be combined with min/max/step"
        )
    if range_requested and not all(value is not None for value in range_values):
        raise ValueError(f"{label}: min, max, and step must be provided together")
    if explicit is not None:
        values = tuple(float(value) for value in explicit)
    elif range_requested:
        assert start is not None and stop is not None and step is not None
        values = inclusive_float_grid(start, stop, step)
    else:
        values = (float(default),)
    if not values:
        raise ValueError(f"{label}: at least one value is required")
    if len(set(values)) != len(values):
        raise ValueError(f"{label}: duplicate values are not allowed")
    return values


def _resolve_grid(args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[float, ...]]:
    angles = _resolve_axis(
        explicit=args.angles_deg,
        start=args.angle_min_deg,
        stop=args.angle_max_deg,
        step=args.angle_step_deg,
        default=0.0,
        label="angles",
    )
    distances = _resolve_axis(
        explicit=args.distances_nm,
        start=args.distance_min_nm,
        stop=args.distance_max_nm,
        step=args.distance_step_nm,
        default=DEFAULT_SEPARATION_NM,
        label="distances",
    )
    if any(value <= 0.0 for value in distances):
        raise ValueError("distances must be positive")
    if float(args.temperature_K) <= 0.0:
        raise ValueError("temperature_K must be positive")
    return angles, distances


def _resources(args: argparse.Namespace):
    return select_runtime_resources(
        reserve_logical_cpus=int(args.reserve_cpus),
        worker_cap=int(args.worker_cap),
    )


def _options(args: argparse.Namespace) -> EnergyRunOptions:
    return EnergyRunOptions(
        output_root=Path(args.output_root),
        log_root=Path(args.log_root),
        temperature_K=float(args.temperature_K),
        separation_nm=DEFAULT_SEPARATION_NM,
        N_candidates=tuple(int(value) for value in args.N_candidates),
        matsubara_cutoffs=tuple(int(value) for value in args.matsubara_cutoffs),
        outer_cutoffs_u=tuple(float(value) for value in args.outer_cutoffs_u),
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
        continue_on_engineering_failure=bool(args.continue_on_engineering_failure),
    )


def build_scan_plan(args: argparse.Namespace) -> dict[str, Any]:
    pairings = validate_pairings(args.pairings)
    angles, distances = _resolve_grid(args)
    cases: list[dict[str, Any]] = []
    names: set[str] = set()
    for pairing in pairings:
        for distance in distances:
            for angle in angles:
                name = case_name(
                    pairing,
                    angle,
                    temperature_K=float(args.temperature_K),
                    separation_nm=distance,
                    profile=str(args.profile),
                )
                if name in names:
                    raise ValueError(f"case-name collision: {name}")
                names.add(name)
                cases.append(
                    {
                        "case": name,
                        "pairing": pairing,
                        "temperature_K": float(args.temperature_K),
                        "separation_nm": distance,
                        "plate_angles_deg": [0.0, angle],
                    }
                )
    return {
        "schema": _PLAN_SCHEMA,
        "profile": str(args.profile),
        "case_count": len(cases),
        "pairings": list(pairings),
        "angles_deg": list(angles),
        "distances_nm": list(distances),
        "cases": cases,
    }


def _print_plan(plan: dict[str, Any]) -> None:
    print(f"profile: {plan['profile']}")
    print(f"cases: {plan['case_count']}")
    print(f"pairings: {plan['pairings']}")
    print(f"distances_nm: {plan['distances_nm']}")
    print(f"angles_deg: {plan['angles_deg']}")
    for index, row in enumerate(plan["cases"], start=1):
        print(
            f"[{index:04d}/{plan['case_count']:04d}] "
            f"{row['case']}"
        )


def _write_plan(path: Path, plan: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"written: {destination}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "resources":
            resources = _resources(args)
            print(f"visible CPUs ({len(resources.visible_cpus)}): {resources.visible_cpus}")
            print(f"selected CPUs ({resources.workers}): {resources.selected_cpus}")
            print(
                f"reserved CPUs ({len(resources.reserved_cpus)}): "
                f"{resources.reserved_cpus}"
            )
            return 0

        plan = build_scan_plan(args)
        _print_plan(plan)
        if args.command == "plan":
            if args.plan_output is not None:
                _write_plan(args.plan_output, plan)
            return 0

        angles, distances = _resolve_grid(args)
        return run_energy_cases(
            pairings=validate_pairings(args.pairings),
            angles_deg=angles,
            distances_nm=distances,
            resources=_resources(args),
            options=_options(args),
            profile=str(args.profile),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"SCAN FAILED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
