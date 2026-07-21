from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from lno327.casimir.certified_matsubara import (
    MATSUBARA_TAIL_CERTIFICATE_CONTRACT,
    PRODUCTION_ERROR_BUDGET_CONTRACT,
    validate_dyadic_matsubara_policy,
)
from lno327.casimir.certified_tail import OUTER_TAIL_CERTIFICATE_CONTRACT
from lno327.casimir.error_budget import error_budget_policy_payload

from .config import (
    DEFAULT_ATOL_J_M2,
    DEFAULT_CERTIFIER_Q_BATCH_SIZE,
    DEFAULT_DEGENERACY,
    DEFAULT_DELTA0_EV,
    DEFAULT_ETA_EV,
    DEFAULT_LOGDET_ATOL,
    DEFAULT_LOGDET_RTOL,
    DEFAULT_MATSUBARA_CUTOFFS,
    DEFAULT_MATSUBARA_TAIL_RATIO_MAX,
    DEFAULT_MATSUBARA_TAIL_START_N,
    DEFAULT_MATSUBARA_TAIL_WINDOW_TERMS,
    DEFAULT_MAX_CONTEXT_WORKERS,
    DEFAULT_MAX_TOTAL_MICROSCOPIC_POINT_ENTRIES,
    DEFAULT_MAX_TOTAL_MICROSCOPIC_Q_NODES,
    DEFAULT_MEMORY_BUDGET_GB,
    DEFAULT_N_CANDIDATES,
    DEFAULT_OUTER_CUTOFFS_U,
    DEFAULT_OUTER_TAIL_RATIO_MAX,
    DEFAULT_OUTER_TAIL_START_U,
    DEFAULT_OUTER_TAIL_WINDOW_SHELLS,
    DEFAULT_PRODUCTION_ROOT,
    DEFAULT_RADIAL_BUDGET_FRACTION,
    DEFAULT_RESERVED_LOGICAL_CPUS,
    DEFAULT_RTOL,
    DEFAULT_SEPARATION_NM,
    DEFAULT_TEMPERATURE_K,
    DEFAULT_WORKER_CAP,
    inclusive_float_grid,
    physical_case_name,
    select_runtime_resources,
    validate_pairings,
)
from .energy import ProductionRunOptions, run_production_plan
from .identity import (
    PLAN_SCHEMA,
    POLICY_SCHEMA,
    build_campaign_identity,
    build_case_identity,
    finalize_plan,
    git_code_identity,
    read_json_object,
    verify_plan_payload,
)


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


def _add_scientific_args(parser: argparse.ArgumentParser) -> None:
    _add_grid_args(parser)
    parser.add_argument(
        "--pairings",
        nargs="+",
        default=("spm", "dwave"),
        choices=("spm", "dwave"),
    )
    parser.add_argument("--temperature-K", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--delta0-eV", type=float, default=DEFAULT_DELTA0_EV)
    parser.add_argument("--eta-eV", type=float, default=DEFAULT_ETA_EV)
    parser.add_argument("--degeneracy", type=float, default=DEFAULT_DEGENERACY)
    parser.add_argument(
        "--N-candidates", nargs="+", type=int, default=DEFAULT_N_CANDIDATES
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
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
    parser.add_argument(
        "--outer-tail-start-u", type=float, default=DEFAULT_OUTER_TAIL_START_U
    )
    parser.add_argument(
        "--outer-tail-window-shells",
        type=int,
        default=DEFAULT_OUTER_TAIL_WINDOW_SHELLS,
    )
    parser.add_argument(
        "--outer-tail-ratio-max", type=float, default=DEFAULT_OUTER_TAIL_RATIO_MAX
    )
    parser.add_argument(
        "--matsubara-tail-start-n", type=int, default=DEFAULT_MATSUBARA_TAIL_START_N
    )
    parser.add_argument(
        "--matsubara-tail-window-terms",
        type=int,
        default=DEFAULT_MATSUBARA_TAIL_WINDOW_TERMS,
        help="Number of dyadic tail blocks, including the final holdout block.",
    )
    parser.add_argument(
        "--matsubara-tail-ratio-max",
        type=float,
        default=DEFAULT_MATSUBARA_TAIL_RATIO_MAX,
        help="Maximum ratio between successive absolute dyadic block envelopes.",
    )
    parser.add_argument(
        "--radial-budget-fraction",
        type=float,
        default=DEFAULT_RADIAL_BUDGET_FRACTION,
    )
    parser.add_argument(
        "--max-total-microscopic-q-nodes",
        type=int,
        default=DEFAULT_MAX_TOTAL_MICROSCOPIC_Q_NODES,
    )
    parser.add_argument(
        "--max-total-microscopic-point-entries",
        type=int,
        default=DEFAULT_MAX_TOTAL_MICROSCOPIC_POINT_ENTRIES,
    )


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    _add_resource_args(parser)
    parser.add_argument(
        "--campaign-root", type=Path, default=DEFAULT_PRODUCTION_ROOT
    )
    parser.add_argument(
        "--certifier-q-batch-size",
        type=int,
        default=DEFAULT_CERTIFIER_Q_BATCH_SIZE,
    )
    parser.add_argument(
        "--memory-budget-gb", type=float, default=DEFAULT_MEMORY_BUDGET_GB
    )
    parser.add_argument(
        "--max-context-workers", type=int, default=DEFAULT_MAX_CONTEXT_WORKERS
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
            "Plan or run fail-closed LNO327 Casimir production campaigns through "
            "one multi-angle, multi-distance interface."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser(
        "plan",
        help="Freeze a scientific policy and physical case matrix without running it.",
    )
    _add_scientific_args(plan)
    plan.add_argument("--plan-output", type=Path)

    run = commands.add_parser(
        "run",
        help="Execute a SHA-confirmed production plan in fresh or resume mode.",
    )
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--confirm-plan-sha256", required=True)
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    _add_execution_args(run)

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


def _strictly_increasing(
    values: Sequence[int | float], *, label: str
) -> tuple[Any, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError(f"{label} must be strictly increasing")
    return normalized


def _scientific_policy(args: argparse.Namespace) -> dict[str, Any]:
    N_candidates = tuple(int(value) for value in args.N_candidates)
    matsubara = tuple(int(value) for value in args.matsubara_cutoffs)
    outer = tuple(float(value) for value in args.outer_cutoffs_u)
    _strictly_increasing(N_candidates, label="N_candidates")
    _strictly_increasing(matsubara, label="matsubara_cutoffs")
    _strictly_increasing(outer, label="outer_cutoffs_u")
    if N_candidates[0] <= 0 or outer[0] <= 0.0:
        raise ValueError("scientific ladders contain invalid non-positive values")
    if int(args.required_consecutive_passes) < 1:
        raise ValueError("required_consecutive_passes must be positive")
    radial_fraction = float(args.radial_budget_fraction)
    if not 0.0 < radial_fraction < 1.0:
        raise ValueError("radial_budget_fraction must lie strictly between zero and one")
    validate_dyadic_matsubara_policy(
        matsubara,
        tail_start_n=int(args.matsubara_tail_start_n),
        tail_window_blocks=int(args.matsubara_tail_window_terms),
    )
    return {
        "schema": POLICY_SCHEMA,
        "model": {
            "delta0_eV": float(args.delta0_eV),
            "eta_eV": float(args.eta_eV),
            "degeneracy": float(args.degeneracy),
        },
        "microscopic": {
            "N_candidates": list(N_candidates),
            "required_consecutive_passes": int(args.required_consecutive_passes),
            "logdet_rtol": float(args.logdet_rtol),
            "logdet_atol": float(args.logdet_atol),
        },
        "outer_integration": {
            "cutoff_u_values": list(outer),
            "tail_start_u": float(args.outer_tail_start_u),
            "tail_window_shells": int(args.outer_tail_window_shells),
            "tail_ratio_max": float(args.outer_tail_ratio_max),
            "radial_budget_fraction": radial_fraction,
            "max_total_microscopic_q_nodes": int(
                args.max_total_microscopic_q_nodes
            ),
            "certificate_contract": OUTER_TAIL_CERTIFICATE_CONTRACT,
            "accepted_certificate_paths": [
                "analytic_passive_vacuum",
                "geometric_numerical_shell_envelope",
            ],
            "static_contraction_norm": "exact_spectral_norm",
        },
        "matsubara": {
            "cutoff_values": list(matsubara),
            "tail_start_n": int(args.matsubara_tail_start_n),
            "tail_window_terms": int(args.matsubara_tail_window_terms),
            "tail_ratio_max": float(args.matsubara_tail_ratio_max),
            "max_total_microscopic_point_entries": int(
                args.max_total_microscopic_point_entries
            ),
            "certificate_contract": MATSUBARA_TAIL_CERTIFICATE_CONTRACT,
            "tail_estimator": "dyadic_absolute_block_envelope",
            "holdout_blocks": 1,
            "per_term_ratio_acceptance_forbidden": True,
        },
        "total_free_energy": {
            "rtol": float(args.rtol),
            "atol_J_m2": float(args.atol_J_m2),
        },
        "error_budget": error_budget_policy_payload(
            radial_budget_fraction=radial_fraction
        ),
        "production_authorization": {
            "contract": PRODUCTION_ERROR_BUDGET_CONTRACT,
            "requires_all_microscopic_nodes_certified": True,
            "requires_outer_tail_certificate": True,
            "requires_matsubara_tail_certificate": True,
            "requires_total_error_budget_closed": True,
            "numerical_convergence_alone_is_insufficient": True,
        },
    }


def _resources(args: argparse.Namespace):
    return select_runtime_resources(
        reserve_logical_cpus=int(args.reserve_cpus),
        worker_cap=int(args.worker_cap),
    )


def build_scan_plan(
    args: argparse.Namespace,
    *,
    code_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pairings = validate_pairings(args.pairings)
    angles, distances = _resolve_grid(args)
    policy = _scientific_policy(args)
    code = git_code_identity() if code_identity is None else dict(code_identity)
    campaign = build_campaign_identity(scientific_policy=policy, code_identity=code)
    cases: list[dict[str, Any]] = []
    names: set[str] = set()
    for pairing in pairings:
        for distance in distances:
            for angle in angles:
                name = physical_case_name(
                    pairing,
                    angle,
                    temperature_K=float(args.temperature_K),
                    separation_nm=distance,
                )
                if name in names:
                    raise ValueError(f"case-name collision: {name}")
                names.add(name)
                identity = build_case_identity(
                    campaign_id=campaign["campaign_id"],
                    pairing=pairing,
                    temperature_K=float(args.temperature_K),
                    separation_nm=distance,
                    plate_angles_deg=(0.0, angle),
                )
                cases.append({"case": name, "case_identity": identity})
    return finalize_plan(
        {
            "schema": PLAN_SCHEMA,
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": campaign["campaign_sha256"],
            "scientific_policy_sha256": campaign[
                "scientific_policy_sha256"
            ],
            "code_identity": code,
            "scientific_policy": policy,
            "case_count": len(cases),
            "pairings": list(pairings),
            "angles_deg": list(angles),
            "distances_nm": list(distances),
            "temperature_K": float(args.temperature_K),
            "cases": cases,
        }
    )


def _print_plan(plan: dict[str, Any]) -> None:
    print(f"campaign_id: {plan['campaign_id']}")
    print(f"campaign_sha256: {plan['campaign_sha256']}")
    print(f"scientific_policy_sha256: {plan['scientific_policy_sha256']}")
    print(f"git_commit: {plan['code_identity']['git_commit']}")
    print(f"plan_sha256: {plan['plan_sha256']}")
    print(f"cases: {plan['case_count']}")
    print(f"pairings: {plan['pairings']}")
    print(f"distances_nm: {plan['distances_nm']}")
    print(f"angles_deg: {plan['angles_deg']}")
    for index, row in enumerate(plan["cases"], start=1):
        print(f"[{index:04d}/{plan['case_count']:04d}] {row['case']}")


def _write_plan(path: Path, plan: dict[str, Any]) -> None:
    from .identity import atomic_json

    atomic_json(Path(path), plan)
    print(f"written: {path}")


def _execution_options(args: argparse.Namespace) -> ProductionRunOptions:
    return ProductionRunOptions(
        campaign_root=Path(args.campaign_root),
        certifier_q_batch_size=int(args.certifier_q_batch_size),
        memory_budget_gb=float(args.memory_budget_gb),
        max_context_workers=int(args.max_context_workers),
        parallel_mode=str(args.parallel_mode),
        retry_unresolved=bool(args.retry_unresolved),
        continue_on_engineering_failure=bool(args.continue_on_engineering_failure),
    )


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

        if args.command == "plan":
            plan = build_scan_plan(args)
            _print_plan(plan)
            if args.plan_output is not None:
                _write_plan(args.plan_output, plan)
            return 0

        plan = read_json_object(args.plan)
        verify_plan_payload(plan, expected_sha256=str(args.confirm_plan_sha256))
        mode = "fresh" if args.fresh else "resume"
        print(f"campaign_id: {plan['campaign_id']}", flush=True)
        print(f"plan_sha256: {plan['plan_sha256']}", flush=True)
        print(f"mode: {mode}", flush=True)
        return run_production_plan(
            plan=plan,
            mode=mode,
            resources=_resources(args),
            options=_execution_options(args),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"SCAN FAILED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
