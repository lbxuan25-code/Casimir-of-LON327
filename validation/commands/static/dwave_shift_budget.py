"""Equal-total-point allocation scan for exact-static d-wave periodic shifts.

The default configurations compare 4x240^2, 8x170^2, and 16x120^2
quadrature points using one nested C4/antithetic Halton shift family.  Each
allocation merges complete-lattice primitive blocks and Ward RHS values before
one Schur complement.  The scan is a budget-allocation diagnostic, not a
converged reference builder.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from validation.lib.dwave_shift_budget import (
    allocation_metrics,
    parse_allocations,
    run_budget_task,
)

DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_shift_budget/raw/"
    "dwave_q003_002_equal_budget.csv"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    wall_seconds: float,
) -> str:
    lines = [
        "d-wave equal-budget periodic-shift allocation scan",
        "=" * 54,
        f"q = ({args.qx:.8g}, {args.qy:.8g}); T = {args.temperature_K:.8g} K",
        f"wall time = {wall_seconds:.3f} s",
        "shift family = nested Halton(2,3) C4/antithetic",
        "",
        "Allocation results",
        "------------------",
        " shifts  base_nk   Ntotal    chi_bar    Dbar_T   raw-long  Ward-prim  proj",
    ]
    for row in rows:
        lines.append(
            f"{int(row['num_shifts']):7d} {int(row['base_nk']):8d} "
            f"{int(row['num_quadrature_points']):8d} "
            f"{float(row['chi_bar']):10.7f} {float(row['dbar_t']):10.7f} "
            f"{float(row['raw_longitudinal']):9.3e} "
            f"{float(row['ward_primitive_mixed_ratio_max']):10.3e} "
            f"{str(bool(row['projection_eligible'])):>5s}"
        )
    lines += ["", "Allocation transitions", "----------------------"]
    for item in metrics["transitions"]:
        lines.append(
            f"{int(item['from_shifts'])}->{int(item['to_shifts'])} shifts: "
            f"relative Delta chi={float(item['relative_chi']):.3e}, "
            f"Delta D_T={float(item['relative_dbar']):.3e}, "
            f"Delta raw-long={float(item['relative_raw_longitudinal']):.3e}"
        )
    lines += [
        "",
        "Screening status",
        "----------------",
        f"relative_chi_span = {metrics['relative_chi_span']:.3e}",
        f"relative_dbar_span = {metrics['relative_dbar_span']:.3e}",
        f"physical_span_max = {metrics['physical_span_max']:.3e}",
        f"equal_budget_agreement = {metrics['equal_budget_agreement']}",
        f"more_shifts_reduce_transition = {metrics['more_shifts_reduce_transition']}",
        f"more_shifts_reduce_raw_longitudinal = {metrics['more_shifts_reduce_raw_longitudinal']}",
        f"allocation_preference = {metrics['allocation_preference']}",
        f"ward_and_inverse_ok = {metrics['ward_and_inverse_ok']}",
        "production_reference_established = False",
        f"agreement_tolerance = {args.agreement_tolerance:.3e}",
        "",
        "This equal-budget comparison only decides how to allocate future global-periodic "
        "quadrature effort.  It does not establish an n=0 production reference.",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocations",
        nargs="+",
        default=["4:240", "8:170", "16:120"],
        metavar="SHIFTS:NK",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-total-points", type=int, default=750_000)
    parser.add_argument("--agreement-tolerance", type=float, default=5e-3)
    parser.add_argument("--qx", type=float, default=0.03)
    parser.add_argument("--qy", type=float, default=0.02)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--raw-longitudinal-ceiling", type=float, default=1e-3)
    parser.add_argument("--longitudinal-tolerance", type=float, default=1e-7)
    parser.add_argument("--mixing-tolerance", type=float, default=1e-7)
    parser.add_argument("--reality-tolerance", type=float, default=1e-9)
    parser.add_argument("--passivity-tolerance", type=float, default=1e-10)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        args.allocations = parse_allocations(args.allocations)
    except ValueError as exc:
        parser.error(str(exc))
    requested = sum(shifts * nk * nk for shifts, nk in args.allocations)
    if args.workers <= 0 or args.max_total_points <= 0:
        parser.error("--workers and --max-total-points must be positive")
    if requested > args.max_total_points:
        parser.error(
            f"requested total points {requested} exceed --max-total-points={args.max_total_points}"
        )
    if not np.isfinite(args.agreement_tolerance) or args.agreement_tolerance <= 0.0:
        parser.error("--agreement-tolerance must be positive and finite")
    if float(np.hypot(args.qx, args.qy)) == 0.0:
        parser.error("q must be nonzero")
    return args


def main() -> None:
    args = _parse_args()
    common = {
        "qx": args.qx,
        "qy": args.qy,
        "temperature_K": args.temperature_K,
        "delta0_eV": args.delta0_eV,
        "eta_eV": args.eta_eV,
        "ward_tolerance": args.ward_tolerance,
        "ward_absolute_tolerance": args.ward_absolute_tolerance,
        "condition_max": args.condition_max,
        "raw_longitudinal_ceiling": args.raw_longitudinal_ceiling,
        "longitudinal_tolerance": args.longitudinal_tolerance,
        "mixing_tolerance": args.mixing_tolerance,
        "reality_tolerance": args.reality_tolerance,
        "passivity_tolerance": args.passivity_tolerance,
        "separation_nm": args.separation_nm,
    }
    tasks = [
        {**common, "num_shifts": shifts, "nk": nk}
        for shifts, nk in args.allocations
    ]

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in tasks:
            row = run_budget_task(task)
            rows.append(row)
            print(
                f"completed shifts={row['num_shifts']} nk={row['base_nk']} "
                f"in {row['total_wall_seconds']:.3f} s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = [executor.submit(run_budget_task, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    f"completed shifts={row['num_shifts']} nk={row['base_nk']} "
                    f"in {row['total_wall_seconds']:.3f} s",
                    flush=True,
                )
    rows.sort(key=lambda row: int(row["num_shifts"]))
    metrics = allocation_metrics(rows, agreement_tolerance=args.agreement_tolerance)
    wall_seconds = time.perf_counter() - started

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name(output.stem + ".summary.txt")
    _write_csv(output, rows)
    summary = _summary(args, rows, metrics, wall_seconds)
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "contract": (
            "equal-total-point nested C4/antithetic shift allocations; complete periodic "
            "lattice per shift; primitive/RHS merge before one Schur; screening only"
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rows": rows,
        "metrics": metrics,
        "wall_seconds": wall_seconds,
        "files": {"csv": str(output), "summary": str(summary_path)},
    }
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + summary)
    print(f"CSV: {output}\nSummary: {summary_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()
