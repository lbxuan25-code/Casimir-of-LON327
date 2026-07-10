"""High-nk complete-periodic reference scan for the exact-static d-wave response.

Every ``nk`` uses the same fixed shift of one complete periodic tensor lattice.
No local mask, nonuniform refinement, or averaging over different rule families is
performed.  The runner reports consecutive-grid drift and a fail-closed family of
fixed-power ``1/nk`` extrapolations for ``chi_bar`` and ``Dbar_T``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from validation.lib.dwave_global_extrapolation import (
    relative_difference,
    static_power_law_fits,
    summarize_fit_ensemble,
)
from validation.lib.dwave_shift_batch import (
    ShiftBatchConfig,
    evaluate_one_shift,
    postprocess_merged,
)


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_global_reference/raw/"
    "dwave_q003_002_midpoint_reference.csv"
)


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _config(task: dict[str, Any]) -> ShiftBatchConfig:
    return ShiftBatchConfig(
        base_nk=int(task["nk"]),
        qx=float(task["qx"]),
        qy=float(task["qy"]),
        temperature_K=float(task["temperature_K"]),
        delta0_eV=float(task["delta0_eV"]),
        eta_eV=float(task["eta_eV"]),
        ward_tolerance=float(task["ward_tolerance"]),
        ward_absolute_tolerance=float(task["ward_absolute_tolerance"]),
        condition_max=float(task["condition_max"]),
        raw_longitudinal_ceiling=float(task["raw_longitudinal_ceiling"]),
        longitudinal_tolerance=float(task["longitudinal_tolerance"]),
        mixing_tolerance=float(task["mixing_tolerance"]),
        reality_tolerance=float(task["reality_tolerance"]),
        passivity_tolerance=float(task["passivity_tolerance"]),
        separation_nm=float(task["separation_nm"]),
    )


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    config = _config(task)
    shift = np.asarray(task["shift"], dtype=float)
    result = evaluate_one_shift(config, 0, shift)
    processed = postprocess_merged(result["components"], result["rhs"], config)
    return {
        "nk": int(config.base_nk),
        "num_k_points": int(config.base_nk) ** 2,
        "shift_x": float(shift[0]),
        "shift_y": float(shift[1]),
        "qx": float(config.qx),
        "qy": float(config.qy),
        "temperature_K": float(config.temperature_K),
        "delta0_eV": float(config.delta0_eV),
        "eta_eV": float(config.eta_eV),
        **processed,
        "total_wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_mb": _peak_rss_mb(),
        "pid": os.getpid(),
    }


def _annotate_drift(rows: list[dict[str, Any]]) -> None:
    previous: dict[str, Any] | None = None
    for row in rows:
        if previous is None:
            row["relative_chi_to_previous"] = float("nan")
            row["relative_dbar_to_previous"] = float("nan")
            row["relative_raw_longitudinal_to_previous"] = float("nan")
        else:
            row["relative_chi_to_previous"] = relative_difference(
                row["chi_bar"], previous["chi_bar"]
            )
            row["relative_dbar_to_previous"] = relative_difference(
                row["dbar_t"], previous["dbar_t"]
            )
            row["relative_raw_longitudinal_to_previous"] = relative_difference(
                row["raw_longitudinal"], previous["raw_longitudinal"]
            )
        previous = row


def _fit_rows(
    rows: list[dict[str, Any]], powers: tuple[int, ...], tails: tuple[int, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nks = [int(row["nk"]) for row in rows]
    all_fits: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for field in ("chi_bar", "dbar_t"):
        fits = static_power_law_fits(
            nks,
            [float(row[field]) for row in rows],
            powers=powers,
            tail_sizes=tails,
        )
        for fit in fits:
            fit["field"] = field
            all_fits.append(fit)
        summary = summarize_fit_ensemble(fits)
        summaries[field] = {
            "estimate": summary.estimate,
            "minimum": summary.minimum,
            "maximum": summary.maximum,
            "relative_spread": summary.relative_spread,
            "best_model": summary.best_model,
            "best_tail_points": summary.best_tail_points,
            "best_normalized_rms": summary.best_normalized_rms,
            "num_accepted_models": summary.num_accepted_models,
        }
    return all_fits, summaries


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


def _summary_text(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    fit_summaries: dict[str, Any],
    numerical_converged: bool,
    casimir_eligible: bool,
    wall_seconds: float,
) -> str:
    lines = [
        "d-wave complete-periodic exact-static reference scan",
        "=" * 57,
        f"q = ({args.qx:.8g}, {args.qy:.8g}); shift = ({args.shift[0]:.8g}, {args.shift[1]:.8g})",
        f"T = {args.temperature_K:.8g} K; wall time = {wall_seconds:.3f} s",
        "",
        "Grid sequence",
        "-------------",
        " nk       Nk    chi_bar    Dbar_T   dchi-prev   dD-prev   raw-long   Ward-prim   proj",
    ]
    for row in rows:
        lines.append(
            f"{int(row['nk']):4d} {int(row['num_k_points']):8d} "
            f"{float(row['chi_bar']):10.7f} {float(row['dbar_t']):10.7f} "
            f"{float(row['relative_chi_to_previous']):11.3e} "
            f"{float(row['relative_dbar_to_previous']):10.3e} "
            f"{float(row['raw_longitudinal']):10.3e} "
            f"{float(row['ward_primitive_mixed_ratio_max']):11.3e} "
            f"{str(bool(row['projection_eligible'])):>5s}"
        )
    lines.extend(["", "Multi-model tail extrapolation", "------------------------------"])
    for field in ("chi_bar", "dbar_t"):
        value = fit_summaries[field]
        lines.append(
            f"{field}: estimate={value['estimate']:.10g}, interval=[{value['minimum']:.10g}, "
            f"{value['maximum']:.10g}], rel-spread={value['relative_spread']:.3e}, "
            f"best={value['best_model']} tail={value['best_tail_points']} "
            f"rms={value['best_normalized_rms']:.3e}, accepted={value['num_accepted_models']}"
        )
    final = rows[-1]
    lines.extend(
        [
            "",
            "Fail-closed status",
            "------------------",
            f"numerical_reference_converged = {numerical_converged}",
            f"final_projection_eligible = {bool(final['projection_eligible'])}",
            f"valid_for_casimir_input = {casimir_eligible}",
            f"drift_tolerance = {args.drift_tolerance:.3e}",
            f"fit_spread_tolerance = {args.fit_spread_tolerance:.3e}",
            "",
            "The extrapolated values remain diagnostic unless numerical_reference_converged is true. "
            "Casimir eligibility additionally requires the final static projection gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nks", type=int, nargs="+", required=True)
    parser.add_argument("--shift", type=float, nargs=2, default=[0.5, 0.5])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-total-points", type=int, default=2_000_000)
    parser.add_argument("--fit-powers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--fit-tail-sizes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--drift-tolerance", type=float, default=1e-3)
    parser.add_argument("--fit-spread-tolerance", type=float, default=2e-3)
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
    nks = sorted(set(int(value) for value in args.nks))
    if len(nks) < 3 or any(value <= 0 for value in nks):
        parser.error("--nks requires at least three distinct positive values")
    args.nks = nks
    shift = np.asarray(args.shift, dtype=float)
    if shift.shape != (2,) or not np.isfinite(shift).all() or np.any(shift < 0.0) or np.any(shift >= 1.0):
        parser.error("--shift coordinates must lie in [0,1)")
    if args.workers <= 0 or args.max_total_points <= 0:
        parser.error("--workers and --max-total-points must be positive")
    requested = sum(value * value for value in nks)
    if requested > args.max_total_points:
        parser.error(
            f"requested total points {requested} exceed --max-total-points={args.max_total_points}"
        )
    if args.drift_tolerance <= 0.0 or args.fit_spread_tolerance <= 0.0:
        parser.error("convergence tolerances must be positive")
    if float(np.hypot(args.qx, args.qy)) == 0.0:
        parser.error("q must be nonzero")
    return args


def main() -> None:
    args = _parse_args()
    common = {
        "shift": tuple(float(value) for value in args.shift),
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
    tasks = [{**common, "nk": nk} for nk in args.nks]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in tasks:
            row = _run_task(task)
            rows.append(row)
            print(f"completed nk={row['nk']} in {row['total_wall_seconds']:.3f} s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = {executor.submit(_run_task, task): task["nk"] for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    f"completed nk={row['nk']} in {row['total_wall_seconds']:.3f} s "
                    f"(RSS {row['peak_rss_mb']:.1f} MiB)",
                    flush=True,
                )
    rows.sort(key=lambda row: int(row["nk"]))
    _annotate_drift(rows)
    fits, fit_summaries = _fit_rows(
        rows,
        tuple(int(value) for value in args.fit_powers),
        tuple(int(value) for value in args.fit_tail_sizes),
    )
    final = rows[-1]
    final_drift = max(
        float(final["relative_chi_to_previous"]),
        float(final["relative_dbar_to_previous"]),
    )
    tail = rows[-min(3, len(rows)) :]
    ward_ok = all(
        bool(row["ward_passed"])
        and str(row["schur_inverse_method"]) == "inv"
        and float(row["ward_primitive_mixed_ratio_max"]) < 1.0
        and float(row["ward_effective_mixed_ratio_max"]) < 1.0
        for row in tail
    )
    spread = max(
        float(fit_summaries["chi_bar"]["relative_spread"]),
        float(fit_summaries["dbar_t"]["relative_spread"]),
    )
    numerical_converged = bool(
        ward_ok
        and final_drift <= args.drift_tolerance
        and spread <= args.fit_spread_tolerance
    )
    casimir_eligible = bool(numerical_converged and final["projection_eligible"])
    wall_seconds = time.perf_counter() - started

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fit_path = output.with_name(output.stem + ".fits.csv")
    summary_path = output.with_name(output.stem + ".summary.txt")
    _write_csv(output, rows)
    _write_csv(fit_path, fits)
    summary = _summary_text(
        args, rows, fit_summaries, numerical_converged, casimir_eligible, wall_seconds
    )
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "contract": (
            "one fixed complete-periodic tensor-lattice shift for every nk; exact xi=0 branch; "
            "primitive Ward validation; multi-model tail extrapolation; fail closed"
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rows": rows,
        "fits": fits,
        "fit_summaries": fit_summaries,
        "reference_estimate": {
            "chi_bar": float(fit_summaries["chi_bar"]["estimate"]),
            "dbar_t": float(fit_summaries["dbar_t"]["estimate"]),
        },
        "reference_status": {
            "numerical_reference_converged": numerical_converged,
            "final_projection_eligible": bool(final["projection_eligible"]),
            "valid_for_casimir_input": casimir_eligible,
            "final_step_relative_drift_max": final_drift,
            "fit_relative_spread_max": spread,
            "tail_ward_and_inverse_ok": ward_ok,
        },
        "wall_seconds": wall_seconds,
        "files": {
            "rows_csv": str(output),
            "fits_csv": str(fit_path),
            "summary_txt": str(summary_path),
        },
    }
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + summary)
    print(f"CSV:     {output}")
    print(f"Fits:    {fit_path}")
    print(f"Summary: {summary_path}")
    print(f"JSON:    {json_path}")


if __name__ == "__main__":
    main()
