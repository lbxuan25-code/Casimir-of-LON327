"""Fixed-rule complete-periodic d-wave exact-static reference scan."""
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

from validation.lib.dwave_shift_ensemble_reference import (
    annotate_drift,
    fit_primary,
    reference_status,
    run_ensemble_task,
)
from validation.lib.dwave_shift_spatial import shift_rule

DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_shift_ensemble_reference/raw/"
    "dwave_q003_002_gauss2_reference.csv"
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


def _summary(args, primary, secondary, fits, status, wall_seconds: float) -> str:
    lines = [
        "d-wave fixed shift-ensemble exact-static reference scan",
        "=" * 60,
        f"q = ({args.qx:.8g}, {args.qy:.8g}); primary = {args.primary_rule}; "
        f"cross rule = {args.cross_rule}",
        f"T = {args.temperature_K:.8g} K; wall time = {wall_seconds:.3f} s",
        "",
        "Primary-rule grid sequence",
        "--------------------------",
        " nk      Ntotal   chi_bar    Dbar_T   dchi-prev  dD-prev  raw-long  Ward-prim  proj",
    ]
    for row in primary:
        lines.append(
            f"{int(row['nk']):4d} {int(row['num_quadrature_points']):10d} "
            f"{float(row['chi_bar']):10.7f} {float(row['dbar_t']):10.7f} "
            f"{float(row['relative_chi_to_previous']):10.3e} "
            f"{float(row['relative_dbar_to_previous']):9.3e} "
            f"{float(row['raw_longitudinal']):9.3e} "
            f"{float(row['ward_primitive_mixed_ratio_max']):10.3e} "
            f"{str(bool(row['projection_eligible'])):>5s}"
        )
    lines += ["", "Primary-rule multi-model extrapolation", "--------------------------------------"]
    for field in ("chi_bar", "dbar_t"):
        value = fits[field]
        lines.append(
            f"{field}: estimate={value['estimate']:.10g}, "
            f"interval=[{value['minimum']:.10g}, {value['maximum']:.10g}], "
            f"rel-spread={value['relative_spread']:.3e}, "
            f"best={value['best_model']} tail={value['best_tail_points']} "
            f"rms={value['best_normalized_rms']:.3e}, accepted={value['num_accepted_models']}"
        )
    final = primary[-1]
    lines += [
        "", "Largest-nk cross-rule check", "---------------------------",
        f"nk = {int(final['nk'])}",
        f"{args.primary_rule}: chi_bar={final['chi_bar']:.10g}, Dbar_T={final['dbar_t']:.10g}, "
        f"raw_long={final['raw_longitudinal']:.3e}, Ward={final['ward_primitive_mixed_ratio_max']:.3e}",
        f"{args.cross_rule}: chi_bar={secondary['chi_bar']:.10g}, Dbar_T={secondary['dbar_t']:.10g}, "
        f"raw_long={secondary['raw_longitudinal']:.3e}, Ward={secondary['ward_primitive_mixed_ratio_max']:.3e}",
        f"relative Delta chi = {status['relative_chi_cross_rule']:.3e}",
        f"relative Delta D_T = {status['relative_dbar_cross_rule']:.3e}",
        "", "Fail-closed status", "------------------",
        f"ensemble_screening_promising = {status['ensemble_screening_promising']}",
        f"numerical_reference_converged = {status['numerical_reference_converged']}",
        f"primary_final_projection_eligible = {status['primary_final_projection_eligible']}",
        f"secondary_projection_eligible = {status['secondary_projection_eligible']}",
        f"valid_for_casimir_input = {status['valid_for_casimir_input']}",
        f"final_step_relative_drift_max = {status['final_step_relative_drift_max']:.3e}",
        f"fit_relative_spread_max = {status['fit_relative_spread_max']:.3e}",
        f"cross_rule_relative_difference_max = {status['cross_rule_relative_difference_max']:.3e}",
        "",
        "A promising screening result only justifies increasing the number of complete-periodic shifts. "
        "It is not a converged reference.",
    ]
    return "\n".join(lines) + "\n"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nks", type=int, nargs="+", required=True)
    parser.add_argument("--primary-rule", choices=("gauss2", "halton4"), default="gauss2")
    parser.add_argument("--cross-rule", choices=("gauss2", "halton4"), default="halton4")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-total-points", type=int, default=1_000_000)
    parser.add_argument("--fit-powers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--fit-tail-sizes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--screening-drift-tolerance", type=float, default=5e-3)
    parser.add_argument("--screening-cross-rule-tolerance", type=float, default=1e-2)
    parser.add_argument("--drift-tolerance", type=float, default=1e-3)
    parser.add_argument("--fit-spread-tolerance", type=float, default=2e-3)
    parser.add_argument("--cross-rule-tolerance", type=float, default=2e-3)
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
    args.nks = sorted(set(args.nks))
    if len(args.nks) < 3 or any(value <= 0 for value in args.nks):
        parser.error("--nks requires at least three distinct positive values")
    if args.primary_rule == args.cross_rule:
        parser.error("--primary-rule and --cross-rule must differ")
    if args.workers <= 0 or args.max_total_points <= 0:
        parser.error("--workers and --max-total-points must be positive")
    primary_count = len(shift_rule(args.primary_rule)[0])
    cross_count = len(shift_rule(args.cross_rule)[0])
    requested = primary_count * sum(nk * nk for nk in args.nks)
    requested += cross_count * args.nks[-1] ** 2
    if requested > args.max_total_points:
        parser.error(f"requested total points {requested} exceed --max-total-points={args.max_total_points}")
    tolerances = [
        args.screening_drift_tolerance, args.screening_cross_rule_tolerance,
        args.drift_tolerance, args.fit_spread_tolerance, args.cross_rule_tolerance,
    ]
    if any(not np.isfinite(value) or value <= 0 for value in tolerances):
        parser.error("all convergence tolerances must be positive finite values")
    if np.hypot(args.qx, args.qy) == 0:
        parser.error("q must be nonzero")
    return args


def main() -> None:
    args = _parse_args()
    common = {
        "qx": args.qx, "qy": args.qy, "temperature_K": args.temperature_K,
        "delta0_eV": args.delta0_eV, "eta_eV": args.eta_eV,
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
    tasks = [{**common, "nk": nk, "rule": args.primary_rule} for nk in args.nks]
    tasks.append({**common, "nk": args.nks[-1], "rule": args.cross_rule})
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in tasks:
            rows.append(run_ensemble_task(task))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = [executor.submit(run_ensemble_task, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
                row = rows[-1]
                print(f"completed rule={row['rule']} nk={row['nk']} in {row['total_wall_seconds']:.3f} s", flush=True)
    primary = sorted([r for r in rows if r["rule"] == args.primary_rule], key=lambda r: r["nk"])
    secondary_rows = [r for r in rows if r["rule"] == args.cross_rule]
    if len(primary) != len(args.nks) or len(secondary_rows) != 1:
        raise RuntimeError("shift-ensemble task accounting failed")
    secondary = secondary_rows[0]
    annotate_drift(primary)
    fit_rows, fit_summaries = fit_primary(primary, args.fit_powers, args.fit_tail_sizes)
    status = reference_status(
        primary, secondary, fit_summaries,
        screening_drift=args.screening_drift_tolerance,
        screening_cross=args.screening_cross_rule_tolerance,
        drift=args.drift_tolerance, fit_spread=args.fit_spread_tolerance,
        cross_rule=args.cross_rule_tolerance,
    )
    wall_seconds = time.perf_counter() - started
    output = args.output
    fit_path = output.with_name(output.stem + ".fits.csv")
    cross_path = output.with_name(output.stem + ".cross_rule.csv")
    summary_path = output.with_name(output.stem + ".summary.txt")
    _write_csv(output, primary)
    _write_csv(fit_path, fit_rows)
    _write_csv(cross_path, [secondary])
    summary = _summary(args, primary, secondary, fit_summaries, status, wall_seconds)
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv), "platform": platform.platform(), "python": sys.version,
        "contract": "same complete-periodic four-shift rule at every nk; primitive/RHS merge before one Schur; largest-nk cross-rule check; fail closed",
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "primary_rows": primary, "cross_rule_row": secondary,
        "fits": fit_rows, "fit_summaries": fit_summaries,
        "reference_estimate": {
            "chi_bar": float(fit_summaries["chi_bar"]["estimate"]),
            "dbar_t": float(fit_summaries["dbar_t"]["estimate"]),
        },
        "reference_status": status, "wall_seconds": wall_seconds,
        "files": {"primary_csv": str(output), "fits_csv": str(fit_path),
                  "cross_rule_csv": str(cross_path), "summary_txt": str(summary_path)},
    }
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + summary)
    print(f"Primary: {output}\nFits: {fit_path}\nCross: {cross_path}\nSummary: {summary_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()
