"""Incremental exact-static d-wave scan over nested periodic shift batches."""

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

import numpy as np

from lno327.workflows.dwave_periodic_shift_ensemble import (
    DWavePeriodicShiftEnsembleOptions,
    build_dwave_periodic_shift_ensemble,
)
from validation.lib.dwave_shift_batch import (
    ShiftBatchConfig,
    evaluate_one_shift,
    evaluate_one_shift_portable,
    jackknife_orbit_errors,
    merge_prefix,
    restore_portable_shift_result,
)


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_shift_batch/raw/"
    "dwave_static_shift_batch_scan.csv"
)


def _relative(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference):
        return float("nan")
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-30)


def _write(rows, output: Path, metadata: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "integration_contract": (
            "cached complete periodic lattices; nested C4-antithetic shift prefixes; "
            "primitive average before one Schur; delete-one-orbit jackknife"
        ),
        "ensemble": metadata,
        "rows": rows,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print(rows) -> None:
    header = (
        " shifts       Nk  raw-long   chi_bar    Dbar_T  dchi-prev    dD-prev "
        "jk-chi-rel  jk-D-rel Ward-prim Ward-eff  proj"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['num_shifts']:7d} "
            f"{row['num_quadrature_points']:8d} "
            f"{row['raw_longitudinal']:10.3e} "
            f"{row['chi_bar']:9.5f} "
            f"{row['dbar_t']:9.5f} "
            f"{row['relative_chi_to_previous']:10.2e} "
            f"{row['relative_dbar_to_previous']:10.2e} "
            f"{row['jackknife_chi_bar_relative']:10.2e} "
            f"{row['jackknife_dbar_t_relative']:9.2e} "
            f"{row['ward_primitive_mixed_ratio_max']:9.2e} "
            f"{row['ward_effective_mixed_ratio_max']:8.2e} "
            f"{str(row['projection_eligible']):>5s}"
        )
        if row["projection_error"]:
            print(f"    projection_error: {row['projection_error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-nk", type=int, default=56)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--qx", type=float, default=0.03)
    parser.add_argument("--qy", type=float, default=0.02)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--max-quadrature-points", type=int, default=120_000)
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

    batches = sorted(set(int(value) for value in args.batch_sizes))
    if not batches or any(value <= 0 or value % 4 != 0 for value in batches):
        raise ValueError("batch sizes must be positive multiples of four")
    maximum = max(batches)
    q = np.asarray([args.qx, args.qy], dtype=float)
    shifts, ensemble = build_dwave_periodic_shift_ensemble(
        q,
        DWavePeriodicShiftEnsembleOptions(
            base_nk=args.base_nk,
            max_shifts=maximum,
            max_quadrature_points=args.max_quadrature_points,
        ),
    )
    config = ShiftBatchConfig(
        base_nk=args.base_nk,
        qx=args.qx,
        qy=args.qy,
        temperature_K=args.temperature_K,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta_eV,
        ward_tolerance=args.ward_tolerance,
        ward_absolute_tolerance=args.ward_absolute_tolerance,
        condition_max=args.condition_max,
        raw_longitudinal_ceiling=args.raw_longitudinal_ceiling,
        longitudinal_tolerance=args.longitudinal_tolerance,
        mixing_tolerance=args.mixing_tolerance,
        reality_tolerance=args.reality_tolerance,
        passivity_tolerance=args.passivity_tolerance,
        separation_nm=args.separation_nm,
    )

    started = time.perf_counter()
    first = evaluate_one_shift(config, 0, shifts[0])
    template_workspace = first.pop("workspace")
    results = [None] * maximum
    results[0] = first
    if int(args.workers) <= 1:
        for index in range(1, maximum):
            results[index] = evaluate_one_shift(config, index, shifts[index])
            print(f"completed shift {index + 1}/{maximum}")
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(
                    evaluate_one_shift_portable, config, index, shifts[index]
                ): index
                for index in range(1, maximum)
            }
            for future in as_completed(futures):
                portable = future.result()
                result = restore_portable_shift_result(portable)
                results[int(result["index"])] = result
                print(f"completed shift {int(result['index']) + 1}/{maximum}")
    if any(item is None for item in results):
        raise RuntimeError("not all shift evaluations completed")
    cached = list(results)

    rows = []
    previous = None
    for batch in batches:
        row = merge_prefix(cached, batch, template_workspace, config)
        row.update(
            {
                "base_nk": int(args.base_nk),
                "num_shifts": batch,
                "num_c4_orbits": batch // 4,
                "num_points_per_shift": int(args.base_nk) ** 2,
                "num_quadrature_points": int(args.base_nk) ** 2 * batch,
                "qx": float(args.qx),
                "qy": float(args.qy),
                "q_abs": float(np.linalg.norm(q)),
                "shift_prefix_json": json.dumps(shifts[:batch].tolist()),
            }
        )
        errors = jackknife_orbit_errors(cached, batch, template_workspace, config)
        row.update(errors)
        row["jackknife_chi_bar_relative"] = errors["jackknife_chi_bar_abs"] / max(
            abs(row["chi_bar"]), 1e-30
        )
        row["jackknife_dbar_t_relative"] = errors["jackknife_dbar_t_abs"] / max(
            abs(row["dbar_t"]), 1e-30
        )
        row["jackknife_reflection_relative"] = (
            errors["jackknife_reflection_norm_abs"] / max(abs(row["reflection_norm"]), 1e-30)
            if np.isfinite(errors["jackknife_reflection_norm_abs"])
            and np.isfinite(row["reflection_norm"])
            else float("nan")
        )
        row["jackknife_logdet_relative"] = (
            errors["jackknife_logdet_abs"] / max(abs(row["logdet"]), 1e-30)
            if np.isfinite(errors["jackknife_logdet_abs"])
            and np.isfinite(row["logdet"])
            else float("nan")
        )
        if previous is None:
            row["relative_chi_to_previous"] = float("nan")
            row["relative_dbar_to_previous"] = float("nan")
            row["relative_reflection_to_previous"] = float("nan")
            row["relative_logdet_to_previous"] = float("nan")
        else:
            row["relative_chi_to_previous"] = _relative(row["chi_bar"], previous["chi_bar"])
            row["relative_dbar_to_previous"] = _relative(row["dbar_t"], previous["dbar_t"])
            row["relative_reflection_to_previous"] = _relative(
                row["reflection_norm"], previous["reflection_norm"]
            )
            row["relative_logdet_to_previous"] = _relative(row["logdet"], previous["logdet"])
        rows.append(row)
        previous = row

    reference = rows[-1]
    for row in rows:
        row["reference_num_shifts"] = int(reference["num_shifts"])
        row["relative_chi_to_reference"] = _relative(row["chi_bar"], reference["chi_bar"])
        row["relative_dbar_to_reference"] = _relative(row["dbar_t"], reference["dbar_t"])

    ensemble.update(
        {
            "batch_sizes": batches,
            "workers": int(args.workers),
            "per_shift_components_computed_once": True,
            "worker_transfer": "pickle-safe numeric payloads; typed objects restored in parent",
            "prefix_recomputation": "primitive merge, one Schur, and postprocessing only",
            "wall_seconds": time.perf_counter() - started,
        }
    )
    _write(rows, args.output, ensemble)
    _print(rows)
    print(f"Sweep wall time: {time.perf_counter() - started:.4f} s")
    print(f"CSV:  {args.output}")
    print(f"JSON: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
