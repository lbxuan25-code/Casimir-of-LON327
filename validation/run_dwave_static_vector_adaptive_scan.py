"""Error-driven shared-cell cubature for the exact-static two-band d-wave response."""

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

from lno327.workflows.dwave_vector_adaptive_cubature import (
    DWaveVectorAdaptiveOptions,
    initial_cubature_cells,
    subdivide_cubature_cell,
    validate_vector_adaptive_options,
)
from validation.lib.dwave_vector_adaptive import (
    VectorAdaptiveConfig,
    aggregate_cubature_cells,
    choose_refinement_indices,
    evaluate_cubature_cell,
    evaluate_cubature_cell_portable,
    restore_portable_cubature_cell_result,
)


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_vector_adaptive/raw/"
    "dwave_static_vector_adaptive_scan.csv"
)


def _relative(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference):
        return float("nan")
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-30)


def _evaluate_many(
    config: VectorAdaptiveConfig,
    cells,
    workers: int,
) -> list[dict[str, Any]]:
    if not cells:
        return []
    if int(workers) <= 1:
        return [evaluate_cubature_cell(config, cell) for cell in cells]
    results: list[dict[str, Any] | None] = [None] * len(cells)
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {
            executor.submit(evaluate_cubature_cell_portable, config, cell): index
            for index, cell in enumerate(cells)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = restore_portable_cubature_cell_result(future.result())
    if any(item is None for item in results):
        raise RuntimeError("not all cubature cells completed")
    return list(results)


def _write(rows: list[dict[str, Any]], output: Path, metadata: dict[str, Any]) -> None:
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
            "shared-cell low/high tensor Gauss rules; complete primitive-vector error; "
            "accepted cell primitives summed before one Schur; Ward is a stop gate, "
            "never an artificial projection"
        ),
        "run": metadata,
        "rows": rows,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print(rows: list[dict[str, Any]]) -> None:
    header = (
        " iter cells maxL eval-pts final-pts err-cons err-sign score-max "
        "Ward-prim raw-long chi_bar Dbar_T dchi dD stop"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['iteration']:5d} "
            f"{row['num_cells']:5d} "
            f"{row['maximum_cell_level']:4d} "
            f"{row['cumulative_evaluation_points']:8d} "
            f"{row['accepted_high_rule_points']:9d} "
            f"{row['conservative_error_ratio_max']:8.2e} "
            f"{row['signed_error_ratio_max']:8.2e} "
            f"{row['maximum_cell_score']:9.2e} "
            f"{row['ward_primitive_mixed_ratio_max']:9.2e} "
            f"{row['raw_longitudinal']:8.2e} "
            f"{row['chi_bar']:7.5f} "
            f"{row['dbar_t']:7.5f} "
            f"{row['relative_chi_to_previous']:6.2e} "
            f"{row['relative_dbar_to_previous']:6.2e} "
            f"{row['stop_reason']}"
        )
        if row["projection_error"]:
            print(f"    projection_error: {row['projection_error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-grid", type=int, default=6)
    parser.add_argument("--low-order", type=int, default=2)
    parser.add_argument("--high-order", type=int, default=3)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-9)
    parser.add_argument("--max-level", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--refine-fraction", type=float, default=0.2)
    parser.add_argument("--min-refine-cells", type=int, default=6)
    parser.add_argument("--max-cells", type=int, default=4000)
    parser.add_argument("--max-evaluation-points", type=int, default=60000)
    parser.add_argument("--workers", type=int, default=2)
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

    options = DWaveVectorAdaptiveOptions(
        coarse_grid=args.coarse_grid,
        low_order=args.low_order,
        high_order=args.high_order,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
        max_level=args.max_level,
        max_iterations=args.max_iterations,
        refine_fraction=args.refine_fraction,
        min_refine_cells=args.min_refine_cells,
        max_cells=args.max_cells,
        max_evaluation_points=args.max_evaluation_points,
    )
    validate_vector_adaptive_options(options)
    config = VectorAdaptiveConfig(
        low_order=args.low_order,
        high_order=args.high_order,
        qx=args.qx,
        qy=args.qy,
        temperature_K=args.temperature_K,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta_eV,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
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
    cells = initial_cubature_cells(args.coarse_grid)
    first = evaluate_cubature_cell(config, cells[0])
    template_workspace = first.pop("workspace")
    remaining = _evaluate_many(config, cells[1:], args.workers)
    results = [first, *remaining]
    cumulative_points = sum(int(item["evaluation_points"]) for item in results)
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    stop_reason = "continue"

    for iteration in range(int(args.max_iterations) + 1):
        physical, scores, errors = aggregate_cubature_cells(
            results, template_workspace, config
        )
        row: dict[str, Any] = {
            "iteration": iteration,
            "num_cells": len(results),
            "maximum_cell_level": max(int(item["cell"].level) for item in results),
            "cumulative_evaluation_points": cumulative_points,
            "accepted_high_rule_points": len(results) * int(args.high_order) ** 2,
            "maximum_cell_score": float(np.max(scores)),
            "median_cell_score": float(np.median(scores)),
            "conservative_error_ratio_max": errors["conservative_error_ratio_max"],
            "signed_error_ratio_max": errors["signed_error_ratio_max"],
            "ward_error_ratio_conservative": errors["ward_error_ratio_conservative"],
            "global_high_vector_norm": errors["global_high_vector_norm"],
            **physical,
        }
        if previous is None:
            row["relative_chi_to_previous"] = float("nan")
            row["relative_dbar_to_previous"] = float("nan")
        else:
            row["relative_chi_to_previous"] = _relative(row["chi_bar"], previous["chi_bar"])
            row["relative_dbar_to_previous"] = _relative(row["dbar_t"], previous["dbar_t"])

        converged = bool(
            row["conservative_error_ratio_max"] <= 1.0 and row["ward_passed"]
        )
        if converged:
            stop_reason = "error_and_ward_pass"
        elif iteration >= int(args.max_iterations):
            stop_reason = "max_iterations"
        else:
            remaining_budget = int(args.max_evaluation_points) - cumulative_points
            points_per_child = int(args.low_order) ** 2 + int(args.high_order) ** 2
            selected = choose_refinement_indices(
                results,
                scores,
                refine_fraction=args.refine_fraction,
                min_refine_cells=args.min_refine_cells,
                max_level=args.max_level,
                max_cells=args.max_cells,
                remaining_evaluation_points=remaining_budget,
                points_per_child=points_per_child,
            )
            if not selected:
                stop_reason = "budget_or_level_exhausted"
            else:
                selected_set = set(selected)
                children = [
                    child
                    for index in selected
                    for child in subdivide_cubature_cell(results[index]["cell"])
                ]
                child_results = _evaluate_many(config, children, args.workers)
                cumulative_points += sum(
                    int(item["evaluation_points"]) for item in child_results
                )
                results = [
                    item for index, item in enumerate(results) if index not in selected_set
                ] + child_results
                stop_reason = "continue"

        row["stop_reason"] = stop_reason
        rows.append(row)
        previous = row
        print(
            f"iteration={iteration} cells={row['num_cells']} "
            f"eval_points={cumulative_points} stop={stop_reason}"
        )
        if stop_reason != "continue":
            break

    metadata = {
        "options": vars(args),
        "wall_seconds": time.perf_counter() - started,
        "template_workspace_source": "first high-rule cell; metadata only",
        "per_cell_low_high_primitives_computed_once": True,
        "parent_child_double_counting": False,
        "ward_used_as_stop_gate_not_projection": True,
    }
    _write(rows, args.output, metadata)
    _print(rows)
    print(f"Sweep wall time: {time.perf_counter() - started:.4f} s")
    print(f"CSV:  {args.output}")
    print(f"JSON: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
