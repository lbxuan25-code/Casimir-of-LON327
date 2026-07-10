"""Diagnose where two Ward-compatible periodic shift rules disagree in the BZ."""

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

from validation.lib.dwave_shift_batch import ShiftBatchConfig, postprocess_merged
from validation.lib.dwave_shift_spatial import (
    SpatialDiagnosticConfig,
    block_mass_table,
    components_from_primitive_vector,
    concentration_area,
    estimate_dwave_nodes,
    evaluate_shift_spatial,
    periodic_node_distances,
    shift_rule,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_shift_spatial/raw/"
    "dwave_q003_002_base56_gauss2_vs_halton4.json"
)


def _shift_key(shift: np.ndarray) -> tuple[float, float]:
    value = np.asarray(shift, dtype=float)
    return round(float(value[0]), 15), round(float(value[1]), 15)


def _evaluate_portable(config: SpatialDiagnosticConfig, shift: np.ndarray) -> dict[str, Any]:
    result = evaluate_shift_spatial(config, shift, keep_workspace=False)
    return {
        "shift": np.asarray(result["shift"], dtype=float),
        "vectors": np.asarray(result["vectors"], dtype=complex),
    }


def _physical_config(args) -> ShiftBatchConfig:
    return ShiftBatchConfig(
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


def _postprocess_vector(vector, workspace, physical_config):
    components, rhs = components_from_primitive_vector(vector, workspace)
    return postprocess_merged(components, rhs, physical_config)


def _rule_vectors(shifts, weights, cache):
    values = [cache[_shift_key(shift)] for shift in shifts]
    stacked = np.stack(values, axis=0)
    return np.tensordot(np.asarray(weights, dtype=float), stacked, axes=(0, 0))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _relative(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference):
        return float("nan")
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-30)


def _summary_text(
    args,
    rule_a_result,
    rule_b_result,
    concentration_rows,
    node_rows,
    prefix_rows,
    node_count: int,
    wall_seconds: float,
) -> str:
    lines = [
        "d-wave periodic-shift spatial localization diagnostic",
        "=" * 58,
        f"q = ({args.qx:.8g}, {args.qy:.8g}), base_nk = {args.base_nk}",
        f"rule A = {args.rule_a}; rule B = {args.rule_b}",
        f"estimated d-wave nodes = {node_count}",
        f"wall time = {wall_seconds:.3f} s",
        "",
        "Rule totals",
        "-----------",
    ]
    for name, result in (("A", rule_a_result), ("B", rule_b_result)):
        lines.append(
            f"{name}: chi_bar={result['chi_bar']:.10g}, Dbar_T={result['dbar_t']:.10g}, "
            f"Ward_prim={result['ward_primitive_mixed_ratio_max']:.3e}, "
            f"Ward_eff={result['ward_effective_mixed_ratio_max']:.3e}, "
            f"raw_long={result['raw_longitudinal']:.3e}"
        )
    lines.extend(
        [
            f"relative |Delta chi| / |chi_B| = {_relative(rule_a_result['chi_bar'], rule_b_result['chi_bar']):.3e}",
            f"relative |Delta D_T| / |D_T_B| = {_relative(rule_a_result['dbar_t'], rule_b_result['dbar_t']):.3e}",
            "",
            "Minimum BZ area containing absolute shift-difference mass",
            "---------------------------------------------------------",
            "block               area50       area80       area90",
        ]
    )
    for row in concentration_rows:
        lines.append(
            f"{row['block']:<18s} {row['area_50']:11.4f} {row['area_80']:12.4f} {row['area_90']:12.4f}"
        )
    lines.extend(
        [
            "",
            "Difference mass inside nodal neighborhoods",
            "--------------------------------------------",
            "radius/cell-step    area_frac   k_ss   k_seta  k_etas  k_etaeta  ward_rhs",
        ]
    )
    for row in node_rows:
        lines.append(
            f"{row['radius_in_cell_steps']:16.1f} {row['area_fraction']:10.4f} "
            f"{row['k_ss_mass_fraction']:6.3f} {row['k_seta_mass_fraction']:8.3f} "
            f"{row['k_etas_mass_fraction']:7.3f} {row['k_etaeta_mass_fraction']:9.3f} "
            f"{row['ward_rhs_mass_fraction']:9.3f}"
        )
    lines.extend(
        [
            "",
            "Replacing rule-A cells by rule-B cells in sensitivity-rank order",
            "----------------------------------------------------------------",
            "area_frac   chi_bar      Dbar_T       chi_resid_B   D_resid_B    Ward_prim",
        ]
    )
    for row in prefix_rows:
        lines.append(
            f"{row['area_fraction']:9.4f} {row['chi_bar']:12.7f} {row['dbar_t']:12.7f} "
            f"{row['relative_chi_residual_to_rule_b']:13.3e} "
            f"{row['relative_dbar_residual_to_rule_b']:12.3e} "
            f"{row['ward_primitive_mixed_ratio_max']:11.3e}"
        )
    area80 = max(float(row["area_80"]) for row in concentration_rows)
    node2 = next((row for row in node_rows if row["radius_in_cell_steps"] == 2.0), None)
    node_capture = (
        min(
            float(node2[f"{name}_mass_fraction"])
            for name in ("k_ss", "k_seta", "k_etas", "k_etaeta", "ward_rhs")
        )
        if node2 is not None
        else float("nan")
    )
    lines.extend(["", "Preliminary localization verdict", "--------------------------------"])
    if area80 <= 0.20 and np.isfinite(node_capture) and node_capture >= 0.60:
        lines.append(
            "Shift sensitivity is strongly localized by this diagnostic; a symmetry-complete "
            "node-patch subtraction method is worth prototyping."
        )
    elif area80 <= 0.35:
        lines.append(
            "Shift sensitivity is moderately localized. Patch methods may help, but the node "
            "neighborhood and Ward-carrying correction must be tested explicitly."
        )
    else:
        lines.append(
            "Shift sensitivity is spatially broad for at least one primitive block. A small "
            "node-only patch is unlikely to provide a reliable production correction."
        )
    lines.append(
        "This verdict is diagnostic only and does not make either rule Casimir-eligible."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-nk", type=int, default=56)
    parser.add_argument("--rule-a", choices=["midpoint", "gauss2", "halton4"], default="gauss2")
    parser.add_argument("--rule-b", choices=["midpoint", "gauss2", "halton4"], default="halton4")
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

    if int(args.base_nk) <= 0:
        raise ValueError("base_nk must be positive")
    if args.rule_a == args.rule_b:
        raise ValueError("rule-a and rule-b must differ")

    started = time.perf_counter()
    config = SpatialDiagnosticConfig(
        base_nk=args.base_nk,
        qx=args.qx,
        qy=args.qy,
        temperature_K=args.temperature_K,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta_eV,
    )
    shifts_a, weights_a = shift_rule(args.rule_a)
    shifts_b, weights_b = shift_rule(args.rule_b)
    all_shifts = []
    seen = set()
    for shift in np.concatenate([shifts_a, shifts_b], axis=0):
        key = _shift_key(shift)
        if key not in seen:
            all_shifts.append(np.asarray(shift, dtype=float))
            seen.add(key)

    first = evaluate_shift_spatial(config, all_shifts[0], keep_workspace=True)
    workspace = first["workspace"]
    cache = {_shift_key(first["shift"]): np.asarray(first["vectors"], dtype=complex)}
    remaining = all_shifts[1:]
    if int(args.workers) <= 1:
        for index, shift in enumerate(remaining, start=2):
            result = _evaluate_portable(config, shift)
            cache[_shift_key(result["shift"])] = result["vectors"]
            print(f"completed grid {index}/{len(all_shifts)}")
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {executor.submit(_evaluate_portable, config, shift): shift for shift in remaining}
            completed = 1
            for future in as_completed(futures):
                result = future.result()
                cache[_shift_key(result["shift"])] = result["vectors"]
                completed += 1
                print(f"completed grid {completed}/{len(all_shifts)}")

    rule_a_cells = _rule_vectors(shifts_a, weights_a, cache)
    rule_b_cells = _rule_vectors(shifts_b, weights_b, cache)
    delta_cells = rule_b_cells - rule_a_cells
    total_a = np.sum(rule_a_cells, axis=0)
    total_b = np.sum(rule_b_cells, axis=0)
    physical_config = _physical_config(args)
    result_a = _postprocess_vector(total_a, workspace, physical_config)
    result_b = _postprocess_vector(total_b, workspace, physical_config)

    masses, score = block_mass_table(delta_cells)
    order = np.argsort(score)[::-1]
    concentration_rows = [
        {
            "block": name,
            "area_50": concentration_area(values, 0.50),
            "area_80": concentration_area(values, 0.80),
            "area_90": concentration_area(values, 0.90),
        }
        for name, values in masses.items()
    ]

    step = 2.0 * np.pi / float(args.base_nk)
    centers_1d = -np.pi + (np.arange(int(args.base_nk), dtype=float) + 0.5) * step
    gx, gy = np.meshgrid(centers_1d, centers_1d, indexing="ij")
    centers = np.column_stack([gx.ravel(), gy.ravel()])
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    nodes = estimate_dwave_nodes(model.spec, fermi_level_eV=0.0)
    node_distance = periodic_node_distances(centers, nodes)
    node_rows = []
    for multiple in (1.0, 2.0, 4.0, 8.0):
        mask = node_distance <= multiple * step
        row = {
            "radius_in_cell_steps": multiple,
            "radius_model": multiple * step,
            "area_fraction": float(np.mean(mask)),
        }
        for name, values in masses.items():
            total = float(np.sum(values))
            row[f"{name}_mass_fraction"] = (
                float(np.sum(values[mask]) / total) if total > 0.0 else float("nan")
            )
        node_rows.append(row)

    prefix_rows = []
    fractions = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
    running = np.zeros_like(total_a)
    previous_count = 0
    for fraction in fractions:
        count = min(len(order), int(np.ceil(fraction * len(order))))
        if count > previous_count:
            running += np.sum(delta_cells[order[previous_count:count]], axis=0)
        mixed = _postprocess_vector(total_a + running, workspace, physical_config)
        prefix_rows.append(
            {
                "area_fraction": float(count / len(order)),
                "num_cells_replaced": count,
                "chi_bar": mixed["chi_bar"],
                "dbar_t": mixed["dbar_t"],
                "raw_longitudinal": mixed["raw_longitudinal"],
                "ward_primitive_mixed_ratio_max": mixed["ward_primitive_mixed_ratio_max"],
                "ward_effective_mixed_ratio_max": mixed["ward_effective_mixed_ratio_max"],
                "relative_chi_residual_to_rule_b": _relative(mixed["chi_bar"], result_b["chi_bar"]),
                "relative_dbar_residual_to_rule_b": _relative(mixed["dbar_t"], result_b["dbar_t"]),
            }
        )
        previous_count = count

    inverse_rank = np.empty_like(order)
    inverse_rank[order] = np.arange(len(order))
    cell_rows = []
    for index, center in enumerate(centers):
        row = {
            "cell_index": index,
            "ix": index // int(args.base_nk),
            "iy": index % int(args.base_nk),
            "kx_center": float(center[0]),
            "ky_center": float(center[1]),
            "node_distance": float(node_distance[index]),
            "sensitivity_score": float(score[index]),
            "sensitivity_rank": int(inverse_rank[index] + 1),
        }
        for name, values in masses.items():
            row[f"{name}_difference_mass"] = float(values[index])
        cell_rows.append(row)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    cell_path = output.with_name(output.stem + ".cells.csv")
    prefix_path = output.with_name(output.stem + ".prefix.csv")
    summary_path = output.with_name(output.stem + ".summary.txt")
    _write_csv(cell_path, cell_rows)
    _write_csv(prefix_path, prefix_rows)
    wall_seconds = time.perf_counter() - started
    summary = _summary_text(
        args,
        result_a,
        result_b,
        concentration_rows,
        node_rows,
        prefix_rows,
        len(nodes),
        wall_seconds,
    )
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "contract": (
            "weighted complete periodic shift rules; common base-cell pointwise primitive "
            "difference; no local quadrature replacement; primitive reconstruction before one Schur"
        ),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "rule_a": {"shifts": shifts_a.tolist(), "weights": weights_a.tolist(), "result": result_a},
        "rule_b": {"shifts": shifts_b.tolist(), "weights": weights_b.tolist(), "result": result_b},
        "estimated_nodes": nodes.tolist(),
        "concentration": concentration_rows,
        "node_neighborhoods": node_rows,
        "prefix_replacement": prefix_rows,
        "wall_seconds": wall_seconds,
        "files": {
            "cells_csv": str(cell_path),
            "prefix_csv": str(prefix_path),
            "summary_txt": str(summary_path),
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSpatial diagnostic completed.")
    print(f"Rule A: chi={result_a['chi_bar']:.7f}, D_T={result_a['dbar_t']:.7f}, Ward={result_a['ward_primitive_mixed_ratio_max']:.3e}")
    print(f"Rule B: chi={result_b['chi_bar']:.7f}, D_T={result_b['dbar_t']:.7f}, Ward={result_b['ward_primitive_mixed_ratio_max']:.3e}")
    print(f"Summary: {summary_path}")
    print(f"Cells:   {cell_path}")
    print(f"JSON:    {output}")


if __name__ == "__main__":
    main()
