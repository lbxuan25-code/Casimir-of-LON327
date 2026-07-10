"""Classify the spectral geometry of exact-static d-wave shift sensitivity."""

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
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation.lib.dwave_shift_batch import ShiftBatchConfig, postprocess_merged
from validation.lib.dwave_shift_spatial import (
    SpatialDiagnosticConfig,
    block_mass_table,
    components_from_primitive_vector,
    estimate_dwave_nodes,
    evaluate_shift_spatial,
    periodic_node_distances,
    shift_rule,
)
from validation.lib.dwave_shift_spectrum import (
    aggregate_rule_spectrum,
    combined_spectral_indicators,
    pointwise_spectral_indicators,
    spearman_correlation_rows,
    spectral_score_fields,
    top_fraction_rows,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_shift_spectrum/raw/"
    "dwave_q003_002_base56_gauss2_vs_halton4.json"
)


def _shift_key(shift: Sequence[float]) -> tuple[float, float]:
    value = np.asarray(shift, dtype=float)
    return round(float(value[0]), 15), round(float(value[1]), 15)


def _evaluate_portable(config: SpatialDiagnosticConfig, shift: np.ndarray) -> dict[str, Any]:
    result = evaluate_shift_spatial(config, shift, keep_workspace=True)
    return {
        "shift": np.asarray(result["shift"], dtype=float),
        "vectors": np.asarray(result["vectors"], dtype=complex),
        "spectrum": pointwise_spectral_indicators(result["workspace"]),
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
    stacked = np.stack([cache[_shift_key(shift)] for shift in shifts], axis=0)
    return np.tensordot(np.asarray(weights, dtype=float), stacked, axes=(0, 0))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference):
        return float("nan")
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-30)


def _top_correlation_rows(
    correlation_rows: list[dict[str, Any]], block: str, count: int = 3
) -> list[dict[str, Any]]:
    values = [
        row
        for row in correlation_rows
        if row["block"] == block and np.isfinite(float(row["abs_spearman_rho"]))
    ]
    values.sort(key=lambda row: float(row["abs_spearman_rho"]), reverse=True)
    return values[:count]


def _top_fraction_row(
    rows: list[dict[str, Any]], block: str, target: float = 0.05
) -> dict[str, Any]:
    candidates = [row for row in rows if row["block"] == block]
    if not candidates:
        raise ValueError(f"missing top-fraction rows for {block}")
    return min(candidates, key=lambda row: abs(float(row["top_area_fraction"]) - target))


def _geometry_verdict(top_rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    blocks = ("k_ss", "k_seta", "k_etas", "k_etaeta", "ward_rhs")
    rows = [_top_fraction_row(top_rows, block, 0.05) for block in blocks]
    node_capture = float(np.mean([row["node_within_2_cells_fraction"] for row in rows]))
    transition_capture = float(
        np.mean([row["transition_gap_le_5T_fraction"] for row in rows])
    )
    low_pair_capture = float(np.mean([row["pair_energy_le_4T_fraction"] for row in rows]))
    kubo_capture = float(
        np.mean([row["kubo_above_global_p90_fraction"] for row in rows])
    )
    strip_votes = sum(
        max(float(row["largest_component_span_x"]), float(row["largest_component_span_y"]))
        >= 0.40
        for row in rows
    )
    fragmented_votes = sum(int(row["num_components"]) >= 4 for row in rows)

    details = [
        f"mean top-5% node-within-2-cells fraction = {node_capture:.3f}",
        f"mean top-5% transition-gap<=5T fraction = {transition_capture:.3f}",
        f"mean top-5% pair-energy<=4T fraction = {low_pair_capture:.3f}",
        f"mean top-5% global-Kubo-p90 fraction = {kubo_capture:.3f}",
        f"strip-span votes among five blocks = {strip_votes}/5",
        f"fragmented-component votes among five blocks = {fragmented_votes}/5",
    ]
    if node_capture >= 0.60 and low_pair_capture >= 0.60:
        verdict = (
            "The most sensitive cells are predominantly conventional-node/low-energy localized. "
            "A symmetry-complete node-patch subtraction is the leading candidate."
        )
    elif transition_capture >= 0.60 or kubo_capture >= 0.50:
        if strip_votes >= 3:
            verdict = (
                "Sensitivity is dominated by finite-q near-degenerate/high-Kubo structures with "
                "extended periodic geometry. A symmetry-complete transition-strip diagnostic is "
                "favored over circular node patches."
            )
        else:
            verdict = (
                "Sensitivity is dominated by finite-q near-degenerate/high-Kubo pockets rather "
                "than conventional nodes. A symmetry-complete transition-region correction is "
                "favored, but its geometry is not yet demonstrably strip-like."
            )
    else:
        verdict = (
            "No single node or transition indicator explains the sensitive cells across all "
            "primitive blocks. The local-correction route remains mixed and should not yet be "
            "promoted to a solver."
        )
    return verdict, details


def _summary_text(
    args,
    result_a: Mapping[str, Any],
    result_b: Mapping[str, Any],
    correlation_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    verdict: str,
    verdict_details: list[str],
    wall_seconds: float,
) -> str:
    lines = [
        "d-wave shift-sensitivity spectrum correlation diagnostic",
        "=" * 62,
        f"q = ({args.qx:.8g}, {args.qy:.8g}), base_nk = {args.base_nk}",
        f"rule A = {args.rule_a}; rule B = {args.rule_b}",
        f"T = {args.temperature_K:.8g} K; wall time = {wall_seconds:.3f} s",
        "",
        "Rule totals",
        "-----------",
        (
            f"A: chi_bar={result_a['chi_bar']:.10g}, Dbar_T={result_a['dbar_t']:.10g}, "
            f"Ward_prim={result_a['ward_primitive_mixed_ratio_max']:.3e}, "
            f"raw_long={result_a['raw_longitudinal']:.3e}"
        ),
        (
            f"B: chi_bar={result_b['chi_bar']:.10g}, Dbar_T={result_b['dbar_t']:.10g}, "
            f"Ward_prim={result_b['ward_primitive_mixed_ratio_max']:.3e}, "
            f"raw_long={result_b['raw_longitudinal']:.3e}"
        ),
        f"relative Delta chi = {_relative(result_a['chi_bar'], result_b['chi_bar']):.3e}",
        f"relative Delta D_T = {_relative(result_a['dbar_t'], result_b['dbar_t']):.3e}",
        "",
        "Strongest Spearman rank correlations",
        "-------------------------------------",
    ]
    blocks = ("k_ss", "k_seta", "k_etas", "k_etaeta", "ward_rhs")
    for block in blocks:
        entries = _top_correlation_rows(correlation_rows, block)
        formatted = ", ".join(
            f"{row['indicator']}={float(row['spearman_rho']):+.3f}" for row in entries
        )
        lines.append(f"{block:<10s}: {formatted}")

    lines.extend(
        [
            "",
            "Top-5% sensitive-cell classification",
            "--------------------------------------",
            (
                "block      mass_cap node2  shift<=5T trans<=5T pair<=4T KuboP90 "
                "components largest span_x span_y"
            ),
        ]
    )
    for block in blocks:
        row = _top_fraction_row(top_rows, block, 0.05)
        lines.append(
            f"{block:<10s} {float(row['difference_mass_captured']):7.3f} "
            f"{float(row['node_within_2_cells_fraction']):5.3f} "
            f"{float(row['shifted_energy_le_5T_fraction']):9.3f} "
            f"{float(row['transition_gap_le_5T_fraction']):9.3f} "
            f"{float(row['pair_energy_le_4T_fraction']):8.3f} "
            f"{float(row['kubo_above_global_p90_fraction']):7.3f} "
            f"{int(row['num_components']):10d} "
            f"{float(row['largest_component_fraction']):7.3f} "
            f"{float(row['largest_component_span_x']):6.3f} "
            f"{float(row['largest_component_span_y']):6.3f}"
        )

    lines.extend(["", "Geometry verdict", "----------------", verdict])
    lines.extend(f"- {value}" for value in verdict_details)
    lines.extend(
        [
            "",
            "This is a diagnostic classification only. It neither restores Ward for a local "
            "replacement nor makes either shift rule Casimir-eligible.",
        ]
    )
    return "\n".join(lines) + "\n"


def _plot_sensitivity(
    masses: Mapping[str, np.ndarray], score: np.ndarray, base_nk: int, path: Path
) -> None:
    fields = [
        ("k_ss", masses["k_ss"]),
        ("k_seta", masses["k_seta"]),
        ("k_etas", masses["k_etas"]),
        ("k_etaeta", masses["k_etaeta"]),
        ("ward_rhs", masses["ward_rhs"]),
        ("combined score", score),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, (title, values) in zip(axes.ravel(), fields, strict=True):
        image = axis.imshow(
            np.log10(np.asarray(values, dtype=float).reshape(base_nk, base_nk) + 1e-300),
            origin="lower",
            extent=(-np.pi, np.pi, -np.pi, np.pi),
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("kx")
        axis.set_ylabel("ky")
        figure.colorbar(image, ax=axis, shrink=0.8, label="log10 mass")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_spectrum(
    indicators: Mapping[str, np.ndarray],
    node_distance: np.ndarray,
    base_nk: int,
    path: Path,
) -> None:
    fields = [
        ("node distance", node_distance, False),
        ("midpoint min |E|", indicators["midpoint_min_abs_eV"], True),
        ("shifted min |E|", indicators["shifted_min_abs_eV"], True),
        ("transition min gap", indicators["transition_min_gap_eV"], True),
        ("pair min energy", indicators["pair_min_energy_eV"], True),
        ("max |Kubo factor|", indicators["max_abs_kubo_factor_eV_inv"], False),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, (title, values, invert) in zip(axes.ravel(), fields, strict=True):
        array = np.asarray(values, dtype=float)
        plotted = -np.log10(array + 1e-300) if invert else np.log10(array + 1e-300)
        image = axis.imshow(
            plotted.reshape(base_nk, base_nk),
            origin="lower",
            extent=(-np.pi, np.pi, -np.pi, np.pi),
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("kx")
        axis.set_ylabel("ky")
        label = "-log10 value" if invert else "log10 value"
        figure.colorbar(image, ax=axis, shrink=0.8, label=label)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-nk", type=int, default=56)
    parser.add_argument("--rule-a", choices=["midpoint", "gauss2", "halton4"], default="gauss2")
    parser.add_argument("--rule-b", choices=["midpoint", "gauss2", "halton4"], default="halton4")
    parser.add_argument("--top-fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10])
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
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if int(args.base_nk) <= 0:
        raise ValueError("base_nk must be positive")
    if args.rule_a == args.rule_b:
        raise ValueError("rule-a and rule-b must differ")
    if any(not 0.0 < float(value) <= 1.0 for value in args.top_fractions):
        raise ValueError("top-fractions must lie in (0, 1]")

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
    all_shifts: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()
    for shift in np.concatenate([shifts_a, shifts_b], axis=0):
        key = _shift_key(shift)
        if key not in seen:
            all_shifts.append(np.asarray(shift, dtype=float))
            seen.add(key)

    first = evaluate_shift_spatial(config, all_shifts[0], keep_workspace=True)
    workspace = first["workspace"]
    vector_cache = {_shift_key(first["shift"]): np.asarray(first["vectors"], dtype=complex)}
    spectrum_cache = {
        _shift_key(first["shift"]): pointwise_spectral_indicators(workspace)
    }
    remaining = all_shifts[1:]
    if int(args.workers) <= 1:
        for index, shift in enumerate(remaining, start=2):
            result = _evaluate_portable(config, shift)
            vector_cache[_shift_key(result["shift"])] = result["vectors"]
            spectrum_cache[_shift_key(result["shift"])] = result["spectrum"]
            print(f"completed grid {index}/{len(all_shifts)}")
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(_evaluate_portable, config, shift): shift for shift in remaining
            }
            completed = 1
            for future in as_completed(futures):
                result = future.result()
                vector_cache[_shift_key(result["shift"])] = result["vectors"]
                spectrum_cache[_shift_key(result["shift"])] = result["spectrum"]
                completed += 1
                print(f"completed grid {completed}/{len(all_shifts)}")

    rule_a_cells = _rule_vectors(shifts_a, weights_a, vector_cache)
    rule_b_cells = _rule_vectors(shifts_b, weights_b, vector_cache)
    delta_cells = rule_b_cells - rule_a_cells
    total_a = np.sum(rule_a_cells, axis=0)
    total_b = np.sum(rule_b_cells, axis=0)
    physical_config = _physical_config(args)
    result_a = _postprocess_vector(total_a, workspace, physical_config)
    result_b = _postprocess_vector(total_b, workspace, physical_config)
    masses, sensitivity_score = block_mass_table(delta_cells)

    step = 2.0 * np.pi / float(args.base_nk)
    centers_1d = -np.pi + (np.arange(int(args.base_nk), dtype=float) + 0.5) * step
    gx, gy = np.meshgrid(centers_1d, centers_1d, indexing="ij")
    centers = np.column_stack([gx.ravel(), gy.ravel()])
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    nodes = estimate_dwave_nodes(model.spec, fermi_level_eV=0.0)
    node_distance = periodic_node_distances(centers, nodes)

    spectrum_a = aggregate_rule_spectrum(
        shifts_a, weights_a, spectrum_cache, key_function=_shift_key
    )
    spectrum_b = aggregate_rule_spectrum(
        shifts_b, weights_b, spectrum_cache, key_function=_shift_key
    )
    indicators = combined_spectral_indicators(spectrum_a, spectrum_b)
    score_fields = spectral_score_fields(
        indicators,
        node_distance,
        cell_step=step,
        energy_floor_eV=max(args.eta_eV, 1e-14),
    )
    correlation_rows = spearman_correlation_rows(masses, score_fields)
    top_rows = top_fraction_rows(
        masses,
        indicators,
        node_distance,
        base_nk=args.base_nk,
        temperature_eV=float(workspace.material.config.temperature_eV),
        fractions=tuple(args.top_fractions),
    )
    verdict, verdict_details = _geometry_verdict(top_rows)

    rank_by_block: dict[str, np.ndarray] = {}
    for block, values in masses.items():
        order = np.argsort(values)[::-1]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        rank_by_block[block] = inverse + 1

    cell_rows: list[dict[str, Any]] = []
    for index, center in enumerate(centers):
        row: dict[str, Any] = {
            "cell_index": index,
            "ix": index // int(args.base_nk),
            "iy": index % int(args.base_nk),
            "kx_center": float(center[0]),
            "ky_center": float(center[1]),
            "node_distance": float(node_distance[index]),
            "combined_sensitivity_score": float(sensitivity_score[index]),
        }
        for name, values in masses.items():
            row[f"{name}_difference_mass"] = float(values[index])
            row[f"{name}_rank"] = int(rank_by_block[name][index])
        for name, values in indicators.items():
            row[name] = float(np.asarray(values)[index])
        cell_rows.append(row)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.stem
    cells_path = output.with_name(stem + ".cells.csv")
    correlation_path = output.with_name(stem + ".correlations.csv")
    top_path = output.with_name(stem + ".top.csv")
    summary_path = output.with_name(stem + ".summary.txt")
    sensitivity_plot = output.with_name(stem + ".sensitivity.png")
    spectrum_plot = output.with_name(stem + ".spectrum.png")
    _write_csv(cells_path, cell_rows)
    _write_csv(correlation_path, correlation_rows)
    _write_csv(top_path, top_rows)
    if not args.no_plots:
        _plot_sensitivity(masses, sensitivity_score, args.base_nk, sensitivity_plot)
        _plot_spectrum(indicators, node_distance, args.base_nk, spectrum_plot)

    wall_seconds = time.perf_counter() - started
    summary = _summary_text(
        args,
        result_a,
        result_b,
        correlation_rows,
        top_rows,
        verdict,
        verdict_details,
        wall_seconds,
    )
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "contract": (
            "two Ward-compatible complete-periodic shift rules; pointwise primitive "
            "difference on common base cells; exact-static spectra from the same shift "
            "workspaces; diagnostic classification only"
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "rule_a": {"shifts": shifts_a.tolist(), "weights": weights_a.tolist(), "result": result_a},
        "rule_b": {"shifts": shifts_b.tolist(), "weights": weights_b.tolist(), "result": result_b},
        "estimated_nodes": nodes.tolist(),
        "correlations": correlation_rows,
        "top_fraction_classification": top_rows,
        "verdict": verdict,
        "verdict_details": verdict_details,
        "wall_seconds": wall_seconds,
        "files": {
            "summary_txt": str(summary_path),
            "cells_csv": str(cells_path),
            "correlations_csv": str(correlation_path),
            "top_csv": str(top_path),
            "sensitivity_png": None if args.no_plots else str(sensitivity_plot),
            "spectrum_png": None if args.no_plots else str(spectrum_plot),
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSpectrum diagnostic completed.")
    print(
        f"Rule A: chi={result_a['chi_bar']:.7f}, D_T={result_a['dbar_t']:.7f}, "
        f"Ward={result_a['ward_primitive_mixed_ratio_max']:.3e}"
    )
    print(
        f"Rule B: chi={result_b['chi_bar']:.7f}, D_T={result_b['dbar_t']:.7f}, "
        f"Ward={result_b['ward_primitive_mixed_ratio_max']:.3e}"
    )
    print(f"Summary:      {summary_path}")
    print(f"Correlations: {correlation_path}")
    print(f"Top cells:    {top_path}")
    if not args.no_plots:
        print(f"Heatmaps:     {sensitivity_plot}, {spectrum_plot}")
    print(f"JSON:         {output}")


if __name__ == "__main__":
    main()
