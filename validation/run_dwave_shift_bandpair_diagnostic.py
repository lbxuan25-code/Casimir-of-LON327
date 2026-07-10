"""Identify the BdG band pairs and shifted-FS strips driving d-wave shift sensitivity."""

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

from validation.lib.dwave_shift_bandpair import (
    aggregate_pair_classification,
    aggregate_rule_pair_strengths,
    bandpair_mass_summary,
    dominant_pair_fields,
    normal_shifted_fs_fields,
    pair_strength_contrast,
    pointwise_bandpair_data,
)
from validation.lib.dwave_shift_batch import ShiftBatchConfig, postprocess_merged
from validation.lib.dwave_shift_spatial import (
    SpatialDiagnosticConfig,
    block_mass_table,
    components_from_primitive_vector,
    evaluate_shift_spatial,
    shift_rule,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_shift_bandpair/raw/"
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
        "bandpair": pointwise_bandpair_data(result["workspace"]),
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _relative(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-30)


def _mass_verdict(rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    crossing = float(np.mean([row["sign_crossing_mass_fraction"] for row in rows]))
    same_normal = float(np.mean([row["same_normal_band_mass_fraction"] for row in rows]))
    same_bdg = float(np.mean([row["same_bdg_index_mass_fraction"] for row in rows]))
    center_strip = float(
        np.mean([row["center_shifted_fs_strip_mass_fraction"] for row in rows])
    )
    top_pair = float(np.mean([row["top_pair_mass_fraction"] for row in rows]))
    top5_crossing = float(np.mean([row["top_sign_crossing_fraction"] for row in rows]))
    top5_same_normal = float(
        np.mean([row["top_same_normal_band_fraction"] for row in rows])
    )
    unique_pairs = sorted({str(row["top_pair"]) for row in rows})
    details = [
        f"mean sign-crossing difference-mass fraction = {crossing:.3f}",
        f"mean same-normal-band difference-mass fraction = {same_normal:.3f}",
        f"mean same-sorted-BdG-index difference-mass fraction = {same_bdg:.3f}",
        f"mean center shifted-FS-strip difference-mass fraction = {center_strip:.3f}",
        f"mean leading-pair difference-mass fraction = {top_pair:.3f}",
        f"mean top-5% sign-crossing cell fraction = {top5_crossing:.3f}",
        f"mean top-5% same-normal-band cell fraction = {top5_same_normal:.3f}",
        f"leading pair labels across five blocks = {', '.join(unique_pairs)}",
    ]
    if crossing >= 0.60 and same_normal >= 0.60 and len(unique_pairs) <= 2:
        verdict = (
            "A small set of same-normal-band finite-q sign-crossing transitions explains most "
            "shift sensitivity. A symmetry-complete shifted-FS strip subtraction is the leading "
            "solver candidate, subject to an independent Ward-defect test of the full primitive correction."
        )
    elif crossing >= 0.40 or center_strip >= 0.50:
        verdict = (
            "Shifted-FS sign-crossing transitions explain a substantial but not dominant share of all "
            "primitive blocks. A strip correction may help, but multiple pair/source sectors must be retained."
        )
    else:
        verdict = (
            "No compact same-band shifted-FS pair class explains the primitive shift sensitivity. "
            "A specialized local strip solver is unlikely to be both simple and reliable."
        )
    return verdict, details


def _summary_text(
    args,
    result_a: Mapping[str, Any],
    result_b: Mapping[str, Any],
    rows: list[dict[str, Any]],
    verdict: str,
    details: list[str],
    wall_seconds: float,
) -> str:
    lines = [
        "d-wave band-pair and shifted-FS diagnostic",
        "=" * 54,
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
        "Band-pair / shifted-FS mass classification",
        "-------------------------------------------",
        (
            "block      pair pairMass signX sameNorm sameBdG centerFS top5Mass "
            "top5SignX top5Norm top5FS n00 n11 inter"
        ),
    ]
    for row in rows:
        lines.append(
            f"{row['block']:<10s} {row['top_pair']:>4s} "
            f"{float(row['top_pair_mass_fraction']):8.3f} "
            f"{float(row['sign_crossing_mass_fraction']):5.3f} "
            f"{float(row['same_normal_band_mass_fraction']):8.3f} "
            f"{float(row['same_bdg_index_mass_fraction']):7.3f} "
            f"{float(row['center_shifted_fs_strip_mass_fraction']):8.3f} "
            f"{float(row['top_area_mass_captured']):8.3f} "
            f"{float(row['top_sign_crossing_fraction']):9.3f} "
            f"{float(row['top_same_normal_band_fraction']):8.3f} "
            f"{float(row['top_center_shifted_fs_strip_fraction']):6.3f} "
            f"{float(row['normal_00_mass_fraction']):4.3f} "
            f"{float(row['normal_11_mass_fraction']):4.3f} "
            f"{float(row['normal_interband_mass_fraction']):5.3f}"
        )
    lines.extend(["", "Solver-direction verdict", "------------------------", verdict])
    lines.extend(f"- {detail}" for detail in details)
    lines.extend(
        [
            "",
            "Definitions:",
            "- pair is the sorted BdG index m->n with the largest rule contrast in vertex-weighted pair strength.",
            "- signX means E_m(k-q/2) E_n(k+q/2) < 0 in at least half of the combined rule weight.",
            "- sameNorm uses particle/hole projections onto the two normal-state bands.",
            "- centerFS is the same-normal-band sign-crossing strip evaluated at base-cell centers.",
            "",
            "This remains diagnostic only. It neither constructs a local correction nor grants Casimir eligibility.",
        ]
    )
    return "\n".join(lines) + "\n"


def _contour_if_zero(axis, x, y, values, *, linestyle: str, linewidth: float = 0.8):
    field = np.asarray(values, dtype=float)
    if float(np.min(field)) <= 0.0 <= float(np.max(field)):
        axis.contour(x, y, field, levels=[0.0], linestyles=linestyle, linewidths=linewidth)


def _plot_overlay(
    masses: Mapping[str, np.ndarray],
    normal_fs: Mapping[str, np.ndarray],
    base_nk: int,
    path: Path,
) -> None:
    n = int(base_nk)
    coordinates = -np.pi + (np.arange(n, dtype=float) + 0.5) * (2.0 * np.pi / n)
    minus = np.asarray(normal_fs["normal_minus_eV"], dtype=float).reshape(n, n, 2)
    plus = np.asarray(normal_fs["normal_plus_eV"], dtype=float).reshape(n, n, 2)
    fields = [(name, masses[name]) for name in ("k_ss", "k_seta", "k_etas", "k_etaeta", "ward_rhs")]
    combined = sum(
        np.asarray(values, dtype=float) / max(float(np.sum(values)), 1e-300)
        for _, values in fields
    ) / len(fields)
    fields.append(("combined", combined))
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, (title, values) in zip(axes.ravel(), fields, strict=True):
        image = axis.imshow(
            np.log10(np.asarray(values, dtype=float).reshape(n, n) + 1e-300),
            origin="lower",
            extent=(-np.pi, np.pi, -np.pi, np.pi),
            aspect="equal",
        )
        for band in range(2):
            _contour_if_zero(axis, coordinates, coordinates, minus[:, :, band], linestyle="--")
            _contour_if_zero(axis, coordinates, coordinates, plus[:, :, band], linestyle="-")
        axis.set_title(title)
        axis.set_xlabel("kx")
        axis.set_ylabel("ky")
        figure.colorbar(image, ax=axis, shrink=0.8, label="log10 difference mass")
    figure.suptitle("Dashed: normal FS at k-q/2; solid: normal FS at k+q/2")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_dominant_pairs(
    dominant: Mapping[str, Mapping[str, np.ndarray]],
    base_nk: int,
    path: Path,
) -> None:
    n = int(base_nk)
    blocks = ("k_ss", "k_seta", "k_etas", "k_etaeta", "equal_forward")
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, block in zip(axes.ravel()[:5], blocks, strict=True):
        fields = dominant[block]
        code = np.asarray(fields["m"], dtype=int) * 10 + np.asarray(fields["n"], dtype=int)
        image = axis.imshow(
            code.reshape(n, n), origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi), aspect="equal"
        )
        axis.set_title("ward equal-forward" if block == "equal_forward" else block)
        axis.set_xlabel("kx")
        axis.set_ylabel("ky")
        figure.colorbar(image, ax=axis, shrink=0.8, label="10*m+n")
    crossing = np.mean(
        [np.asarray(dominant[block]["sign_crossing_fraction"], dtype=float) for block in blocks],
        axis=0,
    )
    image = axes.ravel()[5].imshow(
        crossing.reshape(n, n), origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi), aspect="equal", vmin=0.0, vmax=1.0
    )
    axes.ravel()[5].set_title("mean dominant-pair sign crossing")
    axes.ravel()[5].set_xlabel("kx")
    axes.ravel()[5].set_ylabel("ky")
    figure.colorbar(image, ax=axes.ravel()[5], shrink=0.8, label="fraction")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-nk", type=int, default=56)
    parser.add_argument("--rule-a", choices=["midpoint", "gauss2", "halton4"], default="gauss2")
    parser.add_argument("--rule-b", choices=["midpoint", "gauss2", "halton4"], default="halton4")
    parser.add_argument("--top-area-fraction", type=float, default=0.05)
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
    if args.base_nk <= 0 or not 0.0 < args.top_area_fraction <= 1.0:
        raise ValueError("base-nk must be positive and top-area-fraction must lie in (0,1]")
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
    all_shifts: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()
    for shift in np.concatenate([shifts_a, shifts_b], axis=0):
        key = _shift_key(shift)
        if key not in seen:
            all_shifts.append(np.asarray(shift, dtype=float))
            seen.add(key)

    first_raw = evaluate_shift_spatial(config, all_shifts[0], keep_workspace=True)
    workspace = first_raw["workspace"]
    first = {
        "shift": np.asarray(first_raw["shift"], dtype=float),
        "vectors": np.asarray(first_raw["vectors"], dtype=complex),
        "bandpair": pointwise_bandpair_data(workspace),
    }
    vector_cache = {_shift_key(first["shift"]): first["vectors"]}
    pair_cache = {_shift_key(first["shift"]): first["bandpair"]}
    remaining = all_shifts[1:]
    if args.workers <= 1:
        for index, shift in enumerate(remaining, start=2):
            result = _evaluate_portable(config, shift)
            vector_cache[_shift_key(result["shift"])] = result["vectors"]
            pair_cache[_shift_key(result["shift"])] = result["bandpair"]
            print(f"completed grid {index}/{len(all_shifts)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_evaluate_portable, config, shift): shift for shift in remaining}
            completed = 1
            for future in as_completed(futures):
                result = future.result()
                vector_cache[_shift_key(result["shift"])] = result["vectors"]
                pair_cache[_shift_key(result["shift"])] = result["bandpair"]
                completed += 1
                print(f"completed grid {completed}/{len(all_shifts)}")

    cells_a = _rule_vectors(shifts_a, weights_a, vector_cache)
    cells_b = _rule_vectors(shifts_b, weights_b, vector_cache)
    total_a, total_b = np.sum(cells_a, axis=0), np.sum(cells_b, axis=0)
    physical = _physical_config(args)
    result_a = _postprocess_vector(total_a, workspace, physical)
    result_b = _postprocess_vector(total_b, workspace, physical)
    masses, _ = block_mass_table(cells_b - cells_a)

    pair_a = aggregate_rule_pair_strengths(
        shifts_a, weights_a, pair_cache, key_function=_shift_key
    )
    pair_b = aggregate_rule_pair_strengths(
        shifts_b, weights_b, pair_cache, key_function=_shift_key
    )
    contrast = pair_strength_contrast(pair_a, pair_b)
    classification = aggregate_pair_classification(
        shifts_a,
        weights_a,
        shifts_b,
        weights_b,
        pair_cache,
        key_function=_shift_key,
    )
    dominant = dominant_pair_fields(contrast, classification)

    step = 2.0 * np.pi / args.base_nk
    centers_1d = -np.pi + (np.arange(args.base_nk, dtype=float) + 0.5) * step
    gx, gy = np.meshgrid(centers_1d, centers_1d, indexing="ij")
    centers = np.column_stack([gx.ravel(), gy.ravel()])
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    normal_fs = normal_shifted_fs_fields(model.spec, centers, config.q)
    rows = bandpair_mass_summary(
        masses, dominant, normal_fs, top_area_fraction=args.top_area_fraction
    )
    verdict, details = _mass_verdict(rows)

    cell_rows: list[dict[str, Any]] = []
    for index, center in enumerate(centers):
        row: dict[str, Any] = {
            "cell_index": index,
            "ix": index // args.base_nk,
            "iy": index % args.base_nk,
            "kx_center": float(center[0]),
            "ky_center": float(center[1]),
            "center_same_normal_band_sign_crossing": bool(
                normal_fs["same_normal_band_sign_crossing"][index]
            ),
            "center_minimum_shifted_normal_abs_eV": float(
                normal_fs["minimum_shifted_normal_abs_eV"][index]
            ),
        }
        for block in ("k_ss", "k_seta", "k_etas", "k_etaeta", "ward_rhs"):
            pair_block = "equal_forward" if block == "ward_rhs" else block
            fields = dominant[pair_block]
            row[f"{block}_difference_mass"] = float(masses[block][index])
            row[f"{block}_dominant_m"] = int(fields["m"][index])
            row[f"{block}_dominant_n"] = int(fields["n"][index])
            for name in (
                "sign_crossing_fraction",
                "same_normal_band_fraction",
                "same_bdg_index_fraction",
                "particle_weight_minus",
                "particle_weight_plus",
                "normal_00_fraction",
                "normal_11_fraction",
                "normal_interband_fraction",
                "ph_pp_fraction",
                "ph_ph_fraction",
                "ph_hp_fraction",
                "ph_hh_fraction",
            ):
                row[f"{block}_{name}"] = float(fields[name][index])
        cell_rows.append(row)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.stem
    summary_path = output.with_name(stem + ".summary.txt")
    mass_path = output.with_name(stem + ".mass.csv")
    cells_path = output.with_name(stem + ".cells.csv")
    overlay_path = output.with_name(stem + ".fs_overlay.png")
    pair_plot_path = output.with_name(stem + ".dominant_pairs.png")
    _write_csv(mass_path, rows)
    _write_csv(cells_path, cell_rows)
    if not args.no_plots:
        _plot_overlay(masses, normal_fs, args.base_nk, overlay_path)
        _plot_dominant_pairs(dominant, args.base_nk, pair_plot_path)
    wall_seconds = time.perf_counter() - started
    summary = _summary_text(args, result_a, result_b, rows, verdict, details, wall_seconds)
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "contract": (
            "complete-periodic rule comparison; vertex-weighted sorted-BdG-pair strength contrast; "
            "normal-band and particle-hole classification; no local correction"
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "rule_a": {"shifts": shifts_a.tolist(), "weights": weights_a.tolist(), "result": result_a},
        "rule_b": {"shifts": shifts_b.tolist(), "weights": weights_b.tolist(), "result": result_b},
        "mass_summary": rows,
        "verdict": verdict,
        "verdict_details": details,
        "wall_seconds": wall_seconds,
        "files": {
            "summary_txt": str(summary_path),
            "mass_csv": str(mass_path),
            "cells_csv": str(cells_path),
            "fs_overlay_png": None if args.no_plots else str(overlay_path),
            "dominant_pairs_png": None if args.no_plots else str(pair_plot_path),
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nBand-pair diagnostic completed.")
    print(
        f"Rule A: chi={result_a['chi_bar']:.7f}, D_T={result_a['dbar_t']:.7f}, "
        f"Ward={result_a['ward_primitive_mixed_ratio_max']:.3e}"
    )
    print(
        f"Rule B: chi={result_b['chi_bar']:.7f}, D_T={result_b['dbar_t']:.7f}, "
        f"Ward={result_b['ward_primitive_mixed_ratio_max']:.3e}"
    )
    print(f"Summary: {summary_path}")
    print(f"Mass:    {mass_path}")
    if not args.no_plots:
        print(f"Plots:   {overlay_path}, {pair_plot_path}")
    print(f"JSON:    {output}")


if __name__ == "__main__":
    main()
