#!/usr/bin/env python3
"""Diagnose finite-q formula and vertex consistency."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.finite_q_response import (  # noqa: E402
    FiniteQFormulaConsistencyDiagnostic,
    compare_finite_q_formula_consistency,
)
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("normal", "spm", "dwave")
OUTPUT_ROOT = ROOT / "outputs" / "response" / "finite_q_formula_consistency"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "finite_q_formula_consistency"
SUMMARY_PATH = OUTPUT_ROOT / "finite_q_formula_consistency_summary.md"


def _empty_data() -> dict[str, np.ndarray]:
    return {
        "kind": np.array([], dtype="U16"),
        "matsubara_index": np.array([], dtype=int),
        "temperature_K": np.array([], dtype=float),
        "q_magnitude": np.array([], dtype=float),
        "q_phi": np.array([], dtype=float),
        "nk": np.array([], dtype=int),
        "delta0": np.array([], dtype=float),
        "eta": np.array([], dtype=float),
        "vertex_local_limit_abs_error": np.array([], dtype=float),
        "vertex_local_limit_relative_error": np.array([], dtype=float),
        "overlap_unitarity_error": np.array([], dtype=float),
        "overlap_diagonal_error": np.array([], dtype=float),
        "overlap_offdiag_norm": np.array([], dtype=float),
        "wrapped_fraction": np.array([], dtype=float),
        "possible_bz_wrapping_discontinuity": np.array([], dtype=bool),
        "min_abs_energy_diff": np.array([], dtype=float),
        "max_denominator_weight": np.array([], dtype=float),
        "near_degenerate_count": np.array([], dtype=int),
        "possible_denominator_instability": np.array([], dtype=bool),
        "error_to_local_sigma": np.array([], dtype=float),
        "error_to_K_para": np.array([], dtype=float),
        "error_to_K_total": np.array([], dtype=float),
        "error_to_K_total_over_omega": np.array([], dtype=float),
        "error_to_normal_kubo_sigma": np.array([], dtype=float),
        "best_match_component": np.array([], dtype="U48"),
        "small_q_relative_error": np.array([], dtype=float),
        "diagnostic_status": np.array([], dtype="U192"),
        "gauge_status": np.array([], dtype="U64"),
        "final_casimir_input": np.array([], dtype=bool),
        "not_final_Casimir_conclusion": np.array([], dtype=bool),
        "notes": np.array([], dtype=object),
    }


def _rows_from_results(results: list[FiniteQFormulaConsistencyDiagnostic]) -> list[dict[str, object]]:
    rows = []
    for result in results:
        rows.append(
            {
                "kind": result.kind,
                "matsubara_index": result.matsubara_index,
                "temperature_K": result.temperature_K,
                "q_magnitude": result.q_magnitude,
                "q_phi": result.q_phi,
                "nk": result.nk,
                "delta0": result.delta0,
                "eta": result.eta,
                "vertex_local_limit_abs_error": result.vertex_local_limit_abs_error,
                "vertex_local_limit_relative_error": result.vertex_local_limit_relative_error,
                "overlap_unitarity_error": result.overlap_unitarity_error,
                "overlap_diagonal_error": result.overlap_diagonal_error,
                "overlap_offdiag_norm": result.overlap_offdiag_norm,
                "wrapped_fraction": result.wrapped_fraction,
                "possible_bz_wrapping_discontinuity": result.possible_bz_wrapping_discontinuity,
                "min_abs_energy_diff": result.min_abs_energy_diff,
                "max_denominator_weight": result.max_denominator_weight,
                "near_degenerate_count": result.near_degenerate_count,
                "possible_denominator_instability": result.possible_denominator_instability,
                "error_to_local_sigma": result.component_errors["local_sigma"],
                "error_to_K_para": result.component_errors["local_K_para"],
                "error_to_K_total": result.component_errors["local_K_total"],
                "error_to_K_total_over_omega": result.component_errors["local_K_total_over_omega"],
                "error_to_normal_kubo_sigma": result.component_errors["normal_kubo_sigma"],
                "best_match_component": result.best_match_component,
                "small_q_relative_error": result.small_q_relative_error,
                "diagnostic_status": result.diagnostic_status,
                "gauge_status": result.gauge_status,
                "final_casimir_input": result.final_casimir_input,
                "not_final_Casimir_conclusion": result.not_final_Casimir_conclusion,
                "notes": result.notes,
            }
        )
    return rows


def _append_rows(data: dict[str, np.ndarray], rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    if not rows:
        return data
    new_data = _empty_data()
    for key in new_data:
        values = [row[key] for row in rows]
        if key == "notes":
            note_array = np.empty(len(values), dtype=object)
            note_array[:] = values
            new_data[key] = note_array
        else:
            new_data[key] = np.asarray(values, dtype=new_data[key].dtype)
    return {key: np.concatenate([data[key], new_data[key]]) for key in data}


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    return value


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    np.savez(npz_path, **data)
    fieldnames = list(_empty_data())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(data["kind"].size):
            writer.writerow({name: _csv_value(data[name][index]) for name in fieldnames})
    return npz_path, csv_path


def _nanmin_by_mask(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    return float(np.nanmin(selected)) if np.any(np.isfinite(selected)) else np.nan


def save_figures(data: dict[str, np.ndarray]) -> list[Path]:
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    figure_dir = OUTPUT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / "small_q_error_vs_q.png",
        figure_dir / "vertex_error_vs_q.png",
        figure_dir / "overlap_error_vs_q.png",
        figure_dir / "component_error_comparison.png",
    ]
    q_values = sorted(set(float(item) for item in data["q_magnitude"]))

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in KINDS:
        values = [_nanmin_by_mask(data["small_q_relative_error"], (data["kind"] == kind) & np.isclose(data["q_magnitude"], q)) for q in q_values]
        ax.plot(q_values, values, marker="o", label=kind)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("min small-q relative error")
    ax.set_title("small-q error vs q")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[0])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in KINDS:
        values = [_nanmin_by_mask(data["vertex_local_limit_relative_error"], (data["kind"] == kind) & np.isclose(data["q_magnitude"], q)) for q in q_values]
        ax.plot(q_values, values, marker="o", label=kind)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-16)
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("vertex relative error")
    ax.set_title("vertex local-limit consistency")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in KINDS:
        values = [_nanmin_by_mask(data["overlap_diagonal_error"], (data["kind"] == kind) & np.isclose(data["q_magnitude"], q)) for q in q_values]
        ax.plot(q_values, values, marker="o", label=kind)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("overlap diagonal error")
    ax.set_title("band overlap smoothness")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[2])
    plt.close(fig)

    components = [
        ("error_to_local_sigma", "local_sigma"),
        ("error_to_K_para", "K_para"),
        ("error_to_K_total", "K_total"),
        ("error_to_K_total_over_omega", "K_total/omega"),
        ("error_to_normal_kubo_sigma", "normal_kubo_sigma"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    values = []
    labels = []
    for field, label in components:
        value = float(np.nanmin(data[field])) if np.any(np.isfinite(data[field])) else np.nan
        values.append(value)
        labels.append(label)
    ax.bar(labels, values)
    ax.set_yscale("log")
    ax.set_ylabel("min relative error")
    ax.set_title("component error comparison")
    ax.tick_params(axis="x", rotation=30)
    style_publication_axis(ax)
    save_publication_figure(fig, paths[3])
    plt.close(fig)
    return paths


def _summary_lines(data: dict[str, np.ndarray], args: argparse.Namespace) -> list[str]:
    vertex_mismatch = bool(np.nanmax(data["vertex_local_limit_relative_error"]) > 1e-10)
    overlap_issue = bool(
        np.nanmax(data["overlap_diagonal_error"]) > 1e-2 or np.nanmax(data["overlap_offdiag_norm"]) > 1e-1
    )
    wrapping_issue = bool(np.any(data["possible_bz_wrapping_discontinuity"]))
    denominator_issue = bool(np.any(data["possible_denominator_instability"]))
    best_error = float(np.nanmin(data["small_q_relative_error"]))
    improved = bool(best_error < 1e-2)
    statuses = sorted(set(str(item) for item in data["diagnostic_status"]))
    continue_repair = bool(vertex_mismatch or overlap_issue or wrapping_issue or denominator_issue or not improved)
    return_to_a4 = bool(improved and not continue_repair)
    lines = [
        "# finite-q formula / vertex consistency 诊断摘要",
        "",
        "本轮目标是 formula / vertex consistency repair 诊断：检查 finite-q response 在 small-q 下",
        "不平滑的来源，而不是扩大扫描、接入 Lifshitz/Casimir 或输出 torque 结论。",
        "",
        "当前 finite-q response 仍不是 Ward 完备。",
        f"kinds={list(args.kinds)}",
        f"matsubara_list={list(args.matsubara_list)}",
        f"q_list={list(args.q_list)}",
        f"q_phi_list={list(args.q_phi_list)}",
        f"nk_list={list(args.nk_list)}",
        f"temperature={args.temperature}",
        f"delta0={args.delta0}",
        f"eta={args.eta}",
        "",
        f"vertex_mismatch_detected={vertex_mismatch}",
        f"max_vertex_relative_error={float(np.nanmax(data['vertex_local_limit_relative_error'])):.6g}",
        f"overlap_or_band_phase_issue_detected={overlap_issue}",
        f"max_overlap_diagonal_error={float(np.nanmax(data['overlap_diagonal_error'])):.6g}",
        f"max_overlap_offdiag_norm={float(np.nanmax(data['overlap_offdiag_norm'])):.6g}",
        f"bz_wrapping_issue_detected={wrapping_issue}",
        f"max_wrapped_fraction={float(np.nanmax(data['wrapped_fraction'])):.6g}",
        f"denominator_instability_detected={denominator_issue}",
        f"max_near_degenerate_count={int(np.nanmax(data['near_degenerate_count']))}",
        f"best_small_q_relative_error={best_error:.6g}",
        f"small_q_continuity_improved_candidate={improved}",
        f"recommend_return_to_finite_q_A4_anisotropy_diagnostic={return_to_a4}",
        f"recommend_continue_formula_repair={continue_repair}",
        "",
        "## 诊断状态",
    ]
    for status in statuses:
        lines.append(f"- {status}")
    lines.extend(
        [
            "",
            "## 限制",
            "- gauge_status=prototype_not_ward_verified",
            "- Ward identity / diamagnetic closure 未完成",
            "- n=0 model 未完成",
            "- final_casimir_input=False",
            "- not_final_Casimir_conclusion=True",
        ]
    )
    return lines


def write_summary(data: dict[str, np.ndarray], args: argparse.Namespace) -> Path:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(_summary_lines(data, args)) + "\n", encoding="utf-8")
    return SUMMARY_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--matsubara-list", nargs="+", type=int, default=[1])
    parser.add_argument("--q-list", nargs="+", type=float, default=[1e-4, 5e-4, 1e-3, 5e-3])
    parser.add_argument("--q-phi-list", nargs="+", type=float, default=[0.0, 0.7853981634])
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--nk-list", nargs="+", type=int, default=[6, 8])
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compare_finite_q_formula_consistency(
        args.kinds,
        args.matsubara_list,
        args.q_list,
        args.q_phi_list,
        args.nk_list,
        args.temperature,
        args.delta0,
        args.eta,
    )
    data = _append_rows(_empty_data(), _rows_from_results(rows))
    npz_path, csv_path = save_outputs(data, args.output_prefix)
    figure_paths = save_figures(data)
    summary_path = write_summary(data, args)
    print(f"npz_path = {npz_path}")
    print(f"csv_path = {csv_path}")
    print(f"summary_path = {summary_path}")
    print("figure_paths = " + ", ".join(str(path) for path in figure_paths))
    print("note = finite-q formula consistency diagnostic only; not a Casimir result.")


if __name__ == "__main__":
    main()
