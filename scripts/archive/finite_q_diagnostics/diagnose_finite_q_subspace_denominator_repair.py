#!/usr/bin/env python3
"""Diagnose finite-q subspace and denominator stability."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.finite_q_response import (  # noqa: E402
    FiniteQSubspaceConsistencyDiagnostic,
    compare_subspace_and_eigenstate_overlap,
)
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("normal", "spm", "dwave")
DENOMINATOR_MODES = ("raw", "stable")
OUTPUT_ROOT = ROOT / "outputs" / "archive" / "response" / "finite_q_subspace_repair"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "finite_q_subspace_repair"
SUMMARY_PATH = OUTPUT_ROOT / "finite_q_subspace_repair_summary.md"


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
        "deg_tol": np.array([], dtype=float),
        "denominator_mode": np.array([], dtype="U16"),
        "num_subspaces_minus": np.array([], dtype=int),
        "num_subspaces_plus": np.array([], dtype=int),
        "max_subspace_dimension_minus": np.array([], dtype=int),
        "max_subspace_dimension_plus": np.array([], dtype=int),
        "near_degenerate_group_count": np.array([], dtype=int),
        "eigenstate_overlap_offdiag_norm": np.array([], dtype=float),
        "eigenstate_overlap_diagonal_error": np.array([], dtype=float),
        "projector_overlap_error": np.array([], dtype=float),
        "projector_trace_defect": np.array([], dtype=float),
        "subspace_mixing_norm": np.array([], dtype=float),
        "unmatched_subspace_weight": np.array([], dtype=float),
        "possible_band_phase_or_order_issue": np.array([], dtype=bool),
        "possible_true_subspace_mixing": np.array([], dtype=bool),
        "near_degenerate_count": np.array([], dtype=int),
        "min_abs_energy_diff": np.array([], dtype=float),
        "near_degenerate_weight_raw": np.array([], dtype=float),
        "near_degenerate_weight_stable": np.array([], dtype=float),
        "denominator_regularization_delta": np.array([], dtype=float),
        "stable_denominator_changed_response_norm": np.array([], dtype=float),
        "stable_denominator_improves_continuity": np.array([], dtype=bool),
        "error_to_local_sigma": np.array([], dtype=float),
        "error_to_K_para": np.array([], dtype=float),
        "error_to_K_total": np.array([], dtype=float),
        "error_to_K_total_over_omega": np.array([], dtype=float),
        "error_to_normal_kubo_sigma": np.array([], dtype=float),
        "best_match_component": np.array([], dtype="U48"),
        "small_q_relative_error": np.array([], dtype=float),
        "diagnostic_status": np.array([], dtype="U256"),
        "gauge_status": np.array([], dtype="U64"),
        "final_casimir_input": np.array([], dtype=bool),
        "not_final_Casimir_conclusion": np.array([], dtype=bool),
        "notes": np.array([], dtype=object),
    }


def _rows_from_results(results: list[FiniteQSubspaceConsistencyDiagnostic]) -> list[dict[str, object]]:
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
                "deg_tol": result.deg_tol,
                "denominator_mode": result.denominator_mode,
                "num_subspaces_minus": result.num_subspaces_minus,
                "num_subspaces_plus": result.num_subspaces_plus,
                "max_subspace_dimension_minus": result.max_subspace_dimension_minus,
                "max_subspace_dimension_plus": result.max_subspace_dimension_plus,
                "near_degenerate_group_count": result.near_degenerate_group_count,
                "eigenstate_overlap_offdiag_norm": result.eigenstate_overlap_offdiag_norm,
                "eigenstate_overlap_diagonal_error": result.eigenstate_overlap_diagonal_error,
                "projector_overlap_error": result.projector_overlap_error,
                "projector_trace_defect": result.projector_trace_defect,
                "subspace_mixing_norm": result.subspace_mixing_norm,
                "unmatched_subspace_weight": result.unmatched_subspace_weight,
                "possible_band_phase_or_order_issue": result.possible_band_phase_or_order_issue,
                "possible_true_subspace_mixing": result.possible_true_subspace_mixing,
                "near_degenerate_count": result.near_degenerate_count,
                "min_abs_energy_diff": result.min_abs_energy_diff,
                "near_degenerate_weight_raw": result.near_degenerate_weight_raw,
                "near_degenerate_weight_stable": result.near_degenerate_weight_stable,
                "denominator_regularization_delta": result.denominator_regularization_delta,
                "stable_denominator_changed_response_norm": result.stable_denominator_changed_response_norm,
                "stable_denominator_improves_continuity": result.stable_denominator_improves_continuity,
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


def _nanmin(values: np.ndarray) -> float:
    return float(np.nanmin(values)) if np.any(np.isfinite(values)) else np.nan


def _nanmax(values: np.ndarray) -> float:
    return float(np.nanmax(values)) if np.any(np.isfinite(values)) else np.nan


def _min_by_mask(data: dict[str, np.ndarray], field: str, mask: np.ndarray) -> float:
    values = data[field][mask]
    return _nanmin(values) if values.size else np.nan


def save_figures(data: dict[str, np.ndarray]) -> list[Path]:
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    figure_dir = OUTPUT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / "projector_overlap_error_vs_q.png",
        figure_dir / "eigenstate_vs_projector_overlap.png",
        figure_dir / "small_q_error_raw_vs_stable.png",
        figure_dir / "denominator_regularization_delta.png",
        figure_dir / "deg_tol_sensitivity.png",
    ]
    q_values = sorted(set(float(item) for item in data["q_magnitude"]))

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in KINDS:
        values = [_min_by_mask(data, "projector_overlap_error", (data["kind"] == kind) & np.isclose(data["q_magnitude"], q)) for q in q_values]
        ax.plot(q_values, values, marker="o", label=kind)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("min projector overlap error")
    ax.set_title("projector overlap smoothness")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[0])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    ax.scatter(data["eigenstate_overlap_offdiag_norm"], data["projector_overlap_error"], s=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eigenstate offdiag norm")
    ax.set_ylabel("projector overlap error")
    ax.set_title("eigenstate vs projector overlap")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for mode in DENOMINATOR_MODES:
        values = [_min_by_mask(data, "small_q_relative_error", (data["denominator_mode"] == mode) & np.isclose(data["q_magnitude"], q)) for q in q_values]
        ax.plot(q_values, values, marker="o", label=mode)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("min small-q relative error")
    ax.set_title("raw vs stable denominator")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[2])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in KINDS:
        values = [_min_by_mask(data, "denominator_regularization_delta", (data["kind"] == kind) & np.isclose(data["q_magnitude"], q)) for q in q_values]
        ax.plot(q_values, values, marker="o", label=kind)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-16)
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("regularization delta")
    ax.set_title("stable occupation divided difference")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[3])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for deg_tol in sorted(set(float(item) for item in data["deg_tol"])):
        mask = np.isclose(data["deg_tol"], deg_tol)
        values = [_min_by_mask(data, "small_q_relative_error", mask & np.isclose(data["q_magnitude"], q)) for q in q_values]
        ax.plot(q_values, values, marker="o", label=f"{deg_tol:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("min small-q relative error")
    ax.set_title("deg_tol sensitivity")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[4])
    plt.close(fig)
    return paths


def _trend_error_decreases(data: dict[str, np.ndarray]) -> bool:
    stable = data["denominator_mode"] == "stable"
    q_values = sorted(set(float(item) for item in data["q_magnitude"][stable]))
    if len(q_values) < 2:
        return False
    minima = [_min_by_mask(data, "small_q_relative_error", stable & np.isclose(data["q_magnitude"], q)) for q in q_values]
    if not all(np.isfinite(minima)):
        return False
    return bool(minima[0] <= 1.2 * minima[-1])


def _deg_tol_stable(data: dict[str, np.ndarray]) -> bool:
    stable = data["denominator_mode"] == "stable"
    values = []
    for deg_tol in sorted(set(float(item) for item in data["deg_tol"])):
        mask = stable & np.isclose(data["deg_tol"], deg_tol)
        values.append(_min_by_mask(data, "small_q_relative_error", mask))
    finite = [value for value in values if np.isfinite(value)]
    if len(finite) < 2:
        return False
    return bool(max(finite) / (min(finite) + 1e-300) < 10.0)


def _summary_lines(data: dict[str, np.ndarray], args: argparse.Namespace) -> list[str]:
    stable = data["denominator_mode"] == "stable"
    raw = data["denominator_mode"] == "raw"
    min_q = float(np.min(data["q_magnitude"]))
    min_q_stable = stable & np.isclose(data["q_magnitude"], min_q)
    stable_min_q_error = _min_by_mask(data, "small_q_relative_error", min_q_stable)
    raw_best = _min_by_mask(data, "small_q_relative_error", raw)
    stable_best = _min_by_mask(data, "small_q_relative_error", stable)
    stable_improves = bool(stable_best < 0.8 * raw_best) if np.isfinite(raw_best) and np.isfinite(stable_best) else False
    projector_vs_eigen = bool(_nanmax(data["projector_overlap_error"]) < 0.1 * max(_nanmax(data["eigenstate_overlap_offdiag_norm"]), 1e-300))
    true_subspace_mixing = bool(np.any(data["possible_true_subspace_mixing"]))
    deg_tol_stable = _deg_tol_stable(data)
    error_decreases = _trend_error_decreases(data)
    a4_trend_tested = False
    vertex_ok = True
    wrapping_ok = True
    return_to_a4 = bool(
        stable_min_q_error < 1e-2
        and error_decreases
        and a4_trend_tested
        and projector_vs_eigen
        and deg_tol_stable
        and vertex_ok
        and wrapping_ok
    )
    recommend_subspace_safe = bool(true_subspace_mixing)
    recommend_denominator_refine = bool(stable_improves and not return_to_a4)
    recommend_continue = bool(not return_to_a4)
    lines = [
        "# finite-q subspace / denominator repair 诊断摘要",
        "",
        "本轮目标是 subspace / denominator repair diagnostic：处理 small-q continuity 中",
        "near-degenerate band/subspace 的 overlap、band phase、band order，以及 denominator 数值稳定性。",
        "",
        "上一轮已经看到 vertex_mismatch_detected=False 且 BZ wrapping 未触发，因此本轮不优先改 vertex 或 BZ wrapping。",
        "本轮仍只做 response 层 quick 诊断，不接入 Lifshitz/Casimir，也不输出 torque 结论。",
        "",
        f"kinds={list(args.kinds)}",
        f"matsubara_list={list(args.matsubara_list)}",
        f"q_list={list(args.q_list)}",
        f"q_phi_list={list(args.q_phi_list)}",
        f"nk_list={list(args.nk_list)}",
        f"deg_tol_list={list(args.deg_tol_list)}",
        f"denominator_mode_list={list(args.denominator_mode_list)}",
        "",
        f"max_eigenstate_overlap_offdiag_norm={_nanmax(data['eigenstate_overlap_offdiag_norm']):.6g}",
        f"max_projector_overlap_error={_nanmax(data['projector_overlap_error']):.6g}",
        f"projector_overlap_smaller_than_eigenstate_offdiag={projector_vs_eigen}",
        f"possible_true_subspace_mixing={true_subspace_mixing}",
        f"max_near_degenerate_count={int(np.nanmax(data['near_degenerate_count']))}",
        f"max_denominator_regularization_delta={_nanmax(data['denominator_regularization_delta']):.6g}",
        f"raw_best_small_q_relative_error={raw_best:.6g}",
        f"stable_best_small_q_relative_error={stable_best:.6g}",
        f"stable_min_q_relative_error={stable_min_q_error:.6g}",
        f"stable_denominator_improves_continuity={stable_improves}",
        f"small_q_error_decreases_toward_q0={error_decreases}",
        f"deg_tol_conclusion_stable={deg_tol_stable}",
        f"A4_q_to_zero_trend_tested={a4_trend_tested}",
        f"recommend_return_to_finite_q_A4_anisotropy_diagnostic={return_to_a4}",
        f"recommend_continue_formula_repair={recommend_continue}",
        f"recommend_subspace_safe_response_prototype={recommend_subspace_safe}",
        f"recommend_denominator_stable_mode_refinement={recommend_denominator_refine}",
        "",
        "## 限制",
        "- gauge_status=prototype_not_ward_verified",
        "- Ward identity / diamagnetic closure 未完成",
        "- n=0 model 未完成",
        "- final_casimir_input=False",
        "- not_final_Casimir_conclusion=True",
    ]
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
    parser.add_argument("--deg-tol-list", nargs="+", type=float, default=[1e-8, 1e-7, 1e-6])
    parser.add_argument("--denominator-mode-list", nargs="+", choices=DENOMINATOR_MODES, default=list(DENOMINATOR_MODES))
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compare_subspace_and_eigenstate_overlap(
        args.kinds,
        args.matsubara_list,
        args.q_list,
        args.q_phi_list,
        args.nk_list,
        args.deg_tol_list,
        args.denominator_mode_list,
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
    print("note = finite-q subspace / denominator diagnostic only; not a Casimir result.")


if __name__ == "__main__":
    main()
