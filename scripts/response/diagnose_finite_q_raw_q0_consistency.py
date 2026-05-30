#!/usr/bin/env python3
"""Diagnose raw q=0 finite-q bubble consistency."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.finite_q_response import (  # noqa: E402
    FiniteQRawQ0Consistency,
    compare_raw_q0_bubble_to_local_components,
)
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("normal", "spm", "dwave")
DENOMINATOR_MODES = ("raw", "stable")
COMPONENT_FIELDS = (
    ("error_raw_to_local_sigma", "local_sigma"),
    ("error_raw_to_local_K_para", "K_para"),
    ("error_raw_to_local_K_total", "K_total"),
    ("error_raw_to_local_K_total_over_omega", "K_total/omega"),
    ("error_raw_to_normal_kubo_sigma", "normal_kubo_sigma"),
)
OUTPUT_ROOT = ROOT / "outputs" / "response" / "finite_q_raw_q0_consistency"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "finite_q_raw_q0_consistency"
SUMMARY_PATH = OUTPUT_ROOT / "finite_q_raw_q0_consistency_summary.md"


def _empty_data() -> dict[str, np.ndarray]:
    return {
        "kind": np.array([], dtype="U16"),
        "matsubara_index": np.array([], dtype=int),
        "temperature_K": np.array([], dtype=float),
        "nk": np.array([], dtype=int),
        "delta0": np.array([], dtype=float),
        "eta": np.array([], dtype=float),
        "denominator_mode": np.array([], dtype="U16"),
        "deg_tol": np.array([], dtype=float),
        "raw_q0_bubble": np.array([], dtype=object),
        "local_sigma": np.array([], dtype=object),
        "local_K_para": np.array([], dtype=object),
        "local_K_dia": np.array([], dtype=object),
        "local_K_total": np.array([], dtype=object),
        "local_K_total_over_omega": np.array([], dtype=object),
        "normal_kubo_sigma": np.array([], dtype=object),
        "hook_q0_response": np.array([], dtype=object),
        "error_raw_to_local_sigma": np.array([], dtype=float),
        "error_raw_to_local_K_para": np.array([], dtype=float),
        "error_raw_to_local_K_total": np.array([], dtype=float),
        "error_raw_to_local_K_total_over_omega": np.array([], dtype=float),
        "error_raw_to_normal_kubo_sigma": np.array([], dtype=float),
        "error_hook_to_local_sigma": np.array([], dtype=float),
        "best_raw_q0_match_component": np.array([], dtype="U48"),
        "best_raw_q0_relative_error": np.array([], dtype=float),
        "raw_q0_matches_local_sigma": np.array([], dtype=bool),
        "raw_q0_matches_K_para": np.array([], dtype=bool),
        "raw_q0_matches_K_total_over_omega": np.array([], dtype=bool),
        "formula_layer_diagnosis": np.array([], dtype="U96"),
        "diagnostic_status": np.array([], dtype="U256"),
        "gauge_status": np.array([], dtype="U64"),
        "final_casimir_input": np.array([], dtype=bool),
        "not_final_Casimir_conclusion": np.array([], dtype=bool),
        "notes": np.array([], dtype=object),
    }


def _object_array(values: list[object]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def _rows_from_results(results: list[FiniteQRawQ0Consistency]) -> list[dict[str, object]]:
    rows = []
    for result in results:
        rows.append(
            {
                "kind": result.kind,
                "matsubara_index": result.matsubara_index,
                "temperature_K": result.temperature_K,
                "nk": result.nk,
                "delta0": result.delta0,
                "eta": result.eta,
                "denominator_mode": result.denominator_mode,
                "deg_tol": result.deg_tol,
                "raw_q0_bubble": result.raw_q0_bubble,
                "local_sigma": result.local_sigma,
                "local_K_para": result.local_K_para,
                "local_K_dia": result.local_K_dia,
                "local_K_total": result.local_K_total,
                "local_K_total_over_omega": result.local_K_total_over_omega,
                "normal_kubo_sigma": result.normal_kubo_sigma,
                "hook_q0_response": result.hook_q0_response,
                "error_raw_to_local_sigma": result.error_raw_to_local_sigma,
                "error_raw_to_local_K_para": result.error_raw_to_local_K_para,
                "error_raw_to_local_K_total": result.error_raw_to_local_K_total,
                "error_raw_to_local_K_total_over_omega": result.error_raw_to_local_K_total_over_omega,
                "error_raw_to_normal_kubo_sigma": result.error_raw_to_normal_kubo_sigma,
                "error_hook_to_local_sigma": result.error_hook_to_local_sigma,
                "best_raw_q0_match_component": result.best_raw_q0_match_component,
                "best_raw_q0_relative_error": result.best_raw_q0_relative_error,
                "raw_q0_matches_local_sigma": result.raw_q0_matches_local_sigma,
                "raw_q0_matches_K_para": result.raw_q0_matches_K_para,
                "raw_q0_matches_K_total_over_omega": result.raw_q0_matches_K_total_over_omega,
                "formula_layer_diagnosis": result.formula_layer_diagnosis,
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
    object_fields = {
        "raw_q0_bubble",
        "local_sigma",
        "local_K_para",
        "local_K_dia",
        "local_K_total",
        "local_K_total_over_omega",
        "normal_kubo_sigma",
        "hook_q0_response",
        "notes",
    }
    for key in new_data:
        values = [row[key] for row in rows]
        if key in object_fields:
            new_data[key] = _object_array(values)
        else:
            new_data[key] = np.asarray(values, dtype=new_data[key].dtype)
    return {key: np.concatenate([data[key], new_data[key]]) for key in data}


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=8, separator=" ")
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


def _min_by_mask(data: dict[str, np.ndarray], field: str, mask: np.ndarray) -> float:
    values = data[field][mask]
    return _nanmin(values) if values.size else np.nan


def save_figures(data: dict[str, np.ndarray]) -> list[Path]:
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    figure_dir = OUTPUT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / "raw_q0_component_errors.png",
        figure_dir / "best_match_by_kind.png",
        figure_dir / "raw_vs_hook_error.png",
        figure_dir / "denominator_mode_q0_comparison.png",
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    labels = [label for _field, label in COMPONENT_FIELDS]
    values = [_nanmin(data[field]) for field, _label in COMPONENT_FIELDS]
    ax.bar(labels, values)
    ax.set_yscale("log")
    ax.set_ylabel("min relative error")
    ax.set_title("raw q=0 bubble component errors")
    ax.tick_params(axis="x", rotation=25)
    style_publication_axis(ax)
    save_publication_figure(fig, paths[0])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    kinds = list(KINDS)
    best_values = [_min_by_mask(data, "best_raw_q0_relative_error", data["kind"] == kind) for kind in kinds]
    ax.bar(kinds, best_values)
    ax.set_yscale("log")
    ax.set_ylabel("best raw q=0 relative error")
    ax.set_title("best match by kind")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    raw_values = [_min_by_mask(data, "error_raw_to_local_sigma", data["kind"] == kind) for kind in kinds]
    hook_values = [_min_by_mask(data, "error_hook_to_local_sigma", data["kind"] == kind) for kind in kinds]
    positions = np.arange(len(kinds))
    width = 0.38
    ax.bar(positions - width / 2.0, raw_values, width, label="raw q0")
    ax.bar(positions + width / 2.0, hook_values, width, label="hook q0")
    ax.set_yscale("symlog", linthresh=1e-16)
    ax.set_xticks(positions, kinds)
    ax.set_ylabel("relative error to local_sigma")
    ax.set_title("raw q=0 vs hook q=0")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[2])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for mode in DENOMINATOR_MODES:
        values = [_min_by_mask(data, "best_raw_q0_relative_error", (data["kind"] == kind) & (data["denominator_mode"] == mode)) for kind in kinds]
        ax.plot(kinds, values, marker="o", label=mode)
    ax.set_yscale("log")
    ax.set_ylabel("best raw q=0 relative error")
    ax.set_title("denominator mode q=0 comparison")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[3])
    plt.close(fig)
    return paths


def _summary_lines(data: dict[str, np.ndarray], args: argparse.Namespace) -> list[str]:
    normal = data["kind"] == "normal"
    bdg = data["kind"] != "normal"
    normal_q0_pass = bool(_min_by_mask(data, "error_raw_to_local_sigma", normal) < 1e-3)
    bdg_sigma_like = bool(np.any(data["raw_q0_matches_local_sigma"][bdg]))
    bdg_para_like = bool(np.any((data["best_raw_q0_match_component"] == "local_K_para") & (data["best_raw_q0_relative_error"] < 1e-3)))
    bdg_total_over_omega_like = bool(
        np.any((data["best_raw_q0_match_component"] == "local_K_total_over_omega") & (data["best_raw_q0_relative_error"] < 1e-3))
    )
    unmatched = bool(np.any(data["best_raw_q0_relative_error"] >= 1e-2))
    recommend_normal = bool(not normal_q0_pass)
    recommend_bdg = bool(normal_q0_pass and not bdg_sigma_like)
    recommend_smoothness = bool(normal_q0_pass and bdg_sigma_like)
    diagnoses = sorted(set(str(item) for item in data["formula_layer_diagnosis"]))
    lines = [
        "# finite-q raw q=0 formula consistency 诊断摘要",
        "",
        "本轮目标是检查 raw q=0 finite-q bubble 与已有 local response 的定义层级是否一致。",
        "q=0 hook 会直接返回 local reference；raw q=0 bubble 则强制走与 q>0 相同的 finite-q bubble 公式。",
        "",
        f"kinds={list(args.kinds)}",
        f"matsubara_list={list(args.matsubara_list)}",
        f"nk_list={list(args.nk_list)}",
        f"denominator_mode_list={list(args.denominator_mode_list)}",
        f"deg_tol_list={list(args.deg_tol_list)}",
        f"temperature={args.temperature}",
        f"delta0={args.delta0}",
        f"eta={args.eta}",
        "",
        f"normal_q0_consistency_pass={normal_q0_pass}",
        f"normal_min_error_raw_to_local_sigma={_min_by_mask(data, 'error_raw_to_local_sigma', normal):.6g}",
        f"normal_min_error_hook_to_local_sigma={_min_by_mask(data, 'error_hook_to_local_sigma', normal):.6g}",
        f"bdg_raw_q0_sigma_like={bdg_sigma_like}",
        f"bdg_raw_q0_para_like={bdg_para_like}",
        f"bdg_raw_q0_total_over_omega_like={bdg_total_over_omega_like}",
        f"raw_q0_unmatched={unmatched}",
        f"spm_best_match={_best_match_for_kind(data, 'spm')}",
        f"dwave_best_match={_best_match_for_kind(data, 'dwave')}",
        f"formula_layer_mismatch_detected={bool(recommend_normal or recommend_bdg or unmatched)}",
        f"recommend_normal_kubo_formula_repair={recommend_normal}",
        f"recommend_bdg_layer_alignment={recommend_bdg}",
        f"recommend_formula_rederive={unmatched}",
        f"recommend_return_to_small_q_smoothness_diagnostic={recommend_smoothness}",
        "",
        "## formula_layer_diagnosis",
    ]
    for diagnosis in diagnoses:
        lines.append(f"- {diagnosis}")
    lines.extend(
        [
            "",
            "## 限制",
            "- 当前 finite-q response 仍不是 Ward 完备",
            "- finite-q diamagnetic / Ward closure 未完成",
            "- n=0 model 未完成",
            "- final_casimir_input=False",
            "- not_final_Casimir_conclusion=True",
        ]
    )
    return lines


def _best_match_for_kind(data: dict[str, np.ndarray], kind: str) -> str:
    mask = data["kind"] == kind
    if not np.any(mask):
        return "none"
    indices = np.where(mask)[0]
    best_index = indices[int(np.nanargmin(data["best_raw_q0_relative_error"][indices]))]
    return f"{data['best_raw_q0_match_component'][best_index]} ({data['best_raw_q0_relative_error'][best_index]:.6g})"


def write_summary(data: dict[str, np.ndarray], args: argparse.Namespace) -> Path:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(_summary_lines(data, args)) + "\n", encoding="utf-8")
    return SUMMARY_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--matsubara-list", nargs="+", type=int, default=[1])
    parser.add_argument("--nk-list", nargs="+", type=int, default=[6, 8])
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--denominator-mode-list", nargs="+", choices=DENOMINATOR_MODES, default=list(DENOMINATOR_MODES))
    parser.add_argument("--deg-tol-list", nargs="+", type=float, default=[1e-8, 1e-7])
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compare_raw_q0_bubble_to_local_components(
        args.kinds,
        args.matsubara_list,
        args.nk_list,
        args.denominator_mode_list,
        args.deg_tol_list,
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
    print("note = finite-q raw q=0 formula consistency diagnostic only; not a Casimir result.")


if __name__ == "__main__":
    main()
