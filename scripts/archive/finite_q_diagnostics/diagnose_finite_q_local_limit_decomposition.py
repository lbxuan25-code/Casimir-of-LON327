#!/usr/bin/env python3
"""Diagnose which local component a finite-q bubble approaches."""

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
    FiniteQLocalLimitDecomposition,
    compare_finite_q_to_local_components,
)
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("normal", "spm", "dwave")
OUTPUT_ROOT = ROOT / "outputs" / "archive" / "response" / "finite_q_local_limit"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "finite_q_local_limit"
SUMMARY_PATH = OUTPUT_ROOT / "finite_q_local_limit_summary.md"
COMPONENTS = (
    "local_sigma",
    "local_K_para",
    "local_K_total",
    "local_K_total_over_omega",
    "normal_kubo_sigma",
)


def _empty_data() -> dict[str, np.ndarray]:
    return {
        "kind": np.array([], dtype="U16"),
        "matsubara_index": np.array([], dtype=int),
        "omega_eV": np.array([], dtype=float),
        "temperature_K": np.array([], dtype=float),
        "nk": np.array([], dtype=int),
        "small_q": np.array([], dtype=float),
        "q_phi": np.array([], dtype=float),
        "component": np.array([], dtype="U48"),
        "finite_q_xx": np.array([], dtype=complex),
        "finite_q_yy": np.array([], dtype=complex),
        "finite_q_xy": np.array([], dtype=complex),
        "finite_q_yx": np.array([], dtype=complex),
        "local_sigma_xx": np.array([], dtype=complex),
        "local_K_para_xx": np.array([], dtype=complex),
        "local_K_dia_xx": np.array([], dtype=complex),
        "local_K_total_xx": np.array([], dtype=complex),
        "local_K_total_over_omega_xx": np.array([], dtype=complex),
        "normal_kubo_sigma_xx": np.array([], dtype=complex),
        "error_to_local_sigma": np.array([], dtype=float),
        "error_to_K_para": np.array([], dtype=float),
        "error_to_K_total": np.array([], dtype=float),
        "error_to_K_total_over_omega": np.array([], dtype=float),
        "error_to_normal_kubo_sigma": np.array([], dtype=float),
        "best_match_component": np.array([], dtype="U48"),
        "best_match_relative_error": np.array([], dtype=float),
        "error_monotonic_in_q": np.array([], dtype=bool),
        "error_improves_with_nk": np.array([], dtype=bool),
        "diagnostic_status": np.array([], dtype="U192"),
        "gauge_status": np.array([], dtype="U64"),
        "final_casimir_input": np.array([], dtype=bool),
        "not_final_Casimir_conclusion": np.array([], dtype=bool),
        "notes": np.array([], dtype=object),
    }


def _rows_from_results(results: list[FiniteQLocalLimitDecomposition]) -> list[dict[str, object]]:
    rows = []
    for result in results:
        rows.append(
            {
                "kind": result.kind,
                "matsubara_index": result.matsubara_index,
                "omega_eV": result.omega_eV,
                "temperature_K": result.temperature_K,
                "nk": result.nk,
                "small_q": result.small_q,
                "q_phi": result.q_phi,
                "component": "finite_q_bubble",
                "finite_q_xx": result.finite_q[0, 0],
                "finite_q_yy": result.finite_q[1, 1],
                "finite_q_xy": result.finite_q[0, 1],
                "finite_q_yx": result.finite_q[1, 0],
                "local_sigma_xx": result.local_sigma[0, 0],
                "local_K_para_xx": result.local_K_para[0, 0],
                "local_K_dia_xx": result.local_K_dia[0, 0],
                "local_K_total_xx": result.local_K_total[0, 0],
                "local_K_total_over_omega_xx": result.local_K_total_over_omega[0, 0],
                "normal_kubo_sigma_xx": result.normal_kubo_sigma[0, 0],
                "error_to_local_sigma": result.error_to_local_sigma,
                "error_to_K_para": result.error_to_K_para,
                "error_to_K_total": result.error_to_K_total,
                "error_to_K_total_over_omega": result.error_to_K_total_over_omega,
                "error_to_normal_kubo_sigma": result.error_to_normal_kubo_sigma,
                "best_match_component": result.best_match_component,
                "best_match_relative_error": result.best_match_relative_error,
                "error_monotonic_in_q": False,
                "error_improves_with_nk": False,
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


def _best_error_field(component: str) -> str:
    return {
        "local_sigma": "error_to_local_sigma",
        "local_K_para": "error_to_K_para",
        "local_K_total": "error_to_K_total",
        "local_K_total_over_omega": "error_to_K_total_over_omega",
        "normal_kubo_sigma": "error_to_normal_kubo_sigma",
    }.get(component, "best_match_relative_error")


def _fill_trends(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    data = {key: np.array(value, copy=True) for key, value in data.items()}
    data["error_monotonic_in_q"][:] = False
    data["error_improves_with_nk"][:] = False
    for kind in KINDS:
        for matsubara_index in sorted(set(int(item) for item in data["matsubara_index"])):
            for nk in sorted(set(int(item) for item in data["nk"])):
                for q_phi in sorted(set(float(item) for item in data["q_phi"])):
                    mask = (
                        (data["kind"] == kind)
                        & (data["matsubara_index"] == matsubara_index)
                        & (data["nk"] == nk)
                        & np.isclose(data["q_phi"], q_phi)
                    )
                    indices = np.where(mask)[0]
                    if indices.size < 2:
                        continue
                    ordered = sorted(indices, key=lambda index: float(data["small_q"][index]))
                    errors = data["best_match_relative_error"][ordered]
                    monotonic = bool(np.all(np.diff(errors) >= -1e-12))
                    data["error_monotonic_in_q"][ordered] = monotonic
            for small_q in sorted(set(float(item) for item in data["small_q"])):
                for q_phi in sorted(set(float(item) for item in data["q_phi"])):
                    mask = (
                        (data["kind"] == kind)
                        & (data["matsubara_index"] == matsubara_index)
                        & np.isclose(data["small_q"], small_q)
                        & np.isclose(data["q_phi"], q_phi)
                    )
                    indices = np.where(mask)[0]
                    if indices.size < 2:
                        continue
                    ordered = sorted(indices, key=lambda index: int(data["nk"][index]))
                    errors = data["best_match_relative_error"][ordered]
                    improves = bool(errors[-1] < 0.8 * errors[0])
                    data["error_improves_with_nk"][ordered] = improves
    statuses = []
    for index, status in enumerate(data["diagnostic_status"]):
        parts = [part for part in str(status).split(";") if part]
        if not data["error_monotonic_in_q"][index]:
            parts.append("warning_small_q_not_smooth")
        if data["error_improves_with_nk"][index]:
            parts.append("likely_mesh_sampling_issue")
        elif "warning_small_q_not_smooth" in parts:
            parts.append("likely_formula_or_vertex_mismatch")
        statuses.append(";".join(dict.fromkeys(parts)))
    data["diagnostic_status"] = np.asarray(statuses, dtype=data["diagnostic_status"].dtype)
    return data


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    return value


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path]:
    data = _fill_trends(data)
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


def save_figures(data: dict[str, np.ndarray]) -> list[Path]:
    data = _fill_trends(data)
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    figure_dir = OUTPUT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / "small_q_error_vs_q.png",
        figure_dir / "small_q_error_vs_nk.png",
        figure_dir / "best_local_component_match.png",
        figure_dir / "component_error_heatmap.png",
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in KINDS:
        mask = (data["kind"] == kind) & (data["matsubara_index"] == int(np.min(data["matsubara_index"])))
        if np.any(mask):
            q_values = sorted(set(float(item) for item in data["small_q"][mask]))
            values = []
            for q_value in q_values:
                q_mask = mask & np.isclose(data["small_q"], q_value)
                values.append(float(np.nanmin(data["best_match_relative_error"][q_mask])))
            ax.plot(q_values, values, marker="o", label=kind)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("small q")
    ax.set_ylabel("min best-match relative error")
    ax.set_title("small-q local-limit error")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[0])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in KINDS:
        mask = (data["kind"] == kind) & np.isclose(data["small_q"], np.min(data["small_q"]))
        if np.any(mask):
            nk_values = sorted(set(int(item) for item in data["nk"][mask]))
            values = []
            for nk in nk_values:
                nk_mask = mask & (data["nk"] == nk)
                values.append(float(np.nanmin(data["best_match_relative_error"][nk_mask])))
            ax.plot(nk_values, values, marker="o", label=kind)
    ax.set_yscale("log")
    ax.set_xlabel("Nk")
    ax.set_ylabel("min best-match relative error")
    ax.set_title("local-limit error vs Nk")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    labels = list(COMPONENTS)
    counts = [int(np.count_nonzero(data["best_match_component"] == item)) for item in labels]
    ax.bar(labels, counts)
    ax.set_ylabel("row count")
    ax.set_title("best local component match")
    ax.tick_params(axis="x", rotation=30)
    style_publication_axis(ax)
    save_publication_figure(fig, paths[2])
    plt.close(fig)

    heat = []
    for kind in KINDS:
        row = []
        for component in COMPONENTS:
            field = _best_error_field(component)
            mask = data["kind"] == kind
            values = data[field][mask]
            row.append(float(np.nanmin(values)) if np.any(np.isfinite(values)) else np.nan)
        heat.append(row)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    image = ax.imshow(np.asarray(heat, dtype=float), aspect="auto")
    ax.set_xticks(range(len(COMPONENTS)), COMPONENTS, rotation=30, ha="right")
    ax.set_yticks(range(len(KINDS)), KINDS)
    ax.set_title("minimum relative error by component")
    fig.colorbar(image, ax=ax)
    save_publication_figure(fig, paths[3])
    plt.close(fig)
    return paths


def _summary_lines(data: dict[str, np.ndarray], args: argparse.Namespace) -> list[str]:
    data = _fill_trends(data)
    min_q = float(np.min(data["small_q"]))
    max_nk = int(np.max(data["nk"]))
    candidate_mask = np.isclose(data["small_q"], min_q) & (data["nk"] == max_nk)
    candidate_error = float(np.nanmin(data["best_match_relative_error"][candidate_mask]))
    candidate = bool(candidate_error < 1e-2)
    best_components = sorted(set(str(item) for item in data["best_match_component"][candidate_mask]))
    global_best_index = int(np.nanargmin(data["best_match_relative_error"]))
    global_best = str(data["best_match_component"][global_best_index])
    monotonic = bool(np.all(data["error_monotonic_in_q"]))
    nk_improves = bool(np.any(data["error_improves_with_nk"]))
    statuses = sorted(set(str(item) for item in data["diagnostic_status"]))
    likely_missing_contact = any("likely_missing_contact_or_diamagnetic_completion" in item for item in statuses)
    likely_vertex = any("likely_formula_or_vertex_mismatch" in item for item in statuses)
    lines = [
        "# finite-q local-limit 分解诊断摘要",
        "",
        "本轮目的：拆解 finite-q bubble 在 q->0 时与 local response 不连续的来源，判断它更接近",
        "local sigma、BdG K_para、K_total、K_total/omega，还是 normal Kubo sigma。",
        "",
        "本轮不做 Casimir、不做 torque、不做正式物理结论。q=0 local hook 只是直接引用",
        "local reference；small-q finite-q bubble continuity 才是连续极限诊断。",
        "",
        f"kinds={list(args.kinds)}",
        f"matsubara_list={list(args.matsubara_list)}",
        f"small_q_list={list(args.small_q_list)}",
        f"q_phi_list={list(args.q_phi_list)}",
        f"nk_list={list(args.nk_list)}",
        f"temperature={args.temperature}",
        f"delta0={args.delta0}",
        f"eta={args.eta}",
        "",
        f"global_best_match_component={global_best}",
        f"best_match_components_at_min_q_max_nk={best_components}",
        f"best_match_relative_error_at_min_q_max_nk={candidate_error:.6g}",
        f"local_limit_component_match_candidate={candidate}",
        f"small_q_error_monotonic_in_q={monotonic}",
        f"error_improves_with_nk={nk_improves}",
        f"likely_missing_contact_or_diamagnetic_completion={likely_missing_contact}",
        f"likely_formula_or_vertex_mismatch={likely_vertex}",
        "worth_next_finite_q_casimir_plumbing_smoke=False",
        "",
        "## 诊断状态",
    ]
    for status in statuses:
        lines.append(f"- {status}")
    lines.extend(
        [
            "",
            "## 建议",
            f"recommend_finite_q_formula_repair={bool(not candidate or likely_missing_contact or likely_vertex)}",
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
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--matsubara-list", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--small-q-list", nargs="+", type=float, default=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    parser.add_argument(
        "--q-phi-list",
        nargs="+",
        type=float,
        default=[0.0, 0.3926990817, 0.7853981634, 1.1780972451, 1.5707963268],
    )
    parser.add_argument("--nk-list", nargs="+", type=int, default=[6, 8, 12])
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()
    if args.quick:
        args.matsubara_list = [1]
        args.small_q_list = [1e-4, 1e-3]
        args.q_phi_list = [0.0, 0.7853981634]
        args.nk_list = [6]
    return args


def main() -> None:
    args = parse_args()
    results = compare_finite_q_to_local_components(
        args.kinds,
        args.matsubara_list,
        args.small_q_list,
        args.q_phi_list,
        args.nk_list,
        args.temperature,
        args.delta0,
        args.eta,
    )
    data = _append_rows(_empty_data(), _rows_from_results(results))
    data = _fill_trends(data)
    npz_path, csv_path = save_outputs(data, args.output_prefix)
    figure_paths = save_figures(data)
    summary_path = write_summary(data, args)
    print(f"npz_path = {npz_path}")
    print(f"csv_path = {csv_path}")
    print(f"summary_path = {summary_path}")
    print("figure_paths = " + ", ".join(str(path) for path in figure_paths))
    print("note = finite-q local-limit decomposition diagnostic only; not a Casimir result.")


if __name__ == "__main__":
    main()
