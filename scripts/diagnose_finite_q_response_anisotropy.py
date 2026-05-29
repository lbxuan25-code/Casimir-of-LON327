#!/usr/bin/env python3
"""Diagnose finite-q response angular anisotropy prototypes."""

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
    FiniteQResponseResult,
    finite_q_response_phi_scan,
)
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("normal", "spm", "dwave")
OUTPUT_ROOT = ROOT / "outputs" / "response" / "finite_q_anisotropy"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "finite_q_anisotropy"
SUMMARY_PATH = OUTPUT_ROOT / "finite_q_anisotropy_summary.md"
RATIO_EPS = 1e-300


def _empty_data() -> dict[str, np.ndarray]:
    return {
        "kind": np.array([], dtype="U16"),
        "matsubara_index": np.array([], dtype=int),
        "temperature_K": np.array([], dtype=float),
        "q_magnitude": np.array([], dtype=float),
        "q_phi": np.array([], dtype=float),
        "qx": np.array([], dtype=float),
        "qy": np.array([], dtype=float),
        "nk": np.array([], dtype=int),
        "delta0": np.array([], dtype=float),
        "eta": np.array([], dtype=float),
        "response_xx": np.array([], dtype=complex),
        "response_yy": np.array([], dtype=complex),
        "response_xy": np.array([], dtype=complex),
        "response_yx": np.array([], dtype=complex),
        "sheet_xx_SI": np.array([], dtype=complex),
        "reflection_xx": np.array([], dtype=complex),
        "finite_q_resolved": np.array([], dtype=bool),
        "finite_q_response_diagnostic": np.array([], dtype=bool),
        "final_casimir_input": np.array([], dtype=bool),
        "not_final_Casimir_conclusion": np.array([], dtype=bool),
        "local_limit_abs_error": np.array([], dtype=float),
        "local_limit_relative_error": np.array([], dtype=float),
        "angular_anisotropy_A4_xx": np.array([], dtype=float),
        "angular_anisotropy_A4_trace": np.array([], dtype=float),
        "relative_offdiag": np.array([], dtype=float),
        "relative_eigen_split": np.array([], dtype=float),
        "gauge_status": np.array([], dtype="U64"),
        "diagnostic_status": np.array([], dtype="U64"),
        "pairing_contrast_dwave_minus_spm": np.array([], dtype=float),
        "notes": np.array([], dtype=object),
    }


def _rows_from_results(results: list[FiniteQResponseResult]) -> list[dict[str, object]]:
    rows = []
    for result in results:
        rows.append(
            {
                "kind": result.kind,
                "matsubara_index": result.matsubara_index,
                "temperature_K": result.temperature_K,
                "q_magnitude": result.q_magnitude,
                "q_phi": result.q_phi,
                "qx": result.q_vector[0],
                "qy": result.q_vector[1],
                "nk": result.nk,
                "delta0": result.delta0,
                "eta": result.eta,
                "response_xx": result.response_tensor_model[0, 0],
                "response_yy": result.response_tensor_model[1, 1],
                "response_xy": result.response_tensor_model[0, 1],
                "response_yx": result.response_tensor_model[1, 0],
                "sheet_xx_SI": result.sheet_conductivity_SI[0, 0],
                "reflection_xx": result.reflection_dimensionless[0, 0],
                "finite_q_resolved": result.finite_q_resolved,
                "finite_q_response_diagnostic": result.finite_q_response_diagnostic,
                "final_casimir_input": result.final_casimir_input,
                "not_final_Casimir_conclusion": result.not_final_Casimir_conclusion,
                "local_limit_abs_error": result.local_limit_abs_error,
                "local_limit_relative_error": result.local_limit_relative_error,
                "angular_anisotropy_A4_xx": result.angular_anisotropy_A4_xx,
                "angular_anisotropy_A4_trace": result.angular_anisotropy_A4_trace,
                "relative_offdiag": float(result.symmetry_diagnostics["relative_offdiag"]),
                "relative_eigen_split": float(result.symmetry_diagnostics["relative_eigen_split"]),
                "gauge_status": result.gauge_status,
                "diagnostic_status": result.diagnostic_status,
                "pairing_contrast_dwave_minus_spm": np.nan,
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


def _fill_pairing_contrast(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    data = {key: np.array(value, copy=True) for key, value in data.items()}
    data["pairing_contrast_dwave_minus_spm"][:] = np.nan
    keys = sorted(
        {
            (int(n), float(q), float(phi))
            for n, q, phi in zip(data["matsubara_index"], data["q_magnitude"], data["q_phi"], strict=True)
        }
    )
    for matsubara_index, q_magnitude, q_phi in keys:
        base = (
            (data["matsubara_index"] == matsubara_index)
            & np.isclose(data["q_magnitude"], q_magnitude)
            & np.isclose(data["q_phi"], q_phi)
        )
        masks = {kind: base & (data["kind"] == kind) for kind in KINDS}
        if not all(np.any(mask) for mask in masks.values()):
            continue
        normal = complex(data["response_xx"][masks["normal"]][0])
        spm = complex(data["response_xx"][masks["spm"]][0])
        dwave = complex(data["response_xx"][masks["dwave"]][0])
        contrast = abs((dwave - normal) - (spm - normal))
        scale = abs(dwave - normal) + abs(spm - normal) + RATIO_EPS
        value = float(contrast / scale)
        for kind in KINDS:
            data["pairing_contrast_dwave_minus_spm"][masks[kind]] = value
    return data


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    return value


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path]:
    data = _fill_pairing_contrast(data)
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
    data = _fill_pairing_contrast(data)
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    figure_dir = OUTPUT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / "response_xx_vs_phi.png",
        figure_dir / "A4_vs_q.png",
        figure_dir / "pairing_contrast_vs_q.png",
        figure_dir / "local_limit_error.png",
    ]
    q_values = sorted(set(float(item) for item in data["q_magnitude"]))

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in KINDS:
        for q_value in q_values:
            mask = (data["kind"] == kind) & np.isclose(data["q_magnitude"], q_value) & (data["matsubara_index"] == 1)
            if np.any(mask):
                ax.plot(data["q_phi"][mask], data["response_xx"][mask].real, marker="o", label=f"{kind}, q={q_value:g}")
    ax.set_xlabel("q phi (rad)")
    ax.set_ylabel("Re response_xx")
    ax.set_title("finite-q response_xx angular scan")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[0])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in KINDS:
        values = []
        for q_value in q_values:
            mask = (data["kind"] == kind) & np.isclose(data["q_magnitude"], q_value)
            values.append(float(np.nanmax(np.abs(data["angular_anisotropy_A4_xx"][mask]))))
        ax.plot(q_values, values, marker="o", label=kind)
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("max |A4_xx|")
    ax.set_title("finite-q A4 diagnostic")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    values = []
    for q_value in q_values:
        mask = np.isclose(data["q_magnitude"], q_value)
        values.append(float(np.nanmax(data["pairing_contrast_dwave_minus_spm"][mask])))
    ax.plot(q_values, values, marker="o")
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("max contrast")
    ax.set_title("dwave-normal vs spm-normal contrast")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[2])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in KINDS:
        mask = (data["kind"] == kind) & np.isclose(data["q_magnitude"], 0.0)
        if np.any(mask):
            ax.plot(data["matsubara_index"][mask], data["local_limit_relative_error"][mask], marker="o", label=kind)
    ax.set_xlabel("Matsubara index")
    ax.set_ylabel("q=0 relative local-limit error")
    ax.set_title("local limit check")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[3])
    plt.close(fig)
    return paths


def _summary_lines(data: dict[str, np.ndarray], args: argparse.Namespace) -> list[str]:
    data = _fill_pairing_contrast(data)
    local_mask = np.isclose(data["q_magnitude"], 0.0)
    local_limit_passed = bool(np.any(local_mask)) and bool(np.nanmax(data["local_limit_relative_error"][local_mask]) < 1e-8)
    finite_mask = data["q_magnitude"] > 0.0
    anisotropy_signal = bool(np.any(finite_mask)) and bool(np.nanmax(np.abs(data["angular_anisotropy_A4_xx"][finite_mask])) > 1e-8)
    contrast_signal = bool(np.any(finite_mask)) and bool(np.nanmax(data["pairing_contrast_dwave_minus_spm"][finite_mask]) > 1e-8)
    lines = [
        "# finite-q response 角向各向异性诊断摘要",
        "",
        "本阶段选择 finite-q response，是为了检查 Casimir 几何中的有限 q_parallel 是否能在",
        "不修改 H0、不修改 pairing 的情况下放大 spm/dwave 的 response 层差异，尤其是",
        "dwave 节点相关的角向响应差异。",
        "",
        "本脚本只做 response 层 finite-q diagnostic / prototype，不做 Casimir torque，",
        "也不接入正式 Lifshitz 积分。",
        "",
        "q_magnitude 当前使用 dimensionless BZ momentum，与 k 网格单位一致，不是 SI wavevector。",
        "",
        f"kinds={list(args.kinds)}",
        f"matsubara_list={list(args.matsubara_list)}",
        f"q_list={list(args.q_list)}",
        f"q_phi_list={list(args.q_phi_list)}",
        f"temperature={args.temperature}",
        f"nk={args.nk}",
        f"delta0={args.delta0}",
        f"eta={args.eta}",
        "",
        f"q_to_0_local_limit_passed={local_limit_passed}",
        f"finite_q_angular_anisotropy_signal={anisotropy_signal}",
        f"dwave_normal_vs_spm_normal_contrast_signal={contrast_signal}",
        f"worth_next_finite_q_casimir_plumbing_smoke={bool(local_limit_passed and anisotropy_signal and contrast_signal)}",
        "",
        "## 每个 q 的诊断",
    ]
    for q_value in sorted(set(float(item) for item in data["q_magnitude"])):
        q_mask = np.isclose(data["q_magnitude"], q_value)
        lines.append(f"- q={q_value:g}")
        for kind in KINDS:
            mask = q_mask & (data["kind"] == kind)
            lines.append(f"  {kind}_max_abs_A4_xx={float(np.nanmax(np.abs(data['angular_anisotropy_A4_xx'][mask]))):.6g}")
        lines.append(
            f"  max_abs_contrast_dwave_minus_spm={float(np.nanmax(data['pairing_contrast_dwave_minus_spm'][q_mask])):.6g}"
        )
    lines.extend(
        [
            "",
            "## 限制",
            "- gauge_status=prototype_not_ward_verified",
            "- finite-q diamagnetic/Ward identity 尚未严格闭合",
            "- n=0 zero-frequency model 仍未完成",
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
    parser.add_argument("--matsubara-list", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--q-list", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.1])
    parser.add_argument(
        "--q-phi-list",
        nargs="+",
        type=float,
        default=[0.0, 0.3926990817, 0.7853981634, 1.1780972451, 1.5707963268],
    )
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--nk", type=int, default=16)
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = _empty_data()
    for kind in args.kinds:
        for matsubara_index in args.matsubara_list:
            for q_magnitude in args.q_list:
                print(f"running = {kind}, n={matsubara_index}, q={q_magnitude:g}")
                results = finite_q_response_phi_scan(
                    kind,
                    matsubara_index,
                    args.temperature,
                    q_magnitude,
                    args.q_phi_list,
                    args.nk,
                    args.delta0,
                    args.eta,
                )
                data = _append_rows(data, _rows_from_results(results))
    data = _fill_pairing_contrast(data)
    npz_path, csv_path = save_outputs(data, args.output_prefix)
    figure_paths = save_figures(data)
    summary_path = write_summary(data, args)
    print(f"npz_path = {npz_path}")
    print(f"csv_path = {csv_path}")
    print(f"summary_path = {summary_path}")
    print("figure_paths = " + ", ".join(str(path) for path in figure_paths))
    print("note = finite-q response diagnostic prototype; not a final Casimir torque conclusion.")


if __name__ == "__main__":
    main()
