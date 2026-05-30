#!/usr/bin/env python3
"""Diagnose finite-q response angular anisotropy prototypes."""

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
    FiniteQResponseResult,
    bdg_finite_q_response_imag_axis,
    finite_q_response_phi_scan,
)
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("normal", "spm", "dwave")
OUTPUT_ROOT = ROOT / "outputs" / "archive" / "response" / "finite_q_anisotropy"
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
        "local_reference_hook_passed": np.array([], dtype=bool),
        "local_limit_abs_error": np.array([], dtype=float),
        "local_limit_relative_error": np.array([], dtype=float),
        "small_q_limit_abs_error": np.array([], dtype=float),
        "small_q_limit_relative_error": np.array([], dtype=float),
        "small_q_limit_status": np.array([], dtype="U64"),
        "q_to_0_continuity_tested": np.array([], dtype=bool),
        "q_to_0_continuity_passed": np.array([], dtype=bool),
        "angular_anisotropy_A4_xx": np.array([], dtype=float),
        "angular_anisotropy_A4_trace": np.array([], dtype=float),
        "delta_A4_spm": np.array([], dtype=float),
        "delta_A4_dwave": np.array([], dtype=float),
        "A4_pairing_contrast": np.array([], dtype=float),
        "delta_A4_trace_spm": np.array([], dtype=float),
        "delta_A4_trace_dwave": np.array([], dtype=float),
        "A4_trace_pairing_contrast": np.array([], dtype=float),
        "relative_offdiag": np.array([], dtype=float),
        "relative_eigen_split": np.array([], dtype=float),
        "gauge_status": np.array([], dtype="U64"),
        "diagnostic_status": np.array([], dtype="U64"),
        "legacy_response_xx_contrast": np.array([], dtype=float),
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
                "local_reference_hook_passed": result.local_reference_hook_passed,
                "local_limit_abs_error": result.local_limit_abs_error,
                "local_limit_relative_error": result.local_limit_relative_error,
                "small_q_limit_abs_error": result.small_q_limit_abs_error,
                "small_q_limit_relative_error": result.small_q_limit_relative_error,
                "small_q_limit_status": result.small_q_limit_status,
                "q_to_0_continuity_tested": result.q_to_0_continuity_tested,
                "q_to_0_continuity_passed": result.q_to_0_continuity_passed,
                "angular_anisotropy_A4_xx": result.angular_anisotropy_A4_xx,
                "angular_anisotropy_A4_trace": result.angular_anisotropy_A4_trace,
                "delta_A4_spm": np.nan,
                "delta_A4_dwave": np.nan,
                "A4_pairing_contrast": np.nan,
                "delta_A4_trace_spm": np.nan,
                "delta_A4_trace_dwave": np.nan,
                "A4_trace_pairing_contrast": np.nan,
                "relative_offdiag": float(result.symmetry_diagnostics["relative_offdiag"]),
                "relative_eigen_split": float(result.symmetry_diagnostics["relative_eigen_split"]),
                "gauge_status": result.gauge_status,
                "diagnostic_status": result.diagnostic_status,
                "legacy_response_xx_contrast": np.nan,
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


def _small_q_status(relative_error: float) -> str:
    if relative_error < 1e-2:
        return "good_continuity_candidate"
    if relative_error < 5e-2:
        return "prototype_continuity_candidate"
    return "not_continuous_enough"


def _small_q_not_smooth(errors_by_q: list[tuple[float, float]]) -> bool:
    ordered = sorted(errors_by_q, key=lambda item: item[0])
    finite = [(q, err) for q, err in ordered if np.isfinite(err)]
    if len(finite) < 2:
        return False
    return bool(finite[0][1] > finite[-1][1])


def _small_q_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    for kind in args.kinds:
        for matsubara_index in args.matsubara_list:
            for q_phi in args.q_phi_list:
                errors: list[tuple[float, float]] = []
                results = []
                for small_q in args.small_q_list:
                    result = bdg_finite_q_response_imag_axis(
                        kind,
                        matsubara_index,
                        args.temperature,
                        float(small_q),
                        float(q_phi),
                        args.nk,
                        args.delta0,
                        args.eta,
                    )
                    errors.append((float(small_q), result.local_limit_relative_error))
                    results.append(result)
                min_result = min(results, key=lambda item: item.q_magnitude)
                status = _small_q_status(min_result.local_limit_relative_error)
                rows.append(
                    {
                        "kind": kind,
                        "matsubara_index": matsubara_index,
                        "temperature_K": args.temperature,
                        "q_magnitude": 0.0,
                        "q_phi": float(q_phi),
                        "qx": 0.0,
                        "qy": 0.0,
                        "nk": args.nk,
                        "delta0": args.delta0,
                        "eta": args.eta,
                        "response_xx": np.nan + 0.0j,
                        "response_yy": np.nan + 0.0j,
                        "response_xy": np.nan + 0.0j,
                        "response_yx": np.nan + 0.0j,
                        "sheet_xx_SI": np.nan + 0.0j,
                        "reflection_xx": np.nan + 0.0j,
                        "finite_q_resolved": True,
                        "finite_q_response_diagnostic": True,
                        "final_casimir_input": False,
                        "not_final_Casimir_conclusion": True,
                        "local_reference_hook_passed": False,
                        "local_limit_abs_error": np.nan,
                        "local_limit_relative_error": np.nan,
                        "small_q_limit_abs_error": min_result.local_limit_abs_error,
                        "small_q_limit_relative_error": min_result.local_limit_relative_error,
                        "small_q_limit_status": status,
                        "q_to_0_continuity_tested": True,
                        "q_to_0_continuity_passed": status in {"good_continuity_candidate", "prototype_continuity_candidate"},
                        "angular_anisotropy_A4_xx": np.nan,
                        "angular_anisotropy_A4_trace": np.nan,
                        "delta_A4_spm": np.nan,
                        "delta_A4_dwave": np.nan,
                        "A4_pairing_contrast": np.nan,
                        "delta_A4_trace_spm": np.nan,
                        "delta_A4_trace_dwave": np.nan,
                        "A4_trace_pairing_contrast": np.nan,
                        "relative_offdiag": np.nan,
                        "relative_eigen_split": np.nan,
                        "gauge_status": "prototype_not_ward_verified",
                        "diagnostic_status": "warning_small_q_not_smooth" if _small_q_not_smooth(errors) else "small_q_continuity_test",
                        "legacy_response_xx_contrast": np.nan,
                        "notes": (
                            "small-q finite-q bubble continuity diagnostic",
                            "q=0 hook is separate from true q->0 continuity",
                            "prototype_not_ward_verified",
                            "not a final Casimir torque conclusion",
                        ),
                    }
                )
    return rows


def _fill_pairing_contrast(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    data = {key: np.array(value, copy=True) for key, value in data.items()}
    data["legacy_response_xx_contrast"][:] = np.nan
    for field in (
        "delta_A4_spm",
        "delta_A4_dwave",
        "A4_pairing_contrast",
        "delta_A4_trace_spm",
        "delta_A4_trace_dwave",
        "A4_trace_pairing_contrast",
    ):
        data[field][:] = np.nan
    keys = sorted(
        {
            (int(n), float(q))
            for n, q, a4 in zip(data["matsubara_index"], data["q_magnitude"], data["angular_anisotropy_A4_xx"], strict=True)
            if np.isfinite(a4)
        }
    )
    for matsubara_index, q_magnitude in keys:
        base = (
            (data["matsubara_index"] == matsubara_index)
            & np.isclose(data["q_magnitude"], q_magnitude)
        )
        masks = {kind: base & (data["kind"] == kind) for kind in KINDS}
        if not all(np.any(mask) for mask in masks.values()):
            continue
        normal_a4 = float(data["angular_anisotropy_A4_xx"][masks["normal"]][0])
        spm_a4 = float(data["angular_anisotropy_A4_xx"][masks["spm"]][0])
        dwave_a4 = float(data["angular_anisotropy_A4_xx"][masks["dwave"]][0])
        normal_trace = float(data["angular_anisotropy_A4_trace"][masks["normal"]][0])
        spm_trace = float(data["angular_anisotropy_A4_trace"][masks["spm"]][0])
        dwave_trace = float(data["angular_anisotropy_A4_trace"][masks["dwave"]][0])
        delta_spm = spm_a4 - normal_a4
        delta_dwave = dwave_a4 - normal_a4
        contrast_a4 = delta_dwave - delta_spm
        delta_trace_spm = spm_trace - normal_trace
        delta_trace_dwave = dwave_trace - normal_trace
        contrast_trace = delta_trace_dwave - delta_trace_spm
        for kind in KINDS:
            data["delta_A4_spm"][masks[kind]] = delta_spm
            data["delta_A4_dwave"][masks[kind]] = delta_dwave
            data["A4_pairing_contrast"][masks[kind]] = contrast_a4
            data["delta_A4_trace_spm"][masks[kind]] = delta_trace_spm
            data["delta_A4_trace_dwave"][masks[kind]] = delta_trace_dwave
            data["A4_trace_pairing_contrast"][masks[kind]] = contrast_trace
    legacy_keys = sorted(
        {
            (int(n), float(q), float(phi))
            for n, q, phi in zip(data["matsubara_index"], data["q_magnitude"], data["q_phi"], strict=True)
        }
    )
    for matsubara_index, q_magnitude, q_phi in legacy_keys:
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
        if not all(np.isfinite([normal.real, spm.real, dwave.real])):
            continue
        contrast = abs((dwave - normal) - (spm - normal))
        scale = abs(dwave - normal) + abs(spm - normal) + RATIO_EPS
        value = float(contrast / scale)
        for kind in KINDS:
            data["legacy_response_xx_contrast"][masks[kind]] = value
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
        figure_dir / "small_q_local_limit_error.png",
        figure_dir / "A4_pairing_contrast_vs_q.png",
        figure_dir / "A4_trace_pairing_contrast_vs_q.png",
        figure_dir / "A4_components_vs_q.png",
    ]
    q_values = sorted(set(float(item) for item, a4 in zip(data["q_magnitude"], data["angular_anisotropy_A4_xx"], strict=True) if np.isfinite(a4)))

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
        values.append(float(np.nanmax(data["legacy_response_xx_contrast"][mask])))
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

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in KINDS:
        mask = (data["kind"] == kind) & data["q_to_0_continuity_tested"]
        if np.any(mask):
            ax.plot(data["q_phi"][mask], data["small_q_limit_relative_error"][mask], marker="o", label=kind)
    ax.set_xlabel("q phi (rad)")
    ax.set_ylabel("smallest-q relative error")
    ax.set_title("small-q finite-q bubble continuity")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[4])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    values = []
    for q_value in q_values:
        mask = np.isclose(data["q_magnitude"], q_value)
        values.append(float(np.nanmax(np.abs(data["A4_pairing_contrast"][mask]))))
    ax.plot(q_values, values, marker="o")
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("max |A4 pairing contrast|")
    ax.set_title("A4 pairing contrast")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[5])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    values = []
    for q_value in q_values:
        mask = np.isclose(data["q_magnitude"], q_value)
        values.append(float(np.nanmax(np.abs(data["A4_trace_pairing_contrast"][mask]))))
    ax.plot(q_values, values, marker="o")
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("max |A4 trace pairing contrast|")
    ax.set_title("A4 trace pairing contrast")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[6])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for field, label in (
        ("delta_A4_spm", "delta_A4_spm"),
        ("delta_A4_dwave", "delta_A4_dwave"),
        ("A4_pairing_contrast", "A4_pairing_contrast"),
    ):
        values = []
        for q_value in q_values:
            mask = np.isclose(data["q_magnitude"], q_value)
            values.append(float(np.nanmax(np.abs(data[field][mask]))))
        ax.plot(q_values, values, marker="o", label=label)
    ax.set_xlabel("q magnitude")
    ax.set_ylabel("max abs value")
    ax.set_title("A4 components")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[7])
    plt.close(fig)
    return paths


def _summary_lines(data: dict[str, np.ndarray], args: argparse.Namespace) -> list[str]:
    data = _fill_pairing_contrast(data)
    hook_mask = np.isclose(data["q_magnitude"], 0.0) & ~data["q_to_0_continuity_tested"]
    hook_passed = bool(np.any(hook_mask)) and bool(np.all(data["local_reference_hook_passed"][hook_mask]))
    continuity_mask = data["q_to_0_continuity_tested"]
    continuity_passed = bool(np.any(continuity_mask)) and bool(np.all(data["q_to_0_continuity_passed"][continuity_mask]))
    warning_small_q = bool(np.any(data["diagnostic_status"][continuity_mask] == "warning_small_q_not_smooth"))
    finite_mask = data["q_magnitude"] > 0.0
    anisotropy_signal = bool(np.any(finite_mask)) and bool(np.nanmax(np.abs(data["angular_anisotropy_A4_xx"][finite_mask])) > 1e-8)
    contrast_signal = bool(np.any(finite_mask)) and bool(np.nanmax(np.abs(data["A4_pairing_contrast"][finite_mask])) > 1e-8)
    worth_next = bool(hook_passed and continuity_passed and anisotropy_signal and contrast_signal)
    lines = [
        "# refined finite-q response 角向各向异性诊断摘要",
        "",
        "本轮是 refined finite-q response diagnostic，用于检查 Casimir 几何中的有限 q_parallel 是否能在",
        "不修改 H0、不修改 pairing 的情况下放大 spm/dwave 的 response 层差异，尤其是",
        "dwave 节点相关的角向响应差异。",
        "",
        "本脚本只做 response 层 finite-q diagnostic / prototype，不做 Casimir torque，",
        "也不接入正式 Lifshitz 积分。",
        "q=0 local reference hook 与真正 small-q finite-q bubble continuity 是两件事；",
        "主 pairing contrast 现在使用 A4_pairing_contrast，不再使用 legacy_response_xx_contrast。",
        "",
        "q_magnitude 当前使用 dimensionless BZ momentum，与 k 网格单位一致，不是 SI wavevector。",
        "",
        f"kinds={list(args.kinds)}",
        f"matsubara_list={list(args.matsubara_list)}",
        f"q_list={list(args.q_list)}",
        f"small_q_list={list(args.small_q_list)}",
        f"q_phi_list={list(args.q_phi_list)}",
        f"temperature={args.temperature}",
        f"nk={args.nk}",
        f"delta0={args.delta0}",
        f"eta={args.eta}",
        "",
        f"q0_local_reference_hook_passed={hook_passed}",
        f"small_q_finite_q_bubble_continuity_passed={continuity_passed}",
        f"warning_small_q_not_smooth={warning_small_q}",
        f"finite_q_angular_anisotropy_signal={anisotropy_signal}",
        f"A4_pairing_contrast_signal={contrast_signal}",
        f"worth_next_finite_q_casimir_plumbing_smoke={worth_next}",
        "",
        "## 每个 q 的诊断",
    ]
    for q_value in sorted(set(float(item) for item in data["q_magnitude"])):
        q_mask = np.isclose(data["q_magnitude"], q_value)
        if not np.any(np.isfinite(data["angular_anisotropy_A4_xx"][q_mask])):
            continue
        lines.append(f"- q={q_value:g}")
        for kind in KINDS:
            mask = q_mask & (data["kind"] == kind)
            lines.append(f"  {kind}_max_abs_A4_xx={float(np.nanmax(np.abs(data['angular_anisotropy_A4_xx'][mask]))):.6g}")
        lines.append(f"  max_abs_delta_A4_spm={float(np.nanmax(np.abs(data['delta_A4_spm'][q_mask]))):.6g}")
        lines.append(f"  max_abs_delta_A4_dwave={float(np.nanmax(np.abs(data['delta_A4_dwave'][q_mask]))):.6g}")
        lines.append(f"  max_abs_A4_pairing_contrast={float(np.nanmax(np.abs(data['A4_pairing_contrast'][q_mask]))):.6g}")
        lines.append(f"  max_abs_A4_trace_pairing_contrast={float(np.nanmax(np.abs(data['A4_trace_pairing_contrast'][q_mask]))):.6g}")
    lines.append("")
    lines.append("## small-q continuity")
    if np.any(continuity_mask):
        lines.append(f"small_q_min_relative_error={float(np.nanmin(data['small_q_limit_relative_error'][continuity_mask])):.6g}")
        lines.append(f"small_q_max_relative_error={float(np.nanmax(data['small_q_limit_relative_error'][continuity_mask])):.6g}")
        for status in sorted(set(str(item) for item in data["small_q_limit_status"][continuity_mask])):
            count = int(np.count_nonzero(data["small_q_limit_status"][continuity_mask] == status))
            lines.append(f"- {status}: {count}")
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
    parser.add_argument("--q-list", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--small-q-list", nargs="+", type=float, default=[1e-4, 5e-4, 1e-3, 5e-3])
    parser.add_argument(
        "--q-phi-list",
        nargs="+",
        type=float,
        default=[0.0, 0.3926990817, 0.7853981634, 1.1780972451, 1.5707963268],
    )
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--nk", type=int, default=12)
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
    data = _append_rows(data, _small_q_rows(args))
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
