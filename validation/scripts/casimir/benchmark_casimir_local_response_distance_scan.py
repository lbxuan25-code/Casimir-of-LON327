#!/usr/bin/env python3
"""Distance scan wrapper for the local-response Casimir integral benchmark."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation" / "scripts" / "casimir"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from benchmark_casimir_local_response_integral import (  # noqa: E402
    KINDS,
    N0_POLICY,
    NORMAL_SAMPLING,
    ResponseTensorCache,
    benchmark_casimir_local_response_integral,
)
from casimir_benchmark_config import BENCHMARK_NOTE_PARTS, TORQUE_TOLERANCE  # noqa: E402
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

SCAN_ROOT = ROOT / "validation" / "outputs" / "casimir" / "local_response_integral" / "distance_scan"
DEFAULT_OUTPUT_PREFIX = SCAN_ROOT / "data" / "distance_scan"
DEFAULT_CACHE_DIR = ROOT / "validation" / "cache" / "casimir_local_response" / "response_tensors"
SUMMARY_PATH = SCAN_ROOT / "distance_scan_summary.md"
COMMAND_PATH = SCAN_ROOT / "distance_scan_command.sh"


def progress(message: str) -> None:
    """Print a plain, line-oriented progress message suitable for tee."""

    print(f"[progress] {message}", flush=True)


def _cache_progress_fields(response_cache: ResponseTensorCache | None) -> str:
    if response_cache is None:
        return "cache=disabled"
    return (
        f"cache_entries={response_cache.entry_count()} "
        f"cache_hits={response_cache.hits} "
        f"cache_misses={response_cache.misses} "
        f"cache_writes={response_cache.writes}"
    )


def _full_defaults() -> dict[str, object]:
    return {
        "kinds": list(KINDS),
        "distance_list": [3e-8, 5e-8, 7.5e-8, 1e-7, 1.5e-7, 2e-7],
        "theta_list": [0.0, 0.3926990817, 0.7853981634, 1.1780972451, 1.5707963268],
        "matsubara_max": 64,
        "u_max": 80.0,
        "du": 0.5,
        "phi_num": 32,
        "temperature": 30.0,
        "normal_nk": 96,
        "normal_eta": 1e-4,
        "normal_sampling": "fs_adaptive",
        "normal_refine_factor": 8,
        "bdg_nk": 32,
        "delta0": 0.04,
        "cache_dir": DEFAULT_CACHE_DIR,
        "output_prefix": DEFAULT_OUTPUT_PREFIX,
    }


def _quick_overrides() -> dict[str, object]:
    return {
        "kinds": list(KINDS),
        "distance_list": [5e-8, 1e-7],
        "theta_list": [0.0, 0.7853981634, 1.5707963268],
        "matsubara_max": 2,
        "u_max": 6.0,
        "du": 2.0,
        "phi_num": 8,
        "normal_nk": 12,
        "normal_refine_factor": 2,
        "bdg_nk": 8,
        "include_toy_anisotropic_control": True,
    }


def implied_kparallel_num(u_max: float, du: float) -> int:
    if u_max <= 0.0:
        raise ValueError("u_max must be positive")
    if du <= 0.0:
        raise ValueError("du must be positive")
    return int(u_max / du) + 1


def _format_value(value: object) -> str:
    if isinstance(value, Path):
        try:
            return str(value.relative_to(ROOT))
        except ValueError:
            return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def full_run_command(output_prefix: Path = DEFAULT_OUTPUT_PREFIX) -> str:
    defaults = _full_defaults()
    defaults["output_prefix"] = output_prefix
    parts = ["python", "validation/scripts/casimir/benchmark_casimir_local_response_distance_scan.py"]
    option_map = [
        ("--kinds", "kinds"),
        ("--distance-list", "distance_list"),
        ("--theta-list", "theta_list"),
        ("--matsubara-max", "matsubara_max"),
        ("--u-max", "u_max"),
        ("--du", "du"),
        ("--phi-num", "phi_num"),
        ("--temperature", "temperature"),
        ("--normal-nk", "normal_nk"),
        ("--normal-eta", "normal_eta"),
        ("--normal-sampling", "normal_sampling"),
        ("--normal-refine-factor", "normal_refine_factor"),
        ("--bdg-nk", "bdg_nk"),
        ("--delta0", "delta0"),
        ("--cache-dir", "cache_dir"),
        ("--output-prefix", "output_prefix"),
    ]
    for option, key in option_map:
        value = defaults[key]
        parts.append(option)
        if isinstance(value, list):
            parts.extend(_format_value(item) for item in value)
        else:
            parts.append(_format_value(value))
    parts.append("--use-response-cache")
    return " ".join(shlex.quote(part) for part in parts)


def _write_command_file(command: str) -> Path:
    COMMAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMMAND_PATH.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n\n{command}\n", encoding="utf-8")
    COMMAND_PATH.chmod(0o755)
    return COMMAND_PATH


def _empty_data() -> dict[str, np.ndarray]:
    return {
        "kind": np.array([], dtype="U32"),
        "distance_m": np.array([], dtype=float),
        "theta": np.array([], dtype=float),
        "temperature_K": np.array([], dtype=float),
        "matsubara_max": np.array([], dtype=int),
        "u_max": np.array([], dtype=float),
        "du": np.array([], dtype=float),
        "kparallel_num": np.array([], dtype=int),
        "phi_num": np.array([], dtype=int),
        "energy": np.array([], dtype=complex),
        "torque_fd": np.array([], dtype=float),
        "max_abs_torque_over_theta": np.array([], dtype=float),
        "energy_abs": np.array([], dtype=float),
        "energy_distance_scaling": np.array([], dtype=float),
        "normal_sampling": np.array([], dtype="U24"),
        "normal_nk": np.array([], dtype=int),
        "normal_refine_factor": np.array([], dtype=int),
        "bdg_nk": np.array([], dtype=int),
        "delta0": np.array([], dtype=float),
        "response_cache_used": np.array([], dtype=bool),
        "n0_policy": np.array([], dtype="U16"),
        "local_response": np.array([], dtype=bool),
        "finite_momentum_resolved": np.array([], dtype=bool),
        "benchmark_only": np.array([], dtype=bool),
        "not_final_casimir_conclusion": np.array([], dtype=bool),
        "zero_torque_baseline": np.array([], dtype=bool),
        "warning_possible_spurious_torque": np.array([], dtype=bool),
        "diagnosis": np.array([], dtype="U192"),
        "notes": np.array([], dtype=object),
    }


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


def _parse_bool(value: str) -> bool:
    return value == "True"


def _load_existing(output_prefix: Path) -> dict[str, np.ndarray]:
    npz_path = output_prefix.with_suffix(".npz")
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as loaded:
            return {key: loaded[key] for key in loaded.files}
    csv_path = output_prefix.with_suffix(".csv")
    if not csv_path.exists():
        return _empty_data()
    rows: list[dict[str, object]] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "kind": row["kind"],
                    "distance_m": float(row["distance_m"]),
                    "theta": float(row["theta"]),
                    "temperature_K": float(row["temperature_K"]),
                    "matsubara_max": int(row["matsubara_max"]),
                    "u_max": float(row["u_max"]),
                    "du": float(row["du"]),
                    "kparallel_num": int(row["kparallel_num"]),
                    "phi_num": int(row["phi_num"]),
                    "energy": complex(row["energy"]),
                    "torque_fd": float(row["torque_fd"]),
                    "max_abs_torque_over_theta": float(row["max_abs_torque_over_theta"]),
                    "energy_abs": float(row["energy_abs"]),
                    "energy_distance_scaling": float(row["energy_distance_scaling"]),
                    "normal_sampling": row["normal_sampling"],
                    "normal_nk": int(row["normal_nk"]),
                    "normal_refine_factor": int(row["normal_refine_factor"]),
                    "bdg_nk": int(row["bdg_nk"]),
                    "delta0": float(row["delta0"]),
                    "response_cache_used": _parse_bool(row["response_cache_used"]),
                    "n0_policy": row["n0_policy"],
                    "local_response": _parse_bool(row["local_response"]),
                    "finite_momentum_resolved": _parse_bool(row["finite_momentum_resolved"]),
                    "benchmark_only": _parse_bool(row["benchmark_only"]),
                    "not_final_casimir_conclusion": _parse_bool(row["not_final_casimir_conclusion"]),
                    "zero_torque_baseline": _parse_bool(row["zero_torque_baseline"]),
                    "warning_possible_spurious_torque": _parse_bool(row["warning_possible_spurious_torque"]),
                    "diagnosis": row["diagnosis"],
                    "notes": tuple(row["notes"].split(" | ")),
                }
            )
    return _append_rows(_empty_data(), rows)


def _expected_kinds(args: argparse.Namespace) -> set[str]:
    kinds = set(str(item) for item in args.kinds)
    if args.include_toy_anisotropic_control:
        kinds.add("toy_anisotropic")
    return kinds


def _distance_done(data: dict[str, np.ndarray], distance: float, args: argparse.Namespace) -> bool:
    if data["kind"].size == 0:
        return False
    mask = np.isclose(data["distance_m"], float(distance))
    if not np.any(mask):
        return False
    expected = {(kind, float(theta)) for kind in _expected_kinds(args) for theta in args.theta_list}
    seen = {(str(kind), float(theta)) for kind, theta in zip(data["kind"][mask], data["theta"][mask])}
    return expected <= seen


def _rows_from_integral(raw: dict[str, np.ndarray], args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    kparallel_num = implied_kparallel_num(args.u_max, args.du)
    for index in range(raw["kind"].size):
        kind = str(raw["kind"][index])
        max_torque = float(raw["max_abs_torque_over_theta"][index])
        zero_baseline = bool(kind in KINDS and max_torque <= TORQUE_TOLERANCE)
        spurious = bool(kind in KINDS and max_torque > TORQUE_TOLERANCE)
        rows.append(
            {
                "kind": kind,
                "distance_m": float(raw["distance_m"][index]),
                "theta": float(raw["theta"][index]),
                "temperature_K": float(raw["temperature_K"][index]),
                "matsubara_max": int(raw["matsubara_max"][index]),
                "u_max": float(args.u_max),
                "du": float(args.du),
                "kparallel_num": kparallel_num,
                "phi_num": int(raw["phi_num"][index]),
                "energy": complex(raw["energy"][index]),
                "torque_fd": float(raw["torque_fd"][index]),
                "max_abs_torque_over_theta": max_torque,
                "energy_abs": float(abs(raw["energy"][index])),
                "energy_distance_scaling": np.nan,
                "normal_sampling": str(raw["normal_sampling"][index]),
                "normal_nk": int(raw["normal_nk"][index]),
                "normal_refine_factor": int(raw["normal_refine_factor"][index]),
                "bdg_nk": int(raw["bdg_nk"][index]),
                "delta0": float(raw["delta0"][index]),
                "response_cache_used": bool(args.use_response_cache),
                "n0_policy": N0_POLICY,
                "local_response": True,
                "finite_momentum_resolved": False,
                "benchmark_only": True,
                "not_final_casimir_conclusion": True,
                "zero_torque_baseline": zero_baseline,
                "warning_possible_spurious_torque": spurious,
                "diagnosis": str(raw["diagnosis"][index]),
                "notes": (*BENCHMARK_NOTE_PARTS, "u=k_parallel*d at fixed du"),
            }
        )
    return rows


def _refresh_torque_flags(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if data["kind"].size == 0:
        return data
    data = {key: np.array(value, copy=True) for key, value in data.items()}
    for index, kind_value in enumerate(data["kind"]):
        kind = str(kind_value)
        if kind not in KINDS:
            continue
        max_torque = float(data["max_abs_torque_over_theta"][index])
        zero_baseline = bool(max_torque <= TORQUE_TOLERANCE)
        data["zero_torque_baseline"][index] = zero_baseline
        data["warning_possible_spurious_torque"][index] = not zero_baseline
        diagnosis_parts = [
            part
            for part in str(data["diagnosis"][index]).split(";")
            if part not in {"zero_torque_baseline", "warning_possible_spurious_torque"} and part
        ]
        diagnosis_parts.insert(0, "zero_torque_baseline" if zero_baseline else "warning_possible_spurious_torque")
        data["diagnosis"][index] = ";".join(diagnosis_parts)
        data["notes"][index] = (*BENCHMARK_NOTE_PARTS, "u=k_parallel*d at fixed du")
    return data


def _recompute_energy_distance_scaling(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if data["kind"].size == 0:
        return data
    data = {key: np.array(value, copy=True) for key, value in data.items()}
    data["energy_distance_scaling"][:] = np.nan
    for kind in sorted(set(str(item) for item in data["kind"])):
        for theta in sorted(set(float(item) for item in data["theta"])):
            mask = (data["kind"] == kind) & np.isclose(data["theta"], theta)
            indices = np.where(mask)[0]
            if indices.size == 0:
                continue
            distances = data["distance_m"][indices]
            reference_index = indices[int(np.argmin(distances))]
            reference = float(data["energy_abs"][reference_index])
            scale = reference if reference > 0.0 else 1.0
            for index in indices:
                data["energy_distance_scaling"][index] = float(data["energy_abs"][index] / scale)
    return data


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    return value


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path]:
    data = _recompute_energy_distance_scaling(data)
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


def save_figures(data: dict[str, np.ndarray], include_toy: bool) -> list[Path]:
    if data["kind"].size == 0:
        return []
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    figure_dir = SCAN_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / "energy_vs_distance.png",
        figure_dir / "max_abs_torque_vs_distance.png",
        figure_dir / "energy_vs_theta_by_distance.png",
        figure_dir / "torque_vs_theta_by_distance.png",
    ]
    kinds = [kind for kind in dict.fromkeys(str(item) for item in data["kind"]) if kind != "toy_anisotropic"]
    distances = sorted(set(float(item) for item in data["distance_m"]))

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in kinds:
        values = []
        for distance in distances:
            mask = (data["kind"] == kind) & np.isclose(data["distance_m"], distance)
            values.append(float(np.nanmax(data["energy_abs"][mask])))
        ax.plot(distances, values, marker="o", label=kind)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("max |energy|")
    ax.set_title("local-response energy distance scan")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[0])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in kinds:
        values = []
        for distance in distances:
            mask = (data["kind"] == kind) & np.isclose(data["distance_m"], distance)
            values.append(float(np.nanmax(data["max_abs_torque_over_theta"][mask])))
        ax.plot(distances, values, marker="o", label=kind)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-30)
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("max |torque|")
    ax.set_title("zero-torque baseline distance scan")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in kinds:
        for distance in distances:
            mask = (data["kind"] == kind) & np.isclose(data["distance_m"], distance)
            ax.plot(data["theta"][mask], data["energy"][mask].real, marker="o", label=f"{kind}, d={distance:g}")
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("energy")
    ax.set_title("energy vs theta by distance")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[2])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in kinds:
        for distance in distances:
            mask = (data["kind"] == kind) & np.isclose(data["distance_m"], distance)
            ax.plot(data["theta"][mask], data["torque_fd"][mask], marker="o", label=f"{kind}, d={distance:g}")
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("torque")
    ax.set_title("torque vs theta by distance")
    style_publication_axis(ax)
    save_publication_figure(fig, paths[3])
    plt.close(fig)

    if include_toy and np.any(data["kind"] == "toy_anisotropic"):
        toy_path = figure_dir / "toy_torque_vs_distance.png"
        fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        values = []
        for distance in distances:
            mask = (data["kind"] == "toy_anisotropic") & np.isclose(data["distance_m"], distance)
            values.append(float(np.nanmax(np.abs(data["torque_fd"][mask]))))
        ax.plot(distances, values, marker="o", label="toy_anisotropic")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("distance (m)")
        ax.set_ylabel("max |toy torque|")
        ax.set_title("toy anisotropic plumbing control")
        style_publication_axis(ax)
        save_publication_figure(fig, toy_path)
        plt.close(fig)
        paths.append(toy_path)
    return paths


def _full_scan_completed(data: dict[str, np.ndarray], args: argparse.Namespace) -> bool:
    if args.quick:
        return False
    return all(_distance_done(data, distance, args) for distance in args.distance_list)


def _kind_zero_baseline(data: dict[str, np.ndarray], kind: str) -> bool:
    mask = data["kind"] == kind
    return bool(np.any(mask)) and bool(np.all(data["zero_torque_baseline"][mask]))


def _toy_passed(data: dict[str, np.ndarray]) -> bool:
    mask = data["kind"] == "toy_anisotropic"
    return bool(np.any(mask)) and bool(np.nanmax(np.abs(data["torque_fd"][mask])) > TORQUE_TOLERANCE)


def _summary_lines(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    full_completed: bool,
    response_cache: ResponseTensorCache | None,
) -> list[str]:
    zero_by_kind = {kind: _kind_zero_baseline(data, kind) for kind in KINDS}
    toy_passed = _toy_passed(data)
    all_zero = all(zero_by_kind.values())
    ready = bool(full_completed and all_zero and toy_passed)
    cache_entries = "not_recorded" if response_cache is None else str(response_cache.entry_count())
    cache_hits = "not_recorded" if response_cache is None else str(response_cache.hits)
    cache_misses = "not_recorded" if response_cache is None else str(response_cache.misses)
    cache_writes = "not_recorded" if response_cache is None else str(response_cache.writes)
    lines = [
        "# Local-response distance scan benchmark 摘要",
        "",
        "本脚本用于在已通过数值稳定性检查的推荐参数下，扫描距离 d，建立当前",
        "local-response zero-torque baseline 的距离依赖基准。它只复用已有",
        "local-response integral benchmark，不引入真实各向异性机制，也不输出正式",
        "Casimir torque 结论。",
        "",
        f"full_run_command = `{command}`",
        f"quick_test_only={bool(args.quick)}",
        f"full_distance_scan_completed={full_completed}",
        f"response_cache_used={bool(args.use_response_cache)}",
        f"response_cache_entries={cache_entries}",
        f"response_cache_rebuilt={bool(args.rebuild_response_cache)}",
        f"response_cache_hits={cache_hits}",
        f"response_cache_misses={cache_misses}",
        f"response_cache_writes={cache_writes}",
        "local_response=True",
        "finite_momentum_resolved=False",
        "n0_policy=skip",
        "benchmark_only=True",
        "not_final_Casimir_conclusion=True",
        "",
        "## 推荐参数",
        f"- matsubara_max={args.matsubara_max}",
        f"- u_max={args.u_max:g}",
        f"- du={args.du:g}",
        f"- kparallel_num={implied_kparallel_num(args.u_max, args.du)}",
        f"- phi_num={args.phi_num}",
        f"- normal_sampling={args.normal_sampling}",
        f"- normal_nk={args.normal_nk}",
        f"- normal_refine_factor={args.normal_refine_factor}",
        f"- bdg_nk={args.bdg_nk}",
        f"- delta0={args.delta0:g}",
        "",
        "## 距离扫描范围",
        "- " + ", ".join(f"{distance:g}" for distance in args.distance_list),
        "",
        "## zero-torque baseline",
    ]
    for kind in KINDS:
        lines.append(f"- {kind}: zero_torque_baseline={zero_by_kind[kind]}")
    lines.extend(
        [
            "",
            "## toy anisotropic control",
            f"toy_anisotropic_control_enabled={bool(args.include_toy_anisotropic_control)}",
            f"toy_anisotropic_control_passed={toy_passed}",
            "",
            "## 结论边界",
            "当前仍不是正式 Casimir 结论，原因是：",
            "- local_response=True",
            "- finite_momentum_resolved=False",
            "- n0_policy=skip",
            "- benchmark_only=True",
            f"local_response_distance_scan_benchmark_ready={ready}",
            f"ready_for_anisotropy_mechanism_benchmark={ready}",
            "not_final_Casimir_conclusion=True",
        ]
    )
    if args.quick and not full_completed:
        lines.extend(
            [
                "quick_test_only=True",
                "no_distance_scan_conclusion=True",
                "full_run_pending_user_terminal=True",
            ]
        )
    elif not full_completed:
        lines.extend(
            [
                "partial_distance_scan_only=True",
                "no_distance_scan_conclusion=True",
                "full_run_pending_user_terminal=True",
            ]
        )
    return lines


def _summary_path(output_prefix: Path) -> Path:
    if output_prefix.resolve() == DEFAULT_OUTPUT_PREFIX.resolve():
        return SUMMARY_PATH
    return output_prefix.parent / "distance_scan_summary.md"


def write_summary(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    full_completed: bool,
    response_cache: ResponseTensorCache | None,
) -> Path:
    summary_path = _summary_path(args.output_prefix)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(_summary_lines(data, args, command, full_completed, response_cache)) + "\n",
        encoding="utf-8",
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    defaults = _full_defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=defaults["kinds"])
    parser.add_argument("--distance-list", nargs="+", type=float, default=defaults["distance_list"])
    parser.add_argument("--theta-list", nargs="+", type=float, default=defaults["theta_list"])
    parser.add_argument("--matsubara-max", type=int, default=defaults["matsubara_max"])
    parser.add_argument("--u-max", type=float, default=defaults["u_max"])
    parser.add_argument("--du", type=float, default=defaults["du"])
    parser.add_argument("--phi-num", type=int, default=defaults["phi_num"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--normal-nk", type=int, default=defaults["normal_nk"])
    parser.add_argument("--normal-eta", type=float, default=defaults["normal_eta"])
    parser.add_argument("--normal-sampling", choices=NORMAL_SAMPLING, default=defaults["normal_sampling"])
    parser.add_argument("--normal-refine-factor", type=int, default=defaults["normal_refine_factor"])
    parser.add_argument("--bdg-nk", type=int, default=defaults["bdg_nk"])
    parser.add_argument("--delta0", type=float, default=defaults["delta0"])
    parser.add_argument("--include-toy-anisotropic-control", action="store_true")
    parser.add_argument("--use-response-cache", action="store_true")
    parser.add_argument("--rebuild-response-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=defaults["cache_dir"])
    parser.add_argument("--output-prefix", type=Path, default=defaults["output_prefix"])
    args = parser.parse_args()
    if args.quick:
        for key, value in _quick_overrides().items():
            setattr(args, key, value)
    return args


def main() -> None:
    args = parse_args()
    command = full_run_command(DEFAULT_OUTPUT_PREFIX)
    if args.dry_run:
        print(command)
        return
    _write_command_file(command)
    response_cache = (
        ResponseTensorCache(args.cache_dir, use=True, rebuild=args.rebuild_response_cache)
        if args.use_response_cache
        else None
    )
    data = _load_existing(args.output_prefix) if args.resume else _empty_data()
    kparallel_num = implied_kparallel_num(args.u_max, args.du)
    total_start = perf_counter()
    expected_kinds = sorted(_expected_kinds(args))
    progress(
        "scan_start "
        f"distances={len(args.distance_list)} "
        f"kinds={','.join(expected_kinds)} "
        f"theta_count={len(args.theta_list)} "
        f"matsubara=1-{args.matsubara_max} "
        f"resume={bool(args.resume)} "
        f"detailed_progress={bool(args.progress)} "
        f"{_cache_progress_fields(response_cache)}"
    )
    for distance_index, distance in enumerate(args.distance_list, start=1):
        if args.resume and _distance_done(data, distance, args):
            progress(
                f"distance_skip {distance_index}/{len(args.distance_list)} "
                f"d={distance:g} reason=resume_complete {_cache_progress_fields(response_cache)}"
            )
            continue
        distance_start = perf_counter()
        progress(
            f"distance_start {distance_index}/{len(args.distance_list)} "
            f"d={distance:g} {_cache_progress_fields(response_cache)}"
        )
        integral_progress = None
        if args.progress:
            integral_progress = lambda message, current_distance=distance: progress(
                f"d={current_distance:g} {message}"
            )
        raw = benchmark_casimir_local_response_integral(
            kinds=list(args.kinds),
            distance_list=[float(distance)],
            theta_list=list(args.theta_list),
            matsubara_min=1,
            matsubara_max=args.matsubara_max,
            kparallel_num=kparallel_num,
            kparallel_max_factor=args.u_max,
            phi_num=args.phi_num,
            temperature_K=args.temperature,
            normal_nk=args.normal_nk,
            normal_eta_eV=args.normal_eta,
            normal_sampling=args.normal_sampling,
            normal_refine_factor=args.normal_refine_factor,
            bdg_nk=args.bdg_nk,
            delta0_eV=args.delta0,
            include_toy_anisotropic_control=args.include_toy_anisotropic_control,
            response_cache=response_cache,
            progress_callback=integral_progress,
        )
        data = _append_rows(data, _rows_from_integral(raw, args))
        save_outputs(data, args.output_prefix)
        progress(
            f"distance_done {distance_index}/{len(args.distance_list)} "
            f"d={distance:g} elapsed_s={perf_counter() - distance_start:.3f} "
            f"{_cache_progress_fields(response_cache)}"
        )
    data = _refresh_torque_flags(_recompute_energy_distance_scaling(data))
    npz_path, csv_path = save_outputs(data, args.output_prefix)
    figure_paths = save_figures(data, args.include_toy_anisotropic_control)
    full_completed = _full_scan_completed(data, args)
    summary_path = write_summary(data, args, command, full_completed, response_cache)
    progress(
        f"scan_done elapsed_s={perf_counter() - total_start:.3f} "
        f"completed_distances={sum(_distance_done(data, distance, args) for distance in args.distance_list)}"
        f"/{len(args.distance_list)} {_cache_progress_fields(response_cache)}"
    )
    print(f"npz_path = {npz_path}")
    print(f"csv_path = {csv_path}")
    print(f"summary_path = {summary_path}")
    print(f"command_path = {COMMAND_PATH}")
    print("figure_paths = " + ", ".join(str(path) for path in figure_paths))
    for kind in sorted(set(str(item) for item in data["kind"])):
        mask = data["kind"] == kind
        print(f"{kind}_max_abs_torque = {float(np.nanmax(np.abs(data['torque_fd'][mask])))}")
    print("note = local-response distance scan benchmark only; not a final Casimir conclusion.")


if __name__ == "__main__":
    main()
