#!/usr/bin/env python3
"""Refined convergence diagnostics for local-response Casimir integral blockers.

This runner fixes the old k_parallel cutoff diagnostic by scanning the
dimensionless cutoff u = k_parallel * distance at fixed du. It also extends the
Matsubara cutoff scan. The underlying local-response integral is delegated to
benchmark_casimir_local_response_integral.py.
"""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_casimir_local_response_integral import (  # noqa: E402
    KINDS,
    N0_POLICY,
    ResponseTensorCache,
    benchmark_casimir_local_response_integral,
)

REFINED_ROOT = ROOT / "outputs" / "casimir" / "local_response_integral" / "refined_convergence"
DEFAULT_OUTPUT_PREFIX = REFINED_ROOT / "data" / "refined_local_convergence"
DEFAULT_CACHE_DIR = ROOT / "outputs" / "casimir" / "local_response_integral" / "cache"
SUMMARY_PATH = REFINED_ROOT / "refined_convergence_summary.md"
COMMAND_PATH = REFINED_ROOT / "refined_convergence_command.sh"
SCAN_ORDER = ("cutoff", "matsubara")
ONLY_SCAN_CHOICES = (*SCAN_ORDER, "all")
RATIO_EPS = 1e-300
TORQUE_TOLERANCE = 1e-20


def _full_defaults() -> dict[str, object]:
    return {
        "kinds": list(KINDS),
        "distance": 5e-8,
        "theta_list": [0.0, 0.3926990817, 0.7853981634, 1.1780972451, 1.5707963268],
        "energy_theta_list": [0.0],
        "torque_check_theta_list": [0.0, 0.7853981634, 1.5707963268],
        "u_max_list": [20.0, 40.0, 60.0, 80.0],
        "du": 0.5,
        "matsubara_max_list": [24, 32, 48, 64],
        "temperature": 30.0,
        "normal_nk": 96,
        "normal_eta": 1e-4,
        "normal_sampling": "fs_adaptive",
        "normal_refine_factor": 8,
        "bdg_nk": 32,
        "delta0": 0.04,
        "phi_num": 32,
        "output_prefix": DEFAULT_OUTPUT_PREFIX,
        "cache_dir": DEFAULT_CACHE_DIR,
    }


def _quick_overrides() -> dict[str, object]:
    return {
        "kinds": list(KINDS),
        "u_max_list": [4.0, 6.0],
        "du": 2.0,
        "matsubara_max_list": [1, 2],
        "theta_list": [0.0, 0.7853981634, 1.5707963268],
        "energy_theta_list": [0.0],
        "torque_check_theta_list": [0.0, 0.7853981634, 1.5707963268],
        "phi_num": 8,
        "normal_nk": 12,
        "normal_refine_factor": 2,
        "bdg_nk": 8,
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
    parts = ["python", "scripts/refine_casimir_local_convergence_blockers.py"]
    option_map = [
        ("--kinds", "kinds"),
        ("--distance", "distance"),
        ("--theta-list", "theta_list"),
        ("--energy-theta-list", "energy_theta_list"),
        ("--torque-check-theta-list", "torque_check_theta_list"),
        ("--u-max-list", "u_max_list"),
        ("--du", "du"),
        ("--matsubara-max-list", "matsubara_max_list"),
        ("--temperature", "temperature"),
        ("--normal-nk", "normal_nk"),
        ("--normal-eta", "normal_eta"),
        ("--normal-sampling", "normal_sampling"),
        ("--normal-refine-factor", "normal_refine_factor"),
        ("--bdg-nk", "bdg_nk"),
        ("--delta0", "delta0"),
        ("--phi-num", "phi_num"),
        ("--output-prefix", "output_prefix"),
        ("--cache-dir", "cache_dir"),
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
        "scan_type": np.array([], dtype="U24"),
        "kind": np.array([], dtype="U32"),
        "distance_m": np.array([], dtype=float),
        "theta_grid_size": np.array([], dtype=int),
        "matsubara_max": np.array([], dtype=int),
        "u_max": np.array([], dtype=float),
        "du": np.array([], dtype=float),
        "implied_kparallel_num": np.array([], dtype=int),
        "kparallel_num": np.array([], dtype=int),
        "kparallel_max_factor": np.array([], dtype=float),
        "kparallel_max": np.array([], dtype=float),
        "phi_num": np.array([], dtype=int),
        "energy": np.array([], dtype=float),
        "max_abs_torque_over_theta": np.array([], dtype=float),
        "last_two_relative_change": np.array([], dtype=float),
        "tail_shell_indicator": np.array([], dtype=float),
        "matsubara_tail_indicator": np.array([], dtype=float),
        "cutoff_status": np.array([], dtype="U64"),
        "matsubara_status": np.array([], dtype="U64"),
        "diagnosis": np.array([], dtype="U192"),
        "local_response": np.array([], dtype=bool),
        "finite_q_resolved": np.array([], dtype=bool),
        "n0_policy": np.array([], dtype="U16"),
        "benchmark_only": np.array([], dtype=bool),
        "not_final_casimir_conclusion": np.array([], dtype=bool),
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
                    "scan_type": row["scan_type"],
                    "kind": row["kind"],
                    "distance_m": float(row["distance_m"]),
                    "theta_grid_size": int(row["theta_grid_size"]),
                    "matsubara_max": int(row["matsubara_max"]),
                    "u_max": float(row["u_max"]),
                    "du": float(row["du"]),
                    "implied_kparallel_num": int(row["implied_kparallel_num"]),
                    "kparallel_num": int(row["kparallel_num"]),
                    "kparallel_max_factor": float(row["kparallel_max_factor"]),
                    "kparallel_max": float(row["kparallel_max"]),
                    "phi_num": int(row["phi_num"]),
                    "energy": float(row["energy"]),
                    "max_abs_torque_over_theta": float(row["max_abs_torque_over_theta"]),
                    "last_two_relative_change": float(row["last_two_relative_change"]),
                    "tail_shell_indicator": float(row["tail_shell_indicator"]),
                    "matsubara_tail_indicator": float(row["matsubara_tail_indicator"]),
                    "cutoff_status": row["cutoff_status"],
                    "matsubara_status": row["matsubara_status"],
                    "diagnosis": row["diagnosis"],
                    "local_response": row["local_response"] == "True",
                    "finite_q_resolved": row["finite_q_resolved"] == "True",
                    "n0_policy": row["n0_policy"],
                    "benchmark_only": row["benchmark_only"] == "True",
                    "not_final_casimir_conclusion": row["not_final_casimir_conclusion"] == "True",
                    "notes": tuple(row["notes"].split(" | ")),
                }
            )
    return _append_rows(_empty_data(), rows)


def _relative_change(value: float, reference: float) -> float:
    if abs(value) < RATIO_EPS and abs(reference) < RATIO_EPS:
        return 0.0
    return float(abs(value - reference) / (abs(reference) + RATIO_EPS))


def _status_from_change(change: float) -> str:
    if np.isfinite(change) and change < 0.02:
        return "candidate_converged"
    if np.isfinite(change) and change < 0.05:
        return "loose_converged"
    return "not_converged"


def _selected_scans(only_scan: str) -> tuple[str, ...]:
    if only_scan == "all":
        return SCAN_ORDER
    return (only_scan,)


def _point_done(data: dict[str, np.ndarray], scan_type: str, setting: int | float, args: argparse.Namespace) -> bool:
    if data["kind"].size == 0:
        return False
    mask = data["scan_type"] == scan_type
    if scan_type == "cutoff":
        mask &= np.isclose(data["u_max"], float(setting))
    elif scan_type == "matsubara":
        mask &= data["matsubara_max"] == int(setting)
    return set(str(item) for item in data["kind"][mask]) >= set(args.kinds)


def _scan_settings(args: argparse.Namespace, scan_type: str) -> list[int | float]:
    if scan_type == "cutoff":
        return sorted(float(item) for item in args.u_max_list)
    if scan_type == "matsubara":
        return sorted(int(item) for item in args.matsubara_max_list)
    raise ValueError("unknown scan_type")


def _benchmark_rows(
    *,
    scan_type: str,
    args: argparse.Namespace,
    matsubara_max: int,
    u_max: float,
    response_cache: ResponseTensorCache | None,
) -> list[dict[str, object]]:
    kparallel_num = implied_kparallel_num(u_max, args.du)
    energy_integral = benchmark_casimir_local_response_integral(
        kinds=args.kinds,
        distance_list=[args.distance],
        theta_list=args.energy_theta_list,
        matsubara_min=1,
        matsubara_max=matsubara_max,
        kparallel_num=kparallel_num,
        kparallel_max_factor=u_max,
        phi_num=args.phi_num,
        temperature_K=args.temperature,
        normal_nk=args.normal_nk,
        normal_eta_eV=args.normal_eta,
        normal_sampling=args.normal_sampling,
        normal_refine_factor=args.normal_refine_factor,
        bdg_nk=args.bdg_nk,
        delta0_eV=args.delta0,
        response_cache=response_cache,
    )
    torque_integral = benchmark_casimir_local_response_integral(
        kinds=args.kinds,
        distance_list=[args.distance],
        theta_list=args.torque_check_theta_list,
        matsubara_min=1,
        matsubara_max=matsubara_max,
        kparallel_num=kparallel_num,
        kparallel_max_factor=u_max,
        phi_num=args.phi_num,
        temperature_K=args.temperature,
        normal_nk=args.normal_nk,
        normal_eta_eV=args.normal_eta,
        normal_sampling=args.normal_sampling,
        normal_refine_factor=args.normal_refine_factor,
        bdg_nk=args.bdg_nk,
        delta0_eV=args.delta0,
        response_cache=response_cache,
    )
    rows = []
    for kind in sorted(set(str(item) for item in energy_integral["kind"])):
        energy_mask = energy_integral["kind"] == kind
        torque_mask = torque_integral["kind"] == kind
        rows.append(
            {
                "scan_type": scan_type,
                "kind": kind,
                "distance_m": args.distance,
                "theta_grid_size": len(args.energy_theta_list),
                "matsubara_max": matsubara_max,
                "u_max": u_max,
                "du": args.du,
                "implied_kparallel_num": kparallel_num,
                "kparallel_num": kparallel_num,
                "kparallel_max_factor": u_max,
                "kparallel_max": float(np.nanmax(energy_integral["kparallel_max"][energy_mask])),
                "phi_num": args.phi_num,
                "energy": float(np.nanmax(np.abs(energy_integral["energy"][energy_mask]))),
                "max_abs_torque_over_theta": float(np.nanmax(np.abs(torque_integral["torque_fd"][torque_mask]))),
                "last_two_relative_change": np.nan,
                "tail_shell_indicator": np.nan,
                "matsubara_tail_indicator": (
                    np.nan
                    if np.all(np.isnan(energy_integral["matsubara_tail_indicator"][energy_mask]))
                    else float(np.nanmax(energy_integral["matsubara_tail_indicator"][energy_mask]))
                ),
                "cutoff_status": "not_evaluated",
                "matsubara_status": "not_evaluated",
                "diagnosis": "not_final_casimir_conclusion",
                "local_response": True,
                "finite_q_resolved": False,
                "n0_policy": N0_POLICY,
                "benchmark_only": True,
                "not_final_casimir_conclusion": True,
                "notes": (
                    "refined local-response convergence benchmark only",
                    "n=0 policy: skip",
                    "finite-q response not included",
                    "not a final Casimir conclusion",
                    "cutoff scan uses u=k_parallel*d at fixed du",
                ),
            }
        )
    return rows


def _rows_for_point(
    scan_type: str,
    setting: int | float,
    args: argparse.Namespace,
    response_cache: ResponseTensorCache | None,
) -> list[dict[str, object]]:
    if scan_type == "cutoff":
        return _benchmark_rows(
            scan_type=scan_type,
            args=args,
            matsubara_max=max(args.matsubara_max_list),
            u_max=float(setting),
            response_cache=response_cache,
        )
    if scan_type == "matsubara":
        return _benchmark_rows(
            scan_type=scan_type,
            args=args,
            matsubara_max=int(setting),
            u_max=max(float(item) for item in args.u_max_list),
            response_cache=response_cache,
        )
    raise ValueError("unknown scan_type")


def _setting_value(data: dict[str, np.ndarray], index: int, scan_type: str) -> float:
    if scan_type == "cutoff":
        return float(data["u_max"][index])
    if scan_type == "matsubara":
        return float(data["matsubara_max"][index])
    raise ValueError("unknown scan_type")


def _recompute_status(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if data["kind"].size == 0:
        return data
    data = {key: np.array(value, copy=True) for key, value in data.items()}
    data["last_two_relative_change"][:] = np.nan
    data["tail_shell_indicator"][:] = np.nan
    data["cutoff_status"][:] = "not_evaluated"
    data["matsubara_status"][:] = "not_evaluated"
    for scan_type in SCAN_ORDER:
        for kind in sorted(set(str(item) for item in data["kind"])):
            indices = np.where((data["scan_type"] == scan_type) & (data["kind"] == kind))[0]
            if indices.size == 0:
                continue
            ordered = sorted(indices, key=lambda index: _setting_value(data, int(index), scan_type))
            last_change = np.nan
            if len(ordered) >= 2:
                last_change = _relative_change(float(data["energy"][ordered[-2]]), float(data["energy"][ordered[-1]]))
            status = _status_from_change(last_change)
            for index in ordered:
                data["last_two_relative_change"][index] = last_change
                if scan_type == "cutoff":
                    data["tail_shell_indicator"][index] = last_change
                    data["cutoff_status"][index] = status
                    if index == ordered[-1] and status == "not_converged":
                        data["cutoff_status"][index] = "cutoff_not_converged"
                else:
                    data["matsubara_status"][index] = status
                    tail = float(data["matsubara_tail_indicator"][index])
                    if index == ordered[-1] and np.isfinite(tail) and tail > 0.05:
                        data["matsubara_status"][index] = "matsubara_not_converged"
                diagnosis_parts = []
                if data["max_abs_torque_over_theta"][index] <= TORQUE_TOLERANCE:
                    diagnosis_parts.append("zero_torque_baseline")
                else:
                    diagnosis_parts.append("warning_possible_spurious_torque")
                diagnosis_parts.append("not_final_casimir_conclusion")
                data["diagnosis"][index] = ";".join(diagnosis_parts)
    return data


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    return value


def save_checkpoint(data: dict[str, np.ndarray], output_prefix: Path) -> None:
    data = _recompute_status(data)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_prefix.with_suffix(".npz"), **data)
    fieldnames = list(_empty_data())
    with output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(data["kind"].size):
            writer.writerow({name: _csv_value(data[name][index]) for name in fieldnames})


def _completed_expected_points(data: dict[str, np.ndarray], args: argparse.Namespace, scans: tuple[str, ...]) -> bool:
    for scan_type in scans:
        for setting in _scan_settings(args, scan_type):
            if not _point_done(data, scan_type, setting, args):
                return False
    return True


def _has_scan_at_max_setting(data: dict[str, np.ndarray], scan_type: str, setting: int | float) -> bool:
    if data["kind"].size == 0:
        return False
    mask = data["scan_type"] == scan_type
    if scan_type == "cutoff":
        mask &= np.isclose(data["u_max"], float(setting))
    elif scan_type == "matsubara":
        mask &= data["matsubara_max"] == int(setting)
    else:
        raise ValueError("unknown scan_type")
    return set(str(item) for item in data["kind"][mask]) >= set(KINDS)


def _full_run_completed(data: dict[str, np.ndarray], args: argparse.Namespace) -> bool:
    if args.quick:
        return False
    max_u = max(float(item) for item in args.u_max_list)
    max_matsubara = max(int(item) for item in args.matsubara_max_list)
    return _has_scan_at_max_setting(data, "cutoff", max_u) and _has_scan_at_max_setting(
        data,
        "matsubara",
        max_matsubara,
    )


def _latest_statuses(data: dict[str, np.ndarray], scan_type: str, args: argparse.Namespace) -> list[str]:
    statuses = []
    for kind in args.kinds:
        mask = (data["scan_type"] == scan_type) & (data["kind"] == kind)
        if not np.any(mask):
            statuses.append("missing")
            continue
        indices = np.where(mask)[0]
        latest = int(max(indices, key=lambda index: _setting_value(data, int(index), scan_type)))
        field = "cutoff_status" if scan_type == "cutoff" else "matsubara_status"
        statuses.append(str(data[field][latest]))
    return statuses


def _at_least_loose(status: str) -> bool:
    return status in {"candidate_converged", "loose_converged"}


def _summary_lines(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    full_completed: bool,
    response_cache: ResponseTensorCache | None,
) -> list[str]:
    spurious = any("warning_possible_spurious_torque" in str(item) for item in data["diagnosis"])
    cutoff_statuses = _latest_statuses(data, "cutoff", args)
    matsubara_statuses = _latest_statuses(data, "matsubara", args)
    clean_cutoff_converged = all(_at_least_loose(status) for status in cutoff_statuses)
    extended_matsubara_converged = all(_at_least_loose(status) for status in matsubara_statuses)
    zero_baseline = bool(data["kind"].size) and all("zero_torque_baseline" in str(item) for item in data["diagnosis"])
    ready = bool(
        full_completed
        and clean_cutoff_converged
        and extended_matsubara_converged
        and zero_baseline
        and not spurious
    )
    lines = [
        "# Refined Local-Response Convergence Summary",
        "",
        "This is a refined convergence benchmark, not a final Casimir conclusion.",
        "old_cutoff_scan_issue = fixed kparallel_num caused changing du when cutoff increased",
        "new_cutoff_scan = u=k_parallel*d with fixed du",
        f"full_run_command = `{command}`",
        f"quick_test_only = {bool(args.quick)}",
        f"full_run_completed = {full_completed}",
        f"response_cache_used={bool(args.use_response_cache)}",
        f"response_cache_entries={0 if response_cache is None else response_cache.entry_count()}",
        f"response_cache_rebuilt={bool(args.rebuild_response_cache)}",
        f"response_cache_hits={0 if response_cache is None else response_cache.hits}",
        f"response_cache_misses={0 if response_cache is None else response_cache.misses}",
        f"response_cache_writes={0 if response_cache is None else response_cache.writes}",
        "local_response=True",
        "finite_q_resolved=False",
        "n0_policy=skip",
        "benchmark_only=True",
        "not_final_casimir_conclusion=True",
        "",
        "## Cutoff Scan",
    ]
    for kind in args.kinds:
        mask = (data["scan_type"] == "cutoff") & (data["kind"] == kind)
        if not np.any(mask):
            lines.append(f"- cutoff/{kind}: u_max=nan, du={args.du:g}, last_two_relative_change=nan, cutoff_status=missing")
            continue
        indices = np.where(mask)[0]
        latest = int(max(indices, key=lambda index: _setting_value(data, int(index), "cutoff")))
        lines.append(
            f"- cutoff/{kind}: u_max={data['u_max'][latest]:g}, du={data['du'][latest]:g}, "
            f"implied_kparallel_num={int(data['implied_kparallel_num'][latest])}, "
            f"tail_shell_indicator={data['tail_shell_indicator'][latest]:.6g}, "
            f"last_two_relative_change={data['last_two_relative_change'][latest]:.6g}, "
            f"cutoff_status={data['cutoff_status'][latest]}"
        )
    lines.extend(["", "## Matsubara Scan"])
    for kind in args.kinds:
        mask = (data["scan_type"] == "matsubara") & (data["kind"] == kind)
        if not np.any(mask):
            lines.append(f"- matsubara/{kind}: matsubara_max=nan, last_two_relative_change=nan, matsubara_status=missing")
            continue
        indices = np.where(mask)[0]
        latest = int(max(indices, key=lambda index: _setting_value(data, int(index), "matsubara")))
        lines.append(
            f"- matsubara/{kind}: matsubara_max={int(data['matsubara_max'][latest])}, "
            f"matsubara_tail_indicator={data['matsubara_tail_indicator'][latest]:.6g}, "
            f"last_two_relative_change={data['last_two_relative_change'][latest]:.6g}, "
            f"matsubara_status={data['matsubara_status'][latest]}"
        )
    lines.extend(
        [
            "",
            "## Baseline",
            f"clean_cutoff_converged = {clean_cutoff_converged}",
            f"extended_matsubara_converged = {extended_matsubara_converged}",
            f"zero_torque_baseline = {zero_baseline}",
            f"warning_possible_spurious_torque = {spurious}",
        ]
    )
    for kind in args.kinds:
        mask = data["kind"] == kind
        value = np.nan
        if np.any(mask):
            value = float(np.nanmax(data["max_abs_torque_over_theta"][mask]))
        lines.append(f"{kind}_max_abs_torque_over_theta = {value:.6g}")
    lines.extend(
        [
            f"can_return_to_local_response_distance_scan_benchmark = {ready}",
            f"local_integral_benchmark_ready_for_distance_scan = {ready}",
            "not final Casimir conclusion",
        ]
    )
    if args.quick or not full_completed:
        lines.extend(["quick_test_only=True", "no_full_convergence_conclusion=True", "full_run_pending_user_terminal=True"])
    return lines


def write_summary(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    full_completed: bool,
    response_cache: ResponseTensorCache | None,
) -> Path:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        "\n".join(_summary_lines(data, args, command, full_completed, response_cache)) + "\n",
        encoding="utf-8",
    )
    return SUMMARY_PATH


def parse_args() -> argparse.Namespace:
    defaults = _full_defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-scan", choices=ONLY_SCAN_CHOICES, default="all")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=defaults["kinds"])
    parser.add_argument("--distance", type=float, default=defaults["distance"])
    parser.add_argument("--theta-list", nargs="+", type=float, default=defaults["theta_list"])
    parser.add_argument("--energy-theta-list", nargs="+", type=float, default=defaults["energy_theta_list"])
    parser.add_argument(
        "--torque-check-theta-list",
        nargs="+",
        type=float,
        default=defaults["torque_check_theta_list"],
    )
    parser.add_argument("--u-max-list", nargs="+", type=float, default=defaults["u_max_list"])
    parser.add_argument("--du", type=float, default=defaults["du"])
    parser.add_argument("--matsubara-max-list", nargs="+", type=int, default=defaults["matsubara_max_list"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--normal-nk", type=int, default=defaults["normal_nk"])
    parser.add_argument("--normal-eta", type=float, default=defaults["normal_eta"])
    parser.add_argument("--normal-sampling", type=str, default=defaults["normal_sampling"])
    parser.add_argument("--normal-refine-factor", type=int, default=defaults["normal_refine_factor"])
    parser.add_argument("--bdg-nk", type=int, default=defaults["bdg_nk"])
    parser.add_argument("--delta0", type=float, default=defaults["delta0"])
    parser.add_argument("--phi-num", type=int, default=defaults["phi_num"])
    parser.add_argument("--output-prefix", type=Path, default=defaults["output_prefix"])
    parser.add_argument("--use-response-cache", action="store_true")
    parser.add_argument("--rebuild-response-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=defaults["cache_dir"])
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
    for scan_type in _selected_scans(args.only_scan):
        for setting in _scan_settings(args, scan_type):
            if args.resume and _point_done(data, scan_type, setting, args):
                print(f"resume_skip = {scan_type}:{setting}")
                continue
            print(f"running_scan_point = {scan_type}:{setting}")
            data = _append_rows(data, _rows_for_point(scan_type, setting, args, response_cache))
            save_checkpoint(data, args.output_prefix)
    data = _recompute_status(data)
    save_checkpoint(data, args.output_prefix)
    full_completed = _full_run_completed(data, args)
    summary = write_summary(data, args, command, full_completed, response_cache)
    print(f"npz_path = {args.output_prefix.with_suffix('.npz')}")
    print(f"csv_path = {args.output_prefix.with_suffix('.csv')}")
    print(f"summary_path = {summary}")
    print(f"command_path = {COMMAND_PATH}")
    print("note = refined convergence benchmark only; not a final Casimir conclusion.")


if __name__ == "__main__":
    main()
