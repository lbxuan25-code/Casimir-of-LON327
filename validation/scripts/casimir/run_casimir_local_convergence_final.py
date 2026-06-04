#!/usr/bin/env python3
"""One-command final convergence runner for the local-response integral benchmark.

This script orchestrates existing local-response benchmark functions only. It
does not implement a new Casimir integral, does not include finite momentum response,
and keeps the n=0 Matsubara policy at skip.
"""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation" / "scripts" / "casimir"))
sys.path.insert(0, str(ROOT / "scripts" / "casimir"))

from local_response_integral import (  # noqa: E402
    KINDS,
    N0_POLICY,
    compute_local_response_casimir_integral,
)
import converge_casimir_local_response_integral as convergence  # noqa: E402

FINAL_ROOT = ROOT / "validation" / "outputs" / "archive" / "casimir" / "local_response_integral" / "final_convergence"
DEFAULT_OUTPUT_PREFIX = FINAL_ROOT / "data" / "final_local_convergence"
SUMMARY_PATH = FINAL_ROOT / "final_convergence_summary.md"
COMMAND_PATH = FINAL_ROOT / "final_convergence_command.sh"
SCAN_ORDER = ("matsubara", "kparallel_num", "kparallel_cutoff", "phi")
ONLY_SCAN_CHOICES = (*SCAN_ORDER, "all")
RATIO_EPS = 1e-300
TORQUE_TOLERANCE = 1e-20


def _full_defaults() -> dict[str, object]:
    return {
        "kinds": list(KINDS),
        "distance": 5e-8,
        "theta_list": [0.0, 0.3926990817, 0.7853981634, 1.1780972451, 1.5707963268],
        "matsubara_max_list": [4, 8, 16, 24],
        "kparallel_num_list": [32, 64, 96],
        "kparallel_max_factor_list": [20.0, 40.0, 60.0],
        "phi_num_list": [32, 64, 96],
        "temperature": 30.0,
        "normal_nk": 96,
        "normal_eta": 1e-4,
        "normal_sampling": "fs_adaptive",
        "normal_refine_factor": 8,
        "bdg_nk": 32,
        "delta0": 0.04,
        "output_prefix": DEFAULT_OUTPUT_PREFIX,
    }


def _quick_overrides() -> dict[str, object]:
    return {
        "matsubara_max_list": [1, 2],
        "kparallel_num_list": [6, 8],
        "kparallel_max_factor_list": [10.0, 20.0],
        "phi_num_list": [6, 8],
        "normal_nk": 12,
        "normal_refine_factor": 2,
        "bdg_nk": 8,
    }


def _format_number(value: object) -> str:
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
    parts: list[str] = ["python", "validation/scripts/casimir/run_casimir_local_convergence_final.py"]
    option_map = [
        ("--kinds", "kinds"),
        ("--distance", "distance"),
        ("--theta-list", "theta_list"),
        ("--matsubara-max-list", "matsubara_max_list"),
        ("--kparallel-num-list", "kparallel_num_list"),
        ("--kparallel-max-factor-list", "kparallel_max_factor_list"),
        ("--phi-num-list", "phi_num_list"),
        ("--temperature", "temperature"),
        ("--normal-nk", "normal_nk"),
        ("--normal-eta", "normal_eta"),
        ("--normal-sampling", "normal_sampling"),
        ("--normal-refine-factor", "normal_refine_factor"),
        ("--bdg-nk", "bdg_nk"),
        ("--delta0", "delta0"),
        ("--output-prefix", "output_prefix"),
    ]
    for option, key in option_map:
        value = defaults[key]
        parts.append(option)
        if isinstance(value, list):
            parts.extend(_format_number(item) for item in value)
        else:
            parts.append(_format_number(value))
    return " ".join(shlex.quote(part) for part in parts)


def _write_command_file(command: str) -> Path:
    return _write_command_file_at(command, DEFAULT_OUTPUT_PREFIX)


def _command_path(output_prefix: Path) -> Path:
    if output_prefix.resolve() == DEFAULT_OUTPUT_PREFIX.resolve():
        return COMMAND_PATH
    return output_prefix.parent / "final_convergence_command.sh"


def _summary_path(output_prefix: Path) -> Path:
    if output_prefix.resolve() == DEFAULT_OUTPUT_PREFIX.resolve():
        return SUMMARY_PATH
    return output_prefix.parent / "final_convergence_summary.md"


def _write_command_file_at(command: str, output_prefix: Path) -> Path:
    command_path = _command_path(output_prefix)
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n\n{command}\n", encoding="utf-8")
    command_path.chmod(0o755)
    return command_path


def _selected_scans(only_scan: str) -> tuple[str, ...]:
    if only_scan == "all":
        return SCAN_ORDER
    return (only_scan,)


def _settings_for_scan(args: argparse.Namespace, scan_type: str) -> list[int | float]:
    if scan_type == "matsubara":
        return sorted(args.matsubara_max_list)
    if scan_type == "kparallel_num":
        return sorted(args.kparallel_num_list)
    if scan_type == "kparallel_cutoff":
        return sorted(args.kparallel_max_factor_list)
    if scan_type == "phi":
        return sorted(args.phi_num_list)
    raise ValueError("unknown scan_type")


def _base_settings(args: argparse.Namespace) -> tuple[int, int, float, int]:
    mats = args.matsubara_max_list
    knum = args.kparallel_num_list
    kfac = [float(item) for item in args.kparallel_max_factor_list]
    phi = args.phi_num_list
    return (
        8 if 8 in mats else max(mats),
        32 if 32 in knum else max(knum),
        20.0 if 20.0 in kfac else max(kfac),
        32 if 32 in phi else max(phi),
    )


def _empty_data() -> dict[str, np.ndarray]:
    return {
        "scan_type": np.array([], dtype="U32"),
        "kind": np.array([], dtype="U32"),
        "distance_m": np.array([], dtype=float),
        "theta_grid_size": np.array([], dtype=int),
        "matsubara_max": np.array([], dtype=int),
        "kparallel_num": np.array([], dtype=int),
        "kparallel_max_factor": np.array([], dtype=float),
        "kparallel_max": np.array([], dtype=float),
        "phi_num": np.array([], dtype=int),
        "max_abs_energy_over_theta": np.array([], dtype=float),
        "max_abs_torque_over_theta": np.array([], dtype=float),
        "relative_change_vs_largest_setting": np.array([], dtype=float),
        "last_two_relative_change": np.array([], dtype=float),
        "matsubara_tail_indicator": np.array([], dtype=float),
        "kparallel_cutoff_indicator": np.array([], dtype="U64"),
        "phi_convergence_indicator": np.array([], dtype="U64"),
        "convergence_status": np.array([], dtype="U128"),
        "diagnosis": np.array([], dtype="U192"),
        "local_response": np.array([], dtype=bool),
        "finite_momentum_resolved": np.array([], dtype=bool),
        "n0_policy": np.array([], dtype="U16"),
        "benchmark_only": np.array([], dtype=bool),
        "not_final_casimir_conclusion": np.array([], dtype=bool),
        "notes": np.array([], dtype=object),
    }


def _load_existing(output_prefix: Path) -> dict[str, np.ndarray]:
    path = output_prefix.with_suffix(".npz")
    if path.exists():
        with np.load(path, allow_pickle=True) as loaded:
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
                    "kparallel_num": int(row["kparallel_num"]),
                    "kparallel_max_factor": float(row["kparallel_max_factor"]),
                    "kparallel_max": float(row["kparallel_max"]),
                    "phi_num": int(row["phi_num"]),
                    "max_abs_energy_over_theta": float(row["max_abs_energy_over_theta"]),
                    "max_abs_torque_over_theta": float(row["max_abs_torque_over_theta"]),
                    "relative_change_vs_largest_setting": float(row["relative_change_vs_largest_setting"]),
                    "last_two_relative_change": float(row["last_two_relative_change"]),
                    "matsubara_tail_indicator": float(row["matsubara_tail_indicator"]),
                    "kparallel_cutoff_indicator": row["kparallel_cutoff_indicator"],
                    "phi_convergence_indicator": row["phi_convergence_indicator"],
                    "convergence_status": row["convergence_status"],
                    "diagnosis": row["diagnosis"],
                    "local_response": row["local_response"] == "True",
                    "finite_momentum_resolved": row["finite_momentum_resolved"] == "True",
                    "n0_policy": row["n0_policy"],
                    "benchmark_only": row["benchmark_only"] == "True",
                    "not_final_casimir_conclusion": row["not_final_casimir_conclusion"] == "True",
                    "notes": tuple(row["notes"].split(" | ")),
                }
            )
    return _append_rows(_empty_data(), rows)


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


def _point_done(data: dict[str, np.ndarray], scan_type: str, setting: int | float, args: argparse.Namespace) -> bool:
    if data["kind"].size == 0:
        return False
    mask = data["scan_type"] == scan_type
    if scan_type == "matsubara":
        mask &= data["matsubara_max"] == int(setting)
    elif scan_type == "kparallel_num":
        mask &= data["kparallel_num"] == int(setting)
    elif scan_type == "kparallel_cutoff":
        mask &= np.isclose(data["kparallel_max_factor"], float(setting))
    elif scan_type == "phi":
        mask &= data["phi_num"] == int(setting)
    return set(str(item) for item in data["kind"][mask]) >= set(args.kinds)


def _rows_for_point(
    scan_type: str,
    setting: int | float,
    args: argparse.Namespace,
    base_settings: tuple[int, int, float, int],
) -> list[dict[str, object]]:
    matsubara_max, kparallel_num, kmax_factor, phi_num = convergence._settings_for_scan(  # noqa: SLF001
        scan_type,
        setting,
        base_matsubara_max=base_settings[0],
        base_kparallel_num=base_settings[1],
        base_kparallel_max_factor=base_settings[2],
        base_phi_num=base_settings[3],
    )
    integral = compute_local_response_casimir_integral(
        kinds=args.kinds,
        distance_list=[args.distance],
        theta_list=args.theta_list,
        matsubara_min=1,
        matsubara_max=matsubara_max,
        kparallel_num=kparallel_num,
        kparallel_max_factor=kmax_factor,
        phi_num=phi_num,
        temperature_K=args.temperature,
        normal_nk=args.normal_nk,
        normal_eta_eV=args.normal_eta,
        normal_sampling=args.normal_sampling,
        normal_refine_factor=args.normal_refine_factor,
        bdg_nk=args.bdg_nk,
        delta0_eV=args.delta0,
    )
    rows: list[dict[str, object]] = []
    for kind in sorted(set(str(item) for item in integral["kind"])):
        mask = integral["kind"] == kind
        rows.append(
            {
                "scan_type": scan_type,
                "kind": kind,
                "distance_m": args.distance,
                "theta_grid_size": len(args.theta_list),
                "matsubara_max": matsubara_max,
                "kparallel_num": kparallel_num,
                "kparallel_max_factor": kmax_factor,
                "kparallel_max": float(np.nanmax(integral["kparallel_max"][mask])),
                "phi_num": phi_num,
                "max_abs_energy_over_theta": float(np.nanmax(np.abs(integral["energy"][mask]))),
                "max_abs_torque_over_theta": float(np.nanmax(np.abs(integral["torque_fd"][mask]))),
                "relative_change_vs_largest_setting": np.nan,
                "last_two_relative_change": np.nan,
                "matsubara_tail_indicator": (
                    np.nan
                    if np.all(np.isnan(integral["matsubara_tail_indicator"][mask]))
                    else float(np.nanmax(integral["matsubara_tail_indicator"][mask]))
                ),
                "kparallel_cutoff_indicator": "cutoff_not_final",
                "phi_convergence_indicator": "phi_convergence_not_final",
                "convergence_status": "pending_recomputed_after_checkpoint",
                "diagnosis": "not_final_casimir_conclusion",
                "local_response": True,
                "finite_momentum_resolved": False,
                "n0_policy": N0_POLICY,
                "benchmark_only": True,
                "not_final_casimir_conclusion": True,
                "notes": (
                    "local-response final convergence benchmark only",
                    "n=0 policy: skip",
                    "finite momentum response not included",
                    "not a final Casimir conclusion",
                ),
            }
        )
    return rows


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


def _recompute_convergence(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if data["kind"].size == 0:
        return data
    data = {key: np.array(value, copy=True) for key, value in data.items()}
    data["relative_change_vs_largest_setting"][:] = np.nan
    data["last_two_relative_change"][:] = np.nan
    for scan_type in SCAN_ORDER:
        for kind in sorted(set(str(item) for item in data["kind"])):
            indices = np.where((data["scan_type"] == scan_type) & (data["kind"] == kind))[0]
            if indices.size == 0:
                continue
            ordered = sorted(indices, key=lambda index: convergence._scan_setting_value(data, int(index), scan_type))  # noqa: SLF001
            reference = data["max_abs_energy_over_theta"][ordered[-1]]
            last_change = np.nan
            if len(ordered) >= 2:
                last_change = _relative_change(
                    float(data["max_abs_energy_over_theta"][ordered[-2]]),
                    float(data["max_abs_energy_over_theta"][ordered[-1]]),
                )
            for index in ordered:
                data["relative_change_vs_largest_setting"][index] = _relative_change(
                    float(data["max_abs_energy_over_theta"][index]),
                    float(reference),
                )
                data["last_two_relative_change"][index] = last_change
                status = _status_from_change(last_change)
                status_parts = [status]
                if scan_type == "matsubara" and index == ordered[-1]:
                    tail = data["matsubara_tail_indicator"][index]
                    if np.isfinite(tail) and tail > 0.05:
                        status_parts.append("warning_matsubara_not_converged")
                diagnosis_parts = []
                if str(data["kind"][index]) in KINDS and data["max_abs_torque_over_theta"][index] <= TORQUE_TOLERANCE:
                    diagnosis_parts.append("zero_torque_baseline")
                elif str(data["kind"][index]) in KINDS:
                    diagnosis_parts.append("warning_possible_spurious_torque")
                diagnosis_parts.append("not_final_casimir_conclusion")
                data["convergence_status"][index] = ";".join(status_parts)
                data["diagnosis"][index] = ";".join(diagnosis_parts)
    return data


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    return value


def save_checkpoint(data: dict[str, np.ndarray], output_prefix: Path) -> None:
    data = _recompute_convergence(data)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_prefix.with_suffix(".npz"), **data)
    fieldnames = list(convergence.REQUIRED_NPZ_FIELDS)
    preferred = [
        "scan_type",
        "kind",
        "distance_m",
        "theta_grid_size",
        "matsubara_max",
        "kparallel_num",
        "kparallel_max_factor",
        "kparallel_max",
        "phi_num",
        "max_abs_energy_over_theta",
        "max_abs_torque_over_theta",
        "relative_change_vs_largest_setting",
        "last_two_relative_change",
        "matsubara_tail_indicator",
        "kparallel_cutoff_indicator",
        "phi_convergence_indicator",
        "convergence_status",
        "diagnosis",
        "local_response",
        "finite_momentum_resolved",
        "n0_policy",
        "benchmark_only",
        "not_final_casimir_conclusion",
        "notes",
    ]
    fieldnames = [name for name in preferred if name in fieldnames]
    with output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(data["kind"].size):
            writer.writerow({name: _csv_value(data[name][index]) for name in fieldnames})


def _completed_expected_points(data: dict[str, np.ndarray], args: argparse.Namespace, scans: tuple[str, ...]) -> bool:
    for scan_type in scans:
        for setting in _settings_for_scan(args, scan_type):
            if not _point_done(data, scan_type, setting, args):
                return False
    return True


def _full_run_completed(data: dict[str, np.ndarray], args: argparse.Namespace) -> bool:
    return (not args.quick) and args.only_scan == "all" and _completed_expected_points(data, args, SCAN_ORDER)


def _summary_lines(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    full_completed: bool,
) -> list[str]:
    full_pending = not full_completed
    spurious = any("warning_possible_spurious_torque" in str(item) for item in data["diagnosis"])
    all_loose = True
    lines = [
        "# Final Local-Response Convergence Summary",
        "",
        f"full_run_command = `{command}`",
        f"quick_test_result = {bool(args.quick)}",
        f"full_convergence_result = {'available' if full_completed else 'not_available'}",
        f"full_run_pending = {full_pending}",
        f"full_run_completed = {full_completed}",
        "local_response=True",
        "finite_momentum_resolved=False",
        "n0_policy=skip",
        "benchmark_only=True",
        "not_final_casimir_conclusion=True",
        "",
        "## Scan Status",
    ]
    for scan_type in SCAN_ORDER:
        for kind in args.kinds:
            mask = (data["scan_type"] == scan_type) & (data["kind"] == kind)
            if not np.any(mask):
                all_loose = False
                lines.append(f"- {scan_type}/{kind}: last_two_relative_change=nan, convergence_status=missing")
                continue
            order = np.argsort([convergence._scan_setting_value(data, int(index), scan_type) for index in np.where(mask)[0]])  # noqa: SLF001
            indices = np.where(mask)[0][order]
            last_index = int(indices[-1])
            change = float(data["last_two_relative_change"][last_index])
            status = _status_from_change(change)
            if status == "not_converged":
                all_loose = False
            lines.append(
                f"- {scan_type}/{kind}: last_two_relative_change={change:.6g}, "
                f"convergence_status={status}"
            )
    tail = np.nan
    if data["kind"].size:
        tail = float(np.nanmax(data["matsubara_tail_indicator"]))
    lines.extend(["", "## Diagnostics", f"matsubara_tail_indicator_max = {tail:.6g}"])
    for kind in args.kinds:
        mask = data["kind"] == kind
        value = np.nan
        if np.any(mask):
            value = float(np.nanmax(data["max_abs_torque_over_theta"][mask]))
        lines.append(f"{kind}_max_abs_torque_over_theta = {value:.6g}")
    zero_baseline = bool(data["kind"].size) and all("zero_torque_baseline" in str(item) for item in data["diagnosis"])
    ready = bool(full_completed and all_loose and not spurious)
    lines.extend(
        [
            f"zero_torque_baseline = {zero_baseline}",
            f"warning_possible_spurious_torque = {spurious}",
            f"can_enter_next_stage = {ready}",
            f"local_integral_benchmark_ready_for_distance_scan = {ready}",
            "not final Casimir conclusion",
        ]
    )
    if full_pending:
        lines.extend(["full_run_pending_user_terminal=True", "no_full_convergence_conclusion=True"])
    return lines


def write_summary(data: dict[str, np.ndarray], args: argparse.Namespace, command: str, full_completed: bool) -> Path:
    summary_path = _summary_path(args.output_prefix)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(_summary_lines(data, args, command, full_completed)) + "\n", encoding="utf-8")
    return summary_path


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
    parser.add_argument("--matsubara-max-list", nargs="+", type=int, default=defaults["matsubara_max_list"])
    parser.add_argument("--kparallel-num-list", nargs="+", type=int, default=defaults["kparallel_num_list"])
    parser.add_argument("--kparallel-max-factor-list", nargs="+", type=float, default=defaults["kparallel_max_factor_list"])
    parser.add_argument("--phi-num-list", nargs="+", type=int, default=defaults["phi_num_list"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--normal-nk", type=int, default=defaults["normal_nk"])
    parser.add_argument("--normal-eta", type=float, default=defaults["normal_eta"])
    parser.add_argument("--normal-sampling", type=str, default=defaults["normal_sampling"])
    parser.add_argument("--normal-refine-factor", type=int, default=defaults["normal_refine_factor"])
    parser.add_argument("--bdg-nk", type=int, default=defaults["bdg_nk"])
    parser.add_argument("--delta0", type=float, default=defaults["delta0"])
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
    command_path = _write_command_file_at(command, args.output_prefix)
    data = _load_existing(args.output_prefix) if args.resume else _empty_data()
    base_settings = _base_settings(args)
    for scan_type in _selected_scans(args.only_scan):
        for setting in _settings_for_scan(args, scan_type):
            if args.resume and _point_done(data, scan_type, setting, args):
                print(f"resume_skip = {scan_type}:{setting}")
                continue
            print(f"running_scan_point = {scan_type}:{setting}")
            data = _append_rows(data, _rows_for_point(scan_type, setting, args, base_settings))
            save_checkpoint(data, args.output_prefix)
    data = _recompute_convergence(data)
    save_checkpoint(data, args.output_prefix)
    full_completed = _full_run_completed(data, args)
    if full_completed:
        convergence.save_outputs(data, args.output_prefix)
    summary = write_summary(data, args, command, full_completed)
    print(f"npz_path = {args.output_prefix.with_suffix('.npz')}")
    print(f"csv_path = {args.output_prefix.with_suffix('.csv')}")
    print(f"summary_path = {summary}")
    print(f"command_path = {command_path}")
    print("note = benchmark only; not a final Casimir conclusion.")


if __name__ == "__main__":
    main()
