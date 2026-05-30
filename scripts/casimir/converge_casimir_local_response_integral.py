#!/usr/bin/env python3
"""Convergence benchmark for the local-response Casimir integral pipeline."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "casimir"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from benchmark_casimir_local_response_integral import (  # noqa: E402
    KINDS,
    N0_POLICY,
    TORQUE_TOLERANCE,
    benchmark_casimir_local_response_integral,
)
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

REQUIRED_NPZ_FIELDS = {
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
    "finite_q_resolved",
    "n0_policy",
    "benchmark_only",
    "not_final_casimir_conclusion",
    "notes",
}

SCAN_ORDER = ("matsubara", "kparallel_num", "kparallel_cutoff", "phi")
RATIO_EPS = 1e-300
CONVERGENCE_TORQUE_TOLERANCE = 1e-20


def _relative_change(value: float, reference: float) -> float:
    if abs(value) < RATIO_EPS and abs(reference) < RATIO_EPS:
        return 0.0
    return float(abs(value - reference) / (abs(reference) + RATIO_EPS))


def _settings_for_scan(
    scan_type: str,
    setting: int | float,
    *,
    base_matsubara_max: int,
    base_kparallel_num: int,
    base_kparallel_max_factor: float,
    base_phi_num: int,
) -> tuple[int, int, float, int]:
    if scan_type == "matsubara":
        return int(setting), base_kparallel_num, base_kparallel_max_factor, base_phi_num
    if scan_type == "kparallel_num":
        return base_matsubara_max, int(setting), base_kparallel_max_factor, base_phi_num
    if scan_type == "kparallel_cutoff":
        return base_matsubara_max, base_kparallel_num, float(setting), base_phi_num
    if scan_type == "phi":
        return base_matsubara_max, base_kparallel_num, base_kparallel_max_factor, int(setting)
    raise ValueError("unknown scan_type")


def _scan_setting_value(data: dict[str, np.ndarray], index: int, scan_type: str) -> float:
    if scan_type == "matsubara":
        return float(data["matsubara_max"][index])
    if scan_type == "kparallel_num":
        return float(data["kparallel_num"][index])
    if scan_type == "kparallel_cutoff":
        return float(data["kparallel_max_factor"][index])
    if scan_type == "phi":
        return float(data["phi_num"][index])
    raise ValueError("unknown scan_type")


def converge_casimir_local_response_integral(
    kinds: list[str],
    distance_m: float,
    theta_list: list[float],
    matsubara_max_list: list[int],
    kparallel_num_list: list[int],
    kparallel_max_factor_list: list[float],
    phi_num_list: list[int],
    temperature_K: float,
    normal_nk: int,
    normal_eta_eV: float,
    normal_sampling: str,
    normal_refine_factor: int,
    bdg_nk: int,
    delta0_eV: float,
    include_toy_anisotropic_control: bool = False,
) -> dict[str, np.ndarray]:
    if any(kind not in KINDS for kind in kinds):
        raise ValueError("unknown kind")
    if distance_m <= 0.0:
        raise ValueError("distance must be positive")
    if len(theta_list) < 2:
        raise ValueError("theta_list must contain at least two points")

    base_matsubara_max = 8 if 8 in matsubara_max_list else max(matsubara_max_list)
    base_kparallel_num = 32 if 32 in kparallel_num_list else max(kparallel_num_list)
    base_kparallel_max_factor = 20.0 if 20.0 in [float(v) for v in kparallel_max_factor_list] else max(kparallel_max_factor_list)
    base_phi_num = 32 if 32 in phi_num_list else max(phi_num_list)
    scan_values: dict[str, list[int | float]] = {
        "matsubara": sorted(matsubara_max_list),
        "kparallel_num": sorted(kparallel_num_list),
        "kparallel_cutoff": sorted(kparallel_max_factor_list),
        "phi": sorted(phi_num_list),
    }

    rows: list[dict[str, object]] = []
    for scan_type in SCAN_ORDER:
        for setting in scan_values[scan_type]:
            matsubara_max, kparallel_num, kmax_factor, phi_num = _settings_for_scan(
                scan_type,
                setting,
                base_matsubara_max=base_matsubara_max,
                base_kparallel_num=base_kparallel_num,
                base_kparallel_max_factor=base_kparallel_max_factor,
                base_phi_num=base_phi_num,
            )
            integral = benchmark_casimir_local_response_integral(
                kinds=kinds,
                distance_list=[distance_m],
                theta_list=theta_list,
                matsubara_min=1,
                matsubara_max=matsubara_max,
                kparallel_num=kparallel_num,
                kparallel_max_factor=kmax_factor,
                phi_num=phi_num,
                temperature_K=temperature_K,
                normal_nk=normal_nk,
                normal_eta_eV=normal_eta_eV,
                normal_sampling=normal_sampling,
                normal_refine_factor=normal_refine_factor,
                bdg_nk=bdg_nk,
                delta0_eV=delta0_eV,
                include_toy_anisotropic_control=include_toy_anisotropic_control,
            )
            for kind in sorted(set(str(item) for item in integral["kind"])):
                mask = integral["kind"] == kind
                rows.append(
                    {
                        "scan_type": scan_type,
                        "kind": kind,
                        "distance_m": distance_m,
                        "theta_grid_size": len(theta_list),
                        "matsubara_max": matsubara_max,
                        "kparallel_num": kparallel_num,
                        "kparallel_max_factor": kmax_factor,
                        "kparallel_max": float(np.nanmax(integral["kparallel_max"][mask])),
                        "phi_num": phi_num,
                        "max_abs_energy_over_theta": float(np.nanmax(np.abs(integral["energy"][mask]))),
                        "max_abs_torque_over_theta": float(np.nanmax(np.abs(integral["torque_fd"][mask]))),
                        "matsubara_tail_indicator": (
                            np.nan
                            if np.all(np.isnan(integral["matsubara_tail_indicator"][mask]))
                            else float(np.nanmax(integral["matsubara_tail_indicator"][mask]))
                        ),
                    }
                )

    data: dict[str, np.ndarray] = {
        "scan_type": np.array([str(row["scan_type"]) for row in rows], dtype="U32"),
        "kind": np.array([str(row["kind"]) for row in rows], dtype="U32"),
        "distance_m": np.array([float(row["distance_m"]) for row in rows]),
        "theta_grid_size": np.array([int(row["theta_grid_size"]) for row in rows]),
        "matsubara_max": np.array([int(row["matsubara_max"]) for row in rows]),
        "kparallel_num": np.array([int(row["kparallel_num"]) for row in rows]),
        "kparallel_max_factor": np.array([float(row["kparallel_max_factor"]) for row in rows]),
        "kparallel_max": np.array([float(row["kparallel_max"]) for row in rows]),
        "phi_num": np.array([int(row["phi_num"]) for row in rows]),
        "max_abs_energy_over_theta": np.array([float(row["max_abs_energy_over_theta"]) for row in rows]),
        "max_abs_torque_over_theta": np.array([float(row["max_abs_torque_over_theta"]) for row in rows]),
        "relative_change_vs_largest_setting": np.full(len(rows), np.nan),
        "last_two_relative_change": np.full(len(rows), np.nan),
        "matsubara_tail_indicator": np.array([float(row["matsubara_tail_indicator"]) for row in rows]),
        "kparallel_cutoff_indicator": np.full(len(rows), "cutoff_not_final", dtype="U64"),
        "phi_convergence_indicator": np.full(len(rows), "phi_convergence_not_final", dtype="U64"),
        "convergence_status": np.empty(len(rows), dtype="U128"),
        "diagnosis": np.empty(len(rows), dtype="U192"),
        "local_response": np.full(len(rows), True, dtype=bool),
        "finite_q_resolved": np.full(len(rows), False, dtype=bool),
        "n0_policy": np.full(len(rows), N0_POLICY, dtype="U16"),
        "benchmark_only": np.full(len(rows), True, dtype=bool),
        "not_final_casimir_conclusion": np.full(len(rows), True, dtype=bool),
        "notes": np.empty(len(rows), dtype=object),
    }

    for scan_type in SCAN_ORDER:
        for kind in sorted(set(str(item) for item in data["kind"])):
            indices = np.where((data["scan_type"] == scan_type) & (data["kind"] == kind))[0]
            if indices.size == 0:
                continue
            ordered = sorted(indices, key=lambda index: _scan_setting_value(data, int(index), scan_type))
            reference = data["max_abs_energy_over_theta"][ordered[-1]]
            if len(ordered) >= 2:
                last_change = _relative_change(
                    data["max_abs_energy_over_theta"][ordered[-2]],
                    data["max_abs_energy_over_theta"][ordered[-1]],
                )
            else:
                last_change = np.nan
            for index in ordered:
                data["relative_change_vs_largest_setting"][index] = _relative_change(
                    data["max_abs_energy_over_theta"][index],
                    reference,
                )
                data["last_two_relative_change"][index] = last_change
                status_parts = []
                if np.isfinite(last_change) and last_change < 0.02:
                    status_parts.append("benchmark_converged_candidate")
                elif np.isfinite(last_change) and last_change < 0.05:
                    status_parts.append("benchmark_converged_loose")
                else:
                    status_parts.append("warning_not_converged")
                if scan_type == "matsubara" and _scan_setting_value(data, int(index), scan_type) == max(matsubara_max_list):
                    if data["matsubara_tail_indicator"][index] > 0.05:
                        status_parts.append("warning_matsubara_not_converged")
                if scan_type == "kparallel_cutoff" and _scan_setting_value(data, int(index), scan_type) == max(kparallel_max_factor_list):
                    if not (np.isfinite(last_change) and last_change < 0.05):
                        status_parts.append("warning_kparallel_cutoff_not_converged")
                if scan_type == "phi" and _scan_setting_value(data, int(index), scan_type) == max(phi_num_list):
                    if not (np.isfinite(last_change) and last_change < 0.05):
                        status_parts.append("warning_phi_not_converged")
                diagnosis_parts = []
                if kind == "toy_anisotropic" and data["max_abs_torque_over_theta"][index] > TORQUE_TOLERANCE:
                    diagnosis_parts.append("plumbing_pass_toy_anisotropy")
                elif kind in KINDS and data["max_abs_torque_over_theta"][index] <= CONVERGENCE_TORQUE_TOLERANCE:
                    diagnosis_parts.append("zero_torque_baseline")
                elif kind in KINDS:
                    diagnosis_parts.append("warning_possible_spurious_torque")
                diagnosis_parts.append("not_final_casimir_conclusion")
                data["convergence_status"][index] = ";".join(status_parts)
                data["diagnosis"][index] = ";".join(diagnosis_parts)
                data["notes"][index] = (
                    "local-response convergence benchmark only",
                    "n=0 policy: skip",
                    "finite-q response not included",
                    "not a final Casimir conclusion",
                )
    return data


def output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    resolved_prefix = output_prefix.resolve()
    project_root = (ROOT / "outputs" / "archive" / "casimir" / "local_response_integral" / "convergence" / "data").resolve()
    if resolved_prefix.is_relative_to(project_root):
        figure_dir = ROOT / "outputs" / "archive" / "casimir" / "local_response_integral" / "convergence" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return (
        npz_path,
        csv_path,
        figure_dir / f"{output_prefix.name}_energy_vs_matsubara.png",
        figure_dir / f"{output_prefix.name}_energy_vs_kparallel_num.png",
        figure_dir / f"{output_prefix.name}_energy_vs_kparallel_cutoff.png",
        figure_dir / f"{output_prefix.name}_energy_vs_phi.png",
        figure_dir / f"{output_prefix.name}_torque_summary.png",
    )


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    return value


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    paths = output_paths(output_prefix)
    npz_path, csv_path, matsu_plot, knum_plot, kcut_plot, phi_plot, torque_plot = paths
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    matsu_plot.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)
    fieldnames = [
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
        "finite_q_resolved",
        "n0_policy",
        "benchmark_only",
        "not_final_casimir_conclusion",
        "notes",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(data["kind"].size):
            writer.writerow({name: _csv_value(data[name][index]) for name in fieldnames})

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    plot_specs = [
        ("matsubara", "matsubara_max", matsu_plot, "max |energy| vs Matsubara cutoff"),
        ("kparallel_num", "kparallel_num", knum_plot, "max |energy| vs k grid"),
        ("kparallel_cutoff", "kparallel_max_factor", kcut_plot, "max |energy| vs k cutoff"),
        ("phi", "phi_num", phi_plot, "max |energy| vs phi grid"),
    ]
    for scan_type, x_field, path, title in plot_specs:
        fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        for kind in sorted(set(str(item) for item in data["kind"])):
            mask = (data["scan_type"] == scan_type) & (data["kind"] == kind)
            order = np.argsort(data[x_field][mask])
            ax.plot(data[x_field][mask][order], data["max_abs_energy_over_theta"][mask][order], marker="o", label=kind)
        ax.set_xlabel(x_field)
        ax.set_ylabel("max |energy|")
        ax.set_title(title)
        style_publication_axis(ax)
        save_publication_figure(fig, path)
        plt.close(fig)

    fig_torque, ax_torque = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for scan_type in SCAN_ORDER:
        mask = data["scan_type"] == scan_type
        if np.any(mask):
            ax_torque.scatter(
                np.full(np.count_nonzero(mask), list(SCAN_ORDER).index(scan_type)),
                data["max_abs_torque_over_theta"][mask],
                label=scan_type,
            )
    ax_torque.set_yscale("symlog", linthresh=1e-30)
    ax_torque.set_xticks(range(len(SCAN_ORDER)), SCAN_ORDER, rotation=25, ha="right")
    ax_torque.set_ylabel("max |torque|")
    ax_torque.set_title("local-response benchmark torque summary")
    style_publication_axis(ax_torque)
    save_publication_figure(fig_torque, torque_plot)
    plt.close(fig_torque)
    return paths


def print_summary(data: dict[str, np.ndarray]) -> None:
    for scan_type in SCAN_ORDER:
        mask = data["scan_type"] == scan_type
        statuses = sorted(set(str(item) for item in data["convergence_status"][mask]))
        print(f"{scan_type}_statuses = {statuses}")
    for kind in sorted(set(str(item) for item in data["kind"])):
        mask = data["kind"] == kind
        print(f"{kind}_max_abs_torque = {float(np.nanmax(data['max_abs_torque_over_theta'][mask]))}")
    print("note = convergence benchmark only; not a final Casimir conclusion.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--distance", type=float, default=5e-8)
    parser.add_argument(
        "--theta-list",
        nargs="+",
        type=float,
        default=[0.0, 0.3926990817, 0.7853981634, 1.1780972451, 1.5707963268],
    )
    parser.add_argument("--matsubara-max-list", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--kparallel-num-list", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--kparallel-max-factor-list", nargs="+", type=float, default=[10.0, 20.0, 40.0])
    parser.add_argument("--phi-num-list", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--normal-nk", type=int, default=96)
    parser.add_argument("--normal-eta", type=float, default=1e-4)
    parser.add_argument("--normal-sampling", type=str, default="fs_adaptive")
    parser.add_argument("--normal-refine-factor", type=int, default=8)
    parser.add_argument("--bdg-nk", type=int, default=32)
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--include-toy-anisotropic-control", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT
        / "outputs"
        / "casimir"
        / "local_response_integral"
        / "convergence"
        / "data"
        / "local_integral_convergence",
    )
    args = parser.parse_args()
    data = converge_casimir_local_response_integral(
        args.kinds,
        args.distance,
        args.theta_list,
        args.matsubara_max_list,
        args.kparallel_num_list,
        args.kparallel_max_factor_list,
        args.phi_num_list,
        args.temperature,
        args.normal_nk,
        args.normal_eta,
        args.normal_sampling,
        args.normal_refine_factor,
        args.bdg_nk,
        args.delta0,
        include_toy_anisotropic_control=args.include_toy_anisotropic_control,
    )
    paths = save_outputs(data, args.output_prefix)
    print_summary(data)
    print(f"npz_path = {paths[0]}")
    print(f"csv_path = {paths[1]}")
    print(f"figure_paths = {paths[2]}, {paths[3]}, {paths[4]}, {paths[5]}, {paths[6]}")


if __name__ == "__main__":
    main()
