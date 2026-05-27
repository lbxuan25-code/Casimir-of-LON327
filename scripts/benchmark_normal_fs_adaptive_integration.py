#!/usr/bin/env python3
"""Prototype FS-adaptive BZ integration for normal-state response.

The adaptive mesh changes only the Brillouin-zone quadrature. It evaluates the
same normal-state Kubo integrand through ``kubo_conductivity_imag_axis`` and is
intended as a diagnostic prototype, not a Casimir calculation.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import (  # noqa: E402
    bosonic_matsubara_energy_eV,
    conductivity_matrix_diagnostics,
    model_response_to_sheet_conductivity,
    sheet_conductivity_to_reflection_dimensionless,
)
from lno327.conductivity import ConductivityTensor  # noqa: E402
from lno327.normal_sampling import (  # noqa: E402
    fs_adaptive_mesh,
    multishift_normal_response,
    normal_fs_diagnostics,
    shifted_bz_mesh,
    single_mesh_normal_response,
    uniform_weights,
)
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

SAMPLING_MODES = ("uniform", "multishift_average", "fs_adaptive")
DEFAULT_NK_LIST = (32, 48, 64)
DEFAULT_ETA_LIST = (5e-4, 2e-4, 1e-4)
DEFAULT_MATSUBARA_LIST = (1, 2)
DEFAULT_REFINE_FACTOR_LIST = (2, 4, 6)
HIGH_NK_TOLERANCE = 0.02
SYMMETRY_TOLERANCE = 1e-8
RATIO_EPS = 1e-300

REQUIRED_NPZ_FIELDS = {
    "sampling",
    "nk",
    "eta_eV",
    "matsubara_index",
    "omega_eV",
    "refine_factor",
    "shift_grid",
    "num_kpoints_total",
    "num_fs_cells",
    "num_refined_points",
    "refined_area_fraction",
    "sigma_xx",
    "sigma_yy",
    "sigma_xy",
    "sigma_yx",
    "sigma_xx_std_over_shifts",
    "relative_std_over_shifts",
    "delta",
    "relative_offdiag",
    "relative_eigen_split",
    "sheet_conductivity_xx",
    "reflection_dimensionless_xx",
    "relative_change_vs_largest_nk",
    "relative_change_between_last_two_nk",
    "eta_relative_change",
    "min_abs_band_energy_on_mesh",
    "points_within_eta",
    "points_within_omega",
    "points_within_kBT",
    "fermi_window_weight_sum",
    "estimated_mesh_energy_resolution",
    "fs_sampling_status",
    "convergence_status",
    "diagnosis",
    "notes",
}


def _matrix_to_tensor(matrix: np.ndarray) -> ConductivityTensor:
    return ConductivityTensor(matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0])


def _relative_eigen_split(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvals(matrix)
    scale = 0.5 * (abs(eigenvalues[0]) + abs(eigenvalues[1]))
    if np.isclose(scale, 0.0):
        return 0.0
    return float(abs(eigenvalues[0] - eigenvalues[1]) / scale)


def _relative_change(value: complex, reference: complex) -> float:
    if abs(value) < RATIO_EPS and abs(reference) < RATIO_EPS:
        return 0.0
    return float(abs(value - reference) / (abs(reference) + RATIO_EPS))


def _join_status(parts: list[str]) -> str:
    return "ok" if not parts else ";".join(dict.fromkeys(parts))


def _evaluate_sampling(
    sampling: str,
    nk: int,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
    refine_factor: int,
    shift_grid: int,
    fs_window_factor: float,
) -> tuple[np.ndarray, float, float, dict[str, float | int], dict[str, float | int], str]:
    if sampling == "uniform":
        mesh = shifted_bz_mesh(nk)
        weights = uniform_weights(mesh)
        matrix = single_mesh_normal_response(mesh, weights, eta_eV, omega_eV, temperature_K)
        fs = normal_fs_diagnostics(mesh, weights, eta_eV, omega_eV, temperature_K)
        metadata = {
            "num_kpoints_total": int(mesh.shape[0]),
            "num_fs_cells": 0,
            "num_refined_points": 0,
            "refined_area_fraction": 0.0,
            "weight_sum": float(np.sum(weights)),
        }
        return matrix, 0.0, 0.0, fs, metadata, "uniform midpoint mesh baseline"

    if sampling == "multishift_average":
        matrix, std_xx, rel_std, fs, metadata = multishift_normal_response(
            nk,
            eta_eV,
            omega_eV,
            temperature_K,
            shift_grid,
        )
        return matrix, std_xx, rel_std, fs, metadata, "multi-shift average control"

    if sampling == "fs_adaptive":
        mesh, weights, metadata = fs_adaptive_mesh(
            nk,
            eta_eV,
            omega_eV,
            temperature_K,
            refine_factor,
            fs_window_factor=fs_window_factor,
        )
        matrix = single_mesh_normal_response(mesh, weights, eta_eV, omega_eV, temperature_K)
        fs = normal_fs_diagnostics(mesh, weights, eta_eV, omega_eV, temperature_K)
        note = (
            "FS-adaptive weighted mesh diagnostic; "
            f"fs_window_eV={float(metadata['fs_window_eV']):g}; "
            f"weight_sum={float(metadata['weight_sum']):.12g}"
        )
        return matrix, 0.0, 0.0, fs, metadata, note

    raise ValueError("unknown sampling mode")


def benchmark_normal_fs_adaptive_integration(
    nk_list: list[int],
    eta_list: list[float],
    matsubara_list: list[int],
    temperature_K: float,
    refine_factor_list: list[int],
    fs_window_factor: float,
    sampling_modes: list[str],
    shift_grid: int,
) -> dict[str, np.ndarray]:
    """Run the FS-adaptive integration prototype benchmark."""

    if len(nk_list) < 2 or any(nk <= 0 for nk in nk_list):
        raise ValueError("nk_list must contain at least two positive values")
    if not eta_list or any(eta <= 0.0 for eta in eta_list):
        raise ValueError("eta_list must contain positive values")
    if not matsubara_list or any(n < 1 for n in matsubara_list):
        raise ValueError("matsubara_list must contain n >= 1")
    if not refine_factor_list or any(f <= 1 for f in refine_factor_list):
        raise ValueError("refine_factor_list must contain values greater than one")
    if fs_window_factor <= 0.0:
        raise ValueError("fs_window_factor must be positive")
    if shift_grid <= 0:
        raise ValueError("shift_grid must be positive")
    if any(mode not in SAMPLING_MODES for mode in sampling_modes):
        raise ValueError("unknown sampling mode")

    nk_values = np.asarray(sorted(nk_list), dtype=int)
    eta_values = np.asarray(sorted(eta_list), dtype=float)
    matsubara_values = np.asarray(sorted(matsubara_list), dtype=int)
    refine_values = np.asarray(sorted(refine_factor_list), dtype=int)
    rows: list[tuple[str, int, float, int, int]] = []
    for sampling in sampling_modes:
        factors = refine_values if sampling == "fs_adaptive" else np.array([1], dtype=int)
        for refine_factor in factors:
            for nk in nk_values:
                for eta in eta_values:
                    for n in matsubara_values:
                        rows.append((sampling, int(nk), float(eta), int(n), int(refine_factor)))

    row_count = len(rows)
    data: dict[str, np.ndarray] = {
        "sampling": np.empty(row_count, dtype="U24"),
        "nk": np.empty(row_count, dtype=int),
        "eta_eV": np.empty(row_count, dtype=float),
        "matsubara_index": np.empty(row_count, dtype=int),
        "omega_eV": np.empty(row_count, dtype=float),
        "refine_factor": np.empty(row_count, dtype=int),
        "shift_grid": np.full(row_count, shift_grid, dtype=int),
        "num_kpoints_total": np.empty(row_count, dtype=int),
        "num_fs_cells": np.empty(row_count, dtype=int),
        "num_refined_points": np.empty(row_count, dtype=int),
        "refined_area_fraction": np.empty(row_count, dtype=float),
        "sigma_xx": np.empty(row_count, dtype=complex),
        "sigma_yy": np.empty(row_count, dtype=complex),
        "sigma_xy": np.empty(row_count, dtype=complex),
        "sigma_yx": np.empty(row_count, dtype=complex),
        "sigma_xx_std_over_shifts": np.empty(row_count, dtype=float),
        "relative_std_over_shifts": np.empty(row_count, dtype=float),
        "delta": np.empty(row_count, dtype=complex),
        "relative_offdiag": np.empty(row_count, dtype=float),
        "relative_eigen_split": np.empty(row_count, dtype=float),
        "sheet_conductivity_xx": np.empty(row_count, dtype=complex),
        "reflection_dimensionless_xx": np.empty(row_count, dtype=complex),
        "relative_change_vs_largest_nk": np.full(row_count, np.nan, dtype=float),
        "relative_change_between_last_two_nk": np.full(row_count, np.nan, dtype=float),
        "eta_relative_change": np.full(row_count, np.nan, dtype=float),
        "min_abs_band_energy_on_mesh": np.empty(row_count, dtype=float),
        "points_within_eta": np.empty(row_count, dtype=int),
        "points_within_omega": np.empty(row_count, dtype=int),
        "points_within_kBT": np.empty(row_count, dtype=int),
        "fermi_window_weight_sum": np.empty(row_count, dtype=float),
        "estimated_mesh_energy_resolution": np.empty(row_count, dtype=float),
        "fs_sampling_status": np.empty(row_count, dtype="U192"),
        "convergence_status": np.empty(row_count, dtype="U192"),
        "diagnosis": np.empty(row_count, dtype="U160"),
        "notes": np.empty(row_count, dtype=object),
        "nk_list": nk_values,
        "eta_list": eta_values,
        "matsubara_list": matsubara_values,
        "refine_factor_list": refine_values,
        "temperature_K": np.array(temperature_K),
        "fs_window_factor": np.array(fs_window_factor),
    }

    index_by_key: dict[tuple[str, int, float, int, int], int] = {}
    for index, (sampling, nk, eta, n, refine_factor) in enumerate(rows):
        omega_eV = bosonic_matsubara_energy_eV(n, temperature_K)
        matrix, std_xx, rel_std, fs, metadata, note = _evaluate_sampling(
            sampling,
            nk,
            eta,
            omega_eV,
            temperature_K,
            refine_factor,
            shift_grid,
            fs_window_factor,
        )
        tensor = _matrix_to_tensor(matrix)
        diagnostics = conductivity_matrix_diagnostics(tensor)
        sheet = model_response_to_sheet_conductivity(matrix)
        reflection = sheet_conductivity_to_reflection_dimensionless(sheet)
        index_by_key[(sampling, nk, eta, n, refine_factor)] = index
        data["sampling"][index] = sampling
        data["nk"][index] = nk
        data["eta_eV"][index] = eta
        data["matsubara_index"][index] = n
        data["omega_eV"][index] = omega_eV
        data["refine_factor"][index] = refine_factor
        data["num_kpoints_total"][index] = int(metadata["num_kpoints_total"])
        data["num_fs_cells"][index] = int(metadata["num_fs_cells"])
        data["num_refined_points"][index] = int(metadata["num_refined_points"])
        data["refined_area_fraction"][index] = float(metadata["refined_area_fraction"])
        data["sigma_xx"][index] = matrix[0, 0]
        data["sigma_yy"][index] = matrix[1, 1]
        data["sigma_xy"][index] = matrix[0, 1]
        data["sigma_yx"][index] = matrix[1, 0]
        data["sigma_xx_std_over_shifts"][index] = std_xx
        data["relative_std_over_shifts"][index] = rel_std
        data["delta"][index] = complex(diagnostics["anisotropy_delta"])
        data["relative_offdiag"][index] = float(diagnostics["offdiag_norm"]) / (
            0.5 * (abs(matrix[0, 0]) + abs(matrix[1, 1])) + RATIO_EPS
        )
        data["relative_eigen_split"][index] = _relative_eigen_split(matrix)
        data["sheet_conductivity_xx"][index] = sheet.tensor.xx
        data["reflection_dimensionless_xx"][index] = reflection.tensor.xx
        for key, value in fs.items():
            data[key][index] = value
        data["notes"][index] = (
            "normal-state FS-adaptive BZ integration prototype only",
            note,
            "adaptive sampling changes quadrature only and does not alter the Kubo formula",
            "not a Casimir calculation",
        )

    largest_nk = int(nk_values[-1])
    previous_nk = int(nk_values[-2])
    smallest_eta = float(eta_values[0])
    largest_eta = float(eta_values[-1])
    for index, (sampling, nk, eta, n, refine_factor) in enumerate(rows):
        largest_index = index_by_key.get((sampling, largest_nk, eta, n, refine_factor))
        previous_index = index_by_key.get((sampling, previous_nk, eta, n, refine_factor))
        smallest_eta_index = index_by_key.get((sampling, nk, smallest_eta, n, refine_factor))
        largest_eta_index = index_by_key.get((sampling, nk, largest_eta, n, refine_factor))
        if largest_index is not None:
            data["relative_change_vs_largest_nk"][index] = _relative_change(
                data["sigma_xx"][index],
                data["sigma_xx"][largest_index],
            )
        if previous_index is not None and largest_index is not None:
            data["relative_change_between_last_two_nk"][index] = _relative_change(
                data["sigma_xx"][previous_index],
                data["sigma_xx"][largest_index],
            )
        if smallest_eta_index is not None and largest_eta_index is not None:
            data["eta_relative_change"][index] = _relative_change(
                data["sigma_xx"][largest_eta_index],
                data["sigma_xx"][smallest_eta_index],
            )

    improvement = _adaptive_improvement_status(data)
    unstable_refine = _unstable_refine_status(data)
    for index in range(row_count):
        conv_parts: list[str] = []
        fs_parts: list[str] = []
        diag_parts: list[str] = []
        if data["relative_change_between_last_two_nk"][index] < HIGH_NK_TOLERANCE:
            conv_parts.append("high_nk_converged")
        else:
            conv_parts.append("warning_high_nk_not_converged")
        if data["eta_relative_change"][index] >= HIGH_NK_TOLERANCE:
            conv_parts.append("warning_eta_sensitive")
        key = (
            str(data["sampling"][index]),
            float(data["eta_eV"][index]),
            int(data["matsubara_index"][index]),
            int(data["refine_factor"][index]),
        )
        if improvement.get(key, False):
            conv_parts.append("fs_adaptive_improves_convergence")
        if unstable_refine.get((float(data["eta_eV"][index]), int(data["matsubara_index"][index])), False):
            conv_parts.append("requires_triangle_or_contour_integration")
        if data["points_within_eta"][index] == 0:
            fs_parts.append("warning_fs_underresolved_eta")
        if str(data["sampling"][index]) == "fs_adaptive" and data["num_fs_cells"][index] == 0:
            fs_parts.append("warning_no_fs_cells_detected")
        if (
            abs(data["delta"][index]) > SYMMETRY_TOLERANCE
            or data["relative_offdiag"][index] > SYMMETRY_TOLERANCE
            or data["relative_eigen_split"][index] > SYMMETRY_TOLERANCE
        ):
            diag_parts.append("warning_symmetry")
        if not np.isfinite(data["sigma_xx"][index]):
            diag_parts.append("warning_nonfinite_response")
        data["convergence_status"][index] = _join_status(conv_parts)
        data["fs_sampling_status"][index] = _join_status(fs_parts)
        data["diagnosis"][index] = _join_status(diag_parts)

    return data


def _adaptive_improvement_status(data: dict[str, np.ndarray]) -> dict[tuple[str, float, int, int], bool]:
    status: dict[tuple[str, float, int, int], bool] = {}
    for eta in sorted(set(float(item) for item in data["eta_eV"])):
        for n in sorted(set(int(item) for item in data["matsubara_index"])):
            uniform_mask = (
                (data["sampling"] == "uniform")
                & np.isclose(data["eta_eV"], eta)
                & (data["matsubara_index"] == n)
            )
            multi_mask = (
                (data["sampling"] == "multishift_average")
                & np.isclose(data["eta_eV"], eta)
                & (data["matsubara_index"] == n)
            )
            if not np.any(uniform_mask) or not np.any(multi_mask):
                continue
            uniform_change = float(np.nanmax(data["relative_change_between_last_two_nk"][uniform_mask]))
            multi_change = float(np.nanmax(data["relative_change_between_last_two_nk"][multi_mask]))
            baseline = min(uniform_change, multi_change)
            for refine_factor in sorted(set(int(x) for x in data["refine_factor"][data["sampling"] == "fs_adaptive"])):
                mask = (
                    (data["sampling"] == "fs_adaptive")
                    & np.isclose(data["eta_eV"], eta)
                    & (data["matsubara_index"] == n)
                    & (data["refine_factor"] == refine_factor)
                )
                if not np.any(mask):
                    continue
                adaptive_change = float(np.nanmax(data["relative_change_between_last_two_nk"][mask]))
                status[("fs_adaptive", eta, n, refine_factor)] = adaptive_change < baseline
    return status


def _unstable_refine_status(data: dict[str, np.ndarray]) -> dict[tuple[float, int], bool]:
    status: dict[tuple[float, int], bool] = {}
    for eta in sorted(set(float(item) for item in data["eta_eV"])):
        for n in sorted(set(int(item) for item in data["matsubara_index"])):
            mask = (
                (data["sampling"] == "fs_adaptive")
                & np.isclose(data["eta_eV"], eta)
                & (data["matsubara_index"] == n)
                & (data["nk"] == int(np.max(data["nk"])))
            )
            if np.count_nonzero(mask) < 2:
                status[(eta, n)] = False
                continue
            values = data["sigma_xx"][mask]
            order = np.argsort(data["refine_factor"][mask])
            ordered = values[order]
            changes = [
                _relative_change(ordered[i - 1], ordered[i])
                for i in range(1, ordered.size)
            ]
            status[(eta, n)] = bool(changes and max(changes) >= HIGH_NK_TOLERANCE)
    return status


def output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    resolved_prefix = output_prefix.resolve()
    project_data_root = (ROOT / "outputs" / "normal_state" / "fs_adaptive_integration" / "data").resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "outputs" / "normal_state" / "fs_adaptive_integration" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return (
        npz_path,
        csv_path,
        figure_dir / f"{output_prefix.name}_sigma_xx_vs_nk.png",
        figure_dir / f"{output_prefix.name}_last_two_change_by_sampling.png",
        figure_dir / f"{output_prefix.name}_sigma_xx_vs_refine_factor.png",
        figure_dir / f"{output_prefix.name}_fs_points_vs_nk.png",
        figure_dir / f"{output_prefix.name}_refined_area_fraction_vs_nk.png",
        figure_dir / f"{output_prefix.name}_sigma_vs_fermi_weight.png",
        figure_dir / f"{output_prefix.name}_symmetry_vs_nk.png",
    )


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    return value


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    paths = output_paths(output_prefix)
    npz_path, csv_path, sigma_nk_plot, last_two_plot, refine_plot, fs_points_plot, area_plot, fermi_plot, sym_plot = paths
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    sigma_nk_plot.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)

    fieldnames = [
        "sampling",
        "nk",
        "eta_eV",
        "matsubara_index",
        "omega_eV",
        "refine_factor",
        "shift_grid",
        "num_kpoints_total",
        "num_fs_cells",
        "num_refined_points",
        "refined_area_fraction",
        "sigma_xx",
        "sigma_yy",
        "sigma_xy",
        "sigma_yx",
        "sigma_xx_std_over_shifts",
        "relative_std_over_shifts",
        "delta",
        "relative_offdiag",
        "relative_eigen_split",
        "sheet_conductivity_xx",
        "reflection_dimensionless_xx",
        "relative_change_vs_largest_nk",
        "relative_change_between_last_two_nk",
        "eta_relative_change",
        "min_abs_band_energy_on_mesh",
        "points_within_eta",
        "points_within_omega",
        "points_within_kBT",
        "fermi_window_weight_sum",
        "estimated_mesh_energy_resolution",
        "fs_sampling_status",
        "convergence_status",
        "diagnosis",
        "notes",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(data["sampling"].size):
            writer.writerow({name: _csv_value(data[name][index]) for name in fieldnames})

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    samplings = list(dict.fromkeys(str(item) for item in data["sampling"]))
    reference_eta = min(float(item) for item in data["eta_eV"])
    reference_n = min(int(item) for item in data["matsubara_index"])
    reference_nk = max(int(item) for item in data["nk"])
    reference_refine = max(int(item) for item in data["refine_factor"])

    fig_sigma, ax_sigma = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        refine_factor = reference_refine if sampling == "fs_adaptive" else 1
        mask = (
            (data["sampling"] == sampling)
            & np.isclose(data["eta_eV"], reference_eta)
            & (data["matsubara_index"] == reference_n)
            & (data["refine_factor"] == refine_factor)
        )
        ax_sigma.plot(data["nk"][mask], data["sigma_xx"][mask].real, marker="o", label=sampling)
    ax_sigma.set_xlabel(r"$N_k$")
    ax_sigma.set_ylabel(r"Re $\sigma_{xx}$")
    ax_sigma.set_title(rf"FS-adaptive response at n={reference_n}, $\eta={reference_eta:g}$")
    style_publication_axis(ax_sigma)
    save_publication_figure(fig_sigma, sigma_nk_plot)
    plt.close(fig_sigma)

    fig_last, ax_last = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    labels = []
    values = []
    for sampling in samplings:
        for refine_factor in sorted(set(int(x) for x in data["refine_factor"][data["sampling"] == sampling])):
            mask = (
                (data["sampling"] == sampling)
                & np.isclose(data["eta_eV"], reference_eta)
                & (data["matsubara_index"] == reference_n)
                & (data["refine_factor"] == refine_factor)
            )
            labels.append(f"{sampling}\nr={refine_factor}")
            values.append(float(np.nanmax(data["relative_change_between_last_two_nk"][mask])))
    x = np.arange(len(labels))
    ax_last.bar(x, values)
    ax_last.axhline(HIGH_NK_TOLERANCE, color="black", linestyle="--", linewidth=1.0)
    ax_last.set_xticks(x, labels, rotation=35, ha="right")
    ax_last.set_ylabel("last-two-Nk relative change")
    ax_last.set_title("high-Nk convergence by sampling")
    style_publication_axis(ax_last, legend=False)
    save_publication_figure(fig_last, last_two_plot)
    plt.close(fig_last)

    fig_refine, ax_refine = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for nk in sorted(set(int(item) for item in data["nk"])):
        mask = (
            (data["sampling"] == "fs_adaptive")
            & (data["nk"] == nk)
            & np.isclose(data["eta_eV"], reference_eta)
            & (data["matsubara_index"] == reference_n)
        )
        if np.any(mask):
            ax_refine.plot(data["refine_factor"][mask], data["sigma_xx"][mask].real, marker="o", label=f"Nk={nk}")
    ax_refine.set_xlabel("refine factor")
    ax_refine.set_ylabel(r"Re $\sigma_{xx}$")
    ax_refine.set_title("FS-adaptive response versus refine factor")
    style_publication_axis(ax_refine)
    save_publication_figure(fig_refine, refine_plot)
    plt.close(fig_refine)

    fig_fs, ax_fs = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        refine_factor = reference_refine if sampling == "fs_adaptive" else 1
        mask = (
            (data["sampling"] == sampling)
            & np.isclose(data["eta_eV"], reference_eta)
            & (data["matsubara_index"] == reference_n)
            & (data["refine_factor"] == refine_factor)
        )
        ax_fs.plot(data["nk"][mask], data["points_within_eta"][mask], marker="o", label=f"{sampling} eta")
        ax_fs.plot(data["nk"][mask], data["points_within_kBT"][mask], marker="s", linestyle="--", label=f"{sampling} kBT")
    ax_fs.set_xlabel(r"$N_k$")
    ax_fs.set_ylabel("band-state count")
    ax_fs.set_title("Fermi-window mesh counts")
    style_publication_axis(ax_fs)
    save_publication_figure(fig_fs, fs_points_plot)
    plt.close(fig_fs)

    fig_area, ax_area = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for refine_factor in sorted(set(int(x) for x in data["refine_factor"][data["sampling"] == "fs_adaptive"])):
        mask = (
            (data["sampling"] == "fs_adaptive")
            & (data["refine_factor"] == refine_factor)
            & np.isclose(data["eta_eV"], reference_eta)
            & (data["matsubara_index"] == reference_n)
        )
        ax_area.plot(data["nk"][mask], data["refined_area_fraction"][mask], marker="o", label=f"r={refine_factor}")
    ax_area.set_xlabel(r"$N_k$")
    ax_area.set_ylabel("refined area fraction")
    ax_area.set_title("FS-adaptive refined BZ area")
    style_publication_axis(ax_area)
    save_publication_figure(fig_area, area_plot)
    plt.close(fig_area)

    fig_fermi, ax_fermi = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        mask = (data["sampling"] == sampling) & np.isclose(data["eta_eV"], reference_eta)
        ax_fermi.scatter(data["fermi_window_weight_sum"][mask], data["sigma_xx"][mask].real, label=sampling)
    ax_fermi.set_xlabel("Fermi-window weight sum")
    ax_fermi.set_ylabel(r"Re $\sigma_{xx}$")
    ax_fermi.set_title("normal response versus Fermi-window sampling")
    style_publication_axis(ax_fermi)
    save_publication_figure(fig_fermi, fermi_plot)
    plt.close(fig_fermi)

    fig_sym, ax_sym = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        refine_factor = reference_refine if sampling == "fs_adaptive" else 1
        mask = (
            (data["sampling"] == sampling)
            & np.isclose(data["eta_eV"], reference_eta)
            & (data["matsubara_index"] == reference_n)
            & (data["refine_factor"] == refine_factor)
        )
        ax_sym.plot(data["nk"][mask], np.abs(data["delta"][mask]), marker="o", label=f"{sampling} |delta|")
        ax_sym.plot(data["nk"][mask], data["relative_offdiag"][mask], marker="s", linestyle="--", label=f"{sampling} offdiag")
    ax_sym.set_yscale("symlog", linthresh=1e-16)
    ax_sym.set_xlabel(r"$N_k$")
    ax_sym.set_ylabel("relative diagnostic")
    ax_sym.set_title("normal response C4 diagnostics")
    style_publication_axis(ax_sym)
    save_publication_figure(fig_sym, sym_plot)
    plt.close(fig_sym)

    return paths


def print_summary(data: dict[str, np.ndarray]) -> None:
    print(f"row_count = {data['sampling'].size}")
    print(f"max_relative_change_between_last_two_nk = {float(np.nanmax(data['relative_change_between_last_two_nk']))}")
    print(f"max_eta_relative_change = {float(np.nanmax(data['eta_relative_change']))}")
    print(f"min_points_within_eta = {int(np.nanmin(data['points_within_eta']))}")
    print(f"max_refined_area_fraction = {float(np.nanmax(data['refined_area_fraction']))}")
    print(f"max_num_kpoints_total = {int(np.nanmax(data['num_kpoints_total']))}")
    print(f"max_symmetry_diagnostic = {float(max(np.nanmax(np.abs(data['delta'])), np.nanmax(data['relative_offdiag']), np.nanmax(data['relative_eigen_split'])))}")
    print(f"convergence_statuses = {sorted(set(str(item) for item in data['convergence_status']))}")
    print(f"fs_sampling_statuses = {sorted(set(str(item) for item in data['fs_sampling_status']))}")
    print(f"diagnoses = {sorted(set(str(item) for item in data['diagnosis']))}")
    print("note = normal-state FS-adaptive integration prototype only; not a Casimir result.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--eta-list", nargs="+", type=float, default=list(DEFAULT_ETA_LIST))
    parser.add_argument("--matsubara-list", nargs="+", type=int, default=list(DEFAULT_MATSUBARA_LIST))
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--refine-factor-list", nargs="+", type=int, default=list(DEFAULT_REFINE_FACTOR_LIST))
    parser.add_argument("--fs-window-factor", type=float, default=1.0)
    parser.add_argument("--sampling", nargs="+", choices=SAMPLING_MODES, default=list(SAMPLING_MODES))
    parser.add_argument("--shift-grid", type=int, default=4)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT
        / "outputs"
        / "normal_state"
        / "fs_adaptive_integration"
        / "data"
        / "fs_adaptive",
    )
    args = parser.parse_args()

    data = benchmark_normal_fs_adaptive_integration(
        args.nk_list,
        args.eta_list,
        args.matsubara_list,
        args.temperature,
        args.refine_factor_list,
        args.fs_window_factor,
        args.sampling,
        args.shift_grid,
    )
    paths = save_outputs(data, args.output_prefix)
    print_summary(data)
    print(f"npz_path = {paths[0]}")
    print(f"csv_path = {paths[1]}")
    print(
        "figure_paths = "
        f"{paths[2]}, {paths[3]}, {paths[4]}, {paths[5]}, {paths[6]}, {paths[7]}, {paths[8]}"
    )


if __name__ == "__main__":
    main()
