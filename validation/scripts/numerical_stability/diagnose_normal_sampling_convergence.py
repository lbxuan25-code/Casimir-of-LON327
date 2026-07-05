#!/usr/bin/env python3
"""Diagnose normal-state low-Matsubara k-space sampling convergence.

This script reuses the existing normal-state Kubo implementation without
changing its default behavior. Shifted and averaged meshes are diagnostic
sampling alternatives for understanding Fermi-surface-sensitive convergence.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import (  # noqa: E402
    KuboConfig,
    bosonic_matsubara_energy_eV,
    conductivity_matrix_diagnostics,
    k_weights,
    kubo_conductivity_imag_axis,
)
from lno327 import ConductivityTensor, negative_fermi_derivative  # noqa: E402
from lno327.constants import KB_EV_PER_K  # noqa: E402
from lno327.models.lno327_four_orbital.normal import normal_state_hamiltonian  # noqa: E402
from validation.lib.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

SAMPLING_MODES = ("uniform", "shifted", "average")
DEFAULT_NK_LIST = (32, 48, 64, 80, 96, 128)
DEFAULT_ETA_LIST = (1e-3, 5e-4, 2e-4, 1e-4)
DEFAULT_MATSUBARA_LIST = (1, 2, 5)
HIGH_NK_TOLERANCE = 0.02
SYMMETRY_TOLERANCE = 1e-8
FS_POINT_WARNING_THRESHOLD = 8
RATIO_EPS = 1e-300

AVERAGE_SHIFTS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))

REQUIRED_NPZ_FIELDS = {
    "sampling",
    "nk",
    "eta_eV",
    "matsubara_index",
    "omega_eV",
    "sigma_xx",
    "sigma_yy",
    "sigma_xy",
    "sigma_yx",
    "delta",
    "relative_offdiag",
    "relative_eigen_split",
    "relative_change_vs_largest_nk",
    "relative_change_between_last_two_nk",
    "eta_relative_change",
    "min_abs_band_energy_on_mesh",
    "points_within_eta",
    "points_within_omega",
    "points_within_kBT",
    "fermi_window_weight_sum",
    "estimated_mesh_energy_resolution",
    "sampling_convergence_status",
    "fs_sampling_status",
    "diagnosis",
    "notes",
}


def shifted_bz_mesh(nk: int, shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Return a uniform mesh shifted by fractions of one grid spacing."""

    if nk <= 0:
        raise ValueError("nk must be positive")
    step = 2.0 * np.pi / nk
    kx = -np.pi + (np.arange(nk) + 0.5 + shift[0]) * step
    ky = -np.pi + (np.arange(nk) + 0.5 + shift[1]) * step
    kx = ((kx + np.pi) % (2.0 * np.pi)) - np.pi
    ky = ((ky + np.pi) % (2.0 * np.pi)) - np.pi
    return np.array([(float(x), float(y)) for x in kx for y in ky], dtype=float)


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


def _mesh_band_energies(mesh: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.eigvalsh(normal_state_hamiltonian(kx, ky)) for kx, ky in mesh], dtype=float)


def _estimated_energy_resolution(energies: np.ndarray) -> float:
    flat = np.sort(np.ravel(energies))
    diffs = np.diff(flat)
    diffs = diffs[diffs > 1e-14]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _fs_diagnostics_for_mesh(
    mesh: np.ndarray,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
) -> dict[str, float | int]:
    energies = _mesh_band_energies(mesh)
    abs_energies = np.abs(energies)
    temperature_eV = temperature_K * KB_EV_PER_K
    weights = k_weights(mesh)
    fermi_window = negative_fermi_derivative(energies, 0.0, temperature_eV, eta_eV)
    return {
        "min_abs_band_energy_on_mesh": float(np.min(abs_energies)),
        "points_within_eta": int(np.count_nonzero(abs_energies <= eta_eV)),
        "points_within_omega": int(np.count_nonzero(abs_energies <= omega_eV)),
        "points_within_kBT": int(np.count_nonzero(abs_energies <= temperature_eV)),
        "fermi_window_weight_sum": float(np.sum(fermi_window * weights[:, None])),
        "estimated_mesh_energy_resolution": _estimated_energy_resolution(energies),
    }


def _single_mesh_response(
    mesh: np.ndarray,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
) -> ConductivityTensor:
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    return kubo_conductivity_imag_axis(mesh, config, k_weights(mesh))


def _sampling_response_and_fs(
    sampling: str,
    nk: int,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if sampling == "uniform":
        meshes = [shifted_bz_mesh(nk, (0.0, 0.0))]
    elif sampling == "shifted":
        meshes = [shifted_bz_mesh(nk, (0.5, 0.5))]
    elif sampling == "average":
        meshes = [shifted_bz_mesh(nk, shift) for shift in AVERAGE_SHIFTS]
    else:
        raise ValueError("unknown sampling mode")

    matrices = []
    fs_items = []
    for mesh in meshes:
        matrices.append(_single_mesh_response(mesh, eta_eV, omega_eV, temperature_K).matrix())
        fs_items.append(_fs_diagnostics_for_mesh(mesh, eta_eV, omega_eV, temperature_K))
    matrix = np.mean(np.stack(matrices, axis=0), axis=0)
    fs = {
        "min_abs_band_energy_on_mesh": float(min(item["min_abs_band_energy_on_mesh"] for item in fs_items)),
        "points_within_eta": int(sum(int(item["points_within_eta"]) for item in fs_items)),
        "points_within_omega": int(sum(int(item["points_within_omega"]) for item in fs_items)),
        "points_within_kBT": int(sum(int(item["points_within_kBT"]) for item in fs_items)),
        "fermi_window_weight_sum": float(np.mean([float(item["fermi_window_weight_sum"]) for item in fs_items])),
        "estimated_mesh_energy_resolution": float(
            np.mean([float(item["estimated_mesh_energy_resolution"]) for item in fs_items])
        ),
    }
    return matrix, fs


def _fs_correlation_status(response_values: list[complex], fermi_weight_values: list[float]) -> str | None:
    if len(response_values) < 3:
        return None
    x = np.asarray(fermi_weight_values, dtype=float)
    y = np.asarray([abs(value) for value in response_values], dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    corr = float(np.corrcoef(x, y)[0, 1])
    if np.isfinite(corr) and abs(corr) >= 0.8:
        return "warning_fermi_surface_sampling_sensitive"
    return None


def diagnose_normal_sampling_convergence(
    nk_list: list[int],
    eta_list: list[float],
    matsubara_list: list[int],
    temperature_K: float,
    sampling_modes: list[str],
) -> dict[str, np.ndarray]:
    """Run normal-state k-space sampling diagnostics."""

    if len(nk_list) < 2 or any(nk <= 0 for nk in nk_list):
        raise ValueError("nk_list must contain at least two positive values")
    if not eta_list or any(eta <= 0.0 for eta in eta_list):
        raise ValueError("eta_list must contain positive values")
    if not matsubara_list or any(n < 1 for n in matsubara_list):
        raise ValueError("matsubara_list must contain n >= 1")
    if any(mode not in SAMPLING_MODES for mode in sampling_modes):
        raise ValueError("unknown sampling mode")

    nk_values = np.asarray(sorted(nk_list), dtype=int)
    eta_values = np.asarray(sorted(eta_list), dtype=float)
    matsubara_values = np.asarray(sorted(matsubara_list), dtype=int)
    rows = [
        (sampling, int(nk), float(eta), int(n))
        for sampling in sampling_modes
        for nk in nk_values
        for eta in eta_values
        for n in matsubara_values
    ]
    row_count = len(rows)
    data: dict[str, np.ndarray] = {
        "sampling": np.empty(row_count, dtype="U16"),
        "nk": np.empty(row_count, dtype=int),
        "eta_eV": np.empty(row_count, dtype=float),
        "matsubara_index": np.empty(row_count, dtype=int),
        "omega_eV": np.empty(row_count, dtype=float),
        "sigma_xx": np.empty(row_count, dtype=complex),
        "sigma_yy": np.empty(row_count, dtype=complex),
        "sigma_xy": np.empty(row_count, dtype=complex),
        "sigma_yx": np.empty(row_count, dtype=complex),
        "delta": np.empty(row_count, dtype=complex),
        "relative_offdiag": np.empty(row_count, dtype=float),
        "relative_eigen_split": np.empty(row_count, dtype=float),
        "relative_change_vs_largest_nk": np.full(row_count, np.nan, dtype=float),
        "relative_change_between_last_two_nk": np.full(row_count, np.nan, dtype=float),
        "eta_relative_change": np.full(row_count, np.nan, dtype=float),
        "min_abs_band_energy_on_mesh": np.empty(row_count, dtype=float),
        "points_within_eta": np.empty(row_count, dtype=int),
        "points_within_omega": np.empty(row_count, dtype=int),
        "points_within_kBT": np.empty(row_count, dtype=int),
        "fermi_window_weight_sum": np.empty(row_count, dtype=float),
        "estimated_mesh_energy_resolution": np.empty(row_count, dtype=float),
        "sampling_convergence_status": np.empty(row_count, dtype="U128"),
        "fs_sampling_status": np.empty(row_count, dtype="U192"),
        "diagnosis": np.empty(row_count, dtype="U128"),
        "notes": np.empty(row_count, dtype=object),
        "nk_list": nk_values,
        "eta_list": eta_values,
        "matsubara_list": matsubara_values,
        "temperature_K": np.array(temperature_K),
    }

    index_by_key: dict[tuple[str, int, float, int], int] = {}
    for index, (sampling, nk, eta, n) in enumerate(rows):
        omega_eV = bosonic_matsubara_energy_eV(n, temperature_K)
        matrix, fs = _sampling_response_and_fs(sampling, nk, eta, omega_eV, temperature_K)
        tensor = _matrix_to_tensor(matrix)
        diagnostics = conductivity_matrix_diagnostics(tensor)
        index_by_key[(sampling, nk, eta, n)] = index
        data["sampling"][index] = sampling
        data["nk"][index] = nk
        data["eta_eV"][index] = eta
        data["matsubara_index"][index] = n
        data["omega_eV"][index] = omega_eV
        data["sigma_xx"][index] = matrix[0, 0]
        data["sigma_yy"][index] = matrix[1, 1]
        data["sigma_xy"][index] = matrix[0, 1]
        data["sigma_yx"][index] = matrix[1, 0]
        data["delta"][index] = complex(diagnostics["anisotropy_delta"])
        data["relative_offdiag"][index] = float(diagnostics["offdiag_norm"]) / (
            0.5 * (abs(matrix[0, 0]) + abs(matrix[1, 1])) + RATIO_EPS
        )
        data["relative_eigen_split"][index] = _relative_eigen_split(matrix)
        for key, value in fs.items():
            data[key][index] = value
        data["notes"][index] = (
            "normal-state low-Matsubara sampling diagnostic only",
            "shifted/average sampling does not alter the Kubo formula",
            "not a Casimir calculation",
        )

    largest_nk = int(nk_values[-1])
    previous_nk = int(nk_values[-2])
    smallest_eta = float(eta_values[0])
    largest_eta = float(eta_values[-1])

    for index, (sampling, nk, eta, n) in enumerate(rows):
        largest_index = index_by_key.get((sampling, largest_nk, eta, n))
        previous_index = index_by_key.get((sampling, previous_nk, eta, n))
        smallest_eta_index = index_by_key.get((sampling, nk, smallest_eta, n))
        largest_eta_index = index_by_key.get((sampling, nk, largest_eta, n))
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

    irregular_status: dict[tuple[str, float, int], str | None] = {}
    correlation_status: dict[tuple[str, float, int], str | None] = {}
    for sampling in sampling_modes:
        for eta in eta_values:
            for n in matsubara_values:
                indices = [
                    index_by_key[(sampling, int(nk), float(eta), int(n))]
                    for nk in nk_values
                    if (sampling, int(nk), float(eta), int(n)) in index_by_key
                ]
                min_values = data["min_abs_band_energy_on_mesh"][indices]
                if min_values.size >= 3 and np.ptp(min_values) / (np.nanmedian(min_values) + RATIO_EPS) > 1.0:
                    irregular_status[(sampling, float(eta), int(n))] = "warning_mesh_hits_fermi_surface_irregularly"
                else:
                    irregular_status[(sampling, float(eta), int(n))] = None
                correlation_status[(sampling, float(eta), int(n))] = _fs_correlation_status(
                    [data["sigma_xx"][index] for index in indices],
                    [float(data["fermi_window_weight_sum"][index]) for index in indices],
                )

    for index, (sampling, _nk, eta, n) in enumerate(rows):
        conv_parts: list[str] = []
        fs_parts: list[str] = []
        diag_parts: list[str] = []
        if data["relative_change_between_last_two_nk"][index] >= HIGH_NK_TOLERANCE:
            conv_parts.append("warning_high_nk_not_converged")
        if data["eta_relative_change"][index] >= HIGH_NK_TOLERANCE:
            conv_parts.append("warning_eta_sensitive")
        if data["points_within_eta"][index] < FS_POINT_WARNING_THRESHOLD:
            fs_parts.append("warning_fs_underresolved_eta")
        irregular = irregular_status[(sampling, float(eta), int(n))]
        if irregular is not None:
            fs_parts.append(irregular)
        correlated = correlation_status[(sampling, float(eta), int(n))]
        if correlated is not None:
            fs_parts.append(correlated)
        if (
            abs(data["delta"][index]) > SYMMETRY_TOLERANCE
            or data["relative_offdiag"][index] > SYMMETRY_TOLERANCE
            or data["relative_eigen_split"][index] > SYMMETRY_TOLERANCE
        ):
            diag_parts.append("warning_symmetry")
        if not np.isfinite(data["sigma_xx"][index]):
            diag_parts.append("warning_nonfinite_response")
        data["sampling_convergence_status"][index] = "ok" if not conv_parts else ";".join(conv_parts)
        data["fs_sampling_status"][index] = "ok" if not fs_parts else ";".join(dict.fromkeys(fs_parts))
        data["diagnosis"][index] = "ok" if not diag_parts else ";".join(diag_parts)

    return data


def output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    resolved_prefix = output_prefix.resolve()
    project_data_root = (ROOT / "validation" / "outputs" / "archive" / "normal_state" / "sampling_convergence" / "data").resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "validation" / "outputs" / "archive" / "normal_state" / "sampling_convergence" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return (
        npz_path,
        csv_path,
        figure_dir / f"{output_prefix.name}_sigma_xx_vs_nk.png",
        figure_dir / f"{output_prefix.name}_relative_change_vs_nk.png",
        figure_dir / f"{output_prefix.name}_last_two_summary.png",
        figure_dir / f"{output_prefix.name}_sigma_xx_vs_eta.png",
        figure_dir / f"{output_prefix.name}_fs_points_vs_nk.png",
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
    (
        npz_path,
        csv_path,
        sigma_nk_plot,
        change_plot,
        last_two_plot,
        sigma_eta_plot,
        fs_points_plot,
        sigma_fermi_plot,
        symmetry_plot,
    ) = paths
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    sigma_nk_plot.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)

    fieldnames = [
        "sampling",
        "nk",
        "eta_eV",
        "matsubara_index",
        "omega_eV",
        "sigma_xx",
        "sigma_yy",
        "sigma_xy",
        "sigma_yx",
        "delta",
        "relative_offdiag",
        "relative_eigen_split",
        "relative_change_vs_largest_nk",
        "relative_change_between_last_two_nk",
        "eta_relative_change",
        "min_abs_band_energy_on_mesh",
        "points_within_eta",
        "points_within_omega",
        "points_within_kBT",
        "fermi_window_weight_sum",
        "estimated_mesh_energy_resolution",
        "sampling_convergence_status",
        "fs_sampling_status",
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
    n_values = list(dict.fromkeys(int(item) for item in data["matsubara_index"]))
    eta_values = list(dict.fromkeys(float(item) for item in data["eta_eV"]))
    reference_eta = min(eta_values)
    reference_n = min(n_values)
    reference_nk = max(int(item) for item in data["nk"])

    fig_sigma_nk, ax_sigma_nk = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        for n in n_values:
            mask = (data["sampling"] == sampling) & np.isclose(data["eta_eV"], reference_eta) & (data["matsubara_index"] == n)
            ax_sigma_nk.plot(data["nk"][mask], data["sigma_xx"][mask].real, marker="o", label=f"{sampling} n={n}")
    ax_sigma_nk.set_xlabel(r"$N_k$")
    ax_sigma_nk.set_ylabel(r"Re $\sigma_{xx}$")
    ax_sigma_nk.set_title(rf"normal response sampling convergence at $\eta={reference_eta:g}$")
    style_publication_axis(ax_sigma_nk)
    save_publication_figure(fig_sigma_nk, sigma_nk_plot)
    plt.close(fig_sigma_nk)

    fig_change, ax_change = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        mask = (data["sampling"] == sampling) & np.isclose(data["eta_eV"], reference_eta) & (data["matsubara_index"] == reference_n)
        ax_change.plot(data["nk"][mask], data["relative_change_vs_largest_nk"][mask], marker="o", label=sampling)
    ax_change.axhline(HIGH_NK_TOLERANCE, color="black", linestyle="--", linewidth=1.0)
    ax_change.set_yscale("symlog", linthresh=1e-4)
    ax_change.set_xlabel(r"$N_k$")
    ax_change.set_ylabel("relative change to largest Nk")
    ax_change.set_title(rf"relative convergence at n={reference_n}, $\eta={reference_eta:g}$")
    style_publication_axis(ax_change)
    save_publication_figure(fig_change, change_plot)
    plt.close(fig_change)

    fig_last, ax_last = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    labels = []
    values = []
    for sampling in samplings:
        mask = (data["sampling"] == sampling) & np.isclose(data["eta_eV"], reference_eta)
        labels.append(sampling)
        values.append(float(np.nanmax(data["relative_change_between_last_two_nk"][mask])))
    x = np.arange(len(labels))
    ax_last.bar(x, values)
    ax_last.axhline(HIGH_NK_TOLERANCE, color="black", linestyle="--", linewidth=1.0)
    ax_last.set_xticks(x, labels)
    ax_last.set_ylabel("last-two-Nk relative change")
    ax_last.set_title("sampling last-two-Nk convergence summary")
    style_publication_axis(ax_last, legend=False)
    save_publication_figure(fig_last, last_two_plot)
    plt.close(fig_last)

    fig_sigma_eta, ax_sigma_eta = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        mask = (data["sampling"] == sampling) & (data["nk"] == reference_nk) & (data["matsubara_index"] == reference_n)
        order = np.argsort(data["eta_eV"][mask])
        ax_sigma_eta.plot(data["eta_eV"][mask][order], data["sigma_xx"][mask].real[order], marker="o", label=sampling)
    ax_sigma_eta.set_xscale("log")
    ax_sigma_eta.set_xlabel(r"$\eta$ (eV)")
    ax_sigma_eta.set_ylabel(r"Re $\sigma_{xx}$")
    ax_sigma_eta.set_title(rf"eta sensitivity at $N_k={reference_nk}, n={reference_n}$")
    style_publication_axis(ax_sigma_eta)
    save_publication_figure(fig_sigma_eta, sigma_eta_plot)
    plt.close(fig_sigma_eta)

    fig_fs, ax_fs = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        mask = (data["sampling"] == sampling) & np.isclose(data["eta_eV"], reference_eta) & (data["matsubara_index"] == reference_n)
        ax_fs.plot(data["nk"][mask], data["points_within_eta"][mask], marker="o", label=f"{sampling} eta")
        ax_fs.plot(data["nk"][mask], data["points_within_kBT"][mask], marker="s", linestyle="--", label=f"{sampling} kBT")
    ax_fs.set_xlabel(r"$N_k$")
    ax_fs.set_ylabel("band-state count")
    ax_fs.set_title("Fermi-window mesh counts")
    style_publication_axis(ax_fs)
    save_publication_figure(fig_fs, fs_points_plot)
    plt.close(fig_fs)

    fig_fermi, ax_fermi = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        mask = (data["sampling"] == sampling) & np.isclose(data["eta_eV"], reference_eta)
        ax_fermi.scatter(data["fermi_window_weight_sum"][mask], data["sigma_xx"][mask].real, label=sampling)
    ax_fermi.set_xlabel("Fermi-window weight sum")
    ax_fermi.set_ylabel(r"Re $\sigma_{xx}$")
    ax_fermi.set_title("normal response versus Fermi-window sampling")
    style_publication_axis(ax_fermi)
    save_publication_figure(fig_fermi, sigma_fermi_plot)
    plt.close(fig_fermi)

    fig_sym, ax_sym = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        mask = (data["sampling"] == sampling) & np.isclose(data["eta_eV"], reference_eta) & (data["matsubara_index"] == reference_n)
        ax_sym.plot(data["nk"][mask], np.abs(data["delta"][mask]), marker="o", label=f"{sampling} |delta|")
        ax_sym.plot(data["nk"][mask], data["relative_offdiag"][mask], marker="s", linestyle="--", label=f"{sampling} offdiag")
        ax_sym.plot(data["nk"][mask], data["relative_eigen_split"][mask], marker="^", linestyle=":", label=f"{sampling} eig split")
    ax_sym.set_yscale("symlog", linthresh=1e-16)
    ax_sym.set_xlabel(r"$N_k$")
    ax_sym.set_ylabel("relative diagnostic")
    ax_sym.set_title("normal response C4 diagnostics")
    style_publication_axis(ax_sym)
    save_publication_figure(fig_sym, symmetry_plot)
    plt.close(fig_sym)

    return paths


def print_summary(data: dict[str, np.ndarray]) -> None:
    max_last_two = float(np.nanmax(data["relative_change_between_last_two_nk"]))
    max_eta = float(np.nanmax(data["eta_relative_change"]))
    min_points_eta = int(np.nanmin(data["points_within_eta"]))
    max_symmetry = float(
        max(
            np.nanmax(np.abs(data["delta"])),
            np.nanmax(data["relative_offdiag"]),
            np.nanmax(data["relative_eigen_split"]),
        )
    )
    sampling_statuses = sorted(set(str(item) for item in data["sampling_convergence_status"]))
    fs_statuses = sorted(set(str(item) for item in data["fs_sampling_status"]))
    diagnoses = sorted(set(str(item) for item in data["diagnosis"]))
    print(f"row_count = {data['sampling'].size}")
    print(f"max_relative_change_between_last_two_nk = {max_last_two}")
    print(f"max_eta_relative_change = {max_eta}")
    print(f"min_points_within_eta = {min_points_eta}")
    print(f"max_symmetry_diagnostic = {max_symmetry}")
    print(f"sampling_convergence_statuses = {sampling_statuses}")
    print(f"fs_sampling_statuses = {fs_statuses}")
    print(f"diagnoses = {diagnoses}")
    print("note = normal-state sampling diagnostic only; not a Casimir result.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--eta-list", nargs="+", type=float, default=list(DEFAULT_ETA_LIST))
    parser.add_argument("--matsubara-list", nargs="+", type=int, default=list(DEFAULT_MATSUBARA_LIST))
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--sampling", nargs="+", choices=SAMPLING_MODES, default=["uniform"])
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT
        / "outputs"
        / "normal_state"
        / "sampling_convergence"
        / "data"
        / "normal_sampling_convergence",
    )
    args = parser.parse_args()

    data = diagnose_normal_sampling_convergence(
        args.nk_list,
        args.eta_list,
        args.matsubara_list,
        args.temperature,
        args.sampling,
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
