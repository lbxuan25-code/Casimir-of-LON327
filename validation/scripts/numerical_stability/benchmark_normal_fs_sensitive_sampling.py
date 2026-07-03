#!/usr/bin/env python3
"""Benchmark FS-sensitive sampling for normal-state low-Matsubara response.

This script keeps the existing normal-state Kubo formula intact. The additional
sampling modes are diagnostic alternatives for the k-space quadrature only and
are not Casimir calculations.
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
    kubo_conductivity_imag_axis,
    model_response_to_sheet_conductivity,
    sheet_conductivity_to_reflection_dimensionless,
)
from lno327.conductivity import ConductivityTensor, negative_fermi_derivative  # noqa: E402
from lno327.constants import KB_EV_PER_K  # noqa: E402
from lno327.models.lno327_four_orbital.normal import normal_state_hamiltonian  # noqa: E402
from validation.lib.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

SAMPLING_MODES = ("uniform", "multishift_average", "fs_window_refined")
DEFAULT_NK_LIST = (32, 48, 64, 80)
DEFAULT_ETA_LIST = (5e-4, 2e-4, 1e-4)
DEFAULT_MATSUBARA_LIST = (1, 2)
DEFAULT_SHIFT_GRID_LIST = (1, 2, 4, 8)
HIGH_NK_TOLERANCE = 0.02
SYMMETRY_TOLERANCE = 1e-8
FS_POINT_WARNING_THRESHOLD = 8
RATIO_EPS = 1e-300

REQUIRED_NPZ_FIELDS = {
    "sampling",
    "nk",
    "eta_eV",
    "matsubara_index",
    "omega_eV",
    "shift_grid",
    "num_shifts",
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
    "num_refined_points",
    "fs_sampling_status",
    "convergence_status",
    "diagnosis",
    "notes",
}


def shifted_bz_mesh(nk: int, shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Return a midpoint BZ mesh shifted by fractions of one grid spacing."""

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


def _join_status(parts: list[str]) -> str:
    return "ok" if not parts else ";".join(dict.fromkeys(parts))


def _band_energies(mesh: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.eigvalsh(normal_state_hamiltonian(kx, ky)) for kx, ky in mesh], dtype=float)


def _estimated_energy_resolution(energies: np.ndarray) -> float:
    flat = np.sort(np.ravel(energies))
    diffs = np.diff(flat)
    diffs = diffs[diffs > 1e-14]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _fs_diagnostics(
    mesh: np.ndarray,
    weights: np.ndarray,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
) -> dict[str, float | int]:
    energies = _band_energies(mesh)
    abs_energies = np.abs(energies)
    temperature_eV = temperature_K * KB_EV_PER_K
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
    weights: np.ndarray,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
) -> np.ndarray:
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    return kubo_conductivity_imag_axis(mesh, config, weights).matrix()


def _uniform_weights(mesh: np.ndarray) -> np.ndarray:
    return np.full(mesh.shape[0], 1.0 / mesh.shape[0], dtype=float)


def _shift_grid_shifts(shift_grid: int) -> list[tuple[float, float]]:
    if shift_grid <= 0:
        raise ValueError("shift_grid must be positive")
    return [(i / shift_grid, j / shift_grid) for i in range(shift_grid) for j in range(shift_grid)]


def _wrap_bz(points: np.ndarray) -> np.ndarray:
    wrapped = np.array(points, dtype=float, copy=True)
    wrapped[:, 0] = ((wrapped[:, 0] + np.pi) % (2.0 * np.pi)) - np.pi
    wrapped[:, 1] = ((wrapped[:, 1] + np.pi) % (2.0 * np.pi)) - np.pi
    return wrapped


def fs_window_refined_mesh(
    nk: int,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
    refine_factor: int = 3,
    fs_window_eV: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Replace FS-window coarse cells by local subcells with area weights."""

    if refine_factor <= 1:
        raise ValueError("refine_factor must be greater than one")
    coarse = shifted_bz_mesh(nk)
    coarse_energies = _band_energies(coarse)
    window = max(eta_eV, temperature_K * KB_EV_PER_K, omega_eV) if fs_window_eV is None else fs_window_eV
    marked = np.any(np.abs(coarse_energies) <= window, axis=1)
    base_weight = 1.0 / coarse.shape[0]
    step = 2.0 * np.pi / nk
    sub_offsets = (np.arange(refine_factor) + 0.5) / refine_factor - 0.5

    points: list[np.ndarray] = []
    weights: list[float] = []
    refined_count = 0
    for center, should_refine in zip(coarse, marked, strict=True):
        if not should_refine:
            points.append(center)
            weights.append(base_weight)
            continue
        for dx in sub_offsets:
            for dy in sub_offsets:
                points.append(center + np.array([dx * step, dy * step], dtype=float))
                weights.append(base_weight / (refine_factor * refine_factor))
                refined_count += 1
    mesh = _wrap_bz(np.vstack(points))
    weight_array = np.asarray(weights, dtype=float)
    return mesh, weight_array, refined_count, float(window)


def _sampling_response(
    sampling: str,
    nk: int,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
    shift_grid: int,
    refine_factor: int,
    fs_window_eV: float | None,
) -> tuple[np.ndarray, float, float, dict[str, float | int], int, int, str]:
    if sampling == "uniform":
        mesh = shifted_bz_mesh(nk)
        weights = _uniform_weights(mesh)
        matrix = _single_mesh_response(mesh, weights, eta_eV, omega_eV, temperature_K)
        fs = _fs_diagnostics(mesh, weights, eta_eV, omega_eV, temperature_K)
        return matrix, 0.0, 0.0, fs, 0, 1, "uniform mesh baseline"

    if sampling == "multishift_average":
        matrices = []
        fs_items = []
        for shift in _shift_grid_shifts(shift_grid):
            mesh = shifted_bz_mesh(nk, shift)
            weights = _uniform_weights(mesh)
            matrices.append(_single_mesh_response(mesh, weights, eta_eV, omega_eV, temperature_K))
            fs_items.append(_fs_diagnostics(mesh, weights, eta_eV, omega_eV, temperature_K))
        stack = np.stack(matrices, axis=0)
        matrix = np.mean(stack, axis=0)
        std_xx = float(np.std(stack[:, 0, 0].real))
        rel_std = float(std_xx / (abs(matrix[0, 0]) + RATIO_EPS))
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
        return matrix, std_xx, rel_std, fs, 0, shift_grid * shift_grid, "multi-shift average diagnostic"

    if sampling == "fs_window_refined":
        mesh, weights, refined_count, window = fs_window_refined_mesh(
            nk,
            eta_eV,
            omega_eV,
            temperature_K,
            refine_factor=refine_factor,
            fs_window_eV=fs_window_eV,
        )
        matrix = _single_mesh_response(mesh, weights, eta_eV, omega_eV, temperature_K)
        fs = _fs_diagnostics(mesh, weights, eta_eV, omega_eV, temperature_K)
        return (
            matrix,
            0.0,
            0.0,
            fs,
            refined_count,
            1,
            f"FS-window refined diagnostic; fs_window_eV={window:g}",
        )

    raise ValueError("unknown sampling mode")


def benchmark_normal_fs_sensitive_sampling(
    nk_list: list[int],
    eta_list: list[float],
    matsubara_list: list[int],
    temperature_K: float,
    shift_grid_list: list[int],
    sampling_modes: list[str],
    refine_factor: int = 3,
    fs_window_eV: float | None = None,
) -> dict[str, np.ndarray]:
    """Run the FS-sensitive normal-response sampling benchmark."""

    if len(nk_list) < 2 or any(nk <= 0 for nk in nk_list):
        raise ValueError("nk_list must contain at least two positive values")
    if not eta_list or any(eta <= 0.0 for eta in eta_list):
        raise ValueError("eta_list must contain positive values")
    if not matsubara_list or any(n < 1 for n in matsubara_list):
        raise ValueError("matsubara_list must contain n >= 1")
    if not shift_grid_list or any(s <= 0 for s in shift_grid_list):
        raise ValueError("shift_grid_list must contain positive values")
    if any(mode not in SAMPLING_MODES for mode in sampling_modes):
        raise ValueError("unknown sampling mode")

    nk_values = np.asarray(sorted(nk_list), dtype=int)
    eta_values = np.asarray(sorted(eta_list), dtype=float)
    matsubara_values = np.asarray(sorted(matsubara_list), dtype=int)
    shift_values = np.asarray(sorted(shift_grid_list), dtype=int)
    rows: list[tuple[str, int, float, int, int]] = []
    for sampling in sampling_modes:
        grids = shift_values if sampling == "multishift_average" else np.array([1], dtype=int)
        for shift_grid in grids:
            for nk in nk_values:
                for eta in eta_values:
                    for n in matsubara_values:
                        rows.append((sampling, int(nk), float(eta), int(n), int(shift_grid)))

    row_count = len(rows)
    data: dict[str, np.ndarray] = {
        "sampling": np.empty(row_count, dtype="U24"),
        "nk": np.empty(row_count, dtype=int),
        "eta_eV": np.empty(row_count, dtype=float),
        "matsubara_index": np.empty(row_count, dtype=int),
        "omega_eV": np.empty(row_count, dtype=float),
        "shift_grid": np.empty(row_count, dtype=int),
        "num_shifts": np.empty(row_count, dtype=int),
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
        "num_refined_points": np.empty(row_count, dtype=int),
        "fs_sampling_status": np.empty(row_count, dtype="U192"),
        "convergence_status": np.empty(row_count, dtype="U192"),
        "diagnosis": np.empty(row_count, dtype="U160"),
        "notes": np.empty(row_count, dtype=object),
        "nk_list": nk_values,
        "eta_list": eta_values,
        "matsubara_list": matsubara_values,
        "shift_grid_list": shift_values,
        "temperature_K": np.array(temperature_K),
        "refine_factor": np.array(refine_factor),
        "fs_window_eV": np.array(np.nan if fs_window_eV is None else fs_window_eV),
    }

    index_by_key: dict[tuple[str, int, float, int, int], int] = {}
    for index, (sampling, nk, eta, n, shift_grid) in enumerate(rows):
        omega_eV = bosonic_matsubara_energy_eV(n, temperature_K)
        matrix, std_xx, rel_std, fs, refined_count, num_shifts, note = _sampling_response(
            sampling,
            nk,
            eta,
            omega_eV,
            temperature_K,
            shift_grid,
            refine_factor,
            fs_window_eV,
        )
        tensor = _matrix_to_tensor(matrix)
        diagnostics = conductivity_matrix_diagnostics(tensor)
        sheet = model_response_to_sheet_conductivity(matrix)
        reflection = sheet_conductivity_to_reflection_dimensionless(sheet)
        index_by_key[(sampling, nk, eta, n, shift_grid)] = index
        data["sampling"][index] = sampling
        data["nk"][index] = nk
        data["eta_eV"][index] = eta
        data["matsubara_index"][index] = n
        data["omega_eV"][index] = omega_eV
        data["shift_grid"][index] = shift_grid
        data["num_shifts"][index] = num_shifts
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
        data["num_refined_points"][index] = refined_count
        data["notes"][index] = (
            "normal-state FS-sensitive sampling benchmark only",
            note,
            "sampling changes quadrature only and does not alter the Kubo formula",
            "not a Casimir calculation",
        )

    largest_nk = int(nk_values[-1])
    previous_nk = int(nk_values[-2])
    smallest_eta = float(eta_values[0])
    largest_eta = float(eta_values[-1])
    for index, (sampling, nk, eta, n, shift_grid) in enumerate(rows):
        largest_index = index_by_key.get((sampling, largest_nk, eta, n, shift_grid))
        previous_index = index_by_key.get((sampling, previous_nk, eta, n, shift_grid))
        smallest_eta_index = index_by_key.get((sampling, nk, smallest_eta, n, shift_grid))
        largest_eta_index = index_by_key.get((sampling, nk, largest_eta, n, shift_grid))
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

    improvement_status = _sampling_improvement_status(data, rows)

    for index, (sampling, _nk, eta, n, shift_grid) in enumerate(rows):
        conv_parts: list[str] = []
        fs_parts: list[str] = []
        diag_parts: list[str] = []
        last_two = data["relative_change_between_last_two_nk"][index]
        if last_two < HIGH_NK_TOLERANCE:
            conv_parts.append("high_nk_converged")
        else:
            conv_parts.append("warning_high_nk_not_converged")
        if data["eta_relative_change"][index] >= HIGH_NK_TOLERANCE:
            conv_parts.append("warning_eta_sensitive")
        if sampling == "multishift_average" and data["relative_std_over_shifts"][index] < HIGH_NK_TOLERANCE:
            conv_parts.append("shift_average_stable")
        improvement = improvement_status.get((sampling, float(eta), int(n), int(shift_grid)))
        if improvement is not None:
            conv_parts.append(improvement)
        if data["points_within_eta"][index] == 0:
            fs_parts.append("warning_fs_underresolved_eta")
        if data["points_within_eta"][index] < FS_POINT_WARNING_THRESHOLD:
            fs_parts.append("warning_few_fs_points_within_eta")
        if (
            abs(data["delta"][index]) > SYMMETRY_TOLERANCE
            or data["relative_offdiag"][index] > SYMMETRY_TOLERANCE
            or data["relative_eigen_split"][index] > SYMMETRY_TOLERANCE
        ):
            diag_parts.append("warning_symmetry")
        if not np.isfinite(data["sigma_xx"][index]):
            diag_parts.append("warning_nonfinite_response")
        if not np.isfinite(data["sheet_conductivity_xx"][index]):
            diag_parts.append("warning_nonfinite_sheet_conductivity")
        data["convergence_status"][index] = _join_status(conv_parts)
        data["fs_sampling_status"][index] = _join_status(fs_parts)
        data["diagnosis"][index] = _join_status(diag_parts)

    _mark_requires_advanced_integration(data)
    return data


def _sampling_improvement_status(
    data: dict[str, np.ndarray],
    rows: list[tuple[str, int, float, int, int]],
) -> dict[tuple[str, float, int, int], str | None]:
    status: dict[tuple[str, float, int, int], str | None] = {}
    for eta in sorted(set(float(row[2]) for row in rows)):
        for n in sorted(set(int(row[3]) for row in rows)):
            uniform_mask = (
                (data["sampling"] == "uniform")
                & np.isclose(data["eta_eV"], eta)
                & (data["matsubara_index"] == n)
            )
            if not np.any(uniform_mask):
                continue
            uniform_change = float(np.nanmax(data["relative_change_between_last_two_nk"][uniform_mask]))
            for sampling in ("multishift_average", "fs_window_refined"):
                grids = sorted(set(int(value) for value in data["shift_grid"][data["sampling"] == sampling]))
                for shift_grid in grids:
                    mask = (
                        (data["sampling"] == sampling)
                        & np.isclose(data["eta_eV"], eta)
                        & (data["matsubara_index"] == n)
                        & (data["shift_grid"] == shift_grid)
                    )
                    if not np.any(mask):
                        continue
                    change = float(np.nanmax(data["relative_change_between_last_two_nk"][mask]))
                    key = (sampling, eta, n, shift_grid)
                    if change < 0.8 * uniform_change:
                        status[key] = (
                            "multishift_improves_convergence"
                            if sampling == "multishift_average"
                            else "fs_refinement_improves_convergence"
                        )
                    else:
                        status[key] = None
    return status


def _mark_requires_advanced_integration(data: dict[str, np.ndarray]) -> None:
    for eta in sorted(set(float(item) for item in data["eta_eV"])):
        for n in sorted(set(int(item) for item in data["matsubara_index"])):
            mask = np.isclose(data["eta_eV"], eta) & (data["matsubara_index"] == n)
            if not np.any(mask):
                continue
            converged = np.any(data["relative_change_between_last_two_nk"][mask] < HIGH_NK_TOLERANCE)
            if converged:
                continue
            for index in np.where(mask)[0]:
                current = str(data["convergence_status"][index])
                suffix = "requires_contour_or_tetrahedron_integration"
                data["convergence_status"][index] = current if suffix in current else f"{current};{suffix}"


def output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    resolved_prefix = output_prefix.resolve()
    project_data_root = (ROOT / "validation" / "outputs" / "archive" / "normal_state" / "fs_sensitive_sampling" / "data").resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "validation" / "outputs" / "archive" / "normal_state" / "fs_sensitive_sampling" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return (
        npz_path,
        csv_path,
        figure_dir / f"{output_prefix.name}_sigma_xx_vs_nk.png",
        figure_dir / f"{output_prefix.name}_last_two_change_by_sampling.png",
        figure_dir / f"{output_prefix.name}_shift_std_vs_shift_grid.png",
        figure_dir / f"{output_prefix.name}_sigma_xx_vs_shift_grid.png",
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
        last_two_plot,
        shift_std_plot,
        sigma_shift_plot,
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
        "shift_grid",
        "num_shifts",
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
        "num_refined_points",
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
    n_values = list(dict.fromkeys(int(item) for item in data["matsubara_index"]))
    eta_values = list(dict.fromkeys(float(item) for item in data["eta_eV"]))
    reference_eta = min(eta_values)
    reference_n = min(n_values)
    reference_nk = max(int(item) for item in data["nk"])
    reference_shift_grid = max(int(item) for item in data["shift_grid"])

    fig_sigma_nk, ax_sigma_nk = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        shift_grid = reference_shift_grid if sampling == "multishift_average" else 1
        mask = (
            (data["sampling"] == sampling)
            & np.isclose(data["eta_eV"], reference_eta)
            & (data["matsubara_index"] == reference_n)
            & (data["shift_grid"] == shift_grid)
        )
        ax_sigma_nk.plot(data["nk"][mask], data["sigma_xx"][mask].real, marker="o", label=sampling)
    ax_sigma_nk.set_xlabel(r"$N_k$")
    ax_sigma_nk.set_ylabel(r"Re $\sigma_{xx}$")
    ax_sigma_nk.set_title(rf"FS-sensitive sampling at n={reference_n}, $\eta={reference_eta:g}$")
    style_publication_axis(ax_sigma_nk)
    save_publication_figure(fig_sigma_nk, sigma_nk_plot)
    plt.close(fig_sigma_nk)

    fig_last, ax_last = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    labels = []
    values = []
    for sampling in samplings:
        for shift_grid in sorted(set(int(x) for x in data["shift_grid"][data["sampling"] == sampling])):
            mask = (
                (data["sampling"] == sampling)
                & np.isclose(data["eta_eV"], reference_eta)
                & (data["matsubara_index"] == reference_n)
                & (data["shift_grid"] == shift_grid)
            )
            labels.append(f"{sampling}\ns={shift_grid}")
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

    fig_std, ax_std = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for nk in sorted(set(int(item) for item in data["nk"])):
        mask = (
            (data["sampling"] == "multishift_average")
            & (data["nk"] == nk)
            & np.isclose(data["eta_eV"], reference_eta)
            & (data["matsubara_index"] == reference_n)
        )
        if np.any(mask):
            ax_std.plot(data["shift_grid"][mask], data["relative_std_over_shifts"][mask], marker="o", label=f"Nk={nk}")
    ax_std.axhline(HIGH_NK_TOLERANCE, color="black", linestyle="--", linewidth=1.0)
    ax_std.set_xscale("log", base=2)
    ax_std.set_yscale("symlog", linthresh=1e-4)
    ax_std.set_xlabel("shift grid")
    ax_std.set_ylabel(r"std$(\sigma_{xx})/|\langle\sigma_{xx}\rangle|$")
    ax_std.set_title("multi-shift stability")
    style_publication_axis(ax_std)
    save_publication_figure(fig_std, shift_std_plot)
    plt.close(fig_std)

    fig_shift, ax_shift = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for nk in sorted(set(int(item) for item in data["nk"])):
        mask = (
            (data["sampling"] == "multishift_average")
            & (data["nk"] == nk)
            & np.isclose(data["eta_eV"], reference_eta)
            & (data["matsubara_index"] == reference_n)
        )
        if np.any(mask):
            ax_shift.plot(data["shift_grid"][mask], data["sigma_xx"][mask].real, marker="o", label=f"Nk={nk}")
    ax_shift.set_xscale("log", base=2)
    ax_shift.set_xlabel("shift grid")
    ax_shift.set_ylabel(r"Re $\sigma_{xx}$")
    ax_shift.set_title("response versus number of shifts")
    style_publication_axis(ax_shift)
    save_publication_figure(fig_shift, sigma_shift_plot)
    plt.close(fig_shift)

    fig_fs, ax_fs = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for sampling in samplings:
        shift_grid = reference_shift_grid if sampling == "multishift_average" else 1
        mask = (
            (data["sampling"] == sampling)
            & np.isclose(data["eta_eV"], reference_eta)
            & (data["matsubara_index"] == reference_n)
            & (data["shift_grid"] == shift_grid)
        )
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
        shift_grid = reference_shift_grid if sampling == "multishift_average" else 1
        mask = (
            (data["sampling"] == sampling)
            & np.isclose(data["eta_eV"], reference_eta)
            & (data["matsubara_index"] == reference_n)
            & (data["shift_grid"] == shift_grid)
        )
        ax_sym.plot(data["nk"][mask], np.abs(data["delta"][mask]), marker="o", label=f"{sampling} |delta|")
        ax_sym.plot(data["nk"][mask], data["relative_offdiag"][mask], marker="s", linestyle="--", label=f"{sampling} offdiag")
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
    max_shift_std = float(np.nanmax(data["relative_std_over_shifts"]))
    max_symmetry = float(
        max(
            np.nanmax(np.abs(data["delta"])),
            np.nanmax(data["relative_offdiag"]),
            np.nanmax(data["relative_eigen_split"]),
        )
    )
    print(f"row_count = {data['sampling'].size}")
    print(f"max_relative_change_between_last_two_nk = {max_last_two}")
    print(f"max_eta_relative_change = {max_eta}")
    print(f"max_relative_std_over_shifts = {max_shift_std}")
    print(f"min_points_within_eta = {min_points_eta}")
    print(f"max_symmetry_diagnostic = {max_symmetry}")
    print(f"convergence_statuses = {sorted(set(str(item) for item in data['convergence_status']))}")
    print(f"fs_sampling_statuses = {sorted(set(str(item) for item in data['fs_sampling_status']))}")
    print(f"diagnoses = {sorted(set(str(item) for item in data['diagnosis']))}")
    print("note = normal-state FS-sensitive sampling benchmark only; not a Casimir result.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--eta-list", nargs="+", type=float, default=list(DEFAULT_ETA_LIST))
    parser.add_argument("--matsubara-list", nargs="+", type=int, default=list(DEFAULT_MATSUBARA_LIST))
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--shift-grid-list", nargs="+", type=int, default=list(DEFAULT_SHIFT_GRID_LIST))
    parser.add_argument("--sampling", nargs="+", choices=SAMPLING_MODES, default=list(SAMPLING_MODES))
    parser.add_argument("--refine-factor", type=int, default=3)
    parser.add_argument("--fs-window-eV", type=float, default=None)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT
        / "outputs"
        / "normal_state"
        / "fs_sensitive_sampling"
        / "data"
        / "fs_sensitive_sampling",
    )
    args = parser.parse_args()

    data = benchmark_normal_fs_sensitive_sampling(
        args.nk_list,
        args.eta_list,
        args.matsubara_list,
        args.temperature,
        args.shift_grid_list,
        args.sampling,
        refine_factor=args.refine_factor,
        fs_window_eV=args.fs_window_eV,
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
