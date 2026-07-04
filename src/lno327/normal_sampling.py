"""Reusable normal-state k-space sampling helpers.

These helpers change only the k-space quadrature used to evaluate the existing
normal-state Kubo response. They do not alter the Kubo formula or replace the
default uniform-mesh behavior elsewhere in the package.
"""

from __future__ import annotations

import numpy as np

from .constants import KB_EV_PER_K
from .electrodynamics.conductivity import ConductivityTensor
from .electrodynamics.conventions import require_sheet_conductivity_for_reflection
from .models.lno327_four_orbital.normal import normal_state_hamiltonian
from .models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from .response.config import KuboConfig
from .response.local_normal import kubo_conductivity_imag_axis_from_model
from .response.occupations import negative_fermi_derivative

RATIO_EPS = 1e-300


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


def uniform_weights(mesh: np.ndarray) -> np.ndarray:
    """Return normalized uniform weights for a supplied mesh."""

    return np.full(mesh.shape[0], 1.0 / mesh.shape[0], dtype=float)


def _wrap_bz(points: np.ndarray) -> np.ndarray:
    wrapped = np.array(points, dtype=float, copy=True)
    wrapped[:, 0] = ((wrapped[:, 0] + np.pi) % (2.0 * np.pi)) - np.pi
    wrapped[:, 1] = ((wrapped[:, 1] + np.pi) % (2.0 * np.pi)) - np.pi
    return wrapped


def _band_energies(mesh: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.eigvalsh(normal_state_hamiltonian(kx, ky)) for kx, ky in mesh], dtype=float)


def _estimated_energy_resolution(energies: np.ndarray) -> float:
    flat = np.sort(np.ravel(energies))
    diffs = np.diff(flat)
    diffs = diffs[diffs > 1e-14]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def normal_fs_diagnostics(
    mesh: np.ndarray,
    weights: np.ndarray,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
) -> dict[str, float | int]:
    """Return Fermi-window diagnostics for a weighted normal-state mesh."""

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


def single_mesh_normal_response(
    mesh: np.ndarray,
    weights: np.ndarray,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
) -> np.ndarray:
    """Evaluate the existing normal-state Kubo response on a weighted mesh."""

    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    return kubo_conductivity_imag_axis_from_model(LNO327FourOrbitalSpec(), mesh, config, weights).matrix()


def _shift_grid_shifts(shift_grid: int) -> list[tuple[float, float]]:
    if shift_grid <= 0:
        raise ValueError("shift_grid must be positive")
    return [(i / shift_grid, j / shift_grid) for i in range(shift_grid) for j in range(shift_grid)]


def multishift_normal_response(
    nk: int,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
    shift_grid: int,
) -> tuple[np.ndarray, float, float, dict[str, float | int], dict[str, float | int]]:
    """Average the existing normal-state Kubo response over shifted meshes."""

    matrices = []
    fs_items = []
    for shift in _shift_grid_shifts(shift_grid):
        mesh = shifted_bz_mesh(nk, shift)
        weights = uniform_weights(mesh)
        matrices.append(single_mesh_normal_response(mesh, weights, eta_eV, omega_eV, temperature_K))
        fs_items.append(normal_fs_diagnostics(mesh, weights, eta_eV, omega_eV, temperature_K))
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
    metadata = {
        "num_kpoints_total": int(nk * nk * shift_grid * shift_grid),
        "num_fs_cells": 0,
        "num_refined_points": 0,
        "refined_area_fraction": 0.0,
        "weight_sum": 1.0,
    }
    return matrix, std_xx, rel_std, fs, metadata


def _cell_probe_energies(kx0: float, kx1: float, ky0: float, ky1: float) -> np.ndarray:
    points = np.array(
        [
            [kx0, ky0],
            [kx1, ky0],
            [kx0, ky1],
            [kx1, ky1],
            [0.5 * (kx0 + kx1), 0.5 * (ky0 + ky1)],
        ],
        dtype=float,
    )
    points = _wrap_bz(points)
    return np.array([np.linalg.eigvalsh(normal_state_hamiltonian(kx, ky)) for kx, ky in points], dtype=float)


def _is_fs_cell(energies: np.ndarray, fs_window_eV: float) -> bool:
    band_min = np.min(energies, axis=0)
    band_max = np.max(energies, axis=0)
    crosses = np.any(band_min * band_max < 0.0)
    near_window = np.any(np.min(np.abs(energies), axis=0) < fs_window_eV)
    return bool(crosses or near_window)


def fs_adaptive_mesh(
    nk: int,
    eta_eV: float,
    omega_eV: float,
    temperature_K: float,
    refine_factor: int,
    fs_window_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Build a weighted mesh by refining coarse cells that intersect the FS."""

    if nk <= 0:
        raise ValueError("nk must be positive")
    if refine_factor <= 1:
        raise ValueError("refine_factor must be greater than one")
    if fs_window_factor <= 0.0:
        raise ValueError("fs_window_factor must be positive")

    step = 2.0 * np.pi / nk
    fs_window = fs_window_factor * max(eta_eV, temperature_K * KB_EV_PER_K, omega_eV)
    base_weight = 1.0 / (nk * nk)
    sub_offsets = (np.arange(refine_factor) + 0.5) / refine_factor
    points: list[list[float]] = []
    weights: list[float] = []
    num_fs_cells = 0
    num_refined_points = 0

    for ix in range(nk):
        kx0 = -np.pi + ix * step
        kx1 = kx0 + step
        for iy in range(nk):
            ky0 = -np.pi + iy * step
            ky1 = ky0 + step
            energies = _cell_probe_energies(kx0, kx1, ky0, ky1)
            if _is_fs_cell(energies, fs_window):
                num_fs_cells += 1
                sub_weight = base_weight / (refine_factor * refine_factor)
                for sx in sub_offsets:
                    for sy in sub_offsets:
                        points.append([kx0 + sx * step, ky0 + sy * step])
                        weights.append(sub_weight)
                        num_refined_points += 1
            else:
                points.append([kx0 + 0.5 * step, ky0 + 0.5 * step])
                weights.append(base_weight)

    mesh = _wrap_bz(np.asarray(points, dtype=float))
    weight_array = np.asarray(weights, dtype=float)
    weight_array = weight_array / np.sum(weight_array)
    metadata = {
        "num_kpoints_total": int(mesh.shape[0]),
        "num_fs_cells": int(num_fs_cells),
        "num_refined_points": int(num_refined_points),
        "refined_area_fraction": float(num_fs_cells / (nk * nk)),
        "fs_window_eV": float(fs_window),
        "weight_sum": float(np.sum(weight_array)),
    }
    return mesh, weight_array, metadata


def normal_response_from_sampling(
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
    nk: int,
    sampling: str,
    refine_factor: int,
    shift_grid: int = 4,
    fs_window_factor: float = 1.0,
) -> np.ndarray:
    """Return a normal-state model response matrix from an optional sampler."""

    if sampling == "uniform":
        mesh = shifted_bz_mesh(nk)
        weights = uniform_weights(mesh)
        return single_mesh_normal_response(mesh, weights, eta_eV, omega_eV, temperature_K)
    if sampling == "multishift_average":
        matrix, *_ = multishift_normal_response(nk, eta_eV, omega_eV, temperature_K, shift_grid)
        return matrix
    if sampling == "fs_adaptive":
        mesh, weights, _metadata = fs_adaptive_mesh(
            nk,
            eta_eV,
            omega_eV,
            temperature_K,
            refine_factor=refine_factor,
            fs_window_factor=fs_window_factor,
        )
        return single_mesh_normal_response(mesh, weights, eta_eV, omega_eV, temperature_K)
    raise ValueError("unknown normal sampling")


def normal_sheet_tensor_from_sampling(
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
    nk: int,
    sampling: str,
    refine_factor: int,
    shift_grid: int = 4,
    fs_window_factor: float = 1.0,
) -> ConductivityTensor:
    """Return SI sheet conductivity from a normal-state optional sampler."""

    matrix = normal_response_from_sampling(
        omega_eV,
        temperature_K,
        eta_eV,
        nk,
        sampling,
        refine_factor,
        shift_grid=shift_grid,
        fs_window_factor=fs_window_factor,
    )
    return require_sheet_conductivity_for_reflection(matrix).tensor


# Backward-compatible private aliases for the diagnostic scripts.
_fs_diagnostics = normal_fs_diagnostics
_single_mesh_response = single_mesh_normal_response
_uniform_weights = uniform_weights
