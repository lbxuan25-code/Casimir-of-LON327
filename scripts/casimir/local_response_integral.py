#!/usr/bin/env python3
"""Compute the preliminary local-response Casimir Matsubara/k/phi integral.

This is the production local-response calculation used by the preliminary
distance-scan conclusion. It is not a final Casimir calculation: n=0 is skipped
and finite-momentum response is not included.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from local_response_config import (  # noqa: E402
    BENCHMARK_METADATA,
    BENCHMARK_NOTE_PARTS,
    TORQUE_TOLERANCE,
)
from lno327 import (  # noqa: E402
    CasimirSetup,
    ConductivityTensor,
    PairingAmplitudes,
    bosonic_matsubara_energy_eV,
    casimir_energy_integrand,
    k_weights,
    local_response_imag_axis,
    require_sheet_conductivity_for_reflection,
    uniform_bz_mesh,
)
from lno327.casimir import matsubara_frequency  # noqa: E402
from lno327.constants import KB  # noqa: E402
from lno327.normal_sampling import normal_sheet_tensor_from_sampling  # noqa: E402
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("normal", "spm", "dwave")
NORMAL_SAMPLING = ("uniform", "multishift_average", "fs_adaptive")
N0_POLICY = "skip"
RATIO_EPS = 1e-300
ProgressCallback = Callable[[str], None]

REQUIRED_NPZ_FIELDS = {
    "kind",
    "distance_m",
    "theta",
    "temperature_K",
    "matsubara_min",
    "matsubara_max",
    "kparallel_num",
    "kparallel_max",
    "phi_num",
    "energy",
    "torque_fd",
    "max_abs_torque_over_theta",
    "normal_sampling",
    "normal_refine_factor",
    "normal_nk",
    "bdg_nk",
    "delta0",
    "n0_policy",
    "local_response",
    "finite_momentum_resolved",
    "benchmark_only",
    "preliminary_local_response_conclusion",
    "not_final_casimir_conclusion",
    "matsubara_tail_indicator",
    "kparallel_cutoff_indicator",
    "phi_convergence_indicator",
    "matsubara_partial_sums",
    "diagnosis",
    "notes",
}


class ResponseTensorCache:
    """Small on-disk cache for local-response sheet tensors."""

    def __init__(self, cache_dir: Path, *, use: bool = True, rebuild: bool = False) -> None:
        self.cache_dir = cache_dir
        self.use = use
        self.rebuild = rebuild
        self.memory: dict[str, ConductivityTensor] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @staticmethod
    def _key_payload(
        *,
        kind: str,
        matsubara_index: int,
        temperature_K: float,
        normal_nk: int,
        normal_eta_eV: float,
        normal_sampling: str,
        normal_refine_factor: int,
        bdg_nk: int,
        delta0_eV: float,
    ) -> dict[str, object]:
        return {
            "kind": kind,
            "matsubara_index": int(matsubara_index),
            "temperature_K": float(temperature_K),
            "normal_nk": int(normal_nk),
            "normal_eta_eV": float(normal_eta_eV),
            "normal_sampling": normal_sampling,
            "normal_refine_factor": int(normal_refine_factor),
            "bdg_nk": int(bdg_nk),
            "delta0_eV": float(delta0_eV),
        }

    def _key(self, payload: dict[str, object]) -> str:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"response_tensor_{key}.npz"

    def entry_count(self) -> int:
        if not self.cache_dir.exists():
            return 0
        return sum(1 for _path in self.cache_dir.glob("response_tensor_*.npz"))

    def get_or_compute(
        self,
        *,
        kind: str,
        matsubara_index: int,
        temperature_K: float,
        normal_nk: int,
        normal_eta_eV: float,
        normal_sampling: str,
        normal_refine_factor: int,
        bdg_nk: int,
        delta0_eV: float,
        compute,
    ) -> ConductivityTensor:
        payload = self._key_payload(
            kind=kind,
            matsubara_index=matsubara_index,
            temperature_K=temperature_K,
            normal_nk=normal_nk,
            normal_eta_eV=normal_eta_eV,
            normal_sampling=normal_sampling,
            normal_refine_factor=normal_refine_factor,
            bdg_nk=bdg_nk,
            delta0_eV=delta0_eV,
        )
        key = self._key(payload)
        if key in self.memory:
            self.hits += 1
            return self.memory[key]
        path = self._path(key)
        if self.use and path.exists() and not self.rebuild:
            with np.load(path, allow_pickle=False) as loaded:
                tensor = ConductivityTensor(
                    xx=complex(loaded["xx"].item()),
                    yy=complex(loaded["yy"].item()),
                    xy=complex(loaded["xy"].item()),
                    yx=complex(loaded["yx"].item()),
                )
            self.memory[key] = tensor
            self.hits += 1
            return tensor
        tensor = compute()
        self.memory[key] = tensor
        self.misses += 1
        if self.use:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                xx=np.asarray(tensor.xx),
                yy=np.asarray(tensor.yy),
                xy=np.asarray(tensor.xy),
                yx=np.asarray(tensor.yx),
                key_payload=np.asarray(json.dumps(payload, sort_keys=True)),
            )
            self.writes += 1
        return tensor


def toy_anisotropic_tensor() -> ConductivityTensor:
    return ConductivityTensor(xx=2e-4, yy=1e-4, xy=0.0, yx=0.0)


def _model_matrix_to_sheet_tensor(matrix: np.ndarray) -> ConductivityTensor:
    return require_sheet_conductivity_for_reflection(matrix).tensor


def _normal_sheet_tensor(
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
    nk: int,
    sampling: str,
    refine_factor: int,
) -> ConductivityTensor:
    return normal_sheet_tensor_from_sampling(
        omega_eV,
        temperature_K,
        eta_eV,
        nk,
        sampling,
        refine_factor,
        shift_grid=4,
        fs_window_factor=1.0,
    )


def _bdg_sheet_tensor(
    kind: str,
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
    nk: int,
    delta0_eV: float,
) -> ConductivityTensor:
    mesh = uniform_bz_mesh(nk)
    response = local_response_imag_axis(
        kind,  # type: ignore[arg-type]
        omega_eV,
        mesh,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        pairing_params=PairingAmplitudes(delta0_eV=delta0_eV),
        k_weights=k_weights(mesh),
    )
    return _model_matrix_to_sheet_tensor(response.matrix)


def _sheet_tensor_for_kind(
    kind: str,
    matsubara_index: int,
    temperature_K: float,
    normal_nk: int,
    normal_eta_eV: float,
    normal_sampling: str,
    normal_refine_factor: int,
    bdg_nk: int,
    delta0_eV: float,
    response_cache: ResponseTensorCache | None = None,
) -> ConductivityTensor:
    def compute() -> ConductivityTensor:
        omega_eV = bosonic_matsubara_energy_eV(matsubara_index, temperature_K)
        if kind == "normal":
            return _normal_sheet_tensor(
                omega_eV,
                temperature_K,
                normal_eta_eV,
                normal_nk,
                normal_sampling,
                normal_refine_factor,
            )
        if kind in {"spm", "dwave"}:
            return _bdg_sheet_tensor(kind, omega_eV, temperature_K, normal_eta_eV, bdg_nk, delta0_eV)
        if kind == "toy_anisotropic":
            return toy_anisotropic_tensor()
        raise ValueError("unknown kind")

    if response_cache is None:
        return compute()
    return response_cache.get_or_compute(
        kind=kind,
        matsubara_index=matsubara_index,
        temperature_K=temperature_K,
        normal_nk=normal_nk,
        normal_eta_eV=normal_eta_eV,
        normal_sampling=normal_sampling,
        normal_refine_factor=normal_refine_factor,
        bdg_nk=bdg_nk,
        delta0_eV=delta0_eV,
        compute=compute,
    )


def _integrate_one_matsubara(
    tensor: ConductivityTensor,
    setup: CasimirSetup,
    xi: float,
    theta: float,
    k_values: np.ndarray,
    phi_values: np.ndarray,
) -> complex:
    values = np.empty((k_values.size, phi_values.size), dtype=complex)
    for i, k_parallel in enumerate(k_values):
        for j, phi in enumerate(phi_values):
            values[i, j] = casimir_energy_integrand(
                setup,
                xi,
                float(k_parallel),
                float(phi),
                theta,
                tensor,
                tensor,
            )
    phi_integral = np.trapezoid(values, phi_values, axis=1)
    return complex(np.trapezoid(phi_integral, k_values) / (2.0 * np.pi) ** 2)


def _energy_theta_series(
    kind: str,
    distance_m: float,
    theta_values: np.ndarray,
    matsubara_indices: np.ndarray,
    k_values: np.ndarray,
    phi_values: np.ndarray,
    temperature_K: float,
    normal_nk: int,
    normal_eta_eV: float,
    normal_sampling: str,
    normal_refine_factor: int,
    bdg_nk: int,
    delta0_eV: float,
    response_cache: ResponseTensorCache | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    setup = CasimirSetup(temperature=temperature_K, distance=distance_m)
    tensors: dict[int, ConductivityTensor] = {}
    for n_index, n in enumerate(matsubara_indices, start=1):
        if progress_callback is not None:
            progress_callback(
                f"response_start kind={kind} matsubara={n_index}/{matsubara_indices.size} n={int(n)}"
            )
        tensors[int(n)] = _sheet_tensor_for_kind(
            kind,
            int(n),
            temperature_K,
            normal_nk,
            normal_eta_eV,
            normal_sampling,
            normal_refine_factor,
            bdg_nk,
            delta0_eV,
            response_cache=response_cache,
        )
        if progress_callback is not None:
            progress_callback(
                f"response_done kind={kind} matsubara={n_index}/{matsubara_indices.size} n={int(n)}"
            )
    energies = np.empty(theta_values.size, dtype=complex)
    partials = np.empty((theta_values.size, matsubara_indices.size), dtype=complex)
    prefactor = KB * temperature_K
    for theta_index, theta in enumerate(theta_values):
        if progress_callback is not None:
            progress_callback(
                f"theta_start kind={kind} theta={theta_index + 1}/{theta_values.size} value={float(theta):g} "
                f"matsubara={int(matsubara_indices[0])}-{int(matsubara_indices[-1])}"
            )
        running = 0.0 + 0.0j
        for n_index, n in enumerate(matsubara_indices):
            xi = matsubara_frequency(int(n), temperature_K)
            term = prefactor * _integrate_one_matsubara(
                tensors[int(n)],
                setup,
                xi,
                float(theta),
                k_values,
                phi_values,
            )
            running += term
            partials[theta_index, n_index] = running
        energies[theta_index] = running
        if progress_callback is not None:
            progress_callback(
                f"theta_done kind={kind} theta={theta_index + 1}/{theta_values.size} value={float(theta):g}"
            )
    return energies, partials


def _finite_difference_torque(theta_values: np.ndarray, energies: np.ndarray) -> np.ndarray:
    if theta_values.size < 2:
        return np.zeros_like(energies)
    return -np.gradient(energies.real, theta_values)


def _tail_indicator(partials: np.ndarray) -> float:
    if partials.shape[1] < 2:
        return np.nan
    last_increment = partials[:, -1] - partials[:, -2]
    scale = np.maximum(np.abs(partials[:, -1]), RATIO_EPS)
    return float(np.nanmax(np.abs(last_increment) / scale))


def compute_local_response_casimir_integral(
    kinds: list[str],
    distance_list: list[float],
    theta_list: list[float],
    matsubara_min: int,
    matsubara_max: int,
    kparallel_num: int,
    kparallel_max_factor: float,
    phi_num: int,
    temperature_K: float,
    normal_nk: int,
    normal_eta_eV: float,
    normal_sampling: str,
    normal_refine_factor: int,
    bdg_nk: int,
    delta0_eV: float,
    include_toy_anisotropic_control: bool = False,
    response_cache: ResponseTensorCache | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, np.ndarray]:
    if any(kind not in KINDS for kind in kinds):
        raise ValueError("unknown kind")
    if include_toy_anisotropic_control:
        kinds = [*kinds, "toy_anisotropic"]
    if not distance_list or any(distance <= 0.0 for distance in distance_list):
        raise ValueError("distance_list must contain positive values")
    if len(theta_list) < 1:
        raise ValueError("theta_list must contain at least one value")
    if matsubara_min < 1 or matsubara_max < matsubara_min:
        raise ValueError("Matsubara range must satisfy 1 <= min <= max")
    if kparallel_num < 2 or phi_num < 3:
        raise ValueError("kparallel_num >= 2 and phi_num >= 3 are required")
    if normal_sampling not in NORMAL_SAMPLING:
        raise ValueError("unknown normal sampling")

    theta_values = np.asarray(theta_list, dtype=float)
    matsubara_indices = np.arange(matsubara_min, matsubara_max + 1, dtype=int)
    rows = [(kind, float(distance), float(theta)) for kind in kinds for distance in distance_list for theta in theta_values]
    row_count = len(rows)
    n_count = matsubara_indices.size
    data: dict[str, np.ndarray] = {
        "kind": np.empty(row_count, dtype="U32"),
        "distance_m": np.empty(row_count, dtype=float),
        "theta": np.empty(row_count, dtype=float),
        "temperature_K": np.full(row_count, temperature_K, dtype=float),
        "matsubara_min": np.full(row_count, matsubara_min, dtype=int),
        "matsubara_max": np.full(row_count, matsubara_max, dtype=int),
        "kparallel_num": np.full(row_count, kparallel_num, dtype=int),
        "kparallel_max": np.empty(row_count, dtype=float),
        "phi_num": np.full(row_count, phi_num, dtype=int),
        "energy": np.empty(row_count, dtype=complex),
        "torque_fd": np.empty(row_count, dtype=float),
        "max_abs_torque_over_theta": np.empty(row_count, dtype=float),
        "normal_sampling": np.full(row_count, normal_sampling, dtype="U24"),
        "normal_refine_factor": np.full(row_count, normal_refine_factor, dtype=int),
        "normal_nk": np.full(row_count, normal_nk, dtype=int),
        "bdg_nk": np.full(row_count, bdg_nk, dtype=int),
        "delta0": np.full(row_count, delta0_eV, dtype=float),
        "n0_policy": np.full(row_count, BENCHMARK_METADATA["n0_policy"], dtype="U16"),
        "local_response": np.full(row_count, BENCHMARK_METADATA["local_response"], dtype=bool),
        "finite_momentum_resolved": np.full(row_count, BENCHMARK_METADATA["finite_momentum_resolved"], dtype=bool),
        "benchmark_only": np.full(row_count, BENCHMARK_METADATA["benchmark_only"], dtype=bool),
        "preliminary_local_response_conclusion": np.full(
            row_count, BENCHMARK_METADATA["preliminary_local_response_conclusion"], dtype=bool
        ),
        "not_final_casimir_conclusion": np.full(row_count, BENCHMARK_METADATA["not_final_casimir_conclusion"], dtype=bool),
        "matsubara_tail_indicator": np.empty(row_count, dtype=float),
        "kparallel_cutoff_indicator": np.full(row_count, "cutoff_not_final", dtype="U64"),
        "phi_convergence_indicator": np.full(row_count, "phi_convergence_not_evaluated", dtype="U64"),
        "diagnosis": np.empty(row_count, dtype="U192"),
        "notes": np.empty(row_count, dtype=object),
        "matsubara_partial_sums": np.empty((row_count, n_count), dtype=complex),
        "matsubara_indices": matsubara_indices,
    }

    row_index: dict[tuple[str, float, float], int] = {
        (kind, distance, theta): index for index, (kind, distance, theta) in enumerate(rows)
    }
    phi_values = np.linspace(0.0, 2.0 * np.pi, phi_num, endpoint=True)
    for kind_index, kind in enumerate(kinds, start=1):
        if progress_callback is not None:
            progress_callback(f"kind_start {kind_index}/{len(kinds)} kind={kind}")
        for distance_index, distance in enumerate(distance_list, start=1):
            if progress_callback is not None:
                progress_callback(
                    f"integral_distance_start {distance_index}/{len(distance_list)} kind={kind} d={distance:g}"
                )
            kmax = kparallel_max_factor / distance
            k_values = np.linspace(0.0, kmax, kparallel_num)
            energies, partials = _energy_theta_series(
                kind,
                float(distance),
                theta_values,
                matsubara_indices,
                k_values,
                phi_values,
                temperature_K,
                normal_nk,
                normal_eta_eV,
                normal_sampling,
                normal_refine_factor,
                bdg_nk,
                delta0_eV,
                response_cache=response_cache,
                progress_callback=progress_callback,
            )
            torques = _finite_difference_torque(theta_values, energies)
            max_abs_torque = float(np.nanmax(np.abs(torques)))
            tail = _tail_indicator(partials)
            for theta_index, theta in enumerate(theta_values):
                index = row_index[(kind, float(distance), float(theta))]
                data["kind"][index] = kind
                data["distance_m"][index] = float(distance)
                data["theta"][index] = float(theta)
                data["kparallel_max"][index] = kmax
                data["energy"][index] = energies[theta_index]
                data["torque_fd"][index] = torques[theta_index]
                data["max_abs_torque_over_theta"][index] = max_abs_torque
                data["matsubara_tail_indicator"][index] = tail
                data["matsubara_partial_sums"][index, :] = partials[theta_index, :]
                diagnosis_parts = []
                if kind == "toy_anisotropic" and max_abs_torque > TORQUE_TOLERANCE:
                    diagnosis_parts.append("plumbing_pass_toy_anisotropy")
                elif kind in KINDS and max_abs_torque <= TORQUE_TOLERANCE:
                    diagnosis_parts.append("zero_torque_baseline")
                elif kind in KINDS:
                    diagnosis_parts.append("warning_possible_spurious_torque")
                if np.isfinite(tail) and tail > 0.05:
                    diagnosis_parts.append("warning_matsubara_not_converged")
                diagnosis_parts.append("cutoff_not_final")
                data["diagnosis"][index] = ";".join(diagnosis_parts)
                data["notes"][index] = (*BENCHMARK_NOTE_PARTS, f"kmax = {kparallel_max_factor:g} / distance")
            if progress_callback is not None:
                progress_callback(
                    f"integral_distance_done {distance_index}/{len(distance_list)} kind={kind} d={distance:g}"
                )
        if progress_callback is not None:
            progress_callback(f"kind_done {kind_index}/{len(kinds)} kind={kind}")
    return data


def output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    resolved_prefix = output_prefix.resolve()
    project_data_root = (ROOT / "outputs" / "casimir" / "local_response_integral" / "data").resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "outputs" / "casimir" / "local_response_integral" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return (
        npz_path,
        csv_path,
        figure_dir / f"{output_prefix.name}_energy_vs_theta.png",
        figure_dir / f"{output_prefix.name}_torque_vs_theta.png",
        figure_dir / f"{output_prefix.name}_matsubara_partial_sums.png",
        figure_dir / f"{output_prefix.name}_max_torque_vs_distance.png",
        figure_dir / f"{output_prefix.name}_toy_torque_vs_theta.png",
    )


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    if isinstance(value, np.ndarray):
        return " ".join(str(item) for item in value)
    return value


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    paths = output_paths(output_prefix)
    npz_path, csv_path, energy_plot, torque_plot, partial_plot, distance_plot, toy_plot = paths
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    energy_plot.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)
    fieldnames = [
        "kind",
        "distance_m",
        "theta",
        "temperature_K",
        "matsubara_min",
        "matsubara_max",
        "kparallel_num",
        "kparallel_max",
        "phi_num",
        "energy",
        "torque_fd",
        "max_abs_torque_over_theta",
        "normal_sampling",
        "normal_refine_factor",
        "normal_nk",
        "bdg_nk",
        "delta0",
        "n0_policy",
        "local_response",
        "finite_momentum_resolved",
        "benchmark_only",
        "preliminary_local_response_conclusion",
        "not_final_casimir_conclusion",
        "matsubara_tail_indicator",
        "kparallel_cutoff_indicator",
        "phi_convergence_indicator",
        "diagnosis",
        "notes",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(data["kind"].size):
            writer.writerow({name: _csv_value(data[name][index]) for name in fieldnames})

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    kinds = list(dict.fromkeys(str(item) for item in data["kind"]))
    distances = list(dict.fromkeys(float(item) for item in data["distance_m"]))

    fig_energy, ax_energy = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in kinds:
        for distance in distances:
            mask = (data["kind"] == kind) & np.isclose(data["distance_m"], distance)
            ax_energy.plot(data["theta"][mask], data["energy"][mask].real, marker="o", label=f"{kind}, d={distance:g}")
    ax_energy.set_xlabel(r"$\theta$ (rad)")
    ax_energy.set_ylabel("benchmark energy")
    ax_energy.set_title("local-response benchmark energy")
    style_publication_axis(ax_energy)
    save_publication_figure(fig_energy, energy_plot)
    plt.close(fig_energy)

    fig_torque, ax_torque = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in kinds:
        for distance in distances:
            mask = (data["kind"] == kind) & np.isclose(data["distance_m"], distance)
            ax_torque.plot(data["theta"][mask], data["torque_fd"][mask], marker="o", label=f"{kind}, d={distance:g}")
    ax_torque.set_xlabel(r"$\theta$ (rad)")
    ax_torque.set_ylabel("benchmark torque")
    ax_torque.set_title("local-response finite-difference torque")
    style_publication_axis(ax_torque)
    save_publication_figure(fig_torque, torque_plot)
    plt.close(fig_torque)

    fig_partial, ax_partial = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    reference_theta = float(data["theta"][0])
    for kind in kinds:
        mask = (data["kind"] == kind) & np.isclose(data["theta"], reference_theta)
        if np.any(mask):
            index = np.where(mask)[0][0]
            ax_partial.plot(data["matsubara_indices"], data["matsubara_partial_sums"][index].real, marker="o", label=kind)
    ax_partial.set_xlabel("Matsubara index")
    ax_partial.set_ylabel("partial energy")
    ax_partial.set_title(rf"Matsubara partial sums at $\theta={reference_theta:g}$")
    style_publication_axis(ax_partial)
    save_publication_figure(fig_partial, partial_plot)
    plt.close(fig_partial)

    fig_distance, ax_distance = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in kinds:
        values = []
        for distance in distances:
            mask = (data["kind"] == kind) & np.isclose(data["distance_m"], distance)
            values.append(float(np.nanmax(data["max_abs_torque_over_theta"][mask])))
        ax_distance.plot(distances, values, marker="o", label=kind)
    ax_distance.set_xscale("log")
    ax_distance.set_yscale("symlog", linthresh=1e-30)
    ax_distance.set_xlabel("distance (m)")
    ax_distance.set_ylabel("max |torque|")
    ax_distance.set_title("distance dependence of benchmark torque")
    style_publication_axis(ax_distance)
    save_publication_figure(fig_distance, distance_plot)
    plt.close(fig_distance)

    if "toy_anisotropic" in kinds:
        fig_toy, ax_toy = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        for distance in distances:
            mask = (data["kind"] == "toy_anisotropic") & np.isclose(data["distance_m"], distance)
            ax_toy.plot(data["theta"][mask], data["torque_fd"][mask], marker="o", label=f"d={distance:g}")
        ax_toy.set_xlabel(r"$\theta$ (rad)")
        ax_toy.set_ylabel("toy torque")
        ax_toy.set_title("toy anisotropic control")
        style_publication_axis(ax_toy)
        save_publication_figure(fig_toy, toy_plot)
        plt.close(fig_toy)
    return paths


def print_summary(data: dict[str, np.ndarray]) -> None:
    for kind in sorted(set(str(item) for item in data["kind"])):
        mask = data["kind"] == kind
        print(f"kind = {kind}")
        print(f"max_abs_torque = {float(np.nanmax(data['max_abs_torque_over_theta'][mask]))}")
        print(f"diagnoses = {sorted(set(str(item) for item in data['diagnosis'][mask]))}")
    print("note = local-response benchmark only; n=0 skipped; finite momentum response not resolved; not a final Casimir conclusion.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--distance-list", nargs="+", type=float, default=[3e-8, 5e-8, 1e-7])
    parser.add_argument(
        "--theta-list",
        nargs="+",
        type=float,
        default=[0.0, 0.3926990817, 0.7853981634, 1.1780972451, 1.5707963268],
    )
    parser.add_argument("--matsubara-min", type=int, default=1)
    parser.add_argument("--matsubara-max", type=int, default=8)
    parser.add_argument("--kparallel-num", type=int, default=64)
    parser.add_argument("--kparallel-max-factor", type=float, default=20.0)
    parser.add_argument("--phi-num", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--normal-nk", type=int, default=96)
    parser.add_argument("--normal-eta", type=float, default=1e-4)
    parser.add_argument("--normal-sampling", choices=NORMAL_SAMPLING, default="fs_adaptive")
    parser.add_argument("--normal-refine-factor", type=int, default=8)
    parser.add_argument("--bdg-nk", type=int, default=32)
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--include-toy-anisotropic-control", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "outputs" / "casimir" / "local_response_integral" / "data" / "local_response_integral",
    )
    args = parser.parse_args()
    data = compute_local_response_casimir_integral(
        args.kinds,
        args.distance_list,
        args.theta_list,
        args.matsubara_min,
        args.matsubara_max,
        args.kparallel_num,
        args.kparallel_max_factor,
        args.phi_num,
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
