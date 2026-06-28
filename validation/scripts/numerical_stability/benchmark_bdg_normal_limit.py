#!/usr/bin/env python3
"""Benchmark the Delta0 -> 0 BdG response against a normal-state baseline.

This script checks continuity and numerical stability of the BdG response
layer. It is not a Casimir calculation and does not require BdG Sigma to equal
normal Kubo conductivity term by term.
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
    PairingAmplitudes,
    bdg_total_kernel_imag_axis,
    bosonic_matsubara_energy_eV,
    k_weights,
    kubo_conductivity_imag_axis,
    matrix_symmetry_diagnostics,
    uniform_bz_mesh,
)
from validation.lib.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("spm", "dwave")
DEFAULT_DELTA0_LIST = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.04)
SYMMETRY_TOLERANCE = 1e-8
CANCELLATION_WARNING_RATIO = 1e-2

REQUIRED_NPZ_FIELDS = {
    "delta0_list",
    "delta0",
    "kind",
    "omega_eV",
    "normal_sigma_xx",
    "normal_sigma_yy",
    "normal_sigma_xy",
    "normal_sigma_yx",
    "Sigma_xx",
    "Sigma_yy",
    "Sigma_xy",
    "Sigma_yx",
    "Kpara_xx",
    "Kdia_xx",
    "Ktotal_xx",
    "ratio_sigma_xx_to_normal",
    "spm_dwave_abs_diff_xx",
    "spm_dwave_rel_diff_xx",
    "delta",
    "relative_offdiag",
    "relative_eigen_split",
    "diagnosis",
}


def _relative_eigen_split(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvals(matrix)
    scale = 0.5 * (abs(eigenvalues[0]) + abs(eigenvalues[1]))
    if np.isclose(scale, 0.0):
        return 0.0
    return float(abs(eigenvalues[0] - eigenvalues[1]) / scale)


def _normal_sigma(
    mesh: np.ndarray,
    weights: np.ndarray,
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
) -> np.ndarray:
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    return kubo_conductivity_imag_axis(mesh, config, weights).matrix()


def _bdg_response(
    kind: str,
    delta0_eV: float,
    mesh: np.ndarray,
    weights: np.ndarray,
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    components = bdg_total_kernel_imag_axis(
        mesh,
        config,
        kind,  # type: ignore[arg-type]
        PairingAmplitudes(delta0_eV=delta0_eV),
        weights,
    )
    sigma = components.total / omega_eV
    return components.paramagnetic, components.diamagnetic, components.total, sigma


def _ratio(numerator: complex, denominator: complex) -> complex:
    if np.isclose(denominator, 0.0):
        return complex(np.nan, np.nan)
    return numerator / denominator


def _diagnosis(
    sigma: np.ndarray,
    ratio: complex,
    cancellation_metric: float,
    spm_dwave_rel_diff: float,
    is_small_delta: bool,
) -> str:
    parts: list[str] = []
    diagnostics = matrix_symmetry_diagnostics(sigma)
    if (
        abs(complex(diagnostics["delta"])) > SYMMETRY_TOLERANCE
        or float(diagnostics["relative_offdiag"]) > SYMMETRY_TOLERANCE
        or float(diagnostics["relative_eigen_split"]) > SYMMETRY_TOLERANCE
    ):
        parts.append("warning_symmetry")
    if not np.isfinite(ratio):
        parts.append("warning_nonfinite_ratio")
    if cancellation_metric < CANCELLATION_WARNING_RATIO:
        parts.append("warning_large_Kpara_Kdia_cancellation")
    if is_small_delta and spm_dwave_rel_diff > 1e-8:
        parts.append("warning_small_delta_spm_dwave_difference")
    return "pass" if not parts else ";".join(parts)


def benchmark_bdg_normal_limit(
    kinds: list[str],
    delta0_list: list[float],
    nk: int,
    temperature_K: float,
    matsubara_index: int,
    eta_eV: float,
) -> dict[str, np.ndarray]:
    """Run the BdG-normal-limit benchmark."""

    if nk <= 0:
        raise ValueError("nk must be positive")
    if matsubara_index < 1:
        raise ValueError("matsubara_index must be >= 1")
    if eta_eV <= 0.0:
        raise ValueError("eta must be positive")
    if not delta0_list:
        raise ValueError("delta0_list must be non-empty")
    if any(delta0 < 0.0 for delta0 in delta0_list):
        raise ValueError("delta0 values must be non-negative")

    delta_values = np.asarray(delta0_list, dtype=float)
    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    omega_eV = bosonic_matsubara_energy_eV(matsubara_index, temperature_K)
    normal = _normal_sigma(mesh, weights, omega_eV, temperature_K, eta_eV)
    rows = [(kind, delta0) for kind in kinds for delta0 in delta_values]
    row_count = len(rows)

    data: dict[str, np.ndarray] = {
        "delta0_list": delta_values,
        "delta0": np.empty(row_count, dtype=float),
        "kind": np.empty(row_count, dtype="U16"),
        "omega_eV": np.full(row_count, omega_eV, dtype=float),
        "normal_sigma_xx": np.full(row_count, normal[0, 0], dtype=complex),
        "normal_sigma_yy": np.full(row_count, normal[1, 1], dtype=complex),
        "normal_sigma_xy": np.full(row_count, normal[0, 1], dtype=complex),
        "normal_sigma_yx": np.full(row_count, normal[1, 0], dtype=complex),
        "Sigma_xx": np.empty(row_count, dtype=complex),
        "Sigma_yy": np.empty(row_count, dtype=complex),
        "Sigma_xy": np.empty(row_count, dtype=complex),
        "Sigma_yx": np.empty(row_count, dtype=complex),
        "Kpara_xx": np.empty(row_count, dtype=complex),
        "Kdia_xx": np.empty(row_count, dtype=complex),
        "Ktotal_xx": np.empty(row_count, dtype=complex),
        "ratio_sigma_xx_to_normal": np.empty(row_count, dtype=complex),
        "spm_dwave_abs_diff_xx": np.full(row_count, np.nan, dtype=float),
        "spm_dwave_rel_diff_xx": np.full(row_count, np.nan, dtype=float),
        "delta": np.empty(row_count, dtype=complex),
        "relative_offdiag": np.empty(row_count, dtype=float),
        "relative_eigen_split": np.empty(row_count, dtype=float),
        "diagnosis": np.empty(row_count, dtype="U160"),
        "K_cancellation_metric_xx": np.empty(row_count, dtype=float),
        "nk": np.array(nk),
        "temperature_K": np.array(temperature_K),
        "matsubara_index": np.array(matsubara_index),
        "eta_eV": np.array(eta_eV),
        "notes": np.array(
            "BdG-normal-limit benchmark only; normal Kubo and BdG Sigma need not match term by term",
            dtype=object,
        ),
    }

    sigma_by_pair: dict[tuple[str, float], np.ndarray] = {}
    for index, (kind, delta0) in enumerate(rows):
        kpara, kdia, ktotal, sigma = _bdg_response(
            kind,
            float(delta0),
            mesh,
            weights,
            omega_eV,
            temperature_K,
            eta_eV,
        )
        sigma_by_pair[(kind, float(delta0))] = sigma
        diagnostics = matrix_symmetry_diagnostics(sigma)
        data["kind"][index] = kind
        data["delta0"][index] = float(delta0)
        data["Sigma_xx"][index] = sigma[0, 0]
        data["Sigma_yy"][index] = sigma[1, 1]
        data["Sigma_xy"][index] = sigma[0, 1]
        data["Sigma_yx"][index] = sigma[1, 0]
        data["Kpara_xx"][index] = kpara[0, 0]
        data["Kdia_xx"][index] = kdia[0, 0]
        data["Ktotal_xx"][index] = ktotal[0, 0]
        data["ratio_sigma_xx_to_normal"][index] = _ratio(sigma[0, 0], normal[0, 0])
        data["delta"][index] = complex(diagnostics["delta"])
        data["relative_offdiag"][index] = float(diagnostics["relative_offdiag"])
        data["relative_eigen_split"][index] = float(diagnostics["relative_eigen_split"])
        component_scale = abs(kpara[0, 0]) + abs(kdia[0, 0])
        data["K_cancellation_metric_xx"][index] = (
            1.0 if np.isclose(component_scale, 0.0) else float(abs(ktotal[0, 0]) / component_scale)
        )

    for delta0 in delta_values:
        if ("spm", float(delta0)) not in sigma_by_pair or ("dwave", float(delta0)) not in sigma_by_pair:
            continue
        spm_sigma = sigma_by_pair[("spm", float(delta0))]
        dwave_sigma = sigma_by_pair[("dwave", float(delta0))]
        abs_diff = float(abs(spm_sigma[0, 0] - dwave_sigma[0, 0]))
        scale = 0.5 * (abs(spm_sigma[0, 0]) + abs(dwave_sigma[0, 0]))
        rel_diff = 0.0 if np.isclose(scale, 0.0) else float(abs_diff / scale)
        for index, (kind, row_delta0) in enumerate(rows):
            if np.isclose(row_delta0, delta0):
                data["spm_dwave_abs_diff_xx"][index] = abs_diff
                data["spm_dwave_rel_diff_xx"][index] = rel_diff

    small_positive_delta = min((float(value) for value in delta_values if value > 0.0), default=0.0)
    for index, (_kind, delta0) in enumerate(rows):
        data["diagnosis"][index] = _diagnosis(
            np.array(
                [
                    [data["Sigma_xx"][index], data["Sigma_xy"][index]],
                    [data["Sigma_yx"][index], data["Sigma_yy"][index]],
                ],
                dtype=complex,
            ),
            data["ratio_sigma_xx_to_normal"][index],
            float(data["K_cancellation_metric_xx"][index]),
            float(data["spm_dwave_rel_diff_xx"][index]),
            bool(np.isclose(delta0, 0.0) or np.isclose(delta0, small_positive_delta)),
        )

    return data


def output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    resolved_prefix = output_prefix.resolve()
    project_data_root = (ROOT / "validation" / "outputs" / "archive" / "response" / "bdg_normal_limit" / "data").resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "validation" / "outputs" / "archive" / "response" / "bdg_normal_limit" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return (
        npz_path,
        csv_path,
        figure_dir / f"{output_prefix.name}_sigma_xx.png",
        figure_dir / f"{output_prefix.name}_ratio.png",
        figure_dir / f"{output_prefix.name}_spm_dwave_diff.png",
        figure_dir / f"{output_prefix.name}_kernel_components.png",
        figure_dir / f"{output_prefix.name}_symmetry.png",
    )


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    paths = output_paths(output_prefix)
    npz_path, csv_path, sigma_plot, ratio_plot, diff_plot, kernel_plot, symmetry_plot = paths
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    sigma_plot.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)

    fieldnames = [
        "kind",
        "delta0",
        "omega_eV",
        "normal_sigma_xx",
        "normal_sigma_yy",
        "normal_sigma_xy",
        "normal_sigma_yx",
        "Sigma_xx",
        "Sigma_yy",
        "Sigma_xy",
        "Sigma_yx",
        "Kpara_xx",
        "Kdia_xx",
        "Ktotal_xx",
        "ratio_sigma_xx_to_normal",
        "spm_dwave_abs_diff_xx",
        "spm_dwave_rel_diff_xx",
        "delta",
        "relative_offdiag",
        "relative_eigen_split",
        "K_cancellation_metric_xx",
        "diagnosis",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(data["kind"].size):
            writer.writerow({name: _csv_value(data[name][index]) for name in fieldnames})

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    kinds = list(dict.fromkeys(str(kind) for kind in data["kind"]))
    delta0 = data["delta0_list"]
    x = delta0 + 1e-20
    normal_ref = float(np.real(data["normal_sigma_xx"][0]))

    fig_sigma, ax_sigma = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    for kind in kinds:
        mask = data["kind"] == kind
        ax_sigma.plot(x, data["Sigma_xx"][mask].real, marker="o", label=kind)
    ax_sigma.axhline(normal_ref, color="black", linestyle="--", linewidth=1.0, label="normal sigma_xx")
    ax_sigma.set_xscale("log")
    ax_sigma.set_xlabel(r"$\Delta_0$ (eV)")
    ax_sigma.set_ylabel(r"Re $\Sigma_{xx}$")
    ax_sigma.set_title(r"BdG $\Sigma_{xx}$ normal-limit benchmark")
    style_publication_axis(ax_sigma)
    save_publication_figure(fig_sigma, sigma_plot)
    plt.close(fig_sigma)

    fig_ratio, ax_ratio = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    for kind in kinds:
        mask = data["kind"] == kind
        ax_ratio.plot(x, data["ratio_sigma_xx_to_normal"][mask].real, marker="o", label=kind)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel(r"$\Delta_0$ (eV)")
    ax_ratio.set_ylabel(r"Re $\Sigma_{xx}/\sigma^{normal}_{xx}$")
    ax_ratio.set_title("BdG-to-normal response ratio")
    style_publication_axis(ax_ratio)
    save_publication_figure(fig_ratio, ratio_plot)
    plt.close(fig_ratio)

    fig_diff, ax_diff = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    first_kind_mask = data["kind"] == kinds[0]
    ax_diff.plot(x, data["spm_dwave_abs_diff_xx"][first_kind_mask], marker="o", label="abs diff")
    ax_diff.plot(x, data["spm_dwave_rel_diff_xx"][first_kind_mask], marker="s", linestyle="--", label="rel diff")
    ax_diff.set_xscale("log")
    ax_diff.set_yscale("symlog", linthresh=1e-16)
    ax_diff.set_xlabel(r"$\Delta_0$ (eV)")
    ax_diff.set_ylabel(r"$s_{\pm}$-$d$ difference")
    ax_diff.set_title(r"$s_{\pm}$ and $d$-wave BdG response convergence")
    style_publication_axis(ax_diff)
    save_publication_figure(fig_diff, diff_plot)
    plt.close(fig_diff)

    fig_kernel, ax_kernel = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    for kind in kinds:
        mask = data["kind"] == kind
        ax_kernel.plot(x, data["Kpara_xx"][mask].real, marker="o", label=f"{kind} Kpara")
        ax_kernel.plot(x, data["Kdia_xx"][mask].real, marker="s", linestyle="--", label=f"{kind} Kdia")
        ax_kernel.plot(x, data["Ktotal_xx"][mask].real, marker="^", linestyle=":", label=f"{kind} Ktotal")
    ax_kernel.set_xscale("log")
    ax_kernel.set_xlabel(r"$\Delta_0$ (eV)")
    ax_kernel.set_ylabel("xx component")
    ax_kernel.set_title("BdG kernel component magnitudes")
    style_publication_axis(ax_kernel)
    save_publication_figure(fig_kernel, kernel_plot)
    plt.close(fig_kernel)

    fig_sym, ax_sym = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    for kind in kinds:
        mask = data["kind"] == kind
        ax_sym.plot(x, np.abs(data["delta"][mask]), marker="o", label=f"{kind} |delta|")
        ax_sym.plot(x, data["relative_offdiag"][mask], marker="s", linestyle="--", label=f"{kind} offdiag")
        ax_sym.plot(x, data["relative_eigen_split"][mask], marker="^", linestyle=":", label=f"{kind} eig split")
    ax_sym.set_xscale("log")
    ax_sym.set_yscale("symlog", linthresh=1e-16)
    ax_sym.set_xlabel(r"$\Delta_0$ (eV)")
    ax_sym.set_ylabel("relative diagnostic")
    ax_sym.set_title("BdG response symmetry diagnostics")
    style_publication_axis(ax_sym)
    save_publication_figure(fig_sym, symmetry_plot)
    plt.close(fig_sym)

    return paths


def print_summary(data: dict[str, np.ndarray]) -> None:
    diagnoses = set(str(item) for item in data["diagnosis"])
    finite_ratio = bool(np.isfinite(data["ratio_sigma_xx_to_normal"]).all())
    zero_mask = np.isclose(data["delta0_list"], 0.0)
    if bool(np.any(zero_mask)):
        zero_delta = float(data["delta0_list"][np.flatnonzero(zero_mask)[0]])
    else:
        zero_delta = float(data["delta0_list"][0])
    zero_rows = np.isclose(data["delta0"], zero_delta)
    zero_diff = float(np.nanmax(data["spm_dwave_abs_diff_xx"][zero_rows]))
    max_symmetry = float(
        max(
            np.nanmax(np.abs(data["delta"])),
            np.nanmax(data["relative_offdiag"]),
            np.nanmax(data["relative_eigen_split"]),
        )
    )
    min_cancellation_metric = float(np.nanmin(data["K_cancellation_metric_xx"]))
    print(f"omega_eV = {float(data['omega_eV'][0])}")
    print(f"normal_sigma_xx = {data['normal_sigma_xx'][0]}")
    print(f"delta0_list = {data['delta0_list']}")
    print(f"delta0_zero_spm_dwave_abs_diff_xx = {zero_diff}")
    print(f"ratio_sigma_xx_to_normal_all_finite = {finite_ratio}")
    print(f"max_symmetry_diagnostic = {max_symmetry}")
    print(f"min_K_cancellation_metric_xx = {min_cancellation_metric}")
    print(f"diagnoses = {sorted(diagnoses)}")
    print("note = BdG-normal-limit benchmark only; not a Casimir result.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--delta0-list", nargs="+", type=float, default=list(DEFAULT_DELTA0_LIST))
    parser.add_argument("--nk", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--matsubara-index", type=int, default=1)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "validation" / "outputs" / "archive" / "response" / "bdg_normal_limit" / "data" / "bdg_normal_limit",
    )
    args = parser.parse_args()

    data = benchmark_bdg_normal_limit(
        args.kinds,
        args.delta0_list,
        args.nk,
        args.temperature,
        args.matsubara_index,
        args.eta,
    )
    paths = save_outputs(data, args.output_prefix)
    print_summary(data)
    print(f"npz_path = {paths[0]}")
    print(f"csv_path = {paths[1]}")
    print(f"figure_paths = {paths[2]}, {paths[3]}, {paths[4]}, {paths[5]}, {paths[6]}")


if __name__ == "__main__":
    main()
