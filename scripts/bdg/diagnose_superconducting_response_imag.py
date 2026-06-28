#!/usr/bin/env python3
"""Scan Sigma_SC = K_total / omega_eV on the imaginary axis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import (  # noqa: E402
    KuboConfig,
    PairingAmplitudes,
    bdg_superconducting_response_imag_axis,
    bosonic_matsubara_energy_eV,
    k_weights,
    uniform_bz_mesh,
)
from validation.lib.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

REQUIRED_NPZ_FIELDS = {
    "kind",
    "n",
    "omega_eV",
    "Ktotal_xx",
    "Ktotal_yy",
    "Ktotal_xy",
    "Ktotal_yx",
    "Sigma_xx",
    "Sigma_yy",
    "Sigma_xy",
    "Sigma_yx",
    "delta_Sigma",
    "relative_offdiag_Sigma",
    "relative_eigen_split_Sigma",
    "delta0_eV",
    "nk",
    "temperature_K",
    "eta_eV",
    "total_convention",
    "response_layer",
    "not_final_optical_conductivity",
}


def relative_eigen_split(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvals(matrix)
    scale = 0.5 * (abs(eigenvalues[0]) + abs(eigenvalues[1]))
    if np.isclose(scale, 0.0):
        return 0.0
    return float(abs(eigenvalues[0] - eigenvalues[1]) / scale)


def kernel_diagnostics(kernel: np.ndarray) -> tuple[complex, float, float]:
    diagonal_scale = 0.5 * (abs(kernel[0, 0]) + abs(kernel[1, 1]))
    if np.isclose(kernel[0, 0] + kernel[1, 1], 0.0):
        delta = complex(0.0)
    else:
        delta = (kernel[0, 0] - kernel[1, 1]) / (kernel[0, 0] + kernel[1, 1])
    offdiag_norm = float(np.linalg.norm([kernel[0, 1], kernel[1, 0]]))
    relative_offdiag = 0.0 if np.isclose(diagonal_scale, 0.0) else float(offdiag_norm / diagonal_scale)
    return delta, relative_offdiag, relative_eigen_split(kernel)


def scan_kind(
    kind: str,
    delta0_eV: float,
    nk: int,
    temperature_K: float,
    matsubara_min: int,
    matsubara_max: int,
    eta_eV: float,
) -> dict[str, np.ndarray]:
    if matsubara_min < 1 or matsubara_max < matsubara_min:
        raise ValueError("Matsubara range must satisfy 1 <= min <= max")

    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    matsubara_indices = np.arange(matsubara_min, matsubara_max + 1, dtype=int)
    omega_eV = np.array([bosonic_matsubara_energy_eV(int(n), temperature_K) for n in matsubara_indices])

    data: dict[str, np.ndarray] = {
        "kind": np.array(kind),
        "n": matsubara_indices,
        "omega_eV": omega_eV,
        "delta0_eV": np.array(delta0_eV),
        "nk": np.array(nk),
        "temperature_K": np.array(temperature_K),
        "eta_eV": np.array(eta_eV),
        "total_convention": np.array("K_dia_minus_K_para"),
        "response_layer": np.array("sigma_like_response_from_total_kernel"),
        "not_final_optical_conductivity": np.array(True),
    }
    for prefix in ("Ktotal", "Sigma"):
        for component in ("xx", "yy", "xy", "yx"):
            data[f"{prefix}_{component}"] = np.empty(matsubara_indices.size, dtype=complex)
    data["delta_Sigma"] = np.empty(matsubara_indices.size, dtype=complex)
    data["relative_offdiag_Sigma"] = np.empty(matsubara_indices.size, dtype=float)
    data["relative_eigen_split_Sigma"] = np.empty(matsubara_indices.size, dtype=float)

    for index, omega in enumerate(omega_eV):
        config = KuboConfig.from_kelvin(
            omega_eV=float(omega),
            temperature_K=temperature_K,
            eta_eV=eta_eV,
            output_si=False,
        )
        response = bdg_superconducting_response_imag_axis(
            mesh,
            config,
            kind,  # type: ignore[arg-type]
            PairingAmplitudes(delta0_eV=delta0_eV),
            weights,
        )
        matrices = {
            "Ktotal": response.total,
            "Sigma": response.sigma_like_response,
        }
        for prefix, matrix in matrices.items():
            data[f"{prefix}_xx"][index] = matrix[0, 0]
            data[f"{prefix}_yy"][index] = matrix[1, 1]
            data[f"{prefix}_xy"][index] = matrix[0, 1]
            data[f"{prefix}_yx"][index] = matrix[1, 0]
        (
            data["delta_Sigma"][index],
            data["relative_offdiag_Sigma"][index],
            data["relative_eigen_split_Sigma"][index],
        ) = kernel_diagnostics(response.sigma_like_response)

    return data


def diagnosis(data: dict[str, np.ndarray]) -> str:
    if (
        float(np.max(np.abs(data["delta_Sigma"]))) < 1e-8
        and float(np.max(data["relative_offdiag_Sigma"])) < 1e-8
        and float(np.max(data["relative_eigen_split_Sigma"])) < 1e-8
    ):
        return "pass"
    return "check"


def output_paths(output_prefix: Path, kind: str) -> tuple[Path, Path, Path, Path]:
    data_path = output_prefix.parent / f"{output_prefix.name}_{kind}.npz"
    resolved_prefix = output_prefix.resolve()
    project_data_root = (
        ROOT / "outputs" / "bdg" / "superconducting_response_imag" / "data"
    ).resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "outputs" / "bdg" / "superconducting_response_imag" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return (
        data_path,
        figure_dir / f"{output_prefix.name}_{kind}_response.png",
        figure_dir / f"{output_prefix.name}_{kind}_diagnostics.png",
        figure_dir / f"{output_prefix.name}_{kind}_Ktotal_vs_Sigma.png",
    )


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path, Path, Path]:
    kind = str(data["kind"].item())
    npz_path, response_plot_path, diagnostics_plot_path, compare_plot_path = output_paths(output_prefix, kind)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    response_plot_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    omega = data["omega_eV"]
    fig_response, ax_response = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax_response.plot(omega, data["Sigma_xx"].real, label="Re Sigma_xx")
    ax_response.plot(omega, data["Sigma_yy"].real, label="Re Sigma_yy", linestyle="--")
    ax_response.set_xlabel("imaginary-axis energy (eV)")
    ax_response.set_ylabel("Re Sigma_SC")
    ax_response.set_title(f"{kind}: imaginary-axis BdG response")
    style_publication_axis(ax_response)
    save_publication_figure(fig_response, response_plot_path)
    plt.close(fig_response)

    fig_diag, ax_diag = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax_diag.plot(omega, np.abs(data["delta_Sigma"]), label="|delta_Sigma|")
    ax_diag.plot(omega, data["relative_offdiag_Sigma"], label="relative_offdiag_Sigma")
    ax_diag.plot(omega, data["relative_eigen_split_Sigma"], label="relative_eigen_split_Sigma")
    ax_diag.set_xlabel("imaginary-axis energy (eV)")
    ax_diag.set_ylabel("relative diagnostic")
    ax_diag.set_yscale("log")
    ax_diag.set_title(f"{kind}: $C_4$ symmetry of $\\Sigma_{{\\rm SC}}$")
    style_publication_axis(ax_diag)
    save_publication_figure(fig_diag, diagnostics_plot_path)
    plt.close(fig_diag)

    fig_compare, axes_compare = plt.subplots(2, 1, figsize=(6.0, 5.4), sharex=True, constrained_layout=True)
    axes_compare[0].plot(omega, data["Ktotal_xx"].real, label=r"$\mathrm{Re}\,K_{\rm total,xx}$")
    axes_compare[0].set_ylabel(r"$K_{\rm total,xx}$")
    axes_compare[0].set_title(f"{kind}: total kernel and sigma-like response")
    style_publication_axis(axes_compare[0])
    axes_compare[1].plot(omega, data["Sigma_xx"].real, label=r"$\mathrm{Re}\,\Sigma_{\rm SC,xx}$")
    axes_compare[1].set_xlabel("imaginary-axis energy (eV)")
    axes_compare[1].set_ylabel(r"$\Sigma_{\rm SC,xx}$")
    style_publication_axis(axes_compare[1])
    save_publication_figure(fig_compare, compare_plot_path)
    plt.close(fig_compare)
    return npz_path, response_plot_path, diagnostics_plot_path, compare_plot_path


def print_summary(data: dict[str, np.ndarray]) -> None:
    print(f"kind = {data['kind'].item()}")
    print(f"max_abs_delta_Sigma = {float(np.max(np.abs(data['delta_Sigma'])))}")
    print(f"max_relative_offdiag_Sigma = {float(np.max(data['relative_offdiag_Sigma']))}")
    print(f"max_relative_eigen_split_Sigma = {float(np.max(data['relative_eigen_split_Sigma']))}")
    print(f"diagnosis = {diagnosis(data)}")
    print("total_convention = K_dia_minus_K_para")
    print("note = Sigma_SC = K_total/omega_eV for n>=1; not final optical conductivity.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=("spm", "dwave"), default=["spm", "dwave"])
    parser.add_argument("--delta0", type=float, default=0.04, help="Pairing amplitude Delta0 in eV.")
    parser.add_argument("--nk", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=30.0, help="Temperature in K.")
    parser.add_argument("--matsubara-min", type=int, default=1)
    parser.add_argument("--matsubara-max", type=int, default=8)
    parser.add_argument("--eta", type=float, default=1e-4, help="Imaginary-axis regulator in eV.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT
        / "outputs"
        / "bdg"
        / "superconducting_response_imag"
        / "data"
        / "Sigma_SC_imag",
    )
    args = parser.parse_args()

    if args.nk <= 0:
        raise ValueError("--nk must be positive")
    if args.matsubara_min < 1 or args.matsubara_max < args.matsubara_min:
        raise ValueError("Matsubara range must satisfy 1 <= min <= max")
    if args.eta <= 0.0:
        raise ValueError("--eta must be positive")

    for kind in args.kinds:
        data = scan_kind(
            kind,
            args.delta0,
            args.nk,
            args.temperature,
            args.matsubara_min,
            args.matsubara_max,
            args.eta,
        )
        paths = save_outputs(data, args.output_prefix)
        print_summary(data)
        print(f"npz_path = {paths[0]}")
        print(f"figure_paths = {paths[1]}, {paths[2]}, {paths[3]}")


if __name__ == "__main__":
    main()
