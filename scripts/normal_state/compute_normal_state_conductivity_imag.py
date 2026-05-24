#!/usr/bin/env python3
"""Compute normal-state sigma(i xi) on a uniform Brillouin-zone mesh."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import (
    KuboConfig,
    bosonic_matsubara_energy_eV,
    conductivity_matrix_diagnostics,
    k_weights,
    kubo_conductivity_imag_axis,
    uniform_bz_mesh,
)


def relative_eigen_split(eigenvalues: np.ndarray) -> float:
    scale = 0.5 * (abs(eigenvalues[0]) + abs(eigenvalues[1]))
    if np.isclose(scale, 0.0):
        return 0.0
    return float(abs(eigenvalues[0] - eigenvalues[1]) / scale)


def diagnosis(
    max_abs_delta: float,
    max_relative_offdiag: float,
    max_relative_eigen_split: float,
) -> str:
    if max_abs_delta < 1e-8 and max_relative_offdiag < 1e-8 and max_relative_eigen_split < 1e-8:
        return "ok"
    return "check_relative_diagnostics"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=30.0, help="Temperature in K.")
    parser.add_argument("--matsubara-index", type=int, default=None)
    parser.add_argument("--omega-min", type=float, default=0.001, help="Minimum imaginary-axis energy in eV.")
    parser.add_argument("--omega-max", type=float, default=1.0, help="Maximum imaginary-axis energy in eV.")
    parser.add_argument("--num-omega", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1e-4, help="Energy broadening in eV.")
    parser.add_argument("--fermi-level", type=float, default=0.0, help="Fermi level in eV.")
    parser.add_argument("--dimensionless", action="store_true", help="Do not multiply by e^2/hbar.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT
        / "outputs"
        / "normal_state"
        / "conductivity_imag"
        / "data"
        / "normal_state_conductivity_imag",
    )
    args = parser.parse_args()

    if args.nk <= 0:
        raise ValueError("--nk must be positive")
    if args.num_omega <= 0:
        raise ValueError("--num-omega must be positive")
    if args.omega_max < args.omega_min:
        raise ValueError("--omega-max must be greater than or equal to --omega-min")

    mesh = uniform_bz_mesh(args.nk)
    weights = k_weights(mesh)
    if args.matsubara_index is None:
        omega_eV = np.linspace(args.omega_min, args.omega_max, args.num_omega)
    else:
        omega_eV = np.array([bosonic_matsubara_energy_eV(args.matsubara_index, args.temperature)])

    sigma_xx = np.empty(omega_eV.size, dtype=complex)
    sigma_yy = np.empty(omega_eV.size, dtype=complex)
    sigma_xy = np.empty(omega_eV.size, dtype=complex)
    sigma_yx = np.empty(omega_eV.size, dtype=complex)
    delta = np.empty(omega_eV.size, dtype=complex)
    eigenvalues = np.empty((omega_eV.size, 2), dtype=complex)
    relative_splits = np.empty(omega_eV.size, dtype=float)
    offdiag_norm = np.empty(omega_eV.size, dtype=float)
    diagonal_scale = np.empty(omega_eV.size, dtype=float)

    for index, omega in enumerate(omega_eV):
        config = KuboConfig.from_kelvin(
            omega_eV=float(omega),
            temperature_K=args.temperature,
            fermi_level_eV=args.fermi_level,
            eta_eV=args.eta,
            output_si=not args.dimensionless,
        )
        sigma = kubo_conductivity_imag_axis(mesh, config, weights)
        diagnostics = conductivity_matrix_diagnostics(sigma)
        sigma_xx[index] = sigma.xx
        sigma_yy[index] = sigma.yy
        sigma_xy[index] = sigma.xy
        sigma_yx[index] = sigma.yx
        delta[index] = diagnostics["anisotropy_delta"]
        eigenvalues[index] = diagnostics["eigenvalues"]
        relative_splits[index] = relative_eigen_split(diagnostics["eigenvalues"])
        offdiag_norm[index] = diagnostics["offdiag_norm"]
        diagonal_scale[index] = 0.5 * (abs(sigma.xx) + abs(sigma.yy))

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figures_dir = ROOT / "outputs" / "normal_state" / "conductivity_imag" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    re_plot_path = figures_dir / f"{output_prefix.name}_imag_re.png"
    im_plot_path = figures_dir / f"{output_prefix.name}_imag_im.png"

    np.savez(
        npz_path,
        omega_eV=omega_eV,
        sigma_xx=sigma_xx,
        sigma_yy=sigma_yy,
        sigma_xy=sigma_xy,
        sigma_yx=sigma_yx,
        delta=delta,
        eigenvalues=eigenvalues,
        relative_eigen_split=relative_splits,
        offdiag_norm=offdiag_norm,
    )

    import matplotlib.pyplot as plt

    fig_re, ax_re = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax_re.plot(omega_eV, sigma_xx.real, label="Re sigma_xx")
    ax_re.plot(omega_eV, sigma_yy.real, label="Re sigma_yy", linestyle="--")
    ax_re.set_xlabel("imaginary-axis energy (eV)")
    ax_re.set_ylabel("Re sigma")
    ax_re.legend()
    fig_re.savefig(re_plot_path, dpi=200)
    plt.close(fig_re)

    fig_im, ax_im = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax_im.plot(omega_eV, sigma_xx.imag, label="Im sigma_xx")
    ax_im.plot(omega_eV, sigma_yy.imag, label="Im sigma_yy", linestyle="--")
    ax_im.set_xlabel("imaginary-axis energy (eV)")
    ax_im.set_ylabel("Im sigma")
    ax_im.legend()
    fig_im.savefig(im_plot_path, dpi=200)
    plt.close(fig_im)

    max_abs_delta = float(np.max(np.abs(delta)))
    max_offdiag_norm = float(np.max(offdiag_norm))
    relative_offdiag = np.divide(
        offdiag_norm,
        diagonal_scale,
        out=np.zeros_like(offdiag_norm),
        where=~np.isclose(diagonal_scale, 0.0),
    )
    max_relative_offdiag = float(np.max(relative_offdiag))
    max_relative_eigen_split = float(np.max(relative_splits))
    unit = "e2_over_hbar_scaled conductivity kernel" if not args.dimensionless else "dimensionless"

    print(f"nk = {args.nk} x {args.nk}")
    print(f"T = {args.temperature} K")
    print(f"omega range = [{omega_eV[0]}, {omega_eV[-1]}] eV, num = {omega_eV.size}")
    print(f"eta = {args.eta} eV")
    print(f"unit = {unit}")
    print(f"output path = {npz_path}")
    print(f"figure paths = {re_plot_path}, {im_plot_path}")
    print(f"max_abs_delta = {max_abs_delta}")
    print(f"max_offdiag_norm = {max_offdiag_norm}")
    print(f"max_relative_offdiag = {max_relative_offdiag}")
    print(f"max_relative_eigen_split = {max_relative_eigen_split}")
    print(f"diagnosis = {diagnosis(max_abs_delta, max_relative_offdiag, max_relative_eigen_split)}")


if __name__ == "__main__":
    main()
