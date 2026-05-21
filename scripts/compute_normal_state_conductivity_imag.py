#!/usr/bin/env python3
"""Compute normal-state sigma(i xi) on a uniform Brillouin-zone mesh."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lno327 import (
    KuboConfig,
    bosonic_matsubara_energy_eV,
    conductivity_matrix_diagnostics,
    k_weights,
    kubo_conductivity_imag_axis,
    uniform_bz_mesh,
)


def principal_axis_angle(eigenvectors):
    principal_vector = eigenvectors[:, 0]
    return float(np.angle(principal_vector[0] + 1j * principal_vector[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nk", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=30.0, help="Temperature in K.")
    parser.add_argument("--matsubara-index", type=int, default=1)
    parser.add_argument("--eta", type=float, default=1e-4, help="Energy broadening in eV.")
    parser.add_argument("--fermi-level", type=float, default=0.0, help="Fermi level in eV.")
    parser.add_argument("--dimensionless", action="store_true", help="Do not multiply by e^2/hbar.")
    args = parser.parse_args()

    mesh = uniform_bz_mesh(args.nk)
    omega_eV = bosonic_matsubara_energy_eV(args.matsubara_index, args.temperature)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=args.temperature,
        fermi_level_eV=args.fermi_level,
        eta_eV=args.eta,
        output_si=not args.dimensionless,
    )
    sigma = kubo_conductivity_imag_axis(mesh, config, k_weights(mesh))
    diagnostics = conductivity_matrix_diagnostics(sigma)
    eigenvalues = diagnostics["eigenvalues"]
    eigenvectors = diagnostics["eigenvectors"]
    eigenvalue_scale = 0.5 * (abs(eigenvalues[0]) + abs(eigenvalues[1]))
    relative_eigen_split = 0.0
    if eigenvalue_scale != 0.0:
        relative_eigen_split = abs(eigenvalues[0] - eigenvalues[1]) / eigenvalue_scale

    unit = "SI sheet conductance" if config.output_si else "dimensionless"
    print(f"nk = {args.nk} x {args.nk}")
    print(f"temperature_K = {args.temperature}")
    print(f"matsubara_index = {args.matsubara_index}")
    print(f"omega_eV = {omega_eV:.12g}")
    print(f"unit = {unit}")
    print(f"sigma_matrix = {diagnostics['sigma_matrix']}")
    print(f"anisotropy_delta = {diagnostics['anisotropy_delta']}")
    print(f"offdiag_norm = {diagnostics['offdiag_norm']}")
    print(f"relative_xx_yy_error = {diagnostics['relative_xx_yy_error']}")
    print(f"eigenvalue_1 = {eigenvalues[0]}")
    print(f"eigenvalue_2 = {eigenvalues[1]}")
    print(f"relative_eigen_split = {relative_eigen_split}")
    print(f"principal_axis_angle = {principal_axis_angle(eigenvectors)}")


if __name__ == "__main__":
    main()
