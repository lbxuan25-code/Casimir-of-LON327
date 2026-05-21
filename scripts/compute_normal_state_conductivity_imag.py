#!/usr/bin/env python3
"""Compute normal-state sigma(i xi) on a uniform Brillouin-zone mesh."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lno327 import (
    KuboConfig,
    anisotropy_summary,
    bosonic_matsubara_energy_eV,
    k_weights,
    kubo_conductivity_imag_axis,
    uniform_bz_mesh,
)


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
    summary = anisotropy_summary(sigma)

    unit = "SI sheet conductance" if config.output_si else "dimensionless"
    print(f"nk = {args.nk} x {args.nk}")
    print(f"temperature_K = {args.temperature}")
    print(f"matsubara_index = {args.matsubara_index}")
    print(f"omega_eV = {omega_eV:.12g}")
    print(f"unit = {unit}")
    print(f"sigma_xx = {sigma.xx}")
    print(f"sigma_yy = {sigma.yy}")
    print(f"sigma_xy = {sigma.xy}")
    print(f"sigma_yx = {sigma.yx}")
    print(f"delta = {summary['delta']}")


if __name__ == "__main__":
    main()
