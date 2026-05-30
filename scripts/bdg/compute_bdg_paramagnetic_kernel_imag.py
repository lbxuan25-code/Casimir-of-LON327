#!/usr/bin/env python3
"""Compute a minimal BdG paramagnetic kernel on the imaginary axis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lno327 import (  # noqa: E402
    KuboConfig,
    PairingAmplitudes,
    bdg_paramagnetic_kernel_imag_axis,
    bosonic_matsubara_energy_eV,
    k_weights,
    uniform_bz_mesh,
)


def anisotropy_delta(kernel: np.ndarray) -> complex:
    denom = kernel[0, 0] + kernel[1, 1]
    if np.isclose(denom, 0.0):
        return complex(0.0)
    return (kernel[0, 0] - kernel[1, 1]) / denom


def diagnosis(delta: complex, offdiag_norm: float) -> str:
    if abs(delta) < 1e-8 and offdiag_norm < 1e-8:
        return "ok"
    return "check_relative_diagnostics"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("spm", "dwave"), default="spm")
    parser.add_argument("--delta0", type=float, default=0.04, help="Pairing amplitude Delta0 in eV.")
    parser.add_argument("--nk", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=30.0, help="Temperature in K.")
    parser.add_argument("--matsubara-index", type=int, default=1)
    parser.add_argument("--eta", type=float, default=1e-4, help="Imaginary-axis regulator in eV.")
    args = parser.parse_args()

    if args.nk <= 0:
        raise ValueError("--nk must be positive")
    if args.matsubara_index < 0:
        raise ValueError("--matsubara-index must be non-negative")

    mesh = uniform_bz_mesh(args.nk)
    omega_eV = bosonic_matsubara_energy_eV(args.matsubara_index, args.temperature)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=args.temperature,
        eta_eV=args.eta,
        output_si=False,
    )
    kernel = bdg_paramagnetic_kernel_imag_axis(
        mesh,
        config,
        args.kind,
        PairingAmplitudes(delta0_eV=args.delta0),
        k_weights(mesh),
    )
    delta = anisotropy_delta(kernel)
    offdiag_norm = float(np.linalg.norm([kernel[0, 1], kernel[1, 0]]))

    np.set_printoptions(precision=12, suppress=False)
    print(f"kind = {args.kind}")
    print(f"delta0 = {args.delta0} eV")
    print(f"nk = {args.nk} x {args.nk}")
    print(f"T = {args.temperature} K")
    print(f"matsubara index = {args.matsubara_index}")
    print(f"imaginary-axis energy = {omega_eV} eV")
    print("paramagnetic kernel matrix =")
    print(kernel)
    print(f"delta = {delta}")
    print(f"offdiag_norm = {offdiag_norm}")
    print(f"diagnosis = {diagnosis(delta, offdiag_norm)}")
    print("note = paramagnetic kernel only; no diamagnetic term or full superconducting conductivity")


if __name__ == "__main__":
    main()
