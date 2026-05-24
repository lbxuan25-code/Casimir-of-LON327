#!/usr/bin/env python3
"""Inspect minimal s_pm and d-wave pairing matrices and BdG spectra."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lno327 import (  # noqa: E402
    PairingAmplitudes,
    bdg_hamiltonian,
    dwave_pairing_matrix,
    normal_state_hamiltonian,
    spm_pairing_matrix,
)


def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"{name} dtype = {matrix.dtype}")
    print(f"{name} [eV] =")
    print(matrix)


def particle_hole_residual(energies: np.ndarray) -> float:
    return float(np.max(np.abs(energies + energies[::-1])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kx", type=float, default=0.2)
    parser.add_argument("--ky", type=float, default=-0.5)
    parser.add_argument("--delta0-eV", type=float, default=0.04)
    args = parser.parse_args()

    amp = PairingAmplitudes(delta0_eV=args.delta0_eV)
    h = normal_state_hamiltonian(args.kx, args.ky)
    normal_energies = np.linalg.eigvalsh(h)
    delta_spm = spm_pairing_matrix(args.kx, args.ky, amp)
    delta_dwave = dwave_pairing_matrix(args.kx, args.ky, amp)
    bdg_spm_energies = np.linalg.eigvalsh(bdg_hamiltonian(args.kx, args.ky, delta_spm))
    bdg_dwave_energies = np.linalg.eigvalsh(bdg_hamiltonian(args.kx, args.ky, delta_dwave))
    bdg_zero_energies = np.linalg.eigvalsh(
        bdg_hamiltonian(args.kx, args.ky, np.zeros((4, 4), dtype=complex))
    )

    np.set_printoptions(precision=6, suppress=True)
    print("basis = (dz1, dx1, dz2, dx2)")
    print(f"k = ({args.kx}, {args.ky})")
    print(f"delta0_eV = {args.delta0_eV}")
    print_matrix("Delta_s_pm", delta_spm)
    print_matrix("Delta_dwave", delta_dwave)
    print("normal-state eigenvalues [eV] =")
    print(normal_energies)
    print("BdG zero-pairing eigenvalues [eV] =")
    print(bdg_zero_energies)
    print("BdG s_pm eigenvalues [eV] =")
    print(bdg_spm_energies)
    print("BdG d_wave eigenvalues [eV] =")
    print(bdg_dwave_energies)
    print(f"s_pm particle-hole residual = {particle_hole_residual(bdg_spm_energies):.6e} eV")
    print(f"d_wave particle-hole residual = {particle_hole_residual(bdg_dwave_energies):.6e} eV")


if __name__ == "__main__":
    main()
