#!/usr/bin/env python3
"""Print Qiu normal-state and seed pairing blocks at one momentum."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lno327 import bdg_hamiltonian, dwave_pairing_matrix, qiu_bilayer_hamiltonian, spm_pairing_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kx", type=float, default=0.0)
    parser.add_argument("--ky", type=float, default=0.0)
    args = parser.parse_args()

    h = qiu_bilayer_hamiltonian(args.kx, args.ky)
    delta_spm = spm_pairing_matrix(args.kx, args.ky)
    delta_d = dwave_pairing_matrix(args.kx, args.ky)
    bdg = bdg_hamiltonian(args.kx, args.ky, delta_spm, h)

    np.set_printoptions(precision=6, suppress=True)
    print("basis = (dz1, dx1, dz2, dx2)")
    print("H_qiu(k) [eV] =")
    print(h)
    print("Delta_s_pm(k) =")
    print(delta_spm)
    print("Delta_d_wave(k) =")
    print(delta_d)
    print(f"BdG shape = {bdg.shape}")


if __name__ == "__main__":
    main()
