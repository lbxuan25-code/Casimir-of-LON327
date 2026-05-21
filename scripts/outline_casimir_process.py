#!/usr/bin/env python3
"""Evaluate one Casimir integrand point from supplied toy conductivities."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lno327 import CasimirSetup, ConductivityTensor, casimir_energy_integrand, casimir_torque_integrand
from lno327.casimir import matsubara_frequency


def main() -> None:
    setup = CasimirSetup(temperature=30.0, distance=30e-9, area=1.0)
    xi = matsubara_frequency(1, setup.temperature)
    left = ConductivityTensor(xx=1.1e-4, yy=0.9e-4)
    right = ConductivityTensor(xx=1.1e-4, yy=0.9e-4)

    energy = casimir_energy_integrand(setup, xi, k_parallel=1e6, phi=0.3, theta=0.4, left=left, right=right)
    torque = casimir_torque_integrand(setup, xi, k_parallel=1e6, phi=0.3, theta=0.4, left=left, right=right)
    print("This is a single integrand-point smoke check, not a simulation.")
    print(f"energy_integrand = {energy}")
    print(f"torque_integrand = {torque}")


if __name__ == "__main__":
    main()
