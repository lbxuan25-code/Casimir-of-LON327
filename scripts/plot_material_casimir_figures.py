#!/usr/bin/env python3
"""Plot saved finite-grid material Casimir candidate data without recomputing response."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "material_casimir"


def plot_material_casimir_figures(data_npz: Path, figures_dir: Path) -> dict[str, str]:
    with np.load(data_npz, allow_pickle=True) as data:
        pairings = [str(item) for item in data["pairings"]]
        distance_nm = np.asarray(data["distance_nm"], dtype=float)
        theta_deg = np.asarray(data["theta_deg"], dtype=float)
        energy = np.asarray(data["F_over_A_J_m2"], dtype=complex)
        delta = np.asarray(data["delta_F_over_A_J_m2"], dtype=complex)
        torque = np.asarray(data["tau_over_A_J_m2_rad"], dtype=float)

    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    paths: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for ip, pairing in enumerate(pairings):
        ax.plot(distance_nm, energy[ip, :, 0].real, marker="o", label=f"{pairing}, theta=0")
    ax.set_xlabel("distance (nm)")
    ax.set_ylabel("F/A (J/m^2)")
    ax.set_title("finite-grid material Casimir candidate")
    style_publication_axis(ax)
    path = figures_dir / "material_casimir_energy_vs_distance.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths["energy_vs_distance"] = str(path)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for ip, pairing in enumerate(pairings):
        for idist, distance in enumerate(distance_nm):
            ax.plot(theta_deg, delta[ip, idist].real, marker="o", label=f"{pairing}, d={distance:g} nm")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Delta F/A (J/m^2)")
    ax.set_title("angular energy variation")
    style_publication_axis(ax)
    path = figures_dir / "material_casimir_deltaF_vs_theta.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths["deltaF_vs_theta"] = str(path)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for ip, pairing in enumerate(pairings):
        for idist, distance in enumerate(distance_nm):
            ax.plot(theta_deg, torque[ip, idist], marker="o", label=f"{pairing}, d={distance:g} nm")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("tau/A (J/m^2/rad)")
    ax.set_title("finite-difference torque candidate")
    style_publication_axis(ax)
    path = figures_dir / "material_casimir_torque_vs_theta.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths["torque_vs_theta"] = str(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-npz", type=Path, default=DEFAULT_OUTPUT_DIR / "data" / "material_energy_torque_data.npz")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = plot_material_casimir_figures(args.data_npz, args.figures_dir)
    for label, path in paths.items():
        print(f"{label} = {path}")
    print("report = plot-only path; response was not recomputed")


if __name__ == "__main__":
    main()
