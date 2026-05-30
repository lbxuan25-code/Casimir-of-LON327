#!/usr/bin/env python3
"""Inspect projected minimal pairing gaps near the normal-state Fermi surface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import PairingAmplitudes, gap_statistics_by_band, gap_statistics_on_fermi_surface  # noqa: E402
from lno327.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402


def default_output_prefix(kind: str) -> Path:
    return ROOT / "outputs" / "pairing" / "gap_structure" / "data" / f"gap_structure_{kind}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("spm", "dwave"), default="spm")
    parser.add_argument("--delta0", type=float, default=0.04, help="Pairing amplitude Delta0 in eV.")
    parser.add_argument("--nk", type=int, default=80)
    parser.add_argument("--energy-window", type=float, default=0.05, help="FS energy window in eV.")
    parser.add_argument("--node-tolerance", type=float, default=1e-3, help="Approximate node threshold in eV.")
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()

    if args.nk <= 0:
        raise ValueError("--nk must be positive")

    stats = gap_statistics_on_fermi_surface(
        args.kind,
        PairingAmplitudes(delta0_eV=args.delta0),
        nk=args.nk,
        energy_tolerance_eV=args.energy_window,
        node_tolerance_eV=args.node_tolerance,
    )
    band_summary = gap_statistics_by_band(stats)
    relative_node_mask = stats.gap_abs <= stats.node_tolerance_eV

    output_prefix = args.output_prefix or default_output_prefix(args.kind)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    np.savez(
        npz_path,
        kx=stats.kx,
        ky=stats.ky,
        band_index=stats.band_index,
        energy_eV=stats.energy_eV,
        gap_complex=stats.gap_complex,
        gap_abs=stats.gap_abs,
        gap_sign=stats.gap_sign,
        node_tolerance_eV=np.array(stats.node_tolerance_eV),
        relative_node_mask=relative_node_mask,
    )

    figures_dir = ROOT / "outputs" / "pairing" / "gap_structure" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / f"gap_structure_{args.kind}.png"

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    if len(stats.kx) > 0:
        max_gap = float(np.max(stats.gap_abs))
        sizes = 20.0 + 90.0 * stats.gap_abs / max_gap if max_gap > 0.0 else np.full_like(stats.gap_abs, 30.0)
        scatter = ax.scatter(
            stats.kx,
            stats.ky,
            c=stats.gap_sign,
            s=sizes,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            alpha=0.82,
            linewidths=0.0,
        )
        fig.colorbar(scatter, ax=ax, label="gap sign")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title(
        f"{args.kind} projected gap near normal-state FS\n"
        "sign is preliminary gauge-dependent diagnostic; marker size shows |gap|."
    )
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", adjustable="box")
    style_publication_axis(ax, legend=False)
    save_publication_figure(fig, figure_path)
    plt.close(fig)

    print(f"kind = {args.kind}")
    print(f"delta0_eV = {args.delta0}")
    print(f"nk = {args.nk} x {args.nk}")
    print(f"node_tolerance_eV = {stats.node_tolerance_eV}")
    print(f"number_of_fermi_surface_points = {len(stats.kx)}")
    print(f"min_abs_gap = {stats.min_abs_gap}")
    print(f"max_abs_gap = {stats.max_abs_gap}")
    print(f"mean_abs_gap = {stats.mean_abs_gap}")
    print(f"sign_change_detected = {stats.sign_changes}")
    print(f"approximate_node_count = {stats.approximate_nodes}")
    print(f"relative_node_fraction = {stats.relative_node_fraction}")
    print(f"band_resolved_summary = {band_summary}")
    print(f"output_npz = {npz_path}")
    print(f"figure = {figure_path}")


if __name__ == "__main__":
    main()
