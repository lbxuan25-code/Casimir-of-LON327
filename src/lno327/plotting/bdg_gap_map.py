"""BdG minimum-gap plotting helpers."""

from __future__ import annotations

import os

import numpy as np

from lno327.plotting.io import save_figure


def plot_bdg_min_gap(
    kx_grid,
    ky_grid,
    min_gap,
    output_path,
    *,
    title: str,
    metadata: dict | None = None,
) -> None:
    kx = np.asarray(kx_grid, dtype=float)
    ky = np.asarray(ky_grid, dtype=float)
    values = np.asarray(min_gap, dtype=float)
    if kx.shape != ky.shape or values.shape != kx.shape:
        raise ValueError("kx_grid, ky_grid, and min_gap must have matching 2D shapes")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    mesh = ax.pcolormesh(kx, ky, values, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="min positive BdG energy (eV)")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, output_path, metadata)
    plt.close(fig)
