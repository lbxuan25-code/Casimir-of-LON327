"""Fermi-surface contour plotting helpers."""

from __future__ import annotations

import os
import warnings

import numpy as np

from lno327.plotting.io import save_figure


def plot_fermi_surface(
    kx_grid,
    ky_grid,
    band_energies,
    output_path,
    *,
    title: str,
    fermi_level: float = 0.0,
    metadata: dict | None = None,
) -> None:
    kx = np.asarray(kx_grid, dtype=float)
    ky = np.asarray(ky_grid, dtype=float)
    values = np.asarray(band_energies, dtype=float)
    if kx.shape != ky.shape:
        raise ValueError("kx_grid and ky_grid must have matching shapes")
    if values.ndim != 3 or values.shape[1:] != kx.shape:
        raise ValueError("band_energies must have shape (n_bands, nk, nk)")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    for band_index in range(values.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax.contour(kx, ky, values[band_index], levels=[fermi_level], linewidths=1.0)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, output_path, metadata)
    plt.close(fig)
