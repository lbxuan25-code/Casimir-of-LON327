"""Gap-texture plotting helpers."""

from __future__ import annotations

import os
import warnings

import numpy as np

from lno327.plotting.io import save_figure


def plot_gap_texture(
    kx_grid,
    ky_grid,
    gap_values,
    output_path,
    *,
    title: str,
    fermi_contours=None,
    metadata: dict | None = None,
) -> None:
    kx = np.asarray(kx_grid, dtype=float)
    ky = np.asarray(ky_grid, dtype=float)
    values = np.asarray(gap_values, dtype=float)
    if kx.shape != ky.shape or values.shape != kx.shape:
        raise ValueError("kx_grid, ky_grid, and gap_values must have matching 2D shapes")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    scale = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    mesh = ax.pcolormesh(kx, ky, values, shading="auto", cmap="coolwarm", vmin=-scale, vmax=scale)
    fig.colorbar(mesh, ax=ax, label="Gap")
    if fermi_contours is not None:
        contours = np.asarray(fermi_contours, dtype=float)
        if contours.ndim == 3 and contours.shape[1:] == kx.shape:
            for band in contours:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ax.contour(kx, ky, band, levels=[0.0], colors="black", linewidths=0.6)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, output_path, metadata)
    plt.close(fig)
