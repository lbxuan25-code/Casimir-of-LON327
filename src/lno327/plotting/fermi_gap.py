"""Fermi-pocket projected-gap plotting helpers."""

from __future__ import annotations

import os
import warnings

import numpy as np

from lno327.plotting.io import save_figure

GAP_ZERO_TOL = 1e-10


def _nearest_gap_values(
    kx_grid: np.ndarray,
    ky_grid: np.ndarray,
    gap_grid: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    kx_axis = np.asarray(kx_grid[:, 0], dtype=float)
    ky_axis = np.asarray(ky_grid[0, :], dtype=float)
    ix = np.searchsorted(kx_axis, points[:, 0])
    iy = np.searchsorted(ky_axis, points[:, 1])
    ix = np.clip(ix, 1, kx_axis.size - 1)
    iy = np.clip(iy, 1, ky_axis.size - 1)
    ix -= np.abs(points[:, 0] - kx_axis[ix - 1]) <= np.abs(points[:, 0] - kx_axis[ix])
    iy -= np.abs(points[:, 1] - ky_axis[iy - 1]) <= np.abs(points[:, 1] - ky_axis[iy])
    return gap_grid[ix, iy]


def _summary_row(band_index: int, pocket_index: int, values: np.ndarray) -> dict:
    positive = values > GAP_ZERO_TOL
    negative = values < -GAP_ZERO_TOL
    near_zero = np.abs(values) < GAP_ZERO_TOL
    return {
        "band_index": int(band_index),
        "pocket_index": int(pocket_index),
        "num_points": int(values.size),
        "gap_min": float(np.min(values)),
        "gap_max": float(np.max(values)),
        "gap_mean": float(np.mean(values)),
        "gap_abs_min": float(np.min(np.abs(values))),
        "gap_abs_max": float(np.max(np.abs(values))),
        "has_sign_change": bool(np.any(positive) and np.any(negative)),
        "positive_fraction": float(np.mean(positive)),
        "negative_fraction": float(np.mean(negative)),
        "near_zero_fraction": float(np.mean(near_zero)),
    }


def plot_fermi_surface_gap(
    kx_grid,
    ky_grid,
    band_energies,
    band_gap_values,
    output_path,
    *,
    title: str,
    fermi_level: float = 0.0,
    metadata: dict | None = None,
) -> list[dict]:
    """Plot signed projected gap only on normal-state Fermi contours."""

    kx = np.asarray(kx_grid, dtype=float)
    ky = np.asarray(ky_grid, dtype=float)
    energies = np.asarray(band_energies, dtype=float)
    gaps = np.asarray(band_gap_values, dtype=float)
    if kx.shape != ky.shape:
        raise ValueError("kx_grid and ky_grid must have matching shapes")
    if energies.ndim != 3 or energies.shape[1:] != kx.shape:
        raise ValueError("band_energies must have shape (n_bands, nk, nk)")
    if gaps.shape != energies.shape:
        raise ValueError("band_gap_values must have shape (n_bands, nk, nk)")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    summaries: list[dict] = []
    collections: list[LineCollection] = []

    for band_index in range(energies.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            contour = ax.contour(kx, ky, energies[band_index], levels=[fermi_level], colors="none")
        segments = contour.allsegs[0] if contour.allsegs else []
        for collection in getattr(contour, "collections", ()):
            collection.remove()
        for pocket_index, vertices in enumerate(segments):
            if vertices.shape[0] < 2:
                continue
            point_values = _nearest_gap_values(kx, ky, gaps[band_index], vertices)
            summaries.append(_summary_row(band_index, pocket_index, point_values))
            line_segments = np.stack([vertices[:-1], vertices[1:]], axis=1)
            segment_values = 0.5 * (point_values[:-1] + point_values[1:])
            collection = LineCollection(line_segments, array=segment_values, linewidths=1.8, cmap="coolwarm")
            ax.add_collection(collection)
            collections.append(collection)

    scale = 1.0
    if summaries:
        scale = max(row["gap_abs_max"] for row in summaries)
        if scale == 0.0 or not np.isfinite(scale):
            scale = 1.0
    norm = Normalize(vmin=-scale, vmax=scale)
    for collection in collections:
        collection.set_norm(norm)
    mappable = collections[0] if collections else plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    fig.colorbar(mappable, ax=ax, label="projected gap")

    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(kx)), float(np.max(kx)))
    ax.set_ylim(float(np.min(ky)), float(np.max(ky)))
    save_figure(fig, output_path, metadata)
    plt.close(fig)
    return summaries
