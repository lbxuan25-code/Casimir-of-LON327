"""Band-structure plotting helpers."""

from __future__ import annotations

import os

import numpy as np

from lno327.plotting.io import save_figure


def plot_band_structure(
    k_distance,
    energies,
    label_positions,
    label_names,
    output_path,
    *,
    title: str,
    ylabel: str = "Energy (eV)",
    fermi_level: float = 0.0,
    metadata: dict | None = None,
) -> None:
    distances = np.asarray(k_distance, dtype=float)
    values = np.asarray(energies, dtype=float)
    if values.ndim != 2:
        raise ValueError("energies must have shape (n_k, n_bands)")
    if distances.shape != (values.shape[0],):
        raise ValueError("k_distance must have shape (n_k,)")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for band_index in range(values.shape[1]):
        ax.plot(distances, values[:, band_index], linewidth=1.1)
    ax.axhline(fermi_level, color="0.25", linewidth=0.8, linestyle="--")
    for position in label_positions:
        ax.axvline(float(position), color="0.75", linewidth=0.7)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(float(distances[0]), float(distances[-1]))
    ax.grid(alpha=0.2, linewidth=0.5)
    save_figure(fig, output_path, metadata)
    plt.close(fig)
