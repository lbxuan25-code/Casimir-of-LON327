#!/usr/bin/env python3
"""Plot the adopted normal-state band structure along Gamma-X-M-Gamma."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import normal_state_hamiltonian
from lno327.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis


def interpolate_path(points: list[tuple[str, tuple[float, float]]], samples_per_segment: int):
    k_points: list[tuple[float, float]] = []
    distances: list[float] = []
    tick_positions: list[float] = [0.0]
    tick_labels: list[str] = [points[0][0]]
    total_distance = 0.0

    for (_, start), (label, end) in zip(points[:-1], points[1:], strict=True):
        start_vec = np.array(start, dtype=float)
        end_vec = np.array(end, dtype=float)
        segment = end_vec - start_vec
        segment_length = float(np.linalg.norm(segment))
        endpoint = samples_per_segment + 1
        for i in range(endpoint):
            if k_points and i == 0:
                continue
            frac = i / samples_per_segment
            k_vec = start_vec + frac * segment
            k_points.append((float(k_vec[0]), float(k_vec[1])))
            distances.append(total_distance + frac * segment_length)
        total_distance += segment_length
        tick_positions.append(total_distance)
        tick_labels.append(label)

    return np.array(k_points), np.array(distances), tick_positions, tick_labels


def compute_bands(k_points: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.eigvalsh(normal_state_hamiltonian(kx, ky)) for kx, ky in k_points])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-segment", type=int, default=160)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "smoke" / "figures" / "normal_state_bands.png",
    )
    args = parser.parse_args()

    if args.samples_per_segment < 2:
        raise ValueError("--samples-per-segment must be at least 2")

    high_symmetry_path = [
        ("Gamma", (0.0, 0.0)),
        ("X", (np.pi, 0.0)),
        ("M", (np.pi, np.pi)),
        ("Gamma", (0.0, 0.0)),
    ]
    k_points, distances, tick_positions, tick_labels = interpolate_path(
        high_symmetry_path, args.samples_per_segment
    )
    bands = compute_bands(k_points)

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    for band_index in range(bands.shape[1]):
        ax.plot(distances, bands[:, band_index], color="black", linewidth=1.2)

    for position in tick_positions:
        ax.axvline(position, color="0.82", linewidth=0.8)
    ax.axhline(0.0, color="tab:red", linewidth=0.9, linestyle="--")
    ax.set_xlim(distances[0], distances[-1])
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Normal-State Band Structure")
    style_publication_axis(ax, legend=False)
    save_publication_figure(fig, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
