#!/usr/bin/env python3
"""Generate model-level sanity plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lno327.models.registry import available_models, build_model_spec, get_observables_module  # noqa: E402
from lno327.plotting import (  # noqa: E402
    plot_band_structure,
    plot_bdg_min_gap,
    plot_fermi_surface,
    plot_gap_texture,
)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _high_symmetry_path(points_per_segment: int = 80) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], tuple[str, ...]]:
    nodes = (
        (r"$\Gamma$", np.array([0.0, 0.0])),
        ("X", np.array([np.pi, 0.0])),
        ("M", np.array([np.pi, np.pi])),
        (r"$\Gamma$", np.array([0.0, 0.0])),
    )
    path: list[np.ndarray] = []
    distances: list[float] = []
    label_positions = [0.0]
    current_distance = 0.0
    for start_index, ((_, start), (_, end)) in enumerate(zip(nodes[:-1], nodes[1:], strict=True)):
        segment = end - start
        count = points_per_segment + 1
        for point_index in range(count):
            if start_index > 0 and point_index == 0:
                continue
            t = point_index / float(points_per_segment)
            point = start + t * segment
            if path:
                current_distance += float(np.linalg.norm(point - path[-1]))
            path.append(point)
            distances.append(current_distance)
        label_positions.append(current_distance)
    label_names = tuple(name for name, _ in nodes)
    return np.asarray(path), np.asarray(distances), tuple(label_positions), label_names


def _grid(nk: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.linspace(-np.pi, np.pi, nk)
    return np.meshgrid(values, values, indexing="ij")


def _base_metadata(spec, model_name: str, nk: int, channels: tuple[str, ...]) -> dict:
    metadata = spec.metadata()
    return {
        "model": model_name,
        "basis": list(metadata.basis),
        "channels": list(channels),
        "nk": nk,
    }


def _normal_band_grid(observables, spec, kx_grid: np.ndarray, ky_grid: np.ndarray) -> np.ndarray:
    first = observables.normal_band_energies(float(kx_grid[0, 0]), float(ky_grid[0, 0]), spec)
    output = np.empty((first.shape[0],) + kx_grid.shape, dtype=float)
    output[:, 0, 0] = first
    for index in np.ndindex(kx_grid.shape):
        if index == (0, 0):
            continue
        output[:, index[0], index[1]] = observables.normal_band_energies(
            float(kx_grid[index]),
            float(ky_grid[index]),
            spec,
        )
    return output


def _band_projected_gap_grid(observables, spec, channel: str, kx_grid: np.ndarray, ky_grid: np.ndarray) -> np.ndarray:
    first = observables.band_projected_gap(float(kx_grid[0, 0]), float(ky_grid[0, 0]), channel, spec)
    output = np.empty((first.shape[0],) + kx_grid.shape, dtype=complex)
    output[:, 0, 0] = first
    for index in np.ndindex(kx_grid.shape):
        if index == (0, 0):
            continue
        output[:, index[0], index[1]] = observables.band_projected_gap(
            float(kx_grid[index]),
            float(ky_grid[index]),
            channel,
            spec,
        )
    return output


def _bdg_min_gap_grid(observables, spec, channel: str, kx_grid: np.ndarray, ky_grid: np.ndarray) -> np.ndarray:
    output = np.empty(kx_grid.shape, dtype=float)
    for index in np.ndindex(kx_grid.shape):
        output[index] = observables.min_positive_bdg_energy(
            float(kx_grid[index]),
            float(ky_grid[index]),
            channel,
            spec,
        )
    return output


def generate_plots(
    *,
    model_name: str,
    nk: int,
    channels: tuple[str, ...],
    plots: tuple[str, ...],
    output_root: Path,
) -> None:
    spec = build_model_spec(model_name)
    observables = get_observables_module(model_name)
    all_channels = tuple(channel.name for channel in spec.channels())
    active_channels = tuple(channel for channel in all_channels if channel != "normal") if channels == ("all",) else channels
    output_dir = output_root / model_name
    metadata = _base_metadata(spec, model_name, nk, active_channels)

    if "band" in plots:
        path, distance, label_positions, label_names = _high_symmetry_path()
        energies = observables.band_energies_on_path(spec, path)
        plot_band_structure(
            distance,
            energies,
            label_positions,
            label_names,
            output_dir / "band_structure" / "normal_bands.png",
            title=f"{model_name} normal bands",
            metadata={**metadata, "plot": "band_structure"},
        )

    needs_grid = any(item in plots for item in ("fermi", "gap", "bdg-gap"))
    if not needs_grid:
        return
    kx_grid, ky_grid = _grid(nk)
    band_grid = _normal_band_grid(observables, spec, kx_grid, ky_grid)

    if "fermi" in plots:
        plot_fermi_surface(
            kx_grid,
            ky_grid,
            band_grid,
            output_dir / "fermi_surface" / "normal_fermi_surface.png",
            title=f"{model_name} normal-state Fermi contours",
            metadata={**metadata, "plot": "fermi_surface"},
        )

    if "gap" in plots:
        for channel in active_channels:
            projected = _band_projected_gap_grid(observables, spec, channel, kx_grid, ky_grid)
            for band_index in range(projected.shape[0]):
                plot_gap_texture(
                    kx_grid,
                    ky_grid,
                    np.real(projected[band_index]),
                    output_dir / "gap_texture" / f"{channel}_band_{band_index}.png",
                    title=f"{model_name} {channel} band {band_index} projected gap",
                    fermi_contours=band_grid,
                    metadata={**metadata, "plot": "gap_texture", "channel": channel, "band_index": band_index},
                )

    if "bdg-gap" in plots:
        for channel in active_channels:
            min_gap = _bdg_min_gap_grid(observables, spec, channel, kx_grid, ky_grid)
            plot_bdg_min_gap(
                kx_grid,
                ky_grid,
                min_gap,
                output_dir / "bdg_min_gap" / f"{channel}.png",
                title=f"{model_name} {channel} minimum BdG gap",
                metadata={**metadata, "plot": "bdg_min_gap", "channel": channel},
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="symmetry_bdg_2band", choices=available_models())
    parser.add_argument("--nk", type=int, default=151)
    parser.add_argument("--channels", default="all")
    parser.add_argument("--plots", default="band,fermi,gap,bdg-gap")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/models"))
    args = parser.parse_args(argv)

    if args.nk <= 1:
        raise ValueError("--nk must be greater than 1")
    plots = _parse_csv(args.plots)
    allowed_plots = {"band", "fermi", "gap", "bdg-gap"}
    unknown_plots = set(plots) - allowed_plots
    if unknown_plots:
        raise ValueError(f"unknown plot names: {sorted(unknown_plots)}")

    generate_plots(
        model_name=args.model,
        nk=args.nk,
        channels=_parse_csv(args.channels),
        plots=plots,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
