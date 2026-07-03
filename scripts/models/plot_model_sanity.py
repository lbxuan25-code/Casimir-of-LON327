#!/usr/bin/env python3
"""Generate model-level sanity plots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lno327.models.registry import available_models, build_model_spec, get_observables_module  # noqa: E402
from lno327.plotting import (  # noqa: E402
    plot_band_structure,
    plot_fermi_surface,
    plot_fermi_surface_gap,
    write_metadata_json,
)

OLD_PLOT_MESSAGE = (
    "full-BZ gap texture and BdG min-gap plots are no longer main model sanity outputs; "
    "use fermi-gap for Fermi-pocket order parameter visualization."
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


def _band_projected_gap_grid(
    observables,
    spec,
    channel: str,
    kx_grid: np.ndarray,
    ky_grid: np.ndarray,
    *,
    gauge: str,
) -> np.ndarray:
    first = observables.band_projected_gap(float(kx_grid[0, 0]), float(ky_grid[0, 0]), channel, spec, gauge=gauge)
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
            gauge=gauge,
        )
    return output


def _gap_values_for_plot(projected_gap: np.ndarray, mode: str) -> np.ndarray:
    if mode == "real":
        return np.real(projected_gap)
    if mode == "abs":
        return np.abs(projected_gap)
    if mode == "phase":
        return np.angle(projected_gap)
    raise ValueError("gap value mode must be 'real', 'abs', or 'phase'")


def _validate_plots(plots: tuple[str, ...]) -> None:
    removed = {"gap", "bdg-gap"} & set(plots)
    if removed:
        raise ValueError(OLD_PLOT_MESSAGE)
    allowed_plots = {"band", "fermi", "fermi-gap"}
    unknown_plots = set(plots) - allowed_plots
    if unknown_plots:
        raise ValueError(f"unknown plot names: {sorted(unknown_plots)}")


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "band_index",
        "pocket_index",
        "num_points",
        "gap_min",
        "gap_max",
        "gap_mean",
        "gap_abs_min",
        "gap_abs_max",
        "has_sign_change",
        "positive_fraction",
        "negative_fraction",
        "near_zero_fraction",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _active_channels(model_name: str, requested_channels: tuple[str, ...], all_channels: tuple[str, ...]) -> tuple[str, ...]:
    available = tuple(channel for channel in all_channels if channel != "normal")
    if requested_channels == ("all",):
        return available
    unknown = sorted(set(requested_channels) - set(available))
    if unknown:
        raise ValueError(
            f"unknown channels for {model_name}: {unknown}; "
            f"available non-normal channels: {list(available)}"
        )
    return requested_channels


def generate_plots(
    *,
    model_name: str,
    nk: int,
    channels: tuple[str, ...],
    plots: tuple[str, ...],
    output_root: Path,
    path_points: int = 80,
    gap_projection_gauge: str = "anchor",
    gap_value_mode: str = "real",
) -> None:
    if nk <= 1:
        raise ValueError("nk must be greater than 1")
    if path_points <= 1:
        raise ValueError("path_points must be greater than 1")
    _validate_plots(plots)
    spec = build_model_spec(model_name)
    observables = get_observables_module(model_name)
    all_channels = tuple(channel.name for channel in spec.channels())
    active_channels = _active_channels(model_name, channels, all_channels)
    output_dir = output_root / model_name
    metadata = _base_metadata(spec, model_name, nk, active_channels)

    if "band" in plots:
        path, distance, label_positions, label_names = _high_symmetry_path(points_per_segment=path_points)
        energies = observables.band_energies_on_path(spec, path)
        plot_band_structure(
            distance,
            energies,
            label_positions,
            label_names,
            output_dir / "band_structure" / "normal_bands.png",
            title=f"{model_name} normal bands",
            metadata={
                **metadata,
                "plot": "band_structure",
                "k_path": "Gamma-X-M-Gamma",
                "path_points_per_segment": path_points,
                "path_num_points": int(path.shape[0]),
            },
        )

    needs_grid = any(item in plots for item in ("fermi", "fermi-gap"))
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

    if "fermi-gap" in plots:
        for channel in active_channels:
            projected = _band_projected_gap_grid(
                observables,
                spec,
                channel,
                kx_grid,
                ky_grid,
                gauge=gap_projection_gauge,
            )
            values = _gap_values_for_plot(projected, gap_value_mode)
            fermi_gap_metadata = {
                **metadata,
                "plot": "fermi_gap",
                "domain": "fermi_surface",
                "fermi_level": 0.0,
                "channel": channel,
                "gap_projection_gauge": gap_projection_gauge,
                "gap_value_mode": gap_value_mode,
                "projected_gap_sanity_quantity": True,
            }
            summary = plot_fermi_surface_gap(
                kx_grid,
                ky_grid,
                band_grid,
                values,
                output_dir / "fermi_gap" / f"{channel}_{gap_value_mode}.png",
                title=f"{model_name} {channel} projected gap on Fermi pockets ({gap_value_mode})",
                metadata=fermi_gap_metadata,
            )
            _write_summary_csv(output_dir / "fermi_gap" / f"{channel}_{gap_value_mode}_summary.csv", summary)
            write_metadata_json(
                output_dir / "fermi_gap" / f"{channel}_{gap_value_mode}_summary.json",
                {
                    **fermi_gap_metadata,
                    "plot": "fermi_gap_summary",
                    "summary": summary,
                },
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="symmetry_bdg_2band", choices=available_models())
    parser.add_argument("--nk", type=int, default=151)
    parser.add_argument("--channels", default="all")
    parser.add_argument("--plots", default="band,fermi,fermi-gap")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/models/sanity"))
    parser.add_argument("--path-points", type=int, default=80)
    parser.add_argument("--gap-projection-gauge", choices=("anchor", "raw"), default="anchor")
    parser.add_argument("--gap-value-mode", choices=("real", "abs", "phase"), default="real")
    args = parser.parse_args(argv)

    if args.nk <= 1:
        raise ValueError("--nk must be greater than 1")
    if args.path_points <= 1:
        raise ValueError("--path-points must be greater than 1")
    plots = _parse_csv(args.plots)
    _validate_plots(plots)

    generate_plots(
        model_name=args.model,
        nk=args.nk,
        channels=_parse_csv(args.channels),
        plots=plots,
        output_root=args.output_root,
        path_points=args.path_points,
        gap_projection_gauge=args.gap_projection_gauge,
        gap_value_mode=args.gap_value_mode,
    )


if __name__ == "__main__":
    main()
