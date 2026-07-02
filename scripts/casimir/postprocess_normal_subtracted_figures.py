#!/usr/bin/env python3
"""Generate normal-subtracted smoke-pilot Casimir summary figures."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


REQUIRED_PAIRINGS = ("normal", "spm", "dwave")
EXCESS_PAIRINGS = ("spm", "dwave")


def _read_energy_rows(run_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        run_dir / "data" / "energy_distance_angle_grid.csv",
        run_dir / "data" / "energy_vs_angle.csv",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No energy CSV found under {run_dir / 'data'}; expected energy_distance_angle_grid.csv or energy_vs_angle.csv"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"pairing", "distance_nm", "theta_deg", "energy_per_area_J_m2"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    return [
        {
            "pairing": str(row["pairing"]),
            "distance_nm": float(row["distance_nm"]),
            "theta_deg": float(row["theta_deg"]),
            "energy_per_area_J_m2": float(row["energy_per_area_J_m2"]),
        }
        for row in rows
    ]


def _build_energy_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, float, float], float]:
    pairings = {str(row["pairing"]) for row in rows}
    missing = sorted(set(REQUIRED_PAIRINGS) - pairings)
    if missing:
        raise ValueError(f"Missing required pairing energy data: {missing}; need normal, spm, and dwave")
    return {
        (str(row["pairing"]), float(row["distance_nm"]), float(row["theta_deg"])): float(row["energy_per_area_J_m2"])
        for row in rows
    }


def _common_grid(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    by_pairing = {
        pairing: {
            (float(row["distance_nm"]), float(row["theta_deg"]))
            for row in rows
            if str(row["pairing"]) == pairing
        }
        for pairing in REQUIRED_PAIRINGS
    }
    common = set.intersection(*(set(values) for values in by_pairing.values()))
    if not common:
        raise ValueError("No common distance/theta grid across normal, spm, and dwave")
    distances = sorted({distance for distance, _theta in common})
    angles = sorted({theta for _distance, theta in common})
    return distances, angles


def _nearest_distance(distances: list[float], requested: float | None) -> float:
    target = 50.0 if requested is None else float(requested)
    return min(distances, key=lambda item: abs(float(item) - target))


def _excess_energy_rows(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    lookup = _build_energy_lookup(rows)
    distances, angles = _common_grid(rows)
    excess_rows: list[dict[str, float | str]] = []
    for pairing in EXCESS_PAIRINGS:
        for distance in distances:
            raw_values = []
            for theta in angles:
                energy = lookup[(pairing, distance, theta)]
                normal = lookup[("normal", distance, theta)]
                raw_values.append(energy - normal)
            mean_excess = float(np.mean(raw_values))
            for theta, raw_excess in zip(angles, raw_values, strict=True):
                excess_rows.append(
                    {
                        "pairing": pairing,
                        "distance_nm": distance,
                        "theta_deg": theta,
                        "raw_excess_energy_per_area_J_m2": float(raw_excess),
                        "mean_raw_excess_energy_per_area_J_m2": mean_excess,
                        "anisotropic_excess_energy_per_area_J_m2": float(raw_excess - mean_excess),
                    }
                )
    return excess_rows


def _excess_torque_rows(excess_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    torque_rows: list[dict[str, float | str]] = []
    for pairing in EXCESS_PAIRINGS:
        distances = sorted({float(row["distance_nm"]) for row in excess_rows if row["pairing"] == pairing})
        for distance in distances:
            values = sorted(
                [
                    row
                    for row in excess_rows
                    if row["pairing"] == pairing and abs(float(row["distance_nm"]) - distance) < 1e-12
                ],
                key=lambda row: float(row["theta_deg"]),
            )
            if len(values) < 2:
                raise ValueError("Need at least two theta values to compute finite-difference torque")
            theta_rad = np.deg2rad([float(row["theta_deg"]) for row in values])
            anisotropic_excess = np.asarray(
                [float(row["anisotropic_excess_energy_per_area_J_m2"]) for row in values],
                dtype=float,
            )
            torque = -np.gradient(anisotropic_excess, theta_rad)
            for row, torque_value in zip(values, torque, strict=True):
                torque_rows.append(
                    {
                        "pairing": pairing,
                        "distance_nm": distance,
                        "theta_deg": float(row["theta_deg"]),
                        "excess_torque_per_area_J_m2_rad": float(torque_value),
                    }
                )
    return torque_rows


def _anisotropy_rows(
    excess_rows: list[dict[str, float | str]],
    torque_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for pairing in EXCESS_PAIRINGS:
        distances = sorted({float(row["distance_nm"]) for row in excess_rows if row["pairing"] == pairing})
        for distance in distances:
            energies = [
                float(row["anisotropic_excess_energy_per_area_J_m2"])
                for row in excess_rows
                if row["pairing"] == pairing and abs(float(row["distance_nm"]) - distance) < 1e-12
            ]
            torques = [
                abs(float(row["excess_torque_per_area_J_m2_rad"]))
                for row in torque_rows
                if row["pairing"] == pairing and abs(float(row["distance_nm"]) - distance) < 1e-12
            ]
            rows.append(
                {
                    "pairing": pairing,
                    "distance_nm": distance,
                    "excess_energy_anisotropy_amplitude_J_m2": float(max(energies) - min(energies)),
                    "max_abs_excess_torque_per_area_J_m2_rad": float(max(torques)),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_figures(
    figure_dir: Path,
    excess_rows: list[dict[str, float | str]],
    torque_rows: list[dict[str, float | str]],
    anisotropy_rows: list[dict[str, float | str]],
    reference_distance_nm: float,
    *,
    plot_raw_excess_energy: bool,
    plot_torque_amplitude: bool,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    plt.figure(figsize=(6, 4))
    plotted_values: list[float] = []
    for pairing in EXCESS_PAIRINGS:
        rows = sorted(
            [
                row
                for row in excess_rows
                if row["pairing"] == pairing and abs(float(row["distance_nm"]) - reference_distance_nm) < 1e-12
            ],
            key=lambda row: float(row["theta_deg"]),
        )
        values = [float(row["anisotropic_excess_energy_per_area_J_m2"]) for row in rows]
        plotted_values.extend(values)
        plt.plot(
            [float(row["theta_deg"]) for row in rows],
            values,
            marker="o",
            label=pairing,
        )
    max_abs = max((abs(value) for value in plotted_values), default=1.0)
    ylim = max_abs * 1.15 if max_abs > 0.0 else 1.0
    plt.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")
    plt.ylim(-ylim, ylim)
    plt.xlabel("angle theta (deg)")
    plt.ylabel("δΔE = ΔE - <ΔE>_θ (J/m^2)")
    plt.title(f"Angular excess energy, d = {reference_distance_nm:g} nm")
    plt.legend(frameon=False)
    plt.tight_layout()
    path = figure_dir / "excess_anisotropic_energy_vs_angle.png"
    plt.savefig(path, dpi=180)
    plt.close()
    outputs.append(path)

    plt.figure(figsize=(6, 4))
    for pairing in EXCESS_PAIRINGS:
        rows = sorted(
            [
                row
                for row in torque_rows
                if row["pairing"] == pairing and abs(float(row["distance_nm"]) - reference_distance_nm) < 1e-12
            ],
            key=lambda row: float(row["theta_deg"]),
        )
        plt.plot(
            [float(row["theta_deg"]) for row in rows],
            [float(row["excess_torque_per_area_J_m2_rad"]) for row in rows],
            marker="o",
            label=pairing,
        )
    plt.xlabel("angle theta (deg)")
    plt.ylabel("Δτ = -∂θδΔE (J/(m^2 rad))")
    plt.title(f"Angular excess torque, d = {reference_distance_nm:g} nm")
    plt.legend(frameon=False)
    plt.tight_layout()
    path = figure_dir / "excess_torque_vs_angle.png"
    plt.savefig(path, dpi=180)
    plt.close()
    outputs.append(path)

    plt.figure(figsize=(6, 4))
    for pairing in EXCESS_PAIRINGS:
        rows = sorted([row for row in anisotropy_rows if row["pairing"] == pairing], key=lambda row: float(row["distance_nm"]))
        distances = [float(row["distance_nm"]) for row in rows]
        plt.plot(
            distances,
            [float(row["excess_energy_anisotropy_amplitude_J_m2"]) for row in rows],
            marker="o",
            label=pairing,
        )
    plt.xlabel("distance (nm)")
    plt.ylabel("A_E = max δΔE - min δΔE (J/m^2)")
    plt.title("Angular excess energy amplitude vs distance")
    plt.legend(frameon=False)
    plt.tight_layout()
    path = figure_dir / "excess_anisotropy_amplitude_vs_distance.png"
    plt.savefig(path, dpi=180)
    plt.close()
    outputs.append(path)
    if plot_torque_amplitude:
        extra_dir = figure_dir / "extra"
        extra_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        for pairing in EXCESS_PAIRINGS:
            rows = sorted([row for row in anisotropy_rows if row["pairing"] == pairing], key=lambda row: float(row["distance_nm"]))
            plt.plot(
                [float(row["distance_nm"]) for row in rows],
                [float(row["max_abs_excess_torque_per_area_J_m2_rad"]) for row in rows],
                marker="o",
                label=pairing,
            )
        plt.xlabel("distance (nm)")
        plt.ylabel("A_tau = max |Δτ| (J/(m^2 rad))")
        plt.title("Angular excess torque amplitude vs distance")
        plt.legend(frameon=False)
        plt.tight_layout()
        path = extra_dir / "excess_torque_amplitude_vs_distance.png"
        plt.savefig(path, dpi=180)
        plt.close()
        outputs.append(path)
    if plot_raw_excess_energy:
        extra_dir = figure_dir / "extra"
        extra_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        for pairing in EXCESS_PAIRINGS:
            rows = sorted(
                [
                    row
                    for row in excess_rows
                    if row["pairing"] == pairing and abs(float(row["distance_nm"]) - reference_distance_nm) < 1e-12
                ],
                key=lambda row: float(row["theta_deg"]),
            )
            plt.plot(
                [float(row["theta_deg"]) for row in rows],
                [float(row["raw_excess_energy_per_area_J_m2"]) for row in rows],
                marker="o",
                label=pairing,
            )
        plt.xlabel("angle theta (deg)")
        plt.ylabel("raw ΔE = E_pairing - E_normal (J/m^2)")
        plt.title(f"Raw normal-subtracted excess energy, d = {reference_distance_nm:g} nm\nextra diagnostic plot; offsets may be angle-independent")
        plt.legend()
        plt.tight_layout()
        path = extra_dir / "raw_excess_energy_vs_angle.png"
        plt.savefig(path, dpi=180)
        plt.close()
        outputs.append(path)
    return outputs


def _write_readme(figure_dir: Path, reference_distance_nm: float) -> Path:
    path = figure_dir / "README.md"
    path.write_text(
        "\n".join(
            [
                "# normal-subtracted smoke-pilot figures",
                "",
                "These figures are pure post-processing outputs derived from the existing runtime CSV files.",
                "",
                "- raw Delta E = E_pairing - E_normal may contain an angle-independent energy offset",
                "- angle-independent offsets do not produce torque",
                "- plotted energy is δΔE = ΔE - <ΔE>_θ",
                "- torque is Δτ = -∂θδΔE",
                "- subtracting <ΔE>_θ leaves torque and max-min anisotropy amplitudes unchanged relative to raw ΔE",
                "- default distance summary only shows A_E(d), because it is the more direct and stable angular anisotropy measure",
                "- torque is a finite-difference derivative and can amplify small smoke-grid residuals",
                "- A_tau(d) is available only as an optional extra diagnostic with `--plot-torque-amplitude`",
                "- theta derivatives use radians",
                "- torque is computed by applying numpy.gradient(theta_rad) to δΔE after subtracting the angular mean",
                "- endpoint values use raw finite-difference one-sided gradients and are not strict physical endpoint torque claims",
                "- smoke-pilot diagnostic only",
                "- valid_for_formal_casimir_claim = false",
                f"- reference distance = {reference_distance_nm:g} nm",
                "",
                "Generated figures:",
                "",
                "- `excess_anisotropic_energy_vs_angle.png`",
                "- `excess_torque_vs_angle.png`",
                "- `excess_anisotropy_amplitude_vs_distance.png`",
                "",
                "Optional extra raw ΔE figure is generated only with `--plot-raw-excess-energy`:",
                "",
                "- `extra/raw_excess_energy_vs_angle.png`",
                "",
                "Optional extra torque amplitude figure is generated only with `--plot-torque-amplitude`:",
                "",
                "- `extra/excess_torque_amplitude_vs_distance.png`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run(
    run_dir: Path,
    reference_distance_nm: float | None,
    write_derived_data: bool,
    plot_raw_excess_energy: bool,
    plot_torque_amplitude: bool,
) -> dict[str, Any]:
    rows = _read_energy_rows(run_dir)
    distances, _angles = _common_grid(rows)
    reference = _nearest_distance(distances, reference_distance_nm)
    excess = _excess_energy_rows(rows)
    torque = _excess_torque_rows(excess)
    anisotropy = _anisotropy_rows(excess, torque)
    figure_dir = run_dir / "figures" / "normal_subtracted"
    figures = _plot_figures(
        figure_dir,
        excess,
        torque,
        anisotropy,
        reference,
        plot_raw_excess_energy=plot_raw_excess_energy,
        plot_torque_amplitude=plot_torque_amplitude,
    )
    readme = _write_readme(figure_dir, reference)

    data_files: list[Path] = []
    if write_derived_data:
        data_dir = run_dir / "data" / "normal_subtracted"
        _write_csv(
            data_dir / "excess_energy_grid.csv",
            excess,
            [
                "pairing",
                "distance_nm",
                "theta_deg",
                "raw_excess_energy_per_area_J_m2",
                "mean_raw_excess_energy_per_area_J_m2",
                "anisotropic_excess_energy_per_area_J_m2",
            ],
        )
        _write_csv(
            data_dir / "excess_torque_grid.csv",
            torque,
            ["pairing", "distance_nm", "theta_deg", "excess_torque_per_area_J_m2_rad"],
        )
        _write_csv(
            data_dir / "excess_anisotropy_amplitude_vs_distance.csv",
            anisotropy,
            ["pairing", "distance_nm", "excess_energy_anisotropy_amplitude_J_m2", "max_abs_excess_torque_per_area_J_m2_rad"],
        )
        data_files = sorted(data_dir.glob("*.csv"))

    return {
        "run_dir": str(run_dir),
        "reference_distance_nm": reference,
        "figures": [str(path) for path in figures],
        "readme": str(readme),
        "derived_data_files": [str(path) for path in data_files],
        "diagnostic_only": True,
        "valid_for_formal_casimir_claim": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate normal-subtracted finite-q BdG Casimir smoke-pilot figures.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-distance-nm", type=float)
    parser.add_argument("--write-derived-data", action="store_true")
    parser.add_argument("--plot-raw-excess-energy", action="store_true")
    parser.add_argument("--plot-torque-amplitude", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        args.run_dir,
        args.reference_distance_nm,
        bool(args.write_derived_data),
        bool(args.plot_raw_excess_energy),
        bool(args.plot_torque_amplitude),
    )
    for path in result["figures"]:
        print(path)
    print(result["readme"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
