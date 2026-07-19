from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math

from .config import DEFAULT_POSTPROCESS_ROOT, PROFILE_NAME


_FIGURE_NAMES = (
    "spm_free_energy.png",
    "spm_torque.png",
    "dwave_free_energy.png",
    "dwave_torque.png",
    "combined_free_energy.png",
    "combined_torque.png",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _physical_title(metadata: dict[str, Any]) -> str:
    temperature = metadata.get("temperature_K")
    separation = metadata.get("separation_nm")
    if isinstance(temperature, (int, float)) and isinstance(separation, (int, float)):
        return f"T={float(temperature):g} K, d={float(separation):g} nm"
    return "LNO327 Casimir diagnostic"


def _finite_rows(
    rows: list[dict[str, str]],
    *,
    pairing: str,
    kind: str,
) -> list[dict[str, str]]:
    if kind == "energy":
        selected = [
            row
            for row in rows
            if row.get("pairing") == pairing
            and row.get("usable_for_torque", "").lower() == "true"
            and row.get("energy_J_m2", "") not in ("", "None")
        ]
        keys = ("energy_J_m2", "energy_error_bound_J_m2")
    else:
        selected = [
            row
            for row in rows
            if row.get("pairing") == pairing and row.get("status") == "computed"
        ]
        keys = ("torque_per_area_N_m", "combined_diagnostic_uncertainty_N_m")
    return [
        row
        for row in selected
        if all(
            row.get(key, "") not in ("", "None")
            and math.isfinite(float(row[key]))
            for key in keys
        )
    ]


def _save_atomic(plt: Any, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    plt.savefig(temporary, dpi=220)
    temporary.replace(path)


def plot_results(
    *,
    output_root: Path = DEFAULT_POSTPROCESS_ROOT,
    profile: str = PROFILE_NAME,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = output_root / profile
    energy_rows = _read_csv(root / "free_energy.csv")
    torque_rows = _read_csv(root / "torque.csv")
    metadata = _read_metadata(root / "metadata.json")
    physical_title = _physical_title(metadata)
    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for filename in _FIGURE_NAMES:
        (figure_root / filename).unlink(missing_ok=True)
    outputs: list[Path] = []

    for pairing in ("spm", "dwave"):
        selected = _finite_rows(energy_rows, pairing=pairing, kind="energy")
        if selected:
            selected.sort(key=lambda row: float(row["angle_deg"]))
            angles = [float(row["angle_deg"]) for row in selected]
            energy = [float(row["energy_J_m2"]) for row in selected]
            errors = [float(row["energy_error_bound_J_m2"]) for row in selected]

            plt.figure(figsize=(8, 5))
            plt.errorbar(angles, energy, yerr=errors, marker="o", capsize=2)
            plt.xlabel("Relative angle (deg)")
            plt.ylabel("Free energy per area (J/m²)")
            plt.title(f"{pairing}: {physical_title}")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            path = figure_root / f"{pairing}_free_energy.png"
            _save_atomic(plt, path)
            plt.close()
            outputs.append(path)

        selected_torque = _finite_rows(torque_rows, pairing=pairing, kind="torque")
        if selected_torque:
            selected_torque.sort(key=lambda row: float(row["angle_deg"]))
            angles = [float(row["angle_deg"]) for row in selected_torque]
            torque = [float(row["torque_per_area_N_m"]) for row in selected_torque]
            errors = [
                float(row["combined_diagnostic_uncertainty_N_m"])
                for row in selected_torque
            ]

            plt.figure(figsize=(8, 5))
            plt.errorbar(angles, torque, yerr=errors, marker="o", capsize=2)
            plt.axhline(0.0, linewidth=1)
            plt.xlabel("Relative angle (deg)")
            plt.ylabel("Torque per area (N/m)")
            plt.title(f"{pairing}: {physical_title}")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            path = figure_root / f"{pairing}_torque.png"
            _save_atomic(plt, path)
            plt.close()
            outputs.append(path)

    for kind, rows, y_key, error_key, ylabel, filename in (
        (
            "energy",
            energy_rows,
            "energy_J_m2",
            "energy_error_bound_J_m2",
            "Free energy per area (J/m²)",
            "combined_free_energy.png",
        ),
        (
            "torque",
            torque_rows,
            "torque_per_area_N_m",
            "combined_diagnostic_uncertainty_N_m",
            "Torque per area (N/m)",
            "combined_torque.png",
        ),
    ):
        plt.figure(figsize=(8, 5))
        plotted = False
        for pairing in ("spm", "dwave"):
            selected = _finite_rows(rows, pairing=pairing, kind=kind)
            if not selected:
                continue
            selected.sort(key=lambda row: float(row["angle_deg"]))
            angles = [float(row["angle_deg"]) for row in selected]
            values = [float(row[y_key]) for row in selected]
            errors = [float(row[error_key]) for row in selected]
            plt.errorbar(
                angles,
                values,
                yerr=errors,
                marker="o",
                capsize=2,
                label=pairing,
            )
            plotted = True

        if plotted:
            if kind == "torque":
                plt.axhline(0.0, linewidth=1)
            plt.xlabel("Relative angle (deg)")
            plt.ylabel(ylabel)
            plt.title(f"SPM and d-wave comparison: {physical_title}")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            path = figure_root / filename
            _save_atomic(plt, path)
            outputs.append(path)
        plt.close()

    return outputs
