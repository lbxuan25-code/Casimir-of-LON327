from __future__ import annotations

from pathlib import Path
import csv

from .config import DEFAULT_POSTPROCESS_ROOT, PROFILE_NAME


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for pairing in ("spm", "dwave"):
        selected = [
            row
            for row in energy_rows
            if row["pairing"] == pairing
            and row["usable_for_torque"].lower() == "true"
            and row["energy_J_m2"] not in ("", "None")
        ]
        if selected:
            selected.sort(key=lambda row: float(row["angle_deg"]))
            angles = [float(row["angle_deg"]) for row in selected]
            energy = [float(row["energy_J_m2"]) for row in selected]
            errors = [float(row["energy_error_bound_J_m2"]) for row in selected]

            plt.figure(figsize=(8, 5))
            plt.errorbar(angles, energy, yerr=errors, marker="o", capsize=2)
            plt.xlabel("Relative angle (deg)")
            plt.ylabel("Free energy per area (J/m²)")
            plt.title(f"{pairing}: T=10 K, d=20 nm")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            path = figure_root / f"{pairing}_free_energy.png"
            plt.savefig(path, dpi=220)
            plt.close()
            outputs.append(path)

        selected_torque = [
            row
            for row in torque_rows
            if row["pairing"] == pairing and row["status"] == "computed"
        ]
        if selected_torque:
            selected_torque.sort(key=lambda row: float(row["angle_deg"]))
            angles = [float(row["angle_deg"]) for row in selected_torque]
            torque = [float(row["torque_per_area_N_per_m"]) for row in selected_torque]
            errors = [float(row["torque_error_bound_N_per_m"]) for row in selected_torque]

            plt.figure(figsize=(8, 5))
            plt.errorbar(angles, torque, yerr=errors, marker="o", capsize=2)
            plt.axhline(0.0, linewidth=1)
            plt.xlabel("Relative angle (deg)")
            plt.ylabel("Torque per area (N/m)")
            plt.title(f"{pairing}: T=10 K, d=20 nm")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            path = figure_root / f"{pairing}_torque.png"
            plt.savefig(path, dpi=220)
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
            "torque_per_area_N_per_m",
            "torque_error_bound_N_per_m",
            "Torque per area (N/m)",
            "combined_torque.png",
        ),
    ):
        plt.figure(figsize=(8, 5))
        plotted = False
        for pairing in ("spm", "dwave"):
            if kind == "energy":
                selected = [
                    row
                    for row in rows
                    if row["pairing"] == pairing
                    and row.get("usable_for_torque", "").lower() == "true"
                    and row.get(y_key, "") not in ("", "None")
                ]
            else:
                selected = [
                    row
                    for row in rows
                    if row["pairing"] == pairing
                    and row.get("status") == "computed"
                ]
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
            plt.title("SPM and d-wave comparison: T=10 K, d=20 nm")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            path = figure_root / filename
            plt.savefig(path, dpi=220)
            outputs.append(path)
        plt.close()

    return outputs
