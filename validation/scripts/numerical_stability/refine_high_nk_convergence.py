#!/usr/bin/env python3
"""Focused high-Nk convergence check for imaginary-axis local responses.

This refines the broad convergence benchmark around larger Brillouin-zone
meshes. It is a response-layer diagnostic only, not a Casimir calculation.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import (  # noqa: E402
    PairingAmplitudes,
    bosonic_matsubara_energy_eV,
    k_weights,
    local_response_imag_axis,
    matrix_symmetry_diagnostics,
    model_response_to_sheet_conductivity,
    sheet_conductivity_to_reflection_dimensionless,
    uniform_bz_mesh,
)
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("normal", "spm", "dwave")
DEFAULT_NK_LIST = (32, 48, 64, 80)
DEFAULT_ETA_LIST = (5e-4, 1e-4)
DEFAULT_MATSUBARA_LIST = (1, 2)
HIGH_NK_TOLERANCE = 0.02
SYMMETRY_TOLERANCE = 1e-8
RATIO_EPS = 1e-300

REQUIRED_NPZ_FIELDS = {
    "kind",
    "nk",
    "eta_eV",
    "matsubara_index",
    "omega_eV",
    "response_xx",
    "response_yy",
    "response_xy",
    "response_yx",
    "sheet_conductivity_xx",
    "reflection_dimensionless_xx",
    "delta",
    "relative_offdiag",
    "relative_eigen_split",
    "relative_change_vs_largest_nk",
    "relative_change_between_last_two_nk",
    "eta_relative_change",
    "spm_dwave_abs_diff_xx",
    "spm_dwave_rel_diff_xx",
    "high_nk_convergence_status",
    "pairing_difference_status",
    "diagnosis",
    "notes",
}


def _relative_change(value: complex, reference: complex) -> float:
    if abs(value) < RATIO_EPS and abs(reference) < RATIO_EPS:
        return 0.0
    return float(abs(value - reference) / (abs(reference) + RATIO_EPS))


def _evaluate_response(
    kind: str,
    nk: int,
    eta_eV: float,
    matsubara_index: int,
    temperature_K: float,
    delta0_eV: float,
) -> tuple[float, np.ndarray, complex, complex, dict[str, complex | float | bool]]:
    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    omega_eV = bosonic_matsubara_energy_eV(matsubara_index, temperature_K)
    response = local_response_imag_axis(
        kind,  # type: ignore[arg-type]
        omega_eV,
        mesh,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        pairing_params=PairingAmplitudes(delta0_eV=delta0_eV),
        k_weights=weights,
    )
    sheet = model_response_to_sheet_conductivity(response.matrix)
    reflection = sheet_conductivity_to_reflection_dimensionless(sheet)
    diagnostics = matrix_symmetry_diagnostics(response.matrix)
    return omega_eV, response.matrix, sheet.tensor.xx, reflection.tensor.xx, diagnostics


def _pairing_difference_status(values_by_nk: dict[int, float], nk_values: np.ndarray) -> str:
    if len(values_by_nk) < 2:
        return "pairing_difference_insufficient_nk_points"
    ordered = [(int(nk), values_by_nk[int(nk)]) for nk in nk_values if int(nk) in values_by_nk]
    if len(ordered) < 2:
        return "pairing_difference_insufficient_nk_points"
    last = ordered[-1][1]
    previous = ordered[-2][1]
    first = ordered[0][1]
    if last < previous and last < first:
        return "pairing_difference_not_stable_yet"
    if abs(last - previous) / (abs(last) + RATIO_EPS) < HIGH_NK_TOLERANCE:
        return "pairing_difference_stable_high_nk"
    return "pairing_difference_check_high_nk"


def refine_high_nk_convergence(
    kinds: list[str],
    nk_list: list[int],
    eta_list: list[float],
    matsubara_list: list[int],
    temperature_K: float,
    delta0_eV: float,
) -> dict[str, np.ndarray]:
    """Run focused high-Nk response convergence checks."""

    if not kinds:
        raise ValueError("kinds must be non-empty")
    if any(kind not in KINDS for kind in kinds):
        raise ValueError("unknown kind")
    if len(nk_list) < 2 or any(nk <= 0 for nk in nk_list):
        raise ValueError("nk_list must contain at least two positive values")
    if not eta_list or any(eta <= 0.0 for eta in eta_list):
        raise ValueError("eta_list must contain positive values")
    if not matsubara_list or any(n < 1 for n in matsubara_list):
        raise ValueError("matsubara_list must contain n >= 1")

    nk_values = np.asarray(sorted(nk_list), dtype=int)
    eta_values = np.asarray(sorted(eta_list), dtype=float)
    matsubara_values = np.asarray(sorted(matsubara_list), dtype=int)
    rows = [
        (kind, int(nk), float(eta), int(n))
        for kind in kinds
        for nk in nk_values
        for eta in eta_values
        for n in matsubara_values
    ]
    row_count = len(rows)
    data: dict[str, np.ndarray] = {
        "kind": np.empty(row_count, dtype="U16"),
        "nk": np.empty(row_count, dtype=int),
        "eta_eV": np.empty(row_count, dtype=float),
        "matsubara_index": np.empty(row_count, dtype=int),
        "omega_eV": np.empty(row_count, dtype=float),
        "response_xx": np.empty(row_count, dtype=complex),
        "response_yy": np.empty(row_count, dtype=complex),
        "response_xy": np.empty(row_count, dtype=complex),
        "response_yx": np.empty(row_count, dtype=complex),
        "sheet_conductivity_xx": np.empty(row_count, dtype=complex),
        "reflection_dimensionless_xx": np.empty(row_count, dtype=complex),
        "delta": np.empty(row_count, dtype=complex),
        "relative_offdiag": np.empty(row_count, dtype=float),
        "relative_eigen_split": np.empty(row_count, dtype=float),
        "relative_change_vs_largest_nk": np.full(row_count, np.nan, dtype=float),
        "relative_change_between_last_two_nk": np.full(row_count, np.nan, dtype=float),
        "eta_relative_change": np.full(row_count, np.nan, dtype=float),
        "spm_dwave_abs_diff_xx": np.full(row_count, np.nan, dtype=float),
        "spm_dwave_rel_diff_xx": np.full(row_count, np.nan, dtype=float),
        "high_nk_convergence_status": np.empty(row_count, dtype="U96"),
        "pairing_difference_status": np.empty(row_count, dtype="U96"),
        "diagnosis": np.empty(row_count, dtype="U128"),
        "notes": np.empty(row_count, dtype=object),
        "nk_list": nk_values,
        "eta_list": eta_values,
        "matsubara_list": matsubara_values,
        "temperature_K": np.array(temperature_K),
        "delta0_eV": np.array(delta0_eV),
    }

    index_by_key: dict[tuple[str, int, float, int], int] = {}
    for index, (kind, nk, eta, n) in enumerate(rows):
        omega_eV, matrix, sheet_xx, reflection_xx, diagnostics = _evaluate_response(
            kind,
            nk,
            eta,
            n,
            temperature_K,
            delta0_eV,
        )
        index_by_key[(kind, nk, eta, n)] = index
        data["kind"][index] = kind
        data["nk"][index] = nk
        data["eta_eV"][index] = eta
        data["matsubara_index"][index] = n
        data["omega_eV"][index] = omega_eV
        data["response_xx"][index] = matrix[0, 0]
        data["response_yy"][index] = matrix[1, 1]
        data["response_xy"][index] = matrix[0, 1]
        data["response_yx"][index] = matrix[1, 0]
        data["sheet_conductivity_xx"][index] = sheet_xx
        data["reflection_dimensionless_xx"][index] = reflection_xx
        data["delta"][index] = complex(diagnostics["delta"])
        data["relative_offdiag"][index] = float(diagnostics["relative_offdiag"])
        data["relative_eigen_split"][index] = float(diagnostics["relative_eigen_split"])
        data["notes"][index] = (
            "focused high-Nk local response convergence diagnostic only",
            "finite-q nonlocal response is not included",
            "not a Casimir calculation",
        )

    largest_nk = int(nk_values[-1])
    previous_nk = int(nk_values[-2])
    smallest_eta = float(eta_values[0])
    largest_eta = float(eta_values[-1])

    for index, (kind, nk, eta, n) in enumerate(rows):
        largest_index = index_by_key.get((kind, largest_nk, eta, n))
        previous_index = index_by_key.get((kind, previous_nk, eta, n))
        current_largest_eta_index = index_by_key.get((kind, nk, largest_eta, n))
        current_smallest_eta_index = index_by_key.get((kind, nk, smallest_eta, n))
        if largest_index is not None:
            data["relative_change_vs_largest_nk"][index] = _relative_change(
                data["response_xx"][index],
                data["response_xx"][largest_index],
            )
        if previous_index is not None and largest_index is not None:
            data["relative_change_between_last_two_nk"][index] = _relative_change(
                data["response_xx"][previous_index],
                data["response_xx"][largest_index],
            )
        if current_largest_eta_index is not None and current_smallest_eta_index is not None:
            data["eta_relative_change"][index] = _relative_change(
                data["response_xx"][current_largest_eta_index],
                data["response_xx"][current_smallest_eta_index],
            )

    for nk in nk_values:
        for eta in eta_values:
            for n in matsubara_values:
                spm_index = index_by_key.get(("spm", int(nk), float(eta), int(n)))
                dwave_index = index_by_key.get(("dwave", int(nk), float(eta), int(n)))
                if spm_index is None or dwave_index is None:
                    continue
                diff = float(abs(data["response_xx"][spm_index] - data["response_xx"][dwave_index]))
                scale = 0.5 * (abs(data["response_xx"][spm_index]) + abs(data["response_xx"][dwave_index]))
                rel_diff = 0.0 if np.isclose(scale, 0.0) else float(diff / scale)
                data["spm_dwave_abs_diff_xx"][spm_index] = diff
                data["spm_dwave_abs_diff_xx"][dwave_index] = diff
                data["spm_dwave_rel_diff_xx"][spm_index] = rel_diff
                data["spm_dwave_rel_diff_xx"][dwave_index] = rel_diff

    pairing_status_by_eta_n: dict[tuple[float, int], str] = {}
    for eta in eta_values:
        for n in matsubara_values:
            values_by_nk: dict[int, float] = {}
            for nk in nk_values:
                spm_index = index_by_key.get(("spm", int(nk), float(eta), int(n)))
                if spm_index is not None:
                    values_by_nk[int(nk)] = float(data["spm_dwave_rel_diff_xx"][spm_index])
            pairing_status_by_eta_n[(float(eta), int(n))] = _pairing_difference_status(values_by_nk, nk_values)

    for index, (kind, _nk, eta, n) in enumerate(rows):
        last_two_change = data["relative_change_between_last_two_nk"][index]
        if last_two_change < HIGH_NK_TOLERANCE:
            status = "high_nk_converged"
        elif kind == "normal":
            status = "normal_requires_finer_grid_or_FS_sensitive_integration"
        else:
            status = "high_nk_not_converged"
        data["high_nk_convergence_status"][index] = status
        data["pairing_difference_status"][index] = pairing_status_by_eta_n.get(
            (float(eta), int(n)),
            "pairing_difference_not_applicable",
        )
        warnings: list[str] = []
        if (
            abs(data["delta"][index]) > SYMMETRY_TOLERANCE
            or data["relative_offdiag"][index] > SYMMETRY_TOLERANCE
            or data["relative_eigen_split"][index] > SYMMETRY_TOLERANCE
        ):
            warnings.append("warning_symmetry")
        if not np.isfinite(data["response_xx"][index]):
            warnings.append("warning_nonfinite_response")
        data["diagnosis"][index] = "ok" if not warnings else ";".join(warnings)

    return data


def output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    resolved_prefix = output_prefix.resolve()
    project_data_root = (ROOT / "validation" / "outputs" / "archive" / "response" / "high_nk_convergence" / "data").resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "validation" / "outputs" / "archive" / "response" / "high_nk_convergence" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return (
        npz_path,
        csv_path,
        figure_dir / f"{output_prefix.name}_response_xx_vs_nk.png",
        figure_dir / f"{output_prefix.name}_relative_change_vs_nk.png",
        figure_dir / f"{output_prefix.name}_last_two_summary.png",
        figure_dir / f"{output_prefix.name}_spm_dwave_diff_vs_nk.png",
        figure_dir / f"{output_prefix.name}_eta_sensitivity_vs_nk.png",
        figure_dir / f"{output_prefix.name}_symmetry_vs_nk.png",
    )


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    return value


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    paths = output_paths(output_prefix)
    (
        npz_path,
        csv_path,
        response_plot,
        change_plot,
        last_two_plot,
        diff_plot,
        eta_plot,
        symmetry_plot,
    ) = paths
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    response_plot.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)

    fieldnames = [
        "kind",
        "nk",
        "eta_eV",
        "matsubara_index",
        "omega_eV",
        "response_xx",
        "response_yy",
        "response_xy",
        "response_yx",
        "sheet_conductivity_xx",
        "reflection_dimensionless_xx",
        "delta",
        "relative_offdiag",
        "relative_eigen_split",
        "relative_change_vs_largest_nk",
        "relative_change_between_last_two_nk",
        "eta_relative_change",
        "spm_dwave_abs_diff_xx",
        "spm_dwave_rel_diff_xx",
        "high_nk_convergence_status",
        "pairing_difference_status",
        "diagnosis",
        "notes",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(data["kind"].size):
            writer.writerow({name: _csv_value(data[name][index]) for name in fieldnames})

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    kinds = list(dict.fromkeys(str(kind) for kind in data["kind"]))
    eta_values = list(dict.fromkeys(float(eta) for eta in data["eta_eV"]))
    n_values = list(dict.fromkeys(int(n) for n in data["matsubara_index"]))
    reference_eta = min(eta_values)
    reference_n = min(n_values)

    fig_response, ax_response = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in kinds:
        for n in n_values:
            mask = (data["kind"] == kind) & np.isclose(data["eta_eV"], reference_eta) & (data["matsubara_index"] == n)
            ax_response.plot(data["nk"][mask], data["response_xx"][mask].real, marker="o", label=f"{kind} n={n}")
    ax_response.set_xlabel(r"$N_k$")
    ax_response.set_ylabel(r"Re response$_{xx}$")
    ax_response.set_title(rf"high-$N_k$ response at $\eta={reference_eta:g}$")
    style_publication_axis(ax_response)
    save_publication_figure(fig_response, response_plot)
    plt.close(fig_response)

    fig_change, ax_change = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in kinds:
        for n in n_values:
            mask = (data["kind"] == kind) & np.isclose(data["eta_eV"], reference_eta) & (data["matsubara_index"] == n)
            ax_change.plot(data["nk"][mask], data["relative_change_vs_largest_nk"][mask], marker="o", label=f"{kind} n={n}")
    ax_change.axhline(HIGH_NK_TOLERANCE, color="black", linestyle="--", linewidth=1.0)
    ax_change.set_yscale("symlog", linthresh=1e-4)
    ax_change.set_xlabel(r"$N_k$")
    ax_change.set_ylabel(r"relative change to largest $N_k$")
    ax_change.set_title("high-Nk convergence relative to largest grid")
    style_publication_axis(ax_change)
    save_publication_figure(fig_change, change_plot)
    plt.close(fig_change)

    fig_last, ax_last = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    labels = []
    values = []
    for kind in kinds:
        for n in n_values:
            mask = (data["kind"] == kind) & np.isclose(data["eta_eV"], reference_eta) & (data["matsubara_index"] == n)
            labels.append(f"{kind} n={n}")
            values.append(float(np.nanmax(data["relative_change_between_last_two_nk"][mask])))
    x = np.arange(len(labels))
    ax_last.bar(x, values)
    ax_last.axhline(HIGH_NK_TOLERANCE, color="black", linestyle="--", linewidth=1.0)
    ax_last.set_xticks(x, labels, rotation=45, ha="right")
    ax_last.set_ylabel("last-two-Nk relative change")
    ax_last.set_title(r"last-two grid convergence summary")
    style_publication_axis(ax_last, legend=False)
    save_publication_figure(fig_last, last_two_plot)
    plt.close(fig_last)

    fig_diff, ax_diff = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for eta in eta_values:
        for n in n_values:
            mask = (data["kind"] == "spm") & np.isclose(data["eta_eV"], eta) & (data["matsubara_index"] == n)
            if np.any(mask):
                ax_diff.plot(data["nk"][mask], data["spm_dwave_rel_diff_xx"][mask], marker="o", label=f"eta={eta:g} n={n}")
    ax_diff.set_xlabel(r"$N_k$")
    ax_diff.set_ylabel(r"relative $s_{\pm}$-$d$ diff")
    ax_diff.set_title(r"high-$N_k$ pairing response difference")
    style_publication_axis(ax_diff)
    save_publication_figure(fig_diff, diff_plot)
    plt.close(fig_diff)

    fig_eta, ax_eta = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in kinds:
        mask = (data["kind"] == kind) & (data["matsubara_index"] == reference_n) & np.isclose(data["eta_eV"], reference_eta)
        ax_eta.plot(data["nk"][mask], data["eta_relative_change"][mask], marker="o", label=kind)
    ax_eta.set_yscale("symlog", linthresh=1e-6)
    ax_eta.set_xlabel(r"$N_k$")
    ax_eta.set_ylabel("eta relative change")
    ax_eta.set_title(rf"eta sensitivity at n={reference_n}")
    style_publication_axis(ax_eta)
    save_publication_figure(fig_eta, eta_plot)
    plt.close(fig_eta)

    fig_sym, ax_sym = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for kind in kinds:
        mask = (data["kind"] == kind) & np.isclose(data["eta_eV"], reference_eta) & (data["matsubara_index"] == reference_n)
        ax_sym.plot(data["nk"][mask], np.abs(data["delta"][mask]), marker="o", label=f"{kind} |delta|")
        ax_sym.plot(data["nk"][mask], data["relative_offdiag"][mask], marker="s", linestyle="--", label=f"{kind} offdiag")
        ax_sym.plot(data["nk"][mask], data["relative_eigen_split"][mask], marker="^", linestyle=":", label=f"{kind} eig split")
    ax_sym.set_yscale("symlog", linthresh=1e-16)
    ax_sym.set_xlabel(r"$N_k$")
    ax_sym.set_ylabel("relative diagnostic")
    ax_sym.set_title(r"high-Nk C4 diagnostics")
    style_publication_axis(ax_sym)
    save_publication_figure(fig_sym, symmetry_plot)
    plt.close(fig_sym)

    return paths


def print_summary(data: dict[str, np.ndarray]) -> None:
    max_last_two = float(np.nanmax(data["relative_change_between_last_two_nk"]))
    max_eta = float(np.nanmax(data["eta_relative_change"]))
    max_symmetry = float(
        max(
            np.nanmax(np.abs(data["delta"])),
            np.nanmax(data["relative_offdiag"]),
            np.nanmax(data["relative_eigen_split"]),
        )
    )
    max_spm_dwave = float(np.nanmax(data["spm_dwave_rel_diff_xx"]))
    statuses = sorted(set(str(item) for item in data["high_nk_convergence_status"]))
    pairing_statuses = sorted(set(str(item) for item in data["pairing_difference_status"]))
    diagnoses = sorted(set(str(item) for item in data["diagnosis"]))
    print(f"row_count = {data['kind'].size}")
    print(f"max_relative_change_between_last_two_nk = {max_last_two}")
    print(f"max_eta_relative_change = {max_eta}")
    print(f"max_symmetry_diagnostic = {max_symmetry}")
    print(f"max_spm_dwave_rel_diff_xx = {max_spm_dwave}")
    print(f"high_nk_convergence_statuses = {statuses}")
    print(f"pairing_difference_statuses = {pairing_statuses}")
    print(f"diagnoses = {diagnoses}")
    print("note = high-Nk response convergence diagnostic only; not a Casimir result.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--eta-list", nargs="+", type=float, default=list(DEFAULT_ETA_LIST))
    parser.add_argument("--matsubara-list", nargs="+", type=int, default=list(DEFAULT_MATSUBARA_LIST))
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "validation" / "outputs" / "archive" / "response" / "high_nk_convergence" / "data" / "high_nk_convergence",
    )
    args = parser.parse_args()

    data = refine_high_nk_convergence(
        args.kinds,
        args.nk_list,
        args.eta_list,
        args.matsubara_list,
        args.temperature,
        args.delta0,
    )
    paths = save_outputs(data, args.output_prefix)
    print_summary(data)
    print(f"npz_path = {paths[0]}")
    print(f"csv_path = {paths[1]}")
    print(f"figure_paths = {paths[2]}, {paths[3]}, {paths[4]}, {paths[5]}, {paths[6]}, {paths[7]}")


if __name__ == "__main__":
    main()
