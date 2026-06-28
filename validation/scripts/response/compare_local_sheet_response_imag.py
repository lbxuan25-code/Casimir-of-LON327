#!/usr/bin/env python3
"""Compare local q=0 sheet responses before the formal Casimir stage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import (  # noqa: E402
    PairingAmplitudes,
    bosonic_matsubara_energy_eV,
    compare_local_responses_imag_axis,
    k_weights,
    uniform_bz_mesh,
)
from validation.lib.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

REQUIRED_NPZ_FIELDS = {
    "kind",
    "n",
    "omega_eV",
    "response_xx",
    "response_yy",
    "response_xy",
    "response_yx",
    "abs_xx",
    "delta",
    "relative_offdiag",
    "relative_eigen_split",
    "unit_label",
    "valid_for_casimir_input",
}


def scan_responses(
    kinds: list[str],
    delta0_eV: float,
    nk: int,
    temperature_K: float,
    matsubara_min: int,
    matsubara_max: int,
    eta_eV: float,
) -> dict[str, np.ndarray]:
    if matsubara_min < 1 or matsubara_max < matsubara_min:
        raise ValueError("Matsubara range must satisfy 1 <= min <= max")
    if nk <= 0:
        raise ValueError("nk must be positive")
    if eta_eV <= 0.0:
        raise ValueError("eta_eV must be positive")

    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    matsubara_indices = np.arange(matsubara_min, matsubara_max + 1, dtype=int)
    omega_grid = np.array([bosonic_matsubara_energy_eV(int(n), temperature_K) for n in matsubara_indices])

    rows = compare_local_responses_imag_axis(
        kinds,  # type: ignore[arg-type]
        omega_grid,
        mesh,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        pairing_params=PairingAmplitudes(delta0_eV=delta0_eV),
        k_weights=weights,
    )

    row_count = len(rows)
    data: dict[str, np.ndarray] = {
        "kind": np.empty(row_count, dtype="U16"),
        "n": np.empty(row_count, dtype=int),
        "omega_eV": np.empty(row_count, dtype=float),
        "response_xx": np.empty(row_count, dtype=complex),
        "response_yy": np.empty(row_count, dtype=complex),
        "response_xy": np.empty(row_count, dtype=complex),
        "response_yx": np.empty(row_count, dtype=complex),
        "abs_xx": np.empty(row_count, dtype=float),
        "delta": np.empty(row_count, dtype=complex),
        "relative_offdiag": np.empty(row_count, dtype=float),
        "relative_eigen_split": np.empty(row_count, dtype=float),
        "unit_label": np.empty(row_count, dtype="U64"),
        "valid_for_casimir_input": np.empty(row_count, dtype=bool),
        "source": np.empty(row_count, dtype="U64"),
        "notes": np.empty(row_count, dtype=object),
        "delta0_eV": np.array(delta0_eV),
        "nk": np.array(nk),
        "temperature_K": np.array(temperature_K),
        "eta_eV": np.array(eta_eV),
    }

    row_index = 0
    for kind in kinds:
        for n, omega in zip(matsubara_indices, omega_grid, strict=True):
            row = rows[row_index]
            matrix = np.asarray(row["matrix"], dtype=complex)
            data["kind"][row_index] = str(kind)
            data["n"][row_index] = int(n)
            data["omega_eV"][row_index] = float(omega)
            data["response_xx"][row_index] = matrix[0, 0]
            data["response_yy"][row_index] = matrix[1, 1]
            data["response_xy"][row_index] = matrix[0, 1]
            data["response_yx"][row_index] = matrix[1, 0]
            data["abs_xx"][row_index] = abs(matrix[0, 0])
            data["delta"][row_index] = complex(row["delta"])
            data["relative_offdiag"][row_index] = float(row["relative_offdiag"])
            data["relative_eigen_split"][row_index] = float(row["relative_eigen_split"])
            data["unit_label"][row_index] = str(row["unit_label"])
            data["valid_for_casimir_input"][row_index] = bool(row["valid_for_casimir_input"])
            data["source"][row_index] = str(row["source"])
            data["notes"][row_index] = tuple(row["notes"])  # type: ignore[arg-type]
            row_index += 1

    return data


def output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    resolved_prefix = output_prefix.resolve()
    project_data_root = (ROOT / "validation" / "outputs" / "archive" / "response" / "local_sheet_imag" / "data").resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "validation" / "outputs" / "archive" / "response" / "local_sheet_imag" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return (
        npz_path,
        figure_dir / f"{output_prefix.name}_abs_xx.png",
        figure_dir / f"{output_prefix.name}_re_xx.png",
        figure_dir / f"{output_prefix.name}_symmetry.png",
    )


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path, Path, Path]:
    npz_path, abs_plot_path, re_plot_path, symmetry_plot_path = output_paths(output_prefix)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    abs_plot_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    kinds = list(dict.fromkeys(str(kind) for kind in data["kind"]))

    fig_abs, ax_abs = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    fig_re, ax_re = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    fig_sym, ax_sym = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)

    for kind in kinds:
        mask = data["kind"] == kind
        omega = data["omega_eV"][mask]
        ax_abs.plot(omega, data["abs_xx"][mask], marker="o", label=kind)
        ax_re.plot(omega, data["response_xx"][mask].real, marker="o", label=kind)
        ax_sym.plot(omega, np.abs(data["delta"][mask]), marker="o", label=f"{kind} |delta|")
        ax_sym.plot(omega, data["relative_offdiag"][mask], marker="s", linestyle="--", label=f"{kind} offdiag")
        ax_sym.plot(
            omega,
            data["relative_eigen_split"][mask],
            marker="^",
            linestyle=":",
            label=f"{kind} eig split",
        )

    ax_abs.set_xlabel("imaginary-axis energy (eV)")
    ax_abs.set_ylabel("|response_xx|")
    ax_abs.set_title("Local sheet response magnitude")
    style_publication_axis(ax_abs)
    save_publication_figure(fig_abs, abs_plot_path)
    plt.close(fig_abs)

    ax_re.set_xlabel("imaginary-axis energy (eV)")
    ax_re.set_ylabel("Re response_xx")
    ax_re.set_title("Local sheet response real part")
    style_publication_axis(ax_re)
    save_publication_figure(fig_re, re_plot_path)
    plt.close(fig_re)

    ax_sym.set_xlabel("imaginary-axis energy (eV)")
    ax_sym.set_ylabel("relative diagnostic")
    ax_sym.set_yscale("log")
    ax_sym.set_title("Local response symmetry diagnostics")
    style_publication_axis(ax_sym)
    save_publication_figure(fig_sym, symmetry_plot_path)
    plt.close(fig_sym)

    return npz_path, abs_plot_path, re_plot_path, symmetry_plot_path


def print_summary(data: dict[str, np.ndarray]) -> None:
    for kind in dict.fromkeys(str(item) for item in data["kind"]):
        mask = data["kind"] == kind
        min_abs_xx = float(np.min(data["abs_xx"][mask]))
        max_abs_xx = float(np.max(data["abs_xx"][mask]))
        valid_values = set(bool(item) for item in data["valid_for_casimir_input"][mask])
        print(f"kind = {kind}")
        print(f"max_abs_delta = {float(np.max(np.abs(data['delta'][mask])))}")
        print(f"max_relative_offdiag = {float(np.max(data['relative_offdiag'][mask]))}")
        print(f"max_relative_eigen_split = {float(np.max(data['relative_eigen_split'][mask]))}")
        print(f"response magnitude range = [{min_abs_xx}, {max_abs_xx}]")
        print(f"valid_for_casimir_input = {valid_values == {True}}")
        print("note = local q=0 response only; n=0 unresolved; SI normalization not finalized.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=("normal", "spm", "dwave"), default=["normal", "spm", "dwave"])
    parser.add_argument("--delta0", type=float, default=0.04, help="Pairing amplitude Delta0 in eV.")
    parser.add_argument("--nk", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=30.0, help="Temperature in K.")
    parser.add_argument("--matsubara-min", type=int, default=1)
    parser.add_argument("--matsubara-max", type=int, default=8)
    parser.add_argument("--eta", type=float, default=1e-4, help="Imaginary-axis regulator in eV.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "validation" / "outputs" / "archive" / "response" / "local_sheet_imag" / "data" / "local_sheet_response_imag",
    )
    args = parser.parse_args()

    data = scan_responses(
        args.kinds,
        args.delta0,
        args.nk,
        args.temperature,
        args.matsubara_min,
        args.matsubara_max,
        args.eta,
    )
    paths = save_outputs(data, args.output_prefix)
    print_summary(data)
    print(f"npz_path = {paths[0]}")
    print(f"figure_paths = {paths[1]}, {paths[2]}, {paths[3]}")


if __name__ == "__main__":
    main()
