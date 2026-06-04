#!/usr/bin/env python3
"""Diagnose the BdG diamagnetic kernel before constructing K_total."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import (  # noqa: E402
    KuboConfig,
    PairingAmplitudes,
    bdg_diamagnetic_kernel,
    k_weights,
    uniform_bz_mesh,
)
from lno327.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402


def relative_eigen_split(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvals(matrix)
    scale = 0.5 * (abs(eigenvalues[0]) + abs(eigenvalues[1]))
    if np.isclose(scale, 0.0):
        return 0.0
    return float(abs(eigenvalues[0] - eigenvalues[1]) / scale)


def kernel_diagnostics(kernel: np.ndarray) -> tuple[complex, float, float]:
    diagonal_scale = 0.5 * (abs(kernel[0, 0]) + abs(kernel[1, 1]))
    if np.isclose(kernel[0, 0] + kernel[1, 1], 0.0):
        delta = complex(0.0)
    else:
        delta = (kernel[0, 0] - kernel[1, 1]) / (kernel[0, 0] + kernel[1, 1])
    offdiag_norm = float(np.linalg.norm([kernel[0, 1], kernel[1, 0]]))
    relative_offdiag = 0.0 if np.isclose(diagonal_scale, 0.0) else float(offdiag_norm / diagonal_scale)
    return delta, relative_offdiag, relative_eigen_split(kernel)


def scan_kind(
    kind: str,
    delta0_eV: float,
    nk: int,
    temperature_K: float,
) -> dict[str, np.ndarray]:
    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    config = KuboConfig.from_kelvin(omega_eV=0.0, temperature_K=temperature_K, output_si=False)
    kernel = bdg_diamagnetic_kernel(
        kind,  # type: ignore[arg-type]
        PairingAmplitudes(delta0_eV=delta0_eV),
        mesh,
        config,
        weights,
    )
    delta, relative_offdiag, eigen_split = kernel_diagnostics(kernel)
    return {
        "kind": np.array(kind),
        "Kxx": np.array(kernel[0, 0]),
        "Kyy": np.array(kernel[1, 1]),
        "Kxy": np.array(kernel[0, 1]),
        "Kyx": np.array(kernel[1, 0]),
        "delta_dia": np.array(delta),
        "relative_offdiag": np.array(relative_offdiag),
        "relative_eigen_split": np.array(eigen_split),
        "delta0_eV": np.array(delta0_eV),
        "nk": np.array(nk),
        "temperature_K": np.array(temperature_K),
        "response_layer": np.array("mass_contact_term"),
    }


def diagnosis(data: dict[str, np.ndarray]) -> str:
    if (
        abs(data["delta_dia"].item()) < 1e-8
        and float(data["relative_offdiag"].item()) < 1e-8
        and float(data["relative_eigen_split"].item()) < 1e-8
    ):
        return "pass"
    return "check"


def output_paths(output_prefix: Path, kind: str) -> tuple[Path, Path]:
    data_path = output_prefix.parent / f"{output_prefix.name}_{kind}.npz"
    resolved_prefix = output_prefix.resolve()
    project_data_root = (ROOT / "outputs" / "bdg" / "diamagnetic_kernel" / "data").resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "outputs" / "bdg" / "diamagnetic_kernel" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return data_path, figure_dir / f"{output_prefix.name}_{kind}_diagnostics.png"


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path]:
    kind = str(data["kind"].item())
    npz_path, figure_path = output_paths(output_prefix, kind)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    labels = ["|delta_dia|", "relative_offdiag", "relative_eigen_split"]
    values = [
        abs(data["delta_dia"].item()),
        float(data["relative_offdiag"].item()),
        float(data["relative_eigen_split"].item()),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    ax.bar(labels, values)
    ax.set_yscale("log")
    ax.set_ylabel("relative diagnostic")
    ax.set_title(f"{kind}: BdG diamagnetic kernel symmetry")
    style_publication_axis(ax, legend=False)
    save_publication_figure(fig, figure_path)
    plt.close(fig)
    return npz_path, figure_path


def print_summary(data: dict[str, np.ndarray]) -> None:
    kernel = np.array(
        [
            [data["Kxx"].item(), data["Kxy"].item()],
            [data["Kyx"].item(), data["Kyy"].item()],
        ],
        dtype=complex,
    )
    print(f"kind = {data['kind'].item()}")
    print("K_dia matrix =")
    print(kernel)
    print(f"delta_dia = {data['delta_dia'].item()}")
    print(f"relative_offdiag = {data['relative_offdiag'].item()}")
    print(f"relative_eigen_split = {data['relative_eigen_split'].item()}")
    print(f"diagnosis = {diagnosis(data)}")
    print("note = diamagnetic kernel only; not full superconducting conductivity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=("spm", "dwave"), default=["spm", "dwave"])
    parser.add_argument("--delta0", type=float, default=0.04, help="Pairing amplitude Delta0 in eV.")
    parser.add_argument("--nk", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=30.0, help="Temperature in K.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "outputs" / "bdg" / "diamagnetic_kernel" / "data" / "K_dia",
    )
    args = parser.parse_args()

    if args.nk <= 0:
        raise ValueError("--nk must be positive")

    for kind in args.kinds:
        data = scan_kind(kind, args.delta0, args.nk, args.temperature)
        npz_path, figure_path = save_outputs(data, args.output_prefix)
        print_summary(data)
        print(f"npz_path = {npz_path}")
        print(f"figure_path = {figure_path}")


if __name__ == "__main__":
    main()
