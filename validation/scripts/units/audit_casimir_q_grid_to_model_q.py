#!/usr/bin/env python3
"""Audit Casimir u-grid momenta in model-q units.

This script performs only geometry and unit conversion:

    q_SI = u / d
    q_model = a * q_SI
    qx_model = q_model * cos(phi)
    qy_model = q_model * sin(phi)

No response tensor, finite-q conductivity, reflection matrix, or Casimir
integral is computed here.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "units" / "casimir_q_grid_model_q_audit"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "casimir_q_grid_model_q_audit"
SUMMARY_NAME = "casimir_q_grid_model_q_audit_summary.md"

DEFAULT_DISTANCE_LIST = (3e-8, 5e-8, 7.5e-8, 1e-7, 1.5e-7, 2e-7)
DEFAULT_U_MAX = 80.0
DEFAULT_DU = 0.5
DEFAULT_PHI_NUM = 32
DEFAULT_LATTICE_CONSTANT_M = 3.9e-10
DEFAULT_SMALL_Q_THRESHOLDS = (1e-3, 5e-3, 1e-2, 5e-2, 1e-1)

QUICK_DISTANCE_LIST = (5e-8, 1e-7)
QUICK_U_MAX = 10.0
QUICK_DU = 1.0
QUICK_PHI_NUM = 8

CSV_COLUMNS = (
    "distance_m",
    "u",
    "phi",
    "lattice_constant_m",
    "q_SI_m_inv",
    "q_model",
    "qx_model",
    "qy_model",
    "q_model_over_pi",
    "q_model_over_2pi",
    "small_q_threshold",
    "inside_small_q_regime",
    "inside_pi_BZ_reference",
    "inside_2pi_BZ_reference",
    "unit_audit_only",
    "response_computed",
    "casimir_computed",
    "not_final_casimir_conclusion",
)


def _u_grid(u_max: float, du: float) -> np.ndarray:
    if u_max < 0.0:
        raise ValueError("u-max must be non-negative")
    if du <= 0.0:
        raise ValueError("du must be positive")
    count = int(np.floor(u_max / du + 1e-12)) + 1
    values = du * np.arange(count, dtype=float)
    if values.size == 0 or not np.isclose(values[-1], u_max):
        values = np.append(values, float(u_max))
    return values[values <= u_max + 1e-12]


def _phi_grid(phi_num: int) -> np.ndarray:
    if phi_num <= 0:
        raise ValueError("phi-num must be positive")
    return 2.0 * np.pi * np.arange(phi_num, dtype=float) / float(phi_num)


def _validate_positive_list(name: str, values: list[float]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"{name} values must be positive")


def generate_audit_data(
    *,
    distance_list: list[float],
    u_max: float,
    du: float,
    phi_num: int,
    lattice_constant_m: float,
    small_q_threshold_list: list[float],
) -> dict[str, np.ndarray]:
    """Generate threshold-expanded audit rows for the Casimir q-grid."""

    _validate_positive_list("distance-list", distance_list)
    _validate_positive_list("small-q-threshold-list", small_q_threshold_list)
    if lattice_constant_m <= 0.0:
        raise ValueError("lattice-constant-m must be positive")

    distances = np.asarray(distance_list, dtype=float)
    u_values = _u_grid(u_max, du)
    phi_values = _phi_grid(phi_num)
    thresholds = np.asarray(small_q_threshold_list, dtype=float)

    distance_mesh, u_mesh, phi_mesh, threshold_mesh = np.meshgrid(
        distances,
        u_values,
        phi_values,
        thresholds,
        indexing="ij",
    )
    q_si = u_mesh / distance_mesh
    q_model = lattice_constant_m * q_si
    qx_model = q_model * np.cos(phi_mesh)
    qy_model = q_model * np.sin(phi_mesh)

    return {
        "distance_m": distance_mesh.ravel(),
        "u": u_mesh.ravel(),
        "phi": phi_mesh.ravel(),
        "lattice_constant_m": np.full(distance_mesh.size, lattice_constant_m, dtype=float),
        "q_SI_m_inv": q_si.ravel(),
        "q_model": q_model.ravel(),
        "qx_model": qx_model.ravel(),
        "qy_model": qy_model.ravel(),
        "q_model_over_pi": (q_model / np.pi).ravel(),
        "q_model_over_2pi": (q_model / (2.0 * np.pi)).ravel(),
        "small_q_threshold": threshold_mesh.ravel(),
        "inside_small_q_regime": (q_model <= threshold_mesh).ravel(),
        "inside_pi_BZ_reference": (q_model <= np.pi).ravel(),
        "inside_2pi_BZ_reference": (q_model <= 2.0 * np.pi).ravel(),
        "unit_audit_only": np.full(distance_mesh.size, True, dtype=bool),
        "response_computed": np.full(distance_mesh.size, False, dtype=bool),
        "casimir_computed": np.full(distance_mesh.size, False, dtype=bool),
        "not_final_casimir_conclusion": np.full(distance_mesh.size, True, dtype=bool),
    }


def _base_q_samples(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    threshold = float(np.min(data["small_q_threshold"]))
    mask = np.isclose(data["small_q_threshold"], threshold)
    return {key: value[mask] for key, value in data.items() if key != "small_q_threshold"}


def _write_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for index in range(len(data["distance_m"])):
            writer.writerow({column: data[column][index] for column in CSV_COLUMNS})


def _output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path]:
    csv_path = output_prefix.with_suffix(".csv")
    npz_path = output_prefix.with_suffix(".npz")
    result_root = output_prefix.parent.parent if output_prefix.parent.name == "data" else output_prefix.parent
    figure_dir = result_root / "figures"
    summary_path = result_root / SUMMARY_NAME
    return csv_path, npz_path, figure_dir, summary_path


def _plot_outputs(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    base = _base_q_samples(data)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    distances = np.array(sorted(set(float(value) for value in base["distance_m"])))
    q_max_by_distance = np.array(
        [np.max(base["q_model"][np.isclose(base["distance_m"], distance)]) for distance in distances]
    )
    ax.plot(distances, q_max_by_distance, marker="o")
    ax.set(
        xlabel="distance d (m)",
        ylabel="max q_model",
        title="Casimir q-grid maximum model momentum",
    )
    style_publication_axis(ax, legend=False)
    path = figure_dir / "q_model_max_vs_distance.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(base["q_model"], bins=60, color="#1f77b4", alpha=0.85)
    ax.set(
        xlabel="q_model",
        ylabel="sample count",
        title="Casimir q-grid model-momentum distribution",
    )
    style_publication_axis(ax, legend=False)
    path = figure_dir / "q_model_histogram.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for threshold in sorted(set(float(value) for value in data["small_q_threshold"])):
        coverage = []
        for distance in distances:
            mask = np.isclose(data["distance_m"], distance) & np.isclose(data["small_q_threshold"], threshold)
            coverage.append(float(np.mean(data["inside_small_q_regime"][mask])))
        ax.plot(distances, coverage, marker="o", label=f"q <= {threshold:g}")
    ax.set(
        xlabel="distance d (m)",
        ylabel="small-q coverage fraction",
        ylim=(-0.02, 1.02),
        title="Small-q coverage by distance",
    )
    style_publication_axis(ax)
    path = figure_dir / "q_model_coverage_by_distance.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    return paths


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _list_fmt(values: list[float]) -> str:
    return " ".join(_fmt(value) for value in values)


def _stage1_q_list_from_summary() -> list[float]:
    summary = ROOT / "validation" / "outputs" / "response" / "normal_finite_q_kernel_convergence" / (
        "normal_finite_q_kernel_convergence_summary.md"
    )
    if not summary.exists():
        return [0.0, 1e-3]
    text = summary.read_text(encoding="utf-8")
    marker = "--q-list"
    if marker not in text:
        return [0.0, 1e-3]
    after = text.split(marker, 1)[1]
    tokens = after.split("--", 1)[0].replace("`", " ").split()
    values: list[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            break
    return values or [0.0, 1e-3]


def _recommended_q_lists(q_max: float) -> tuple[list[float], list[float], list[float]]:
    small_q_regression = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    casimir_relevant = [0.0, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]
    if q_max > 0.5:
        casimir_relevant.extend([0.75, 1.0])
    if q_max > 1.0:
        casimir_relevant.append(float(np.ceil(10.0 * q_max) / 10.0))
    casimir_relevant = sorted({value for value in casimir_relevant if value <= max(q_max * 1.05, 1e-2)})
    if casimir_relevant[-1] < q_max:
        casimir_relevant.append(float(q_max))
    bz_stress = [np.pi / 8.0, np.pi / 4.0, np.pi / 2.0, np.pi, 2.0 * np.pi]
    return small_q_regression, casimir_relevant, bz_stress


def _summary_lines(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    figure_paths: list[Path],
) -> list[str]:
    base = _base_q_samples(data)
    q_min = float(np.min(base["q_model"]))
    q_max = float(np.max(base["q_model"]))
    distances = np.array(sorted(set(float(value) for value in base["distance_m"])))
    thresholds = sorted(set(float(value) for value in data["small_q_threshold"]))
    stage1_q_list = _stage1_q_list_from_summary()
    stage1_q_max = max(stage1_q_list)
    covers_stage2 = stage1_q_max >= q_max
    small_q_regression, casimir_relevant, bz_stress = _recommended_q_lists(q_max)

    lines = [
        "# Casimir q-grid to model-q unit/sampling audit",
        "",
        "This is a unit/sampling audit only.",
        "No response tensor is computed.",
        "No finite-q conductivity is produced.",
        "No Casimir conclusion is made.",
        "",
        f"run_command = `{command}`",
        f"quick_mode = {bool(args.quick)}",
        f"lattice_constant_m = {_fmt(args.lattice_constant_m)}",
        (
            "lattice_constant_m is a configurable assumption for this audit "
            "and is not a final material parameter."
        ),
        f"distance_list_m = {_list_fmt(list(args.distance_list))}",
        f"u_max = {_fmt(args.u_max)}",
        f"du = {_fmt(args.du)}",
        f"phi_num = {int(args.phi_num)}",
        f"small_q_threshold_list = {_list_fmt(list(args.small_q_threshold_list))}",
        "",
        "## Scope flags",
        "unit_audit_only=True",
        "response_computed=False",
        "casimir_computed=False",
        "not_final_casimir_conclusion=True",
        "",
        "## Full grid q_model range",
        f"- q_model_min = {_fmt(q_min)}",
        f"- q_model_max = {_fmt(q_max)}",
        f"- q_model_max/pi = {_fmt(q_max / np.pi)}",
        f"- q_model_max/(2pi) = {_fmt(q_max / (2.0 * np.pi))}",
        "",
        "## q_model_max by distance",
    ]
    for distance in distances:
        mask = np.isclose(base["distance_m"], distance)
        local_max = float(np.max(base["q_model"][mask]))
        lines.append(
            f"- d = {_fmt(distance)} m: q_model_max = {_fmt(local_max)}, "
            f"q_model_max/pi = {_fmt(local_max / np.pi)}, "
            f"q_model_max/(2pi) = {_fmt(local_max / (2.0 * np.pi))}"
        )

    lines.extend(["", "## Small-q coverage"])
    for threshold in thresholds:
        mask = np.isclose(data["small_q_threshold"], threshold)
        coverage = float(np.mean(data["inside_small_q_regime"][mask]))
        count = int(np.count_nonzero(data["inside_small_q_regime"][mask]))
        total = int(np.count_nonzero(mask))
        lines.append(
            f"- threshold q <= {_fmt(threshold)}: {coverage:.6%} "
            f"({count}/{total} sampled points)"
        )

    lines.extend(["", "## Stage 1 coverage check"])
    lines.append(f"- Stage 1 sampled q_model list found in repository: {_list_fmt(stage1_q_list)}")
    lines.append(f"- Stage 1 q_model_max = {_fmt(stage1_q_max)}")
    lines.append(f"- Current audit q_model_max = {_fmt(q_max)}")
    if covers_stage2:
        lines.append("- Stage 1 sampled q range covers the current Casimir-relevant q_model range.")
    else:
        lines.append(
            "- Stage 1 sampled q range does not cover the current Casimir-relevant q_model range; "
            "it only tests the small-q limit."
        )

    lines.extend(
        [
            "",
            "## Stage 3 recommended q-list",
            f"- small-q regression list: {_list_fmt(small_q_regression)}",
            f"- Casimir-relevant q list: {_list_fmt(casimir_relevant)}",
            f"- BZ stress list: {_list_fmt(bz_stress)}",
            "",
            "The BZ stress list is for numerical stress testing only; it is not a statement that "
            "the audited local Casimir grid reaches those momenta.",
            "",
            "## Output files",
            f"- CSV: {args.output_prefix.with_suffix('.csv')}",
            f"- NPZ: {args.output_prefix.with_suffix('.npz')}",
        ]
    )
    lines.extend(f"- figure: {path}" for path in figure_paths)
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-list", nargs="+", type=float, default=list(DEFAULT_DISTANCE_LIST))
    parser.add_argument("--u-max", type=float, default=DEFAULT_U_MAX)
    parser.add_argument("--du", type=float, default=DEFAULT_DU)
    parser.add_argument("--phi-num", type=int, default=DEFAULT_PHI_NUM)
    parser.add_argument("--lattice-constant-m", type=float, default=DEFAULT_LATTICE_CONSTANT_M)
    parser.add_argument(
        "--small-q-threshold-list",
        nargs="+",
        type=float,
        default=list(DEFAULT_SMALL_Q_THRESHOLDS),
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.distance_list = list(QUICK_DISTANCE_LIST)
        args.u_max = QUICK_U_MAX
        args.du = QUICK_DU
        args.phi_num = QUICK_PHI_NUM

    data = generate_audit_data(
        distance_list=list(args.distance_list),
        u_max=float(args.u_max),
        du=float(args.du),
        phi_num=int(args.phi_num),
        lattice_constant_m=float(args.lattice_constant_m),
        small_q_threshold_list=list(args.small_q_threshold_list),
    )
    csv_path, npz_path, figure_dir, summary_path = _output_paths(args.output_prefix)
    _write_csv(csv_path, data)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)
    figure_paths = _plot_outputs(data, figure_dir)
    command = "python " + " ".join(shlex.quote(value) for value in sys.argv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(_summary_lines(data, args, command, figure_paths)) + "\n",
        encoding="utf-8",
    )
    print(f"csv_path = {csv_path}")
    print(f"npz_path = {npz_path}")
    print(f"summary_path = {summary_path}")
    print("figure_paths = " + ", ".join(str(path) for path in figure_paths))


if __name__ == "__main__":
    main()
