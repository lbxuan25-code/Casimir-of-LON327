#!/usr/bin/env python3
"""Audit Fourier/hopping reconstruction of the normal-state H0(k)."""

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

from lno327.conductivity import uniform_bz_mesh  # noqa: E402
from lno327.model import normal_state_hamiltonian  # noqa: E402
from lno327.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402
from lno327.tb_fourier import (  # noqa: E402
    normal_state_hamiltonian_from_hoppings,
    normal_state_hopping_terms,
    validate_hopping_hermiticity,
)

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "tb_fourier_reconstruction_audit"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "tb_fourier_reconstruction"
SUMMARY_NAME = "tb_fourier_reconstruction_audit_summary.md"
EPS = 1e-300

COMPACT_COLUMNS = (
    "hamiltonian_representation",
    "num_k_points",
    "max_abs_reconstruction_error",
    "max_rel_reconstruction_error",
    "max_hopping_hermiticity_error",
    "quick_mode",
    "expanded_data_written",
    "response_computed",
    "conductivity_computed",
    "ward_identity_checked",
    "casimir_computed",
    "not_final_casimir_conclusion",
)

EXPANDED_COLUMNS = (
    "k_index",
    "kx",
    "ky",
    "abs_reconstruction_error",
    "rel_reconstruction_error",
    "h_trig_norm",
)


def _k_points(mesh_n: int, random_num: int, seed: int) -> np.ndarray:
    if mesh_n <= 0:
        raise ValueError("mesh-n must be positive")
    if random_num < 0:
        raise ValueError("random-num must be non-negative")
    mesh = uniform_bz_mesh(mesh_n)
    rng = np.random.default_rng(seed)
    random_points = rng.uniform(-np.pi, np.pi, size=(random_num, 2))
    return np.vstack([mesh, random_points]) if random_num else mesh


def run_audit(*, mesh_n: int, random_num: int, seed: int, hermiticity_atol: float) -> tuple[dict[str, np.ndarray], float]:
    points = _k_points(mesh_n, random_num, seed)
    hopping_terms = normal_state_hopping_terms()
    hermiticity_error = validate_hopping_hermiticity(hopping_terms, atol=hermiticity_atol)
    rows: list[dict[str, float | int]] = []
    for index, (kx, ky) in enumerate(points):
        h_trig = normal_state_hamiltonian(float(kx), float(ky))
        h_hop = normal_state_hamiltonian_from_hoppings(float(kx), float(ky), hopping_terms=hopping_terms)
        abs_error = float(np.linalg.norm(h_hop - h_trig))
        trig_norm = float(np.linalg.norm(h_trig))
        rows.append(
            {
                "k_index": int(index),
                "kx": float(kx),
                "ky": float(ky),
                "abs_reconstruction_error": abs_error,
                "rel_reconstruction_error": abs_error / max(trig_norm, EPS),
                "h_trig_norm": trig_norm,
            }
        )
    return {column: np.array([row[column] for row in rows]) for column in EXPANDED_COLUMNS}, hermiticity_error


def _output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path]:
    compact_csv = output_prefix.with_name(f"{output_prefix.name}_compact").with_suffix(".csv")
    expanded_csv = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".csv")
    expanded_npz = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".npz")
    result_root = output_prefix.parent.parent if output_prefix.parent.name == "data" else output_prefix.parent
    figure_dir = result_root / "figures"
    summary_path = result_root / SUMMARY_NAME
    return compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path


def _write_rows(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_expanded(path: Path, data: dict[str, np.ndarray]) -> None:
    rows = [{column: data[column][index] for column in EXPANDED_COLUMNS} for index in range(len(data["k_index"]))]
    _write_rows(path, EXPANDED_COLUMNS, rows)


def _compact_row(
    data: dict[str, np.ndarray],
    *,
    hermiticity_error: float,
    quick: bool,
    expanded_data_written: bool,
) -> dict[str, object]:
    return {
        "hamiltonian_representation": "H0_hop(k)=sum_R t_R exp(i k.R); equivalent to existing H0_trig(k)",
        "num_k_points": int(len(data["k_index"])),
        "max_abs_reconstruction_error": float(np.max(data["abs_reconstruction_error"])),
        "max_rel_reconstruction_error": float(np.max(data["rel_reconstruction_error"])),
        "max_hopping_hermiticity_error": float(hermiticity_error),
        "quick_mode": bool(quick),
        "expanded_data_written": bool(expanded_data_written),
        "response_computed": False,
        "conductivity_computed": False,
        "ward_identity_checked": False,
        "casimir_computed": False,
        "not_final_casimir_conclusion": True,
    }


def _plot_outputs(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(data["k_index"], np.maximum(data["rel_reconstruction_error"], EPS), marker=".", linestyle="none")
    ax.set(
        xlabel="k-point index",
        ylabel="relative reconstruction error",
        title="Hopping/Fourier H0 reconstruction error",
    )
    style_publication_axis(ax, legend=False)
    path = figure_dir / "tb_fourier_reconstruction_error.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    return [path]


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _summary_lines(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    hermiticity_error: float,
    compact_csv: Path,
    expanded_csv: Path,
    expanded_npz: Path,
    figure_paths: list[Path],
) -> list[str]:
    max_abs = float(np.max(data["abs_reconstruction_error"]))
    max_rel = float(np.max(data["rel_reconstruction_error"]))
    passed = bool(max_rel < args.reconstruction_rtol and hermiticity_error < args.hermiticity_atol)
    return [
        "# TB Fourier reconstruction audit",
        "",
        "This is an H0(k) representation reconstruction audit.",
        "It is not a response calculation, not conductivity, not a Ward identity check, and not Casimir.",
        (
            "The hopping/Fourier representation and the trigonometric representation are the "
            "same Hamiltonian written in equivalent forms."
        ),
        "The hopping/Fourier representation is not a new model and not a higher-precision model.",
        "",
        f"run_command = `{command}`",
        f"quick_mode={bool(args.quick)}",
        f"expanded_data_written={bool(args.write_expanded_data)}",
        f"mesh_n={int(args.mesh_n)}",
        f"random_num={int(args.random_num)}",
        f"random_seed={int(args.seed)}",
        "",
        "## Reconstruction status",
        f"- passed = {passed}",
        f"- num_k_points = {len(data['k_index'])}",
        f"- max absolute reconstruction error = {_fmt(max_abs)}",
        f"- max relative reconstruction error = {_fmt(max_rel)}",
        f"- max hopping Hermiticity error = {_fmt(float(hermiticity_error))}",
        "",
        "## Scope flags",
        "response_computed=False",
        "conductivity_computed=False",
        "ward_identity_checked=False",
        "casimir_computed=False",
        "not_final_casimir_conclusion=True",
        "",
        "## Output files",
        f"- compact CSV: {compact_csv}",
        f"- expanded_data_written={bool(args.write_expanded_data)}",
        *(
            [f"- expanded CSV: {expanded_csv}", f"- expanded NPZ: {expanded_npz}"]
            if args.write_expanded_data
            else ["- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally."]
        ),
        *(f"- figure: {path}" for path in figure_paths),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-n", type=int, default=24)
    parser.add_argument("--random-num", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--hermiticity-atol", type=float, default=1e-12)
    parser.add_argument("--reconstruction-rtol", type=float, default=1e-12)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--write-expanded-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.mesh_n = 8
        args.random_num = 8
    data, hermiticity_error = run_audit(
        mesh_n=int(args.mesh_n),
        random_num=int(args.random_num),
        seed=int(args.seed),
        hermiticity_atol=float(args.hermiticity_atol),
    )
    compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path = _output_paths(args.output_prefix)
    compact = _compact_row(
        data,
        hermiticity_error=hermiticity_error,
        quick=bool(args.quick),
        expanded_data_written=bool(args.write_expanded_data),
    )
    _write_rows(compact_csv, COMPACT_COLUMNS, [compact])
    if args.write_expanded_data:
        _write_expanded(expanded_csv, data)
        expanded_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(expanded_npz, **data)
    figure_paths = _plot_outputs(data, figure_dir)
    command = "python " + " ".join(shlex.quote(value) for value in sys.argv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(_summary_lines(data, args, command, hermiticity_error, compact_csv, expanded_csv, expanded_npz, figure_paths))
        + "\n",
        encoding="utf-8",
    )
    print(f"compact_csv_path = {compact_csv}")
    print(f"expanded_data_written = {bool(args.write_expanded_data)}")
    if args.write_expanded_data:
        print(f"expanded_csv_path = {expanded_csv}")
        print(f"expanded_npz_path = {expanded_npz}")
    print(f"summary_path = {summary_path}")
    print("figure_paths = " + ", ".join(str(path) for path in figure_paths))


if __name__ == "__main__":
    main()
