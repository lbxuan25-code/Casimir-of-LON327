#!/usr/bin/env python3
"""Audit the finite-q Peierls contact vertex Lambda_ij(k,q)."""

from __future__ import annotations

import argparse
import ast
import csv
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.conductivity import uniform_bz_mesh  # noqa: E402
from lno327.models.lno327_four_orbital.vertices import normal_state_mass_operator  # noqa: E402
from validation.lib.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402
from lno327.tb_fourier import normal_state_hopping_terms, peierls_hamiltonian_contact_vertex  # noqa: E402

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "peierls_contact_vertex_audit"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "peierls_contact_vertex_audit"
SUMMARY_NAME = "summary.md"
EPS = 1e-300

DEFAULT_MESH_N = 24
DEFAULT_RANDOM_NUM = 64
DEFAULT_Q_LIST = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
DEFAULT_Q_ANGLE_LIST = (0.0, np.pi / 8.0, np.pi / 4.0, 3.0 * np.pi / 8.0, np.pi / 2.0)
DEFAULT_DIRECTIONS = ("xx", "xy", "yx", "yy")

QUICK_MESH_N = 8
QUICK_RANDOM_NUM = 8
QUICK_Q_LIST = (0.0, 0.01, 0.1, 1.0)
QUICK_Q_ANGLE_LIST = (0.0, np.pi / 4.0)

COMPACT_COLUMNS = (
    "q_model",
    "q_angle_rad",
    "direction_i",
    "direction_j",
    "num_k_points",
    "max_abs_mass_limit_error",
    "max_rel_mass_limit_error",
    "median_rel_mass_limit_error",
    "max_hermiticity_error",
    "max_index_symmetry_error",
    "quick_mode",
    "expanded_data_written",
    "response_computed",
    "conductivity_computed",
    "ward_identity_checked",
    "casimir_computed",
    "not_final_finite_q_contact",
    "not_final_finite_q_conductivity",
    "not_final_casimir_conclusion",
    "diagnosis",
)

EXPANDED_COLUMNS = (
    "q_model",
    "q_angle_rad",
    "direction_i",
    "direction_j",
    "k_index",
    "kx",
    "ky",
    "abs_mass_limit_error",
    "rel_mass_limit_error",
    "mass_norm",
    "contact_norm",
    "hermiticity_error",
    "index_symmetry_error",
)


def _angle_expression(value: str) -> float:
    normalized = re.sub(r"(?<=\d)\s*pi\b", "*pi", value)
    node = ast.parse(normalized, mode="eval")
    allowed_binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
    }
    allowed_unary = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}

    def evaluate(item: ast.AST) -> float:
        if isinstance(item, ast.Expression):
            return evaluate(item.body)
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return float(item.value)
        if isinstance(item, ast.Name) and item.id == "pi":
            return float(np.pi)
        if isinstance(item, ast.BinOp) and type(item.op) in allowed_binary:
            return float(allowed_binary[type(item.op)](evaluate(item.left), evaluate(item.right)))
        if isinstance(item, ast.UnaryOp) and type(item.op) in allowed_unary:
            return float(allowed_unary[type(item.op)](evaluate(item.operand)))
        raise argparse.ArgumentTypeError(f"unsupported angle expression: {value}")

    return evaluate(node)


def _direction_pair(value: str) -> tuple[str, str]:
    if value not in {"xx", "xy", "yx", "yy"}:
        raise argparse.ArgumentTypeError("directions must be one of xx xy yx yy")
    return value[0], value[1]


def _k_points(mesh_n: int, random_num: int, seed: int) -> np.ndarray:
    if mesh_n <= 0:
        raise ValueError("mesh-n must be positive")
    if random_num < 0:
        raise ValueError("random-num must be non-negative")
    mesh = uniform_bz_mesh(mesh_n)
    if random_num == 0:
        return mesh
    rng = np.random.default_rng(seed)
    random_points = rng.uniform(-np.pi, np.pi, size=(random_num, 2))
    return np.vstack([mesh, random_points])


def _diagnosis(max_rel: float, max_hermiticity: float, max_symmetry: float, q_model: float) -> str:
    if q_model == 0.0 and max_rel < 1e-12 and max_hermiticity < 1e-12 and max_symmetry < 1e-12:
        return "pass_q0_machine_precision"
    if max_hermiticity < 1e-12 and max_symmetry < 1e-12:
        return "pass_vertex_audit"
    return "warning_contact_vertex_audit_tolerance"


def run_audit(
    *,
    mesh_n: int,
    random_num: int,
    seed: int,
    q_list: list[float],
    q_angle_list: list[float],
    directions: list[str],
) -> dict[str, np.ndarray]:
    if any(q < 0.0 for q in q_list):
        raise ValueError("q-list values must be non-negative")
    direction_pairs = [_direction_pair(direction) for direction in directions]
    points = _k_points(mesh_n, random_num, seed)
    hopping_terms = normal_state_hopping_terms()
    rows: list[dict[str, object]] = []

    for q_model in q_list:
        for q_angle in q_angle_list:
            qx = float(q_model * np.cos(q_angle))
            qy = float(q_model * np.sin(q_angle))
            for index, (kx_value, ky_value) in enumerate(points):
                kx = float(kx_value)
                ky = float(ky_value)
                contact_by_direction: dict[tuple[str, str], np.ndarray] = {}
                for direction_i, direction_j in {("x", "x"), ("x", "y"), ("y", "x"), ("y", "y")}:
                    contact_by_direction[(direction_i, direction_j)] = peierls_hamiltonian_contact_vertex(
                        kx,
                        ky,
                        qx,
                        qy,
                        direction_i,
                        direction_j,
                        hopping_terms=hopping_terms,
                    )
                xy_yx_error = float(np.linalg.norm(contact_by_direction[("x", "y")] - contact_by_direction[("y", "x")]))
                for direction_i, direction_j in direction_pairs:
                    contact = contact_by_direction[(direction_i, direction_j)]
                    mass = normal_state_mass_operator(kx, ky, direction_i, direction_j)
                    abs_error = float(np.linalg.norm(contact - mass))
                    mass_norm = float(np.linalg.norm(mass))
                    rows.append(
                        {
                            "q_model": float(q_model),
                            "q_angle_rad": float(q_angle),
                            "direction_i": direction_i,
                            "direction_j": direction_j,
                            "k_index": int(index),
                            "kx": kx,
                            "ky": ky,
                            "abs_mass_limit_error": abs_error,
                            "rel_mass_limit_error": abs_error / max(mass_norm, EPS),
                            "mass_norm": mass_norm,
                            "contact_norm": float(np.linalg.norm(contact)),
                            "hermiticity_error": float(np.linalg.norm(contact - contact.conjugate().T)),
                            "index_symmetry_error": xy_yx_error,
                        }
                    )

    return {column: np.array([row[column] for row in rows]) for column in EXPANDED_COLUMNS}


def compact_rows(data: dict[str, np.ndarray], *, quick: bool, expanded_data_written: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = sorted(
        {
            (
                float(data["q_model"][i]),
                float(data["q_angle_rad"][i]),
                str(data["direction_i"][i]),
                str(data["direction_j"][i]),
            )
            for i in range(len(data["q_model"]))
        }
    )
    for q_model, q_angle, direction_i, direction_j in keys:
        mask = (
            np.isclose(data["q_model"].astype(float), q_model)
            & np.isclose(data["q_angle_rad"].astype(float), q_angle)
            & (data["direction_i"] == direction_i)
            & (data["direction_j"] == direction_j)
        )
        rel_values = data["rel_mass_limit_error"][mask].astype(float)
        max_rel = float(np.max(rel_values))
        max_hermiticity = float(np.max(data["hermiticity_error"][mask].astype(float)))
        max_symmetry = float(np.max(data["index_symmetry_error"][mask].astype(float)))
        rows.append(
            {
                "q_model": q_model,
                "q_angle_rad": q_angle,
                "direction_i": direction_i,
                "direction_j": direction_j,
                "num_k_points": int(np.count_nonzero(mask)),
                "max_abs_mass_limit_error": float(np.max(data["abs_mass_limit_error"][mask].astype(float))),
                "max_rel_mass_limit_error": max_rel,
                "median_rel_mass_limit_error": float(np.median(rel_values)),
                "max_hermiticity_error": max_hermiticity,
                "max_index_symmetry_error": max_symmetry,
                "quick_mode": bool(quick),
                "expanded_data_written": bool(expanded_data_written),
                "response_computed": False,
                "conductivity_computed": False,
                "ward_identity_checked": False,
                "casimir_computed": False,
                "not_final_finite_q_contact": True,
                "not_final_finite_q_conductivity": True,
                "not_final_casimir_conclusion": True,
                "diagnosis": _diagnosis(max_rel, max_hermiticity, max_symmetry, q_model),
            }
        )
    return rows


def _write_rows(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_expanded(path: Path, data: dict[str, np.ndarray]) -> None:
    rows = [{column: data[column][index] for column in EXPANDED_COLUMNS} for index in range(len(data["q_model"]))]
    _write_rows(path, EXPANDED_COLUMNS, rows)


def _write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(data)
    np.savez(path, **payload)


def _output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path]:
    compact_csv = output_prefix.with_name(f"{output_prefix.name}_compact").with_suffix(".csv")
    expanded_csv = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".csv")
    expanded_npz = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".npz")
    result_root = output_prefix.parent.parent if output_prefix.parent.name == "data" else output_prefix.parent
    figure_dir = result_root / "figures"
    summary_path = result_root / SUMMARY_NAME
    return compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path


def _plot_metric_by_q(
    compact: list[dict[str, object]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for direction in sorted({str(row["direction_i"]) + str(row["direction_j"]) for row in compact}):
        q_values = sorted({float(row["q_model"]) for row in compact})
        values = []
        for q_model in q_values:
            selected = [
                float(row[metric])
                for row in compact
                if str(row["direction_i"]) + str(row["direction_j"]) == direction
                and np.isclose(float(row["q_model"]), q_model)
            ]
            values.append(max(selected))
        ax.semilogy(q_values, np.maximum(values, EPS), marker="o", label=direction)
    ax.set(xlabel="q_model", ylabel=ylabel, title=title)
    style_publication_axis(ax)
    save_publication_figure(fig, path)
    plt.close(fig)
    return path


def _plot_outputs(compact: list[dict[str, object]], figure_dir: Path) -> list[Path]:
    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_metric_by_q(
            compact,
            metric="max_rel_mass_limit_error",
            ylabel="max relative error vs q=0 mass",
            title="Peierls contact q=0 mass-limit error",
            path=figure_dir / "peierls_contact_q0_mass_error_vs_q.png",
        ),
        _plot_metric_by_q(
            compact,
            metric="max_hermiticity_error",
            ylabel="max Hermiticity error",
            title="Peierls contact Hermiticity audit",
            path=figure_dir / "peierls_contact_hermiticity_error_vs_q.png",
        ),
    ]


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _max_metric(compact: list[dict[str, object]], metric: str, *, q_model: float | None = None) -> float:
    values = [
        float(row[metric])
        for row in compact
        if q_model is None or np.isclose(float(row["q_model"]), q_model)
    ]
    return max(values)


def _q_trend(compact: list[dict[str, object]], metric: str) -> str:
    parts = []
    for q_model in sorted({float(row["q_model"]) for row in compact}):
        parts.append(f"q={_fmt(q_model)}:{_fmt(_max_metric(compact, metric, q_model=q_model))}")
    return ", ".join(parts)


def _summary_lines(
    compact: list[dict[str, object]],
    args: argparse.Namespace,
    command: str,
    compact_csv: Path,
    expanded_csv: Path,
    expanded_npz: Path,
    figure_paths: list[Path],
) -> list[str]:
    max_q0_abs = _max_metric(compact, "max_abs_mass_limit_error", q_model=0.0)
    max_q0_rel = _max_metric(compact, "max_rel_mass_limit_error", q_model=0.0)
    max_hermiticity = _max_metric(compact, "max_hermiticity_error")
    max_symmetry = _max_metric(compact, "max_index_symmetry_error")
    q0_passed = max_q0_rel < 1e-12
    hermiticity_passed = max_hermiticity < 1e-12
    symmetry_passed = max_symmetry < 1e-12
    return [
        "# Peierls contact vertex audit",
        "",
        "This is a finite-q Peierls contact vertex audit.",
        "It is not a Pi_mu_nu response calculation.",
        "It is not conductivity.",
        "It is not Casimir.",
        "This result is a contact vertex-level validation only and is not yet connected to response.",
        "",
        "Formula audited:",
        r"$\Lambda_{ij}^P(k,q)=-\sum_R R_iR_j t_R e^{i k\cdot R}\,\mathrm{sinc}(q\cdot R/2)^2$.",
        "The minus sign follows from the existing convention H0(k)=sum_R t_R exp(i k.R), so q=0 gives d2H0/dk_i dk_j.",
        "The formula comes from the second-order Peierls phase expansion along the same straight-bond hopping path.",
        "",
        f"run_command = `{command}`",
        f"quick_mode={bool(args.quick)}",
        f"expanded_data_written={bool(args.write_expanded_data)}",
        f"mesh_n={int(args.mesh_n)}",
        f"random_num={int(args.random_num)}",
        f"random_seed={int(args.seed)}",
        f"q_list = {' '.join(_fmt(float(q)) for q in args.q_list)}",
        f"q_angle_list = {' '.join(_fmt(float(a)) for a in args.q_angle_list)}",
        f"directions = {' '.join(args.directions)}",
        "",
        "## Audit status",
        f"- q0_mass_limit_passed = {q0_passed}",
        f"- max_abs_q0_mass_error = {_fmt(max_q0_abs)}",
        f"- max_rel_q0_mass_error = {_fmt(max_q0_rel)}",
        f"- hermiticity_passed = {hermiticity_passed}",
        f"- max_hermiticity_error = {_fmt(max_hermiticity)}",
        f"- xy_yx_index_symmetry_passed = {symmetry_passed}",
        f"- max_index_symmetry_error = {_fmt(max_symmetry)}",
        "",
        "## q_model trends",
        f"- max relative mass-limit error: {_q_trend(compact, 'max_rel_mass_limit_error')}",
        f"- max Hermiticity error: {_q_trend(compact, 'max_hermiticity_error')}",
        f"- max xy/yx index-symmetry error: {_q_trend(compact, 'max_index_symmetry_error')}",
        "",
        "## Scope flags",
        "response_computed=False",
        "conductivity_computed=False",
        "ward_identity_checked=False",
        "casimir_computed=False",
        "not_final_finite_q_contact=True",
        "not_final_finite_q_conductivity=True",
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
    parser.add_argument("--mesh-n", type=int, default=DEFAULT_MESH_N)
    parser.add_argument("--random-num", type=int, default=DEFAULT_RANDOM_NUM)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--q-list", nargs="+", type=float, default=list(DEFAULT_Q_LIST))
    parser.add_argument("--q-angle-list", nargs="+", type=_angle_expression, default=list(DEFAULT_Q_ANGLE_LIST))
    parser.add_argument("--directions", nargs="+", choices=DEFAULT_DIRECTIONS, default=list(DEFAULT_DIRECTIONS))
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--write-expanded-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.mesh_n = QUICK_MESH_N
        args.random_num = QUICK_RANDOM_NUM
        args.q_list = list(QUICK_Q_LIST)
        args.q_angle_list = list(QUICK_Q_ANGLE_LIST)
    data = run_audit(
        mesh_n=int(args.mesh_n),
        random_num=int(args.random_num),
        seed=int(args.seed),
        q_list=list(args.q_list),
        q_angle_list=list(args.q_angle_list),
        directions=list(args.directions),
    )
    compact = compact_rows(data, quick=bool(args.quick), expanded_data_written=bool(args.write_expanded_data))
    compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path = _output_paths(args.output_prefix)
    _write_rows(compact_csv, COMPACT_COLUMNS, compact)
    if args.write_expanded_data:
        _write_expanded(expanded_csv, data)
        _write_npz(expanded_npz, data)
    figure_paths = _plot_outputs(compact, figure_dir)
    command = "python " + " ".join(shlex.quote(value) for value in sys.argv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(_summary_lines(compact, args, command, compact_csv, expanded_csv, expanded_npz, figure_paths)) + "\n",
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
