#!/usr/bin/env python3
"""Diagnose vertex-level Ward identity for Peierls finite-q current vertices."""

from __future__ import annotations

import argparse
import ast
import csv
import os
from pathlib import Path
import re
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.conductivity import uniform_bz_mesh  # noqa: E402
from lno327.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402
from lno327.tb_fourier import normal_state_hopping_terms, peierls_vertex_ward_residual  # noqa: E402

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "peierls_vertex_ward_identity"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "peierls_vertex_ward_identity"
SUMMARY_NAME = "summary.md"
EPS = 1e-300

DEFAULT_Q_LIST = (0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
DEFAULT_Q_ANGLE_LIST = (0.0, np.pi / 8.0, np.pi / 4.0, 3.0 * np.pi / 8.0, np.pi / 2.0)
DEFAULT_SIGN_CONVENTIONS = ("plus", "minus")
DEFAULT_MESH_N = 16
DEFAULT_RANDOM_NUM = 32

QUICK_Q_LIST = (0.001, 0.1, 1.0)
QUICK_Q_ANGLE_LIST = (0.0, np.pi / 4.0)
QUICK_MESH_N = 8
QUICK_RANDOM_NUM = 8

COMPACT_COLUMNS = (
    "sign_convention",
    "q_model",
    "q_angle_rad",
    "num_k_points",
    "max_abs_vertex_ward_error",
    "max_rel_vertex_ward_error",
    "median_rel_vertex_ward_error",
    "max_lhs_norm",
    "max_rhs_norm",
    "quick_mode",
    "expanded_data_written",
    "response_computed",
    "conductivity_computed",
    "ward_identity_checked",
    "casimir_computed",
    "not_final_casimir_conclusion",
    "diagnosis",
)

EXPANDED_COLUMNS = (
    "sign_convention",
    "q_model",
    "q_angle_rad",
    "k_index",
    "kx",
    "ky",
    "abs_vertex_ward_error",
    "rel_vertex_ward_error",
    "lhs_norm",
    "rhs_norm",
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


def run_diagnostic(
    *,
    mesh_n: int,
    random_num: int,
    seed: int,
    q_list: list[float],
    q_angle_list: list[float],
    sign_conventions: list[str],
) -> dict[str, np.ndarray]:
    if any(q < 0.0 for q in q_list):
        raise ValueError("q-list values must be non-negative")
    for sign in sign_conventions:
        if sign not in {"plus", "minus"}:
            raise ValueError("sign conventions must be plus or minus")
    points = _k_points(mesh_n, random_num, seed)
    hopping_terms = normal_state_hopping_terms()
    rows: list[dict[str, object]] = []
    for sign in sign_conventions:
        for q_model in q_list:
            for q_angle in q_angle_list:
                qx = float(q_model * np.cos(q_angle))
                qy = float(q_model * np.sin(q_angle))
                for index, (kx, ky) in enumerate(points):
                    abs_error, rel_error, lhs_norm, rhs_norm = peierls_vertex_ward_residual(
                        float(kx),
                        float(ky),
                        qx,
                        qy,
                        hopping_terms=hopping_terms,
                        sign_convention=sign,
                    )
                    rows.append(
                        {
                            "sign_convention": sign,
                            "q_model": float(q_model),
                            "q_angle_rad": float(q_angle),
                            "k_index": int(index),
                            "kx": float(kx),
                            "ky": float(ky),
                            "abs_vertex_ward_error": abs_error,
                            "rel_vertex_ward_error": rel_error,
                            "lhs_norm": lhs_norm,
                            "rhs_norm": rhs_norm,
                        }
                    )
    return {column: np.array([row[column] for row in rows]) for column in EXPANDED_COLUMNS}


def _diagnosis(max_rel_error: float) -> str:
    if max_rel_error < 1e-12:
        return "pass_machine_precision"
    if max_rel_error < 1e-8:
        return "pass"
    return "warning_sign_or_vertex_convention_mismatch"


def compact_rows(data: dict[str, np.ndarray], *, quick: bool, expanded_data_written: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = sorted(
        {
            (str(data["sign_convention"][i]), float(data["q_model"][i]), float(data["q_angle_rad"][i]))
            for i in range(len(data["q_model"]))
        }
    )
    for sign, q_model, q_angle in keys:
        mask = (
            (data["sign_convention"] == sign)
            & np.isclose(data["q_model"].astype(float), q_model)
            & np.isclose(data["q_angle_rad"].astype(float), q_angle)
        )
        rel_values = data["rel_vertex_ward_error"][mask].astype(float)
        max_rel = float(np.max(rel_values))
        rows.append(
            {
                "sign_convention": sign,
                "q_model": q_model,
                "q_angle_rad": q_angle,
                "num_k_points": int(np.count_nonzero(mask)),
                "max_abs_vertex_ward_error": float(np.max(data["abs_vertex_ward_error"][mask].astype(float))),
                "max_rel_vertex_ward_error": max_rel,
                "median_rel_vertex_ward_error": float(np.median(rel_values)),
                "max_lhs_norm": float(np.max(data["lhs_norm"][mask].astype(float))),
                "max_rhs_norm": float(np.max(data["rhs_norm"][mask].astype(float))),
                "quick_mode": bool(quick),
                "expanded_data_written": bool(expanded_data_written),
                "response_computed": False,
                "conductivity_computed": False,
                "ward_identity_checked": True,
                "casimir_computed": False,
                "not_final_casimir_conclusion": True,
                "diagnosis": _diagnosis(max_rel),
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


def _output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path]:
    compact_csv = output_prefix.with_name(f"{output_prefix.name}_compact").with_suffix(".csv")
    expanded_csv = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".csv")
    expanded_npz = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".npz")
    result_root = output_prefix.parent.parent if output_prefix.parent.name == "data" else output_prefix.parent
    figure_dir = result_root / "figures"
    summary_path = result_root / SUMMARY_NAME
    return compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path


def _plot_outputs(compact: list[dict[str, object]], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for sign in sorted({str(row["sign_convention"]) for row in compact}):
        q_values = sorted({float(row["q_model"]) for row in compact if row["sign_convention"] == sign})
        max_errors = []
        for q_model in q_values:
            values = [
                float(row["max_rel_vertex_ward_error"])
                for row in compact
                if row["sign_convention"] == sign and np.isclose(float(row["q_model"]), q_model)
            ]
            max_errors.append(max(values))
        ax.semilogy(q_values, np.maximum(max_errors, EPS), marker="o", label=sign)
    ax.set(
        xlabel="q_model",
        ylabel="max relative vertex Ward error",
        title="Peierls vertex-level Ward check",
    )
    style_publication_axis(ax)
    path = figure_dir / "peierls_vertex_ward_error_vs_q.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    return [path]


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _best_sign_summary(compact: list[dict[str, object]]) -> tuple[str, dict[str, float]]:
    by_sign: dict[str, dict[str, float]] = {}
    for sign in sorted({str(row["sign_convention"]) for row in compact}):
        rel = [float(row["max_rel_vertex_ward_error"]) for row in compact if row["sign_convention"] == sign]
        med = [float(row["median_rel_vertex_ward_error"]) for row in compact if row["sign_convention"] == sign]
        by_sign[sign] = {"max": max(rel), "median": float(np.median(med))}
    best = min(by_sign, key=lambda item: by_sign[item]["max"])
    return best, by_sign


def _summary_lines(
    compact: list[dict[str, object]],
    args: argparse.Namespace,
    command: str,
    compact_csv: Path,
    expanded_csv: Path,
    expanded_npz: Path,
    figure_paths: list[Path],
) -> list[str]:
    best, by_sign = _best_sign_summary(compact)
    lines = [
        "# Peierls vertex Ward identity diagnostic",
        "",
        "This is a vertex-level Peierls current vertex Ward identity check.",
        "It is not a Pi_mu_nu response calculation.",
        "It is not conductivity.",
        "It is not Casimir.",
        "The contact term is not involved in this vertex-level check.",
        "",
        "Formula checked:",
        r"$\Gamma_i^P(k,q)= i \sum_R R_i t_R e^{i k\cdot R}\,\mathrm{sinc}(q\cdot R/2)$.",
        r"$q_x\Gamma_x^P+q_y\Gamma_y^P=H_0(k+q/2)-H_0(k-q/2)$.",
        "",
        f"run_command = `{command}`",
        f"quick_mode={bool(args.quick)}",
        f"expanded_data_written={bool(args.write_expanded_data)}",
        f"mesh_n={int(args.mesh_n)}",
        f"random_num={int(args.random_num)}",
        f"random_seed={int(args.seed)}",
        "",
        "## Sign convention summary",
        f"- best sign_convention = {best}",
    ]
    for sign, values in by_sign.items():
        lines.append(
            f"- {sign}: max relative error = {_fmt(values['max'])}, "
            f"median relative error = {_fmt(values['median'])}"
        )
    lines.extend(
        [
            "",
            "## Scope flags",
            "response_computed=False",
            "conductivity_computed=False",
            "ward_identity_checked=True",
            "casimir_computed=False",
            "not_final_casimir_conclusion=True",
            "",
            "## Output files",
            f"- compact CSV: {compact_csv}",
            f"- expanded_data_written={bool(args.write_expanded_data)}",
        ]
    )
    if args.write_expanded_data:
        lines.extend([f"- expanded CSV: {expanded_csv}", f"- expanded NPZ: {expanded_npz}"])
    else:
        lines.append("- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.")
    lines.extend(f"- figure: {path}" for path in figure_paths)
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-n", type=int, default=DEFAULT_MESH_N)
    parser.add_argument("--random-num", type=int, default=DEFAULT_RANDOM_NUM)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--q-list", nargs="+", type=float, default=list(DEFAULT_Q_LIST))
    parser.add_argument("--q-angle-list", nargs="+", type=_angle_expression, default=list(DEFAULT_Q_ANGLE_LIST))
    parser.add_argument("--sign-conventions", nargs="+", choices=("plus", "minus"), default=list(DEFAULT_SIGN_CONVENTIONS))
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
    data = run_diagnostic(
        mesh_n=int(args.mesh_n),
        random_num=int(args.random_num),
        seed=int(args.seed),
        q_list=list(args.q_list),
        q_angle_list=list(args.q_angle_list),
        sign_conventions=list(args.sign_conventions),
    )
    compact = compact_rows(data, quick=bool(args.quick), expanded_data_written=bool(args.write_expanded_data))
    compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path = _output_paths(args.output_prefix)
    _write_rows(compact_csv, COMPACT_COLUMNS, compact)
    if args.write_expanded_data:
        _write_expanded(expanded_csv, data)
        expanded_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(expanded_npz, **data)
    figure_paths = _plot_outputs(compact, figure_dir)
    command = "python " + " ".join(shlex.quote(value) for value in sys.argv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(_summary_lines(compact, args, command, compact_csv, expanded_csv, expanded_npz, figure_paths))
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
