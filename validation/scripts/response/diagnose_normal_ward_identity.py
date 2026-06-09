#!/usr/bin/env python3
"""Diagnose normal-state Pi_mu_nu Ward-identity residuals."""

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

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402
from lno327.ward_response import normal_density_current_response_imag_axis, ward_errors  # noqa: E402

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "normal_ward_identity"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "normal_ward_identity"
SUMMARY_NAME = "summary.md"
EPS = 1e-300

DEFAULT_MATSUBARA_N_LIST = (1, 2, 4)
DEFAULT_TEMPERATURE_K = 30.0
DEFAULT_Q_LIST = (0.001, 0.005, 0.01, 0.05, 0.1)
DEFAULT_Q_ANGLE_LIST = (0.0, np.pi / 4.0, np.pi / 2.0)
DEFAULT_NK_LIST = (8, 12, 16)
DEFAULT_DEGENERACY_TOL_EV = 1e-10
DEFAULT_VERTEX_SCHEMES = ("midpoint", "peierls")

QUICK_MATSUBARA_N_LIST = (1,)
QUICK_Q_LIST = (0.001, 0.01, 0.1)
QUICK_Q_ANGLE_LIST = (0.0, np.pi / 4.0)
QUICK_NK_LIST = (8,)

COMPACT_COLUMNS = (
    "vertex_scheme",
    "current_vertex_sign_convention",
    "density_vertex_scheme",
    "contact_scheme",
    "matsubara_n",
    "omega_eV",
    "nk",
    "q_model",
    "q_angle",
    "qx_model",
    "qy_model",
    "left_ward_error",
    "right_ward_error",
    "max_ward_error",
    "density_current_included",
    "current_current_included",
    "diamagnetic_contact_included",
    "normal_state_only",
    "bdg_computed",
    "conductivity_computed",
    "casimir_computed",
    "not_final_casimir_conclusion",
    "not_final_finite_q_conductivity",
    "diagnosis",
)

EXPANDED_COLUMNS = COMPACT_COLUMNS + (
    "Pi_00",
    "Pi_0x",
    "Pi_0y",
    "Pi_x0",
    "Pi_xx",
    "Pi_xy",
    "Pi_y0",
    "Pi_yx",
    "Pi_yy",
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


def _diagnosis(max_error: float, vertex_scheme: str) -> str:
    prefix = f"{vertex_scheme}_"
    if max_error < 1e-6:
        return prefix + "prototype_residual_small"
    if max_error < 1e-3:
        return prefix + "prototype_residual_moderate"
    return prefix + "warning_large_ward_residual_contact_or_vertex_not_closed"


def run_diagnostic(
    *,
    matsubara_n_list: list[int],
    temperature_K: float,
    q_list: list[float],
    q_angle_list: list[float],
    nk_list: list[int],
    degeneracy_tol_eV: float,
    vertex_schemes: list[str],
) -> dict[str, np.ndarray]:
    if any(n < 1 for n in matsubara_n_list):
        raise ValueError("matsubara-n-list must contain only n >= 1")
    if any(q < 0.0 for q in q_list):
        raise ValueError("q-list values must be non-negative")
    if any(nk <= 0 for nk in nk_list):
        raise ValueError("nk-list values must be positive")
    if temperature_K < 0.0:
        raise ValueError("temperature must be non-negative")
    if degeneracy_tol_eV <= 0.0:
        raise ValueError("degeneracy-tol must be positive")
    for scheme in vertex_schemes:
        if scheme not in {"midpoint", "peierls"}:
            raise ValueError("vertex schemes must be midpoint or peierls")

    rows: list[dict[str, object]] = []
    for nk in nk_list:
        mesh = uniform_bz_mesh(nk)
        weights = k_weights(mesh)
        for matsubara_n in matsubara_n_list:
            omega_eV = bosonic_matsubara_energy_eV(matsubara_n, temperature_K)
            config = KuboConfig.from_kelvin(
                omega_eV=omega_eV,
                temperature_K=temperature_K,
                eta_eV=degeneracy_tol_eV,
                output_si=False,
            )
            for vertex_scheme in vertex_schemes:
                for q_model in q_list:
                    for q_angle in q_angle_list:
                        qx = float(q_model * np.cos(q_angle))
                        qy = float(q_model * np.sin(q_angle))
                        matrix = normal_density_current_response_imag_axis(
                            mesh,
                            config,
                            np.array([qx, qy], dtype=float),
                            weights,
                            vertex_scheme=vertex_scheme,
                        )
                        left_error, right_error, max_error = ward_errors(matrix, omega_eV, np.array([qx, qy], dtype=float))
                        rows.append(
                            {
                                "vertex_scheme": vertex_scheme,
                                "current_vertex_sign_convention": "plus" if vertex_scheme == "peierls" else "not_applicable",
                                "density_vertex_scheme": "identity_4_orbitals_shared_in_plane_position",
                                "contact_scheme": "none",
                                "matsubara_n": int(matsubara_n),
                                "omega_eV": float(omega_eV),
                                "nk": int(nk),
                                "q_model": float(q_model),
                                "q_angle": float(q_angle),
                                "qx_model": qx,
                                "qy_model": qy,
                                "left_ward_error": left_error,
                                "right_ward_error": right_error,
                                "max_ward_error": max_error,
                                "density_current_included": True,
                                "current_current_included": True,
                                "diamagnetic_contact_included": False,
                                "normal_state_only": True,
                                "bdg_computed": False,
                                "conductivity_computed": False,
                                "casimir_computed": False,
                                "not_final_casimir_conclusion": True,
                                "not_final_finite_q_conductivity": True,
                                "diagnosis": _diagnosis(max_error, vertex_scheme),
                                "Pi_00": complex(matrix[0, 0]),
                                "Pi_0x": complex(matrix[0, 1]),
                                "Pi_0y": complex(matrix[0, 2]),
                                "Pi_x0": complex(matrix[1, 0]),
                                "Pi_xx": complex(matrix[1, 1]),
                                "Pi_xy": complex(matrix[1, 2]),
                                "Pi_y0": complex(matrix[2, 0]),
                                "Pi_yx": complex(matrix[2, 1]),
                                "Pi_yy": complex(matrix[2, 2]),
                            }
                        )

    return {column: np.array([row[column] for row in rows]) for column in EXPANDED_COLUMNS}


def _write_csv(path: Path, data: dict[str, np.ndarray], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for index in range(len(data["matsubara_n"])):
            writer.writerow({column: data[column][index] for column in columns})


def _write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = dict(data)
    np.savez(path, **payload)


def _output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path]:
    compact_csv = output_prefix.with_name(f"{output_prefix.name}_compact").with_suffix(".csv")
    expanded_csv = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".csv")
    expanded_npz = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".npz")
    result_root = output_prefix.parent.parent if output_prefix.parent.name == "data" else output_prefix.parent
    figure_dir = result_root / "figures"
    summary_path = result_root / SUMMARY_NAME
    return compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path


def _scheme_q_max(data: dict[str, np.ndarray], field: str, scheme: str, q_model: float) -> float:
    mask = (data["vertex_scheme"] == scheme) & np.isclose(data["q_model"].astype(float), q_model)
    return float(np.max(data[field][mask].astype(float)))


def _plot_outputs(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    q_values = np.array(sorted(set(float(q) for q in data["q_model"])))

    schemes = sorted(set(str(item) for item in data["vertex_scheme"]))

    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scheme in schemes:
        max_error = [_scheme_q_max(data, "max_ward_error", scheme, q_model) for q_model in q_values]
        ax.semilogy(q_values, np.maximum(max_error, EPS), marker="o", label=scheme)
    ax.set(xlabel="q_model", ylabel="max Ward error", title="Normal-state Ward residual by vertex scheme")
    style_publication_axis(ax)
    path = figure_dir / "ward_error_vs_q.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scheme in schemes:
        left_error = [_scheme_q_max(data, "left_ward_error", scheme, q_model) for q_model in q_values]
        right_error = [_scheme_q_max(data, "right_ward_error", scheme, q_model) for q_model in q_values]
        ax.semilogy(q_values, np.maximum(left_error, EPS), marker="o", label=f"{scheme} left")
        ax.semilogy(q_values, np.maximum(right_error, EPS), marker="s", linestyle="--", label=f"{scheme} right")
    ax.set(xlabel="q_model", ylabel="Ward error", title="Left/right Ward prototype residuals")
    style_publication_axis(ax)
    path = figure_dir / "left_right_ward_error_vs_q.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scheme in schemes:
        max_error = [_scheme_q_max(data, "max_ward_error", scheme, q_model) for q_model in q_values]
        ax.semilogy(q_values, np.maximum(max_error, EPS), marker="o", label=scheme)
    ax.set(xlabel="q_model", ylabel="max Ward error", title="Midpoint vs Peierls Ward residual")
    style_publication_axis(ax)
    path = figure_dir / "ward_error_vs_q_by_vertex_scheme.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    return paths


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _summary_lines(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    compact_csv: Path,
    expanded_csv: Path,
    expanded_npz: Path,
    figure_paths: list[Path],
) -> list[str]:
    schemes = sorted(set(str(item) for item in data["vertex_scheme"]))
    scheme_summary: dict[str, dict[str, float]] = {}
    for scheme in schemes:
        mask = data["vertex_scheme"] == scheme
        scheme_summary[scheme] = {
            "left": float(np.max(data["left_ward_error"][mask].astype(float))),
            "right": float(np.max(data["right_ward_error"][mask].astype(float))),
            "max": float(np.max(data["max_ward_error"][mask].astype(float))),
        }
    improvement_text = "Peierls current vertex comparison unavailable."
    if "midpoint" in scheme_summary and "peierls" in scheme_summary:
        midpoint = scheme_summary["midpoint"]["max"]
        peierls = scheme_summary["peierls"]["max"]
        ratio = midpoint / max(peierls, EPS)
        if ratio > 1.01:
            improvement_text = f"Peierls current vertex lowers the max Ward residual by a factor of {_fmt(ratio)}."
        else:
            improvement_text = (
                f"Peierls current vertex does not materially lower the max Ward residual in this prototype "
                f"(midpoint/Peierls factor = {_fmt(ratio)}); possible reasons include the missing contact term "
                "or remaining vertex/contact closure gaps."
            )
    lines = [
        "# Normal-state Pi_mu_nu Ward identity prototype",
        "",
        "This is a normal-state Pi_mu_nu Ward diagnostic.",
        "It compares midpoint velocity and Peierls current vertex schemes.",
        "It is not conductivity and not a reflection/Casimir input.",
        "The contact term is not included.",
        (
            "Large Ward residuals may reflect finite-q vertex/contact-term closure gaps, "
            "not a material conclusion."
        ),
        "",
        f"run_command = `{command}`",
        f"quick_mode={bool(args.quick)}",
        f"expanded_data_written={bool(args.write_expanded_data)}",
        "density_current_included=True",
        "current_current_included=True",
        "diamagnetic_contact_included=False",
        "contact_scheme=none",
        "normal_state_only=True",
        "bdg_computed=False",
        "conductivity_computed=False",
        "casimir_computed=False",
        "not_final_casimir_conclusion=True",
        "not_final_finite_q_conductivity=True",
        "",
        "## Parameter grid",
        f"- vertex_schemes = {' '.join(args.vertex_schemes)}",
        f"- matsubara_n_list = {' '.join(str(int(n)) for n in args.matsubara_n_list)}",
        f"- temperature_K = {_fmt(float(args.temperature))}",
        f"- q_list = {' '.join(_fmt(float(q)) for q in args.q_list)}",
        f"- q_angle_list = {' '.join(_fmt(float(a)) for a in args.q_angle_list)}",
        f"- nk_list = {' '.join(str(int(nk)) for nk in args.nk_list)}",
        f"- degeneracy_tol_eV = {_fmt(float(args.degeneracy_tol))}",
        "",
        "## Ward residual summary by vertex scheme",
    ]
    for scheme in schemes:
        lines.extend(
            [
                f"- {scheme}: max left Ward error = {_fmt(scheme_summary[scheme]['left'])}",
                f"- {scheme}: max right Ward error = {_fmt(scheme_summary[scheme]['right'])}",
                f"- {scheme}: max Ward error = {_fmt(scheme_summary[scheme]['max'])}",
            ]
        )
    lines.extend(
        [
        "",
        "## q_model max-error trend",
    ]
    )
    for scheme in schemes:
        trend = []
        for q_model in sorted(set(float(q) for q in data["q_model"])):
            trend.append(f"q={_fmt(q_model)}:{_fmt(_scheme_q_max(data, 'max_ward_error', scheme, q_model))}")
        lines.append(f"- {scheme}: " + ", ".join(trend))
    lines.extend(
        [
        "",
        "## Comparison",
        f"- {improvement_text}",
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
    )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matsubara-n-list", nargs="+", type=int, default=list(DEFAULT_MATSUBARA_N_LIST))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--q-list", nargs="+", type=float, default=list(DEFAULT_Q_LIST))
    parser.add_argument("--q-angle-list", nargs="+", type=_angle_expression, default=list(DEFAULT_Q_ANGLE_LIST))
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--degeneracy-tol", type=float, default=DEFAULT_DEGENERACY_TOL_EV)
    parser.add_argument("--vertex-schemes", nargs="+", choices=("midpoint", "peierls"), default=list(DEFAULT_VERTEX_SCHEMES))
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--write-expanded-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.matsubara_n_list = list(QUICK_MATSUBARA_N_LIST)
        args.q_list = list(QUICK_Q_LIST)
        args.q_angle_list = list(QUICK_Q_ANGLE_LIST)
        args.nk_list = list(QUICK_NK_LIST)
        args.temperature = DEFAULT_TEMPERATURE_K
    data = run_diagnostic(
        matsubara_n_list=list(args.matsubara_n_list),
        temperature_K=float(args.temperature),
        q_list=list(args.q_list),
        q_angle_list=list(args.q_angle_list),
        nk_list=list(args.nk_list),
        degeneracy_tol_eV=float(args.degeneracy_tol),
        vertex_schemes=list(args.vertex_schemes),
    )
    compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path = _output_paths(args.output_prefix)
    _write_csv(compact_csv, data, COMPACT_COLUMNS)
    if args.write_expanded_data:
        _write_csv(expanded_csv, data, EXPANDED_COLUMNS)
        _write_npz(expanded_npz, data)
    figure_paths = _plot_outputs(data, figure_dir)
    command = "python " + " ".join(shlex.quote(value) for value in sys.argv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(_summary_lines(data, args, command, compact_csv, expanded_csv, expanded_npz, figure_paths)) + "\n",
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
