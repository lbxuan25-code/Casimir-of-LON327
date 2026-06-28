#!/usr/bin/env python3
"""Diagnose normal finite-q current-current kernel convergence."""

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.nonlocal_response import (  # noqa: E402
    c4_covariance_error,
    normal_current_current_kernel_imag_axis,
)
from validation.lib.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

DEFAULT_MATSUBARA_N_LIST = (1,)
DEFAULT_Q_LIST = (0.0, 1e-3)
DEFAULT_Q_ANGLE_LIST = (0.0, np.pi / 4.0)
DEFAULT_NK_LIST = (8,)
DEFAULT_TEMPERATURE_K = 30.0
DEFAULT_DEGENERACY_TOL_EV = 1e-10
OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "normal_finite_q_kernel_convergence"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "normal_finite_q_kernel_convergence"
EPS = 1e-300

REQUIRED_COLUMNS = (
    "matsubara_n",
    "omega_eV",
    "q_model",
    "q_angle",
    "qx_model",
    "qy_model",
    "nk",
    "temperature_K",
    "K_xx",
    "K_yy",
    "K_xy",
    "K_yx",
    "K_trace",
    "K_anisotropy_delta",
    "offdiag_ratio",
    "K0_xx",
    "K0_yy",
    "K0_xy",
    "K0_yx",
    "q_to_zero_same_interface_error",
    "c4_covariance_error",
    "finite_momentum_resolved",
    "normal_state",
    "current_current_kernel_only",
    "midpoint_vertex_approximation",
    "not_peierls_exact_vertex",
    "ward_identity_not_yet_checked",
    "not_final_casimir_input",
)


def _angle_expression(value: str) -> float:
    """Parse a numeric angle or a simple expression containing pi."""

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


def _matrix_components(prefix: str, matrix: np.ndarray) -> dict[str, complex]:
    return {
        f"{prefix}_xx": complex(matrix[0, 0]),
        f"{prefix}_yy": complex(matrix[1, 1]),
        f"{prefix}_xy": complex(matrix[0, 1]),
        f"{prefix}_yx": complex(matrix[1, 0]),
    }


def _relative_error(matrix: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(matrix - reference) / max(float(np.linalg.norm(reference)), EPS))


def _anisotropy(matrix: np.ndarray) -> complex:
    denominator = matrix[0, 0] + matrix[1, 1]
    if abs(denominator) <= EPS:
        return 0.0 + 0.0j
    return complex((matrix[0, 0] - matrix[1, 1]) / denominator)


def _offdiag_ratio(matrix: np.ndarray) -> float:
    numerator = float(np.linalg.norm([matrix[0, 1], matrix[1, 0]]))
    denominator = float(np.linalg.norm([matrix[0, 0], matrix[1, 1]]))
    return float(numerator / max(denominator, EPS))


def run_diagnostic(
    *,
    matsubara_n_list: list[int],
    q_list: list[float],
    q_angle_list: list[float],
    nk_list: list[int],
    temperature_K: float,
    degeneracy_tol_eV: float,
) -> dict[str, np.ndarray]:
    """Evaluate requested K(i omega_n, q) convergence cases."""

    if not matsubara_n_list or not q_list or not q_angle_list or not nk_list:
        raise ValueError("matsubara-n-list, q-list, q-angle-list, and nk-list must not be empty")
    if any(n < 1 for n in matsubara_n_list):
        raise ValueError("matsubara-n-list must contain only n >= 1")
    if any(q < 0.0 for q in q_list):
        raise ValueError("q-list values must be non-negative")
    if any(nk <= 0 for nk in nk_list):
        raise ValueError("nk-list values must be positive")
    if temperature_K < 0.0:
        raise ValueError("temperature must be non-negative")
    if degeneracy_tol_eV <= 0.0:
        raise ValueError("degeneracy tolerance must be positive")

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
            K0 = normal_current_current_kernel_imag_axis(
                mesh,
                config,
                np.zeros(2, dtype=float),
                weights,
            )
            for q_model in q_list:
                for q_angle in q_angle_list:
                    q = np.array(
                        [q_model * np.cos(q_angle), q_model * np.sin(q_angle)],
                        dtype=float,
                    )
                    matrix = normal_current_current_kernel_imag_axis(
                        mesh,
                        config,
                        q,
                        weights,
                    )
                    rotated_q = np.array([-q[1], q[0]], dtype=float)
                    rotated_matrix = normal_current_current_kernel_imag_axis(
                        mesh,
                        config,
                        rotated_q,
                        weights,
                    )
                    trace = complex(matrix[0, 0] + matrix[1, 1])
                    rows.append(
                        {
                            "matsubara_n": int(matsubara_n),
                            "omega_eV": float(omega_eV),
                            "q_model": float(q_model),
                            "q_angle": float(q_angle),
                            "qx_model": float(q[0]),
                            "qy_model": float(q[1]),
                            "nk": int(nk),
                            "temperature_K": float(temperature_K),
                            **_matrix_components("K", matrix),
                            "K_trace": trace,
                            "K_anisotropy_delta": _anisotropy(matrix),
                            "offdiag_ratio": _offdiag_ratio(matrix),
                            **_matrix_components("K0", K0),
                            "q_to_zero_same_interface_error": _relative_error(matrix, K0),
                            "c4_covariance_error": c4_covariance_error(matrix, rotated_matrix),
                            "finite_momentum_resolved": True,
                            "normal_state": True,
                            "current_current_kernel_only": True,
                            "midpoint_vertex_approximation": True,
                            "not_peierls_exact_vertex": True,
                            "ward_identity_not_yet_checked": True,
                            "not_final_casimir_input": True,
                        }
                    )

    return {column: np.array([row[column] for row in rows]) for column in REQUIRED_COLUMNS}


def _write_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for index in range(len(data["matsubara_n"])):
            writer.writerow({column: data[column][index] for column in REQUIRED_COLUMNS})


def _plot_outputs(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    selectors = sorted(set(zip(data["matsubara_n"], data["nk"], strict=True)))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for matsubara_n, nk in selectors:
        mask = (
            (data["matsubara_n"] == matsubara_n)
            & (data["nk"] == nk)
            & np.isclose(data["q_angle"], 0.0)
        )
        order = np.argsort(data["q_model"][mask])
        ax.plot(
            data["q_model"][mask][order],
            np.real(data["K_trace"][mask][order]),
            marker="o",
            label=rf"$n={int(matsubara_n)}$, $N_k={int(nk)}$",
        )
    ax.set(
        xlabel=r"$q$ (model units)",
        ylabel=r"$\mathrm{Re}\,\mathrm{Tr}\,K$",
        title="Normal current-current kernel trace",
    )
    style_publication_axis(ax)
    path = figure_dir / "normal_finite_q_kernel_trace.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for matsubara_n, nk in selectors:
        mask = (
            (data["matsubara_n"] == matsubara_n)
            & (data["nk"] == nk)
            & np.isclose(data["q_angle"], 0.0)
            & (data["q_model"] > 0.0)
        )
        order = np.argsort(data["q_model"][mask])
        ax.semilogy(
            data["q_model"][mask][order],
            np.maximum(data["q_to_zero_same_interface_error"][mask][order], EPS),
            marker="o",
            label=rf"$n={int(matsubara_n)}$, $N_k={int(nk)}$",
        )
    ax.set(
        xlabel=r"$q$ (model units)",
        ylabel=r"$\|K(q)-K(0)\|/\|K(0)\|$",
        title=r"$q\to0$ same-interface kernel error",
    )
    style_publication_axis(ax)
    path = figure_dir / "q_to_zero_same_interface_error.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for matsubara_n, nk in selectors:
        mask = (
            (data["matsubara_n"] == matsubara_n)
            & (data["nk"] == nk)
            & np.isclose(data["q_angle"], 0.0)
        )
        order = np.argsort(data["q_model"][mask])
        ax.semilogy(
            data["q_model"][mask][order],
            np.maximum(data["c4_covariance_error"][mask][order], EPS),
            marker="o",
            label=rf"$n={int(matsubara_n)}$, $N_k={int(nk)}$",
        )
    ax.set(
        xlabel=r"$q$ (model units)",
        ylabel=r"$C_4$ covariance error",
        title=r"Normal finite-q $C_4$ covariance",
    )
    style_publication_axis(ax)
    path = figure_dir / "c4_covariance_error.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    return paths


def _summary_lines(
    data: dict[str, np.ndarray],
    command: str,
    quick: bool,
) -> list[str]:
    q_zero = np.isclose(data["q_model"], 0.0)
    q_nonzero = ~q_zero
    smallest_q_error = float("nan")
    smallest_q = float("nan")
    if np.any(q_nonzero):
        smallest_q = float(np.min(data["q_model"][q_nonzero]))
        smallest_q_error = float(
            np.max(data["q_to_zero_same_interface_error"][np.isclose(data["q_model"], smallest_q)])
        )
    max_q0_same_interface_error = float(np.max(data["q_to_zero_same_interface_error"][q_zero]))
    max_c4_error = float(np.max(data["c4_covariance_error"]))
    finite = bool(
        np.all(np.isfinite(np.real(data["K_xx"])))
        and np.all(np.isfinite(np.real(data["K_yy"])))
        and np.all(np.isfinite(np.real(data["K_xy"])))
        and np.all(np.isfinite(np.real(data["K_yx"])))
    )
    return [
        "# Normal finite-q current-current kernel convergence diagnostic",
        "",
        "本脚本只测试 normal finite-q current-current kernel (K)。",
        "q=0 与 q!=0 都通过同一 K 接口 normal_current_current_kernel_imag_axis 计算。",
        "q=0 分支不再调用 public local sigma。",
        "public sigma 不进入 K 字段或主判据；若另行比较只能作为 auxiliary/debug。",
        "默认只测 n>=1 positive Matsubara；本阶段不处理 n=0 true static。",
        "Matsubara 频率直接使用 bosonic_matsubara_energy_eV(n, temperature_K)；不使用 omega+eta 频率展宽。",
        "current-current-only 不是 gauge-closed finite-q conductivity。",
        "Ward identity 尚未检查。",
        "本脚本不修改 BdG、Casimir、reflection matrix。",
        "本脚本不输出最终 finite-q conductivity 或 Casimir 结论。",
        "",
        f"run_command = `{command}`",
        f"quick_mode={quick}",
        "finite_momentum_resolved=True",
        "normal_state=True",
        "current_current_kernel_only=True",
        "midpoint_vertex_approximation=True",
        "not_peierls_exact_vertex=True",
        "ward_identity_not_yet_checked=True",
        "not_final_casimir_input=True",
        "",
        "## Quick diagnostic status",
        f"- q=0 same-interface error: {max_q0_same_interface_error:.6g}",
        f"- smallest sampled nonzero q: {smallest_q:.6g}",
        f"- maximum same-interface error at smallest nonzero q: {smallest_q_error:.6g}",
        f"- maximum C4 covariance error: {max_c4_error:.6g}",
        f"- all K components finite: {finite}",
    ]


def _output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path]:
    csv_path = output_prefix.with_suffix(".csv")
    npz_path = output_prefix.with_suffix(".npz")
    result_root = output_prefix.parent.parent if output_prefix.parent.name == "data" else output_prefix.parent
    figure_dir = result_root / "figures"
    summary_path = result_root / "normal_finite_q_kernel_convergence_summary.md"
    return csv_path, npz_path, figure_dir, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matsubara-n-list",
        nargs="+",
        type=int,
        default=list(DEFAULT_MATSUBARA_N_LIST),
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--q-list", nargs="+", type=float, default=list(DEFAULT_Q_LIST))
    parser.add_argument(
        "--q-angle-list",
        nargs="+",
        type=_angle_expression,
        default=list(DEFAULT_Q_ANGLE_LIST),
    )
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--degeneracy-tol", type=float, default=DEFAULT_DEGENERACY_TOL_EV)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.matsubara_n_list = [1]
        args.q_list = [0.0, 1e-3]
        args.q_angle_list = [0.0, np.pi / 4.0]
        args.nk_list = [8]
        args.temperature = 30.0
    data = run_diagnostic(
        matsubara_n_list=list(args.matsubara_n_list),
        q_list=list(args.q_list),
        q_angle_list=list(args.q_angle_list),
        nk_list=list(args.nk_list),
        temperature_K=args.temperature,
        degeneracy_tol_eV=args.degeneracy_tol,
    )
    csv_path, npz_path, figure_dir, summary_path = _output_paths(args.output_prefix)
    _write_csv(csv_path, data)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)
    figure_paths = _plot_outputs(data, figure_dir)
    command = "python " + " ".join(shlex.quote(value) for value in sys.argv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(_summary_lines(data, command, args.quick)) + "\n",
        encoding="utf-8",
    )
    print(f"csv_path = {csv_path}")
    print(f"npz_path = {npz_path}")
    print(f"summary_path = {summary_path}")
    print("figure_paths = " + ", ".join(str(path) for path in figure_paths))


if __name__ == "__main__":
    main()
