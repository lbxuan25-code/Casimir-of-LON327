#!/usr/bin/env python3
"""Diagnose the normal-state finite-q current-current response.

This first-stage diagnostic uses the existing local normal Kubo response at
exactly q=0 and a shifted-state midpoint-vertex current-current bubble at
nonzero q.  It is not a gauge-closed finite-q conductivity or a Casimir input.
"""

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

from lno327 import KuboConfig, k_weights, kubo_conductivity_imag_axis, uniform_bz_mesh  # noqa: E402
from lno327.nonlocal_response import (  # noqa: E402
    c4_covariance_error,
    normal_finite_q_current_current_kernel_imag_axis,
    normal_local_current_current_kernel_imag_axis,
)
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

DEFAULT_OMEGA_LIST = (0.0, 1e-6, 1e-5, 1e-4)
DEFAULT_Q_LIST = (0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)
DEFAULT_Q_ANGLE_LIST = (0.0, np.pi / 8.0, np.pi / 4.0, 3.0 * np.pi / 8.0, np.pi / 2.0)
DEFAULT_NK_LIST = (16, 24, 32)
OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "normal_finite_q_response"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "normal_finite_q_response"
EPS = 1e-300

REQUIRED_COLUMNS = (
    "omega_eV",
    "q_model",
    "q_angle",
    "qx_model",
    "qy_model",
    "nk",
    "temperature_K",
    "eta_eV",
    "K_xx",
    "K_yy",
    "K_xy",
    "K_yx",
    "K_trace",
    "K_anisotropy_delta",
    "offdiag_ratio",
    "public_local_sigma_xx",
    "public_local_sigma_yy",
    "public_local_sigma_xy",
    "public_local_sigma_yx",
    "relative_error_to_public_local_sigma",
    "local_kernel_interband_xx",
    "local_kernel_interband_yy",
    "local_kernel_interband_xy",
    "local_kernel_interband_yx",
    "local_kernel_intraband_static_xx",
    "local_kernel_intraband_static_yy",
    "local_kernel_intraband_static_xy",
    "local_kernel_intraband_static_yx",
    "local_kernel_static_xx",
    "local_kernel_static_yy",
    "local_kernel_static_xy",
    "local_kernel_static_yx",
    "relative_error_to_local_kernel_interband",
    "relative_error_to_local_kernel_static",
    "expected_q_to_zero_reference",
    "q_to_zero_kernel_relative_error",
    "c4_covariance_error",
    "cos4_trace_harmonic",
    "sin4_trace_harmonic",
    "cos4_anisotropy_harmonic",
    "sin4_anisotropy_harmonic",
    "finite_momentum_resolved",
    "normal_state",
    "current_current_only",
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


def _harmonic_coefficients(angles: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    design = np.column_stack(
        [np.ones_like(angles), np.cos(4.0 * angles), np.sin(4.0 * angles)]
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(coefficients[1]), float(coefficients[2])


def _add_angular_harmonics(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[float, float, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (float(row["omega_eV"]), float(row["q_model"]), int(row["nk"]))
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        angles = np.array([float(row["q_angle"]) for row in group])
        traces = np.array([float(np.real(row["K_trace"])) for row in group])
        anisotropies = np.array([float(np.real(row["K_anisotropy_delta"])) for row in group])
        cos_trace, sin_trace = _harmonic_coefficients(angles, traces)
        cos_anisotropy, sin_anisotropy = _harmonic_coefficients(angles, anisotropies)
        for row in group:
            row["cos4_trace_harmonic"] = cos_trace
            row["sin4_trace_harmonic"] = sin_trace
            row["cos4_anisotropy_harmonic"] = cos_anisotropy
            row["sin4_anisotropy_harmonic"] = sin_anisotropy


def run_diagnostic(
    *,
    omega_list: list[float],
    q_list: list[float],
    q_angle_list: list[float],
    nk_list: list[int],
    temperature_K: float,
    eta_eV: float,
) -> dict[str, np.ndarray]:
    """Evaluate all requested normal finite-q diagnostic cases."""

    if not omega_list or not q_list or not q_angle_list or not nk_list:
        raise ValueError("omega-list, q-list, q-angle-list, and nk-list must not be empty")
    if any(omega < 0.0 for omega in omega_list) or any(q < 0.0 for q in q_list):
        raise ValueError("omega and q values must be non-negative")
    if any(nk <= 0 for nk in nk_list):
        raise ValueError("nk values must be positive")
    if eta_eV <= 0.0:
        raise ValueError("eta must be positive")

    rows: list[dict[str, object]] = []
    for nk in nk_list:
        mesh = uniform_bz_mesh(nk)
        weights = k_weights(mesh)
        for omega_eV in omega_list:
            config = KuboConfig.from_kelvin(
                omega_eV=omega_eV,
                temperature_K=temperature_K,
                eta_eV=eta_eV,
                output_si=False,
            )
            public_local_sigma = kubo_conductivity_imag_axis(mesh, config, weights).matrix()
            local_kernel = normal_local_current_current_kernel_imag_axis(
                mesh, config, weights
            )
            expected_reference_name = (
                "local_kernel_interband" if omega_eV > 0.0 else "local_kernel_static"
            )
            expected_reference = (
                local_kernel.interband if omega_eV > 0.0 else local_kernel.static
            )
            for q_model in q_list:
                for q_angle in q_angle_list:
                    q = np.array(
                        [q_model * np.cos(q_angle), q_model * np.sin(q_angle)], dtype=float
                    )
                    matrix = normal_finite_q_current_current_kernel_imag_axis(
                        mesh, config, q, weights
                    )
                    rotated_q = np.array([-q[1], q[0]], dtype=float)
                    rotated_matrix = normal_finite_q_current_current_kernel_imag_axis(
                        mesh, config, rotated_q, weights
                    )
                    trace = complex(matrix[0, 0] + matrix[1, 1])
                    row: dict[str, object] = {
                        "omega_eV": float(omega_eV),
                        "q_model": float(q_model),
                        "q_angle": float(q_angle),
                        "qx_model": float(q[0]),
                        "qy_model": float(q[1]),
                        "nk": int(nk),
                        "temperature_K": float(temperature_K),
                        "eta_eV": float(eta_eV),
                        **_matrix_components("K", matrix),
                        "K_trace": trace,
                        "K_anisotropy_delta": _anisotropy(matrix),
                        "offdiag_ratio": _offdiag_ratio(matrix),
                        **_matrix_components("public_local_sigma", public_local_sigma),
                        "relative_error_to_public_local_sigma": _relative_error(
                            matrix, public_local_sigma
                        ),
                        **_matrix_components("local_kernel_interband", local_kernel.interband),
                        **_matrix_components(
                            "local_kernel_intraband_static", local_kernel.intraband_static
                        ),
                        **_matrix_components("local_kernel_static", local_kernel.static),
                        "relative_error_to_local_kernel_interband": _relative_error(
                            matrix, local_kernel.interband
                        ),
                        "relative_error_to_local_kernel_static": _relative_error(
                            matrix, local_kernel.static
                        ),
                        "expected_q_to_zero_reference": expected_reference_name,
                        "q_to_zero_kernel_relative_error": _relative_error(
                            matrix, expected_reference
                        ),
                        "c4_covariance_error": c4_covariance_error(matrix, rotated_matrix),
                        "finite_momentum_resolved": True,
                        "normal_state": True,
                        "current_current_only": True,
                        "midpoint_vertex_approximation": True,
                        "not_peierls_exact_vertex": True,
                        "ward_identity_not_yet_checked": True,
                        "not_final_casimir_input": True,
                    }
                    rows.append(row)
    _add_angular_harmonics(rows)
    return {
        column: np.array([row[column] for row in rows])
        for column in REQUIRED_COLUMNS
    }


def _write_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=REQUIRED_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for index in range(len(data["omega_eV"])):
            writer.writerow({column: data[column][index] for column in REQUIRED_COLUMNS})


def _plot_outputs(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    selectors = sorted(set(zip(data["omega_eV"], data["nk"], strict=True)))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for omega, nk in selectors:
        mask = (
            (data["omega_eV"] == omega)
            & (data["nk"] == nk)
            & np.isclose(data["q_angle"], 0.0)
            & (data["q_model"] > 0.0)
        )
        order = np.argsort(data["q_model"][mask])
        ax.plot(
            data["q_model"][mask][order],
            np.real(data["K_trace"][mask][order]),
            marker="o",
            label=rf"$\omega={omega:g}$, $N_k={nk}$",
        )
    ax.set(xlabel=r"$q$ (model units)", ylabel=r"$\mathrm{Re}\,\mathrm{Tr}\,K$", title="Normal finite-q trace")
    style_publication_axis(ax)
    path = figure_dir / "normal_finite_q_trace.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for omega, nk in selectors:
        mask = (
            (data["omega_eV"] == omega)
            & (data["nk"] == nk)
            & np.isclose(data["q_angle"], 0.0)
            & (data["q_model"] > 0.0)
        )
        order = np.argsort(data["q_model"][mask])
        ax.semilogy(
            data["q_model"][mask][order],
            np.maximum(data["q_to_zero_kernel_relative_error"][mask][order], EPS),
            marker="o",
            label=rf"$\omega={omega:g}$, $N_k={nk}$",
        )
    ax.set(
        xlabel=r"$q$ (model units)",
        ylabel="relative error to expected local kernel",
        title=r"$q\to0$ kernel-level diagnostic",
    )
    style_publication_axis(ax)
    path = figure_dir / "q_to_zero_kernel_relative_error.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)
    obsolete_path = figure_dir / "q_to_zero_relative_error.png"
    if obsolete_path.exists():
        obsolete_path.unlink()

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for omega, nk in selectors:
        mask = (data["omega_eV"] == omega) & (data["nk"] == nk) & np.isclose(data["q_angle"], 0.0)
        order = np.argsort(data["q_model"][mask])
        ax.semilogy(
            data["q_model"][mask][order],
            np.maximum(data["relative_error_to_public_local_sigma"][mask][order], EPS),
            marker="o",
            label=rf"$\omega={omega:g}$, $N_k={nk}$",
        )
    ax.set(
        xlabel=r"$q$ (model units)",
        ylabel="relative error to public local sigma",
        title="Conductivity-level comparison (not closure criterion)",
    )
    style_publication_axis(ax)
    path = figure_dir / "relative_error_to_public_local_sigma.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for omega, nk in selectors:
        mask = (data["omega_eV"] == omega) & (data["nk"] == nk) & np.isclose(data["q_angle"], 0.0)
        order = np.argsort(data["q_model"][mask])
        ax.semilogy(
            data["q_model"][mask][order],
            np.maximum(data["c4_covariance_error"][mask][order], EPS),
            marker="o",
            label=rf"$\omega={omega:g}$, $N_k={nk}$",
        )
    ax.set(xlabel=r"$q$ (model units)", ylabel=r"$C_4$ covariance error", title=r"Normal finite-q $C_4$ covariance")
    style_publication_axis(ax)
    path = figure_dir / "c4_covariance_error.png"
    save_publication_figure(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    for omega, nk in selectors:
        mask = (data["omega_eV"] == omega) & (data["nk"] == nk) & np.isclose(data["q_angle"], 0.0)
        order = np.argsort(data["q_model"][mask])
        label = rf"$\omega={omega:g}$, $N_k={nk}$"
        axes[0].plot(data["q_model"][mask][order], data["cos4_trace_harmonic"][mask][order], marker="o", label=label)
        axes[1].plot(data["q_model"][mask][order], data["cos4_anisotropy_harmonic"][mask][order], marker="o", label=label)
    axes[0].set(xlabel=r"$q$ (model units)", ylabel=r"$\cos(4\phi_q)$ trace coefficient", title="Trace harmonic")
    axes[1].set(xlabel=r"$q$ (model units)", ylabel=r"$\cos(4\phi_q)$ anisotropy coefficient", title="Anisotropy harmonic")
    for ax in axes:
        style_publication_axis(ax)
    path = figure_dir / "angular_harmonics.png"
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
    max_q0_public_sigma_error = float(
        np.max(data["relative_error_to_public_local_sigma"][q_zero])
    )
    minimum_nonzero_q = float(np.min(data["q_model"][q_nonzero]))
    minimum_nonzero_q_mask = np.isclose(data["q_model"], minimum_nonzero_q)
    max_minimum_nonzero_q_kernel_error = float(
        np.max(data["q_to_zero_kernel_relative_error"][minimum_nonzero_q_mask])
    )
    max_minimum_nonzero_q_public_sigma_error = float(
        np.max(data["relative_error_to_public_local_sigma"][minimum_nonzero_q_mask])
    )
    max_c4_error = float(np.max(data["c4_covariance_error"]))
    angle_count = len(np.unique(data["q_angle"]))
    nonzero_finite = bool(
        np.all(np.isfinite(np.real(data["K_xx"][q_nonzero])))
        and np.all(np.isfinite(np.real(data["K_yy"][q_nonzero])))
        and np.all(np.isfinite(np.real(data["K_xy"][q_nonzero])))
        and np.all(np.isfinite(np.real(data["K_yx"][q_nonzero])))
    )
    return [
        "# Normal finite-q current-current response diagnostic",
        "",
        "这是 normal-state finite-q current-current diagnostic，不是完整 finite-q conductivity，",
        "也不是 Casimir input。q=0 行仍使用现有 public local normal sigma fallback；",
        "q!=0 行使用 shifted-state midpoint velocity approximation。",
        "",
        "public local sigma 是 conductivity-level reference，包含 intraband/omega；",
        "finite-q shifted-state result 是 current-current kernel-level quantity。因此",
        "relative_error_to_public_local_sigma 只作辅助对照，不作为 q-to-zero 闭合判据。",
        "新的主判据是 q_to_zero_kernel_relative_error：omega>0 使用 local interband",
        "kernel，omega=0 使用 local static kernel。",
        "",
        f"run_command = `{command}`",
        f"quick_mode={quick}",
        "finite_momentum_resolved=True",
        "normal_state=True",
        "current_current_only=True",
        "midpoint_vertex_approximation=True",
        "not_peierls_exact_vertex=True",
        "ward_identity_not_yet_checked=True",
        "not_final_casimir_input=True",
        "",
        "## Quick diagnostic status",
        f"- q=0 maximum relative mismatch to public local sigma fallback: {max_q0_public_sigma_error:.6g}",
        f"- smallest sampled nonzero q: {minimum_nonzero_q:.6g}",
        f"- maximum kernel-relative mismatch at smallest nonzero q: {max_minimum_nonzero_q_kernel_error:.6g}",
        f"- maximum public-sigma mismatch at smallest nonzero q: {max_minimum_nonzero_q_public_sigma_error:.6g}",
        f"- maximum C4 covariance error: {max_c4_error:.6g}",
        f"- all q!=0 response components finite: {nonzero_finite}",
        f"- angular samples per q: {angle_count}",
        f"- q-to-zero kernel continuity established by this run: {max_minimum_nonzero_q_kernel_error < 1e-2}",
        "",
        "## Boundary",
        "- current-current-only response is not gauge closed;",
        "- Ward identity has not been checked;",
        "- midpoint vertex is not a Peierls-exact finite-q vertex;",
        "- public sigma and kernel-level references are both reported without empirical rescaling;",
        "- quick-mode harmonic coefficients use only two angles and are smoke-level diagnostics;",
        "- this script does not modify BdG, Casimir, or reflection-matrix logic;",
        "- no final finite-q conductivity or Casimir conclusion is claimed.",
    ]


def _output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path]:
    csv_path = output_prefix.with_suffix(".csv")
    npz_path = output_prefix.with_suffix(".npz")
    result_root = output_prefix.parent.parent if output_prefix.parent.name == "data" else output_prefix.parent
    figure_dir = result_root / "figures"
    summary_path = result_root / "normal_finite_q_response_summary.md"
    return csv_path, npz_path, figure_dir, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega-list", nargs="+", type=float, default=list(DEFAULT_OMEGA_LIST))
    parser.add_argument("--q-list", nargs="+", type=float, default=list(DEFAULT_Q_LIST))
    parser.add_argument("--q-angle-list", nargs="+", type=_angle_expression, default=list(DEFAULT_Q_ANGLE_LIST))
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.omega_list = [1e-4]
        args.q_list = [0.0, 1e-3]
        args.q_angle_list = [0.0, np.pi / 4.0]
        args.nk_list = [8]
    data = run_diagnostic(
        omega_list=list(args.omega_list),
        q_list=list(args.q_list),
        q_angle_list=list(args.q_angle_list),
        nk_list=list(args.nk_list),
        temperature_K=args.temperature,
        eta_eV=args.eta,
    )
    csv_path, npz_path, figure_dir, summary_path = _output_paths(args.output_prefix)
    _write_csv(csv_path, data)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)
    figure_paths = _plot_outputs(data, figure_dir)
    command = "python " + " ".join(shlex.quote(value) for value in sys.argv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(_summary_lines(data, command, args.quick)) + "\n", encoding="utf-8")
    print(f"csv_path = {csv_path}")
    print(f"npz_path = {npz_path}")
    print(f"summary_path = {summary_path}")
    print("figure_paths = " + ", ".join(str(path) for path in figure_paths))


if __name__ == "__main__":
    main()
