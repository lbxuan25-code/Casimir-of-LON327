#!/usr/bin/env python3
"""Diagnose BdG finite-q current-current kernel contract."""

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

from lno327.response.nonlocal_bdg import bdg_current_current_kernel_imag_axis  # noqa: E402
from lno327 import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.response.nonlocal_normal import c4_covariance_error, normal_current_current_kernel_imag_axis  # noqa: E402
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes  # noqa: E402
from validation.lib.plotting import configure_publication_matplotlib, save_publication_figure, style_publication_axis  # noqa: E402

OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q_kernel_contract"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "bdg_finite_q_kernel_contract"
SUMMARY_NAME = "bdg_finite_q_kernel_contract_summary.md"
EPS = 1e-300

DEFAULT_KINDS = ("spm", "dwave")
DEFAULT_DELTA0_LIST = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.04)
DEFAULT_MATSUBARA_N_LIST = (1, 2, 4)
DEFAULT_TEMPERATURE_K = 30.0
DEFAULT_Q_LIST = (0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5, 1.0)
DEFAULT_Q_ANGLE_LIST = (0.0, np.pi / 8.0, np.pi / 4.0, 3.0 * np.pi / 8.0, np.pi / 2.0)
DEFAULT_NK_LIST = (8, 12, 16)
DEFAULT_DEGENERACY_TOL_EV = 1e-10

QUICK_DELTA0_LIST = (0.0, 1e-4, 0.04)
QUICK_MATSUBARA_N_LIST = (1, 2)
QUICK_Q_LIST = (0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1)
QUICK_Q_ANGLE_LIST = (0.0, np.pi / 4.0, np.pi / 2.0)
QUICK_NK_LIST = (8, 12)

ROW_COLUMNS = (
    "kind",
    "delta0",
    "matsubara_n",
    "omega_eV",
    "nk",
    "q_model",
    "q_angle",
    "qx_model",
    "qy_model",
    "K_xx",
    "K_yy",
    "K_xy",
    "K_yx",
    "K0_xx",
    "K0_yy",
    "K0_xy",
    "K0_yx",
    "q_to_zero_same_interface_error",
    "normal_limit_error_delta0_zero",
    "c4_covariance_error",
    "kernel_block_only",
    "current_current_only",
    "positive_matsubara_only",
    "response_computed",
    "conductivity_computed",
    "pi_mu_nu_computed",
    "ward_identity_checked",
    "casimir_computed",
    "not_final_casimir_conclusion",
    "diagnosis",
)

COMPACT_COLUMNS = (
    "kind",
    "delta0",
    "matsubara_n",
    "nk",
    "q_model",
    "max_q_to_zero_same_interface_error",
    "max_normal_limit_error_delta0_zero",
    "max_c4_covariance_error",
    "num_rows",
    "kernel_block_only",
    "current_current_only",
    "positive_matsubara_only",
    "response_computed",
    "conductivity_computed",
    "pi_mu_nu_computed",
    "ward_identity_checked",
    "casimir_computed",
    "not_final_casimir_conclusion",
    "diagnosis",
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


def _relative_error(matrix: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(matrix - reference) / max(float(np.linalg.norm(reference)), EPS))


def _matrix_fields(prefix: str, matrix: np.ndarray) -> dict[str, complex]:
    return {
        f"{prefix}_xx": complex(matrix[0, 0]),
        f"{prefix}_yy": complex(matrix[1, 1]),
        f"{prefix}_xy": complex(matrix[0, 1]),
        f"{prefix}_yx": complex(matrix[1, 0]),
    }


def _diagnosis(q_error: float, normal_error: float, c4_error: float) -> str:
    parts = ["bdg_finite_q_current_current_kernel_contract"]
    if np.isfinite(q_error) and q_error < 1e-2:
        parts.append("small_q_close_to_q0")
    if np.isfinite(normal_error) and normal_error < 1e-2:
        parts.append("delta0_zero_close_to_normal")
    if np.isfinite(c4_error) and c4_error < 1e-8:
        parts.append("c4_covariant")
    return ";".join(parts)


def run_diagnostic(
    *,
    kinds: list[str],
    delta0_list: list[float],
    matsubara_n_list: list[int],
    temperature_K: float,
    q_list: list[float],
    q_angle_list: list[float],
    nk_list: list[int],
    degeneracy_tol_eV: float,
) -> dict[str, np.ndarray]:
    if any(n < 1 for n in matsubara_n_list):
        raise ValueError("matsubara-n-list must contain only n >= 1")
    if any(delta0 < 0.0 for delta0 in delta0_list):
        raise ValueError("delta0-list must contain non-negative values")
    if any(q < 0.0 for q in q_list):
        raise ValueError("q-list values must be non-negative")
    if any(nk <= 0 for nk in nk_list):
        raise ValueError("nk-list values must be positive")
    if temperature_K < 0.0:
        raise ValueError("temperature must be non-negative")
    if degeneracy_tol_eV <= 0.0:
        raise ValueError("degeneracy-tol must be positive")

    rows: list[dict[str, object]] = []
    bdg_cache: dict[tuple[object, ...], np.ndarray] = {}
    normal_cache: dict[tuple[object, ...], np.ndarray] = {}

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
            for kind in kinds:
                for delta0 in delta0_list:
                    pairing_params = PairingAmplitudes(delta0_eV=delta0)

                    def bdg_kernel(qx: float, qy: float) -> np.ndarray:
                        key = (kind, delta0, matsubara_n, nk, qx, qy)
                        if key not in bdg_cache:
                            bdg_cache[key] = bdg_current_current_kernel_imag_axis(
                                mesh,
                                config,
                                np.array([qx, qy], dtype=float),
                                kind,  # type: ignore[arg-type]
                                pairing_params,
                                weights,
                            )
                        return bdg_cache[key]

                    K0 = bdg_kernel(0.0, 0.0)
                    for q_model in q_list:
                        for q_angle in q_angle_list:
                            qx = float(q_model * np.cos(q_angle))
                            qy = float(q_model * np.sin(q_angle))
                            matrix = bdg_kernel(qx, qy)
                            rotated_matrix = bdg_kernel(-qy, qx)
                            q_error = _relative_error(matrix, K0)
                            c4_error = c4_covariance_error(matrix, rotated_matrix)
                            normal_error = float("nan")
                            if np.isclose(delta0, 0.0):
                                normal_key = (matsubara_n, nk, qx, qy)
                                if normal_key not in normal_cache:
                                    normal_cache[normal_key] = normal_current_current_kernel_imag_axis(
                                        mesh,
                                        config,
                                        np.array([qx, qy], dtype=float),
                                        weights,
                                    )
                                normal_error = _relative_error(matrix, normal_cache[normal_key])
                            rows.append(
                                {
                                    "kind": kind,
                                    "delta0": float(delta0),
                                    "matsubara_n": int(matsubara_n),
                                    "omega_eV": float(omega_eV),
                                    "nk": int(nk),
                                    "q_model": float(q_model),
                                    "q_angle": float(q_angle),
                                    "qx_model": qx,
                                    "qy_model": qy,
                                    **_matrix_fields("K", matrix),
                                    **_matrix_fields("K0", K0),
                                    "q_to_zero_same_interface_error": q_error,
                                    "normal_limit_error_delta0_zero": normal_error,
                                    "c4_covariance_error": c4_error,
                                    "kernel_block_only": True,
                                    "current_current_only": True,
                                    "positive_matsubara_only": True,
                                    "response_computed": True,
                                    "conductivity_computed": False,
                                    "pi_mu_nu_computed": False,
                                    "ward_identity_checked": False,
                                    "casimir_computed": False,
                                    "not_final_casimir_conclusion": True,
                                    "diagnosis": _diagnosis(q_error, normal_error, c4_error),
                                }
                            )

    return {column: np.array([row[column] for row in rows]) for column in ROW_COLUMNS}


def _max_finite(values: np.ndarray) -> float:
    finite = np.isfinite(values.astype(float))
    if not np.any(finite):
        return float("nan")
    return float(np.max(values.astype(float)[finite]))


def _compact_diagnosis(q_error: float, normal_error: float, c4_error: float) -> str:
    parts: list[str] = []
    if np.isfinite(q_error) and q_error < 1e-2:
        parts.append("pass_small_q")
    if np.isfinite(normal_error) and normal_error < 1e-2:
        parts.append("pass_normal_limit")
    elif not np.isfinite(normal_error):
        parts.append("normal_limit_not_applicable")
    if np.isfinite(c4_error) and c4_error < 1e-8:
        parts.append("pass_c4")
    if not parts or any(part.startswith("pass") for part in parts) is False:
        parts.append("warning")
    if (np.isfinite(q_error) and q_error >= 1e-2) or (np.isfinite(c4_error) and c4_error >= 1e-8):
        parts.append("warning")
    if np.isfinite(normal_error) and normal_error >= 1e-2:
        parts.append("warning")
    return ";".join(dict.fromkeys(parts))


def compact_bdg_finite_q_contract(data: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Aggregate detailed diagnostic rows into a compact summary table."""

    rows: list[dict[str, object]] = []
    group_keys = sorted(
        {
            (
                str(data["kind"][index]),
                float(data["delta0"][index]),
                int(data["matsubara_n"][index]),
                int(data["nk"][index]),
                float(data["q_model"][index]),
            )
            for index in range(len(data["kind"]))
        }
    )
    for kind, delta0, matsubara_n, nk, q_model in group_keys:
        mask = (
            (data["kind"] == kind)
            & np.isclose(data["delta0"].astype(float), delta0)
            & (data["matsubara_n"].astype(int) == matsubara_n)
            & (data["nk"].astype(int) == nk)
            & np.isclose(data["q_model"].astype(float), q_model)
        )
        q_error = _max_finite(data["q_to_zero_same_interface_error"][mask])
        normal_error = _max_finite(data["normal_limit_error_delta0_zero"][mask])
        c4_error = _max_finite(data["c4_covariance_error"][mask])
        rows.append(
            {
                "kind": kind,
                "delta0": delta0,
                "matsubara_n": matsubara_n,
                "nk": nk,
                "q_model": q_model,
                "max_q_to_zero_same_interface_error": q_error,
                "max_normal_limit_error_delta0_zero": normal_error,
                "max_c4_covariance_error": c4_error,
                "num_rows": int(np.count_nonzero(mask)),
                "kernel_block_only": True,
                "current_current_only": True,
                "positive_matsubara_only": True,
                "response_computed": True,
                "conductivity_computed": False,
                "pi_mu_nu_computed": False,
                "ward_identity_checked": False,
                "casimir_computed": False,
                "not_final_casimir_conclusion": True,
                "diagnosis": _compact_diagnosis(q_error, normal_error, c4_error),
            }
        )
    return rows


def _write_rows_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_expanded_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    rows = [{column: data[column][index] for column in ROW_COLUMNS} for index in range(len(data["kind"]))]
    _write_rows_csv(path, ROW_COLUMNS, rows)


def _write_compact_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    _write_rows_csv(path, COMPACT_COLUMNS, compact_bdg_finite_q_contract(data))


def _output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path]:
    compact_csv = output_prefix.with_name(f"{output_prefix.name}_compact").with_suffix(".csv")
    expanded_csv = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".csv")
    expanded_npz = output_prefix.with_name(f"{output_prefix.name}_expanded").with_suffix(".npz")
    result_root = output_prefix.parent.parent if output_prefix.parent.name == "data" else output_prefix.parent
    figure_dir = result_root / "figures"
    summary_path = result_root / SUMMARY_NAME
    return compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path


def _plot_error_by_q(
    data: dict[str, np.ndarray],
    values: np.ndarray,
    path: Path,
    *,
    ylabel: str,
    title: str,
    finite_mask: np.ndarray | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    mask = np.ones(len(data["q_model"]), dtype=bool) if finite_mask is None else finite_mask
    q_values = np.array(sorted(set(float(q) for q in data["q_model"][mask])))
    max_values = []
    for q_model in q_values:
        q_mask = mask & np.isclose(data["q_model"], q_model)
        finite = np.isfinite(values[q_mask])
        max_values.append(float(np.max(values[q_mask][finite])) if np.any(finite) else float("nan"))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(q_values, np.maximum(np.asarray(max_values, dtype=float), EPS), marker="o")
    ax.set(xlabel="q_model", ylabel=ylabel, title=title)
    style_publication_axis(ax, legend=False)
    save_publication_figure(fig, path)
    plt.close(fig)
    return path


def _plot_outputs(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    configure_publication_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_error_by_q(
            data,
            data["q_to_zero_same_interface_error"].astype(float),
            figure_dir / "q_to_zero_error_vs_q.png",
            ylabel=r"$\|K(q)-K(0)\|/\|K(0)\|$",
            title=r"BdG finite-q same-interface $q\to0$ error",
        ),
        _plot_error_by_q(
            data,
            data["normal_limit_error_delta0_zero"].astype(float),
            figure_dir / "normal_limit_error_vs_q.png",
            ylabel=r"$\|K_{BdG,\Delta=0}(q)-K_N(q)\|/\|K_N(q)\|$",
            title=r"BdG $\Delta_0=0$ normal-kernel comparison",
            finite_mask=np.isfinite(data["normal_limit_error_delta0_zero"].astype(float)),
        ),
        _plot_error_by_q(
            data,
            data["c4_covariance_error"].astype(float),
            figure_dir / "c4_covariance_error_vs_q.png",
            ylabel=r"$C_4$ covariance error",
            title=r"BdG finite-q $C_4$ covariance",
        ),
    ]
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
    q_nonzero = data["q_model"].astype(float) > 0.0
    q_small = q_nonzero & (data["q_model"].astype(float) <= 1e-3)
    max_small_q_error = float(np.max(data["q_to_zero_same_interface_error"][q_small])) if np.any(q_small) else float("nan")
    normal_values = data["normal_limit_error_delta0_zero"].astype(float)
    finite_normal = np.isfinite(normal_values)
    max_normal_error = float(np.max(normal_values[finite_normal])) if np.any(finite_normal) else float("nan")
    max_c4_error = float(np.max(data["c4_covariance_error"].astype(float)))
    max_q = float(np.max(data["q_model"].astype(float)))

    lines = [
        "# BdG finite-q current-current kernel contract diagnostic",
        "",
        "This is BdG finite-q current-current kernel contract only.",
        "It is not gauge-closed finite-q conductivity.",
        "It is not Pi_mu_nu and not Ward identity.",
        "It is not Casimir input.",
        (
            "Stage 2 showed Casimir-relevant q_model reaches O(1), so this diagnostic "
            "includes q up to O(1) but remains a kernel contract test."
        ),
        "",
        f"run_command = `{command}`",
        f"quick_mode={bool(args.quick)}",
        f"expanded_data_written={bool(args.write_expanded_data)}",
        "kernel_block_only=True",
        "current_current_only=True",
        "positive_matsubara_only=True",
        "response_computed=True",
        "conductivity_computed=False",
        "pi_mu_nu_computed=False",
        "ward_identity_checked=False",
        "casimir_computed=False",
        "not_final_casimir_conclusion=True",
        "",
        "## Parameter grid",
        f"- kinds = {' '.join(args.kinds)}",
        f"- delta0_list = {' '.join(_fmt(float(v)) for v in args.delta0_list)}",
        f"- matsubara_n_list = {' '.join(str(int(v)) for v in args.matsubara_n_list)}",
        f"- temperature_K = {_fmt(float(args.temperature))}",
        f"- q_list = {' '.join(_fmt(float(v)) for v in args.q_list)}",
        f"- q_angle_list = {' '.join(_fmt(float(v)) for v in args.q_angle_list)}",
        f"- nk_list = {' '.join(str(int(v)) for v in args.nk_list)}",
        f"- degeneracy_tol_eV = {_fmt(float(args.degeneracy_tol))}",
        f"- q_model_max = {_fmt(max_q)}",
        "",
        "## Contract results",
        f"- max q->0 same-interface error for 0<q<=1e-3: {_fmt(max_small_q_error)}",
        f"- max Delta0=0 normal finite-q kernel comparison error: {_fmt(max_normal_error)}",
        f"- max C4 covariance error: {_fmt(max_c4_error)}",
        "",
        "The Delta0=0 comparison is against normal_current_current_kernel_imag_axis and is reported as a kernel comparison, not conductivity.",
        "No legacy local BdG comparison is used in the pass/fail path.",
        "",
        "## Output files",
        f"- compact CSV: {compact_csv}",
        f"- expanded_data_written={bool(args.write_expanded_data)}",
    ]
    if args.write_expanded_data:
        lines.extend([f"- expanded CSV: {expanded_csv}", f"- expanded NPZ: {expanded_npz}"])
    else:
        lines.append("- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.")
    lines.extend(f"- figure: {path}" for path in figure_paths)
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", nargs="+", choices=("spm", "dwave"), default=list(DEFAULT_KINDS))
    parser.add_argument("--delta0-list", nargs="+", type=float, default=list(DEFAULT_DELTA0_LIST))
    parser.add_argument("--matsubara-n-list", nargs="+", type=int, default=list(DEFAULT_MATSUBARA_N_LIST))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--q-list", nargs="+", type=float, default=list(DEFAULT_Q_LIST))
    parser.add_argument("--q-angle-list", nargs="+", type=_angle_expression, default=list(DEFAULT_Q_ANGLE_LIST))
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--degeneracy-tol", type=float, default=DEFAULT_DEGENERACY_TOL_EV)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--write-expanded-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.kinds = list(DEFAULT_KINDS)
        args.delta0_list = list(QUICK_DELTA0_LIST)
        args.matsubara_n_list = list(QUICK_MATSUBARA_N_LIST)
        args.temperature = DEFAULT_TEMPERATURE_K
        args.q_list = list(QUICK_Q_LIST)
        args.q_angle_list = list(QUICK_Q_ANGLE_LIST)
        args.nk_list = list(QUICK_NK_LIST)

    data = run_diagnostic(
        kinds=list(args.kinds),
        delta0_list=list(args.delta0_list),
        matsubara_n_list=list(args.matsubara_n_list),
        temperature_K=float(args.temperature),
        q_list=list(args.q_list),
        q_angle_list=list(args.q_angle_list),
        nk_list=list(args.nk_list),
        degeneracy_tol_eV=float(args.degeneracy_tol),
    )
    compact_csv, expanded_csv, expanded_npz, figure_dir, summary_path = _output_paths(args.output_prefix)
    _write_compact_csv(compact_csv, data)
    if args.write_expanded_data:
        _write_expanded_csv(expanded_csv, data)
        expanded_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(expanded_npz, **data)
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
