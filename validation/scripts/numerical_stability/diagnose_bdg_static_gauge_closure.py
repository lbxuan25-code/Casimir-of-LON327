#!/usr/bin/env python3
"""Diagnose static gauge closure of the local BdG electromagnetic kernel.

This script checks whether K_para + K_dia behaves consistently before the BdG
response is interpreted as a superconducting conductivity-like response. It is
local q=0 only, benchmark-only, and not a Casimir input.
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

from lno327 import (  # noqa: E402
    KuboConfig,
    PairingAmplitudes,
    bdg_total_kernel_imag_axis,
    k_weights,
    uniform_bz_mesh,
)
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("spm", "dwave")
DEFAULT_DELTA0_LIST = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.04)
DEFAULT_OMEGA_LIST = (0.0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4)
EPS = 1e-300
OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "bdg_static_gauge_closure"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "bdg_static_gauge_closure"
SUMMARY_PATH = OUTPUT_ROOT / "bdg_static_gauge_closure_summary.md"

REQUIRED_COLUMNS = (
    "kind",
    "delta0_eV",
    "omega_eV",
    "K_para_xx",
    "K_para_yy",
    "K_para_xy",
    "K_para_yx",
    "K_dia_xx",
    "K_dia_yy",
    "K_dia_xy",
    "K_dia_yx",
    "K_total_xx",
    "K_total_yy",
    "K_total_xy",
    "K_total_yx",
    "norm_para",
    "norm_dia",
    "norm_total",
    "gauge_residual",
    "rho_s_xx",
    "rho_s_yy",
    "rho_s_anisotropy",
    "offdiag_ratio",
    "nk",
    "temperature_K",
    "eta_eV",
    "benchmark_only",
    "local_response",
    "static_gauge_closure_diagnostic",
    "not_final_optical_conductivity",
    "not_final_Casimir_input",
)


def _quick_overrides() -> dict[str, object]:
    return {
        "nk": 6,
        "delta0_list": [0.0, 0.04],
        "omega_list": [0.0, 1e-4],
    }


def _matrix_components(prefix: str, matrix: np.ndarray) -> dict[str, complex]:
    return {
        f"{prefix}_xx": complex(matrix[0, 0]),
        f"{prefix}_yy": complex(matrix[1, 1]),
        f"{prefix}_xy": complex(matrix[0, 1]),
        f"{prefix}_yx": complex(matrix[1, 0]),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(float(denominator), EPS))


def _rho_anisotropy(rho_xx: float, rho_yy: float) -> float:
    denominator = rho_xx + rho_yy
    if abs(denominator) <= EPS:
        return 0.0
    return float((rho_xx - rho_yy) / denominator)


def _row_for_kernel(
    *,
    kind: str,
    delta0_eV: float,
    omega_eV: float,
    nk: int,
    temperature_K: float,
    eta_eV: float,
    mesh: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    components = bdg_total_kernel_imag_axis(
        mesh,
        config,
        kind,  # type: ignore[arg-type]
        PairingAmplitudes(delta0_eV=delta0_eV),
        weights,
    )
    k_para = components.paramagnetic
    k_dia = components.diamagnetic
    k_total = components.total
    norm_para = float(np.linalg.norm(k_para))
    norm_dia = float(np.linalg.norm(k_dia))
    norm_total = float(np.linalg.norm(k_total))
    rho_s_xx = float(np.real(k_total[0, 0]))
    rho_s_yy = float(np.real(k_total[1, 1]))
    offdiag_norm = float(np.linalg.norm([k_total[0, 1], k_total[1, 0]]))
    diag_norm = float(np.linalg.norm([k_total[0, 0], k_total[1, 1]]))
    row: dict[str, object] = {
        "kind": kind,
        "delta0_eV": float(delta0_eV),
        "omega_eV": float(omega_eV),
        **_matrix_components("K_para", k_para),
        **_matrix_components("K_dia", k_dia),
        **_matrix_components("K_total", k_total),
        "norm_para": norm_para,
        "norm_dia": norm_dia,
        "norm_total": norm_total,
        "gauge_residual": _safe_ratio(norm_total, max(norm_para, norm_dia)),
        "rho_s_xx": rho_s_xx,
        "rho_s_yy": rho_s_yy,
        "rho_s_anisotropy": _rho_anisotropy(rho_s_xx, rho_s_yy),
        "offdiag_ratio": _safe_ratio(offdiag_norm, diag_norm),
        "nk": int(nk),
        "temperature_K": float(temperature_K),
        "eta_eV": float(eta_eV),
        "benchmark_only": True,
        "local_response": True,
        "static_gauge_closure_diagnostic": True,
        "not_final_optical_conductivity": True,
        "not_final_Casimir_input": True,
    }
    return row


def run_diagnostic(
    *,
    kinds: list[str],
    delta0_list: list[float],
    omega_list: list[float],
    nk: int,
    temperature_K: float,
    eta_eV: float,
) -> dict[str, np.ndarray]:
    if nk <= 0:
        raise ValueError("nk must be positive")
    if eta_eV <= 0.0:
        raise ValueError("eta must be positive")
    if not delta0_list:
        raise ValueError("delta0-list must not be empty")
    if not omega_list:
        raise ValueError("omega-list must not be empty")
    if any(delta0 < 0.0 for delta0 in delta0_list):
        raise ValueError("delta0 values must be non-negative")
    if any(omega < 0.0 for omega in omega_list):
        raise ValueError("omega values must be non-negative")

    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    rows = [
        _row_for_kernel(
            kind=kind,
            delta0_eV=delta0,
            omega_eV=omega,
            nk=nk,
            temperature_K=temperature_K,
            eta_eV=eta_eV,
            mesh=mesh,
            weights=weights,
        )
        for kind in kinds
        for delta0 in delta0_list
        for omega in omega_list
    ]
    data: dict[str, np.ndarray] = {}
    for column in REQUIRED_COLUMNS:
        values = [row[column] for row in rows]
        if column == "kind":
            data[column] = np.asarray(values, dtype="U16")
        elif column.startswith("K_"):
            data[column] = np.asarray(values, dtype=complex)
        elif column in {
            "benchmark_only",
            "local_response",
            "static_gauge_closure_diagnostic",
            "not_final_optical_conductivity",
            "not_final_Casimir_input",
        }:
            data[column] = np.asarray(values, dtype=bool)
        elif column == "nk":
            data[column] = np.asarray(values, dtype=int)
        else:
            data[column] = np.asarray(values, dtype=float)
    data["delta0_list"] = np.asarray(delta0_list, dtype=float)
    data["omega_list"] = np.asarray(omega_list, dtype=float)
    return data


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        return repr(value)
    return value


def _output_paths(output_prefix: Path) -> tuple[Path, Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    figure_dir = OUTPUT_ROOT / "figures" if output_prefix.resolve() == DEFAULT_OUTPUT_PREFIX.resolve() else output_prefix.parent / "figures"
    return npz_path, csv_path, figure_dir


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path, list[Path]]:
    npz_path, csv_path, figure_dir = _output_paths(output_prefix)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        for index in range(data["kind"].size):
            writer.writerow({column: _csv_value(data[column][index]) for column in REQUIRED_COLUMNS})
    figure_paths = save_figures(data, figure_dir)
    return npz_path, csv_path, figure_paths


def _plot_by_delta(
    *,
    data: dict[str, np.ndarray],
    figure_dir: Path,
    values: list[tuple[str, np.ndarray]],
    ylabel: str,
    title: str,
    filename: str,
    yscale: str | None = None,
) -> Path:
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    path = figure_dir / filename
    kinds = sorted(set(str(item) for item in data["kind"]))
    omega_values = list(data["omega_list"])
    ncols = min(2, max(len(omega_values), 1))
    nrows = int(np.ceil(max(len(omega_values), 1) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.4, max(3.0, 2.3 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    delta_values = data["delta0_list"]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    line_styles = ("-", "--", ":", "-.")
    markers = ("o", "s", "^", "D")
    for omega_index, omega in enumerate(omega_values):
        ax = axes[omega_index]
        for kind_index, kind in enumerate(kinds):
            color = color_cycle[kind_index % max(len(color_cycle), 1)] if color_cycle else None
            mask = (data["kind"] == kind) & np.isclose(data["omega_eV"], omega)
            order = np.argsort(data["delta0_eV"][mask])
            x = data["delta0_eV"][mask][order]
            for value_index, (label, array) in enumerate(values):
                y = array[mask][order]
                ax.plot(
                    x,
                    y,
                    marker=markers[value_index % len(markers)],
                    color=color,
                    linestyle=line_styles[value_index % len(line_styles)],
                    alpha=0.9,
                    label=f"{kind} {label}" if omega_index == 0 else None,
                )
        ax.set_ylabel(ylabel)
        ax.set_title(f"omega={omega:g}")
        if yscale is not None:
            ax.set_yscale(yscale)
        if np.any(delta_values == 0.0):
            ax.set_xlim(left=-0.002 * max(float(np.max(delta_values)), 1.0))
        style_publication_axis(ax, legend=False)
    for ax in axes[len(omega_values) :]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("Delta0 (eV)")
    fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    save_publication_figure(fig, path)
    plt.close(fig)
    return path


def save_figures(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    return [
        _plot_by_delta(
            data=data,
            figure_dir=figure_dir,
            values=[("gauge_residual", data["gauge_residual"])],
            ylabel="gauge residual",
            title="BdG static gauge residual",
            filename="gauge_residual_vs_delta0.png",
            yscale="log",
        ),
        _plot_by_delta(
            data=data,
            figure_dir=figure_dir,
            values=[("rho_s_xx", data["rho_s_xx"]), ("rho_s_yy", data["rho_s_yy"])],
            ylabel="candidate rho_s",
            title="Candidate static stiffness components",
            filename="rho_s_xx_yy_vs_delta0.png",
        ),
        _plot_by_delta(
            data=data,
            figure_dir=figure_dir,
            values=[("rho_s_anisotropy", data["rho_s_anisotropy"])],
            ylabel="rho_s anisotropy",
            title="Candidate rho_s C4 anisotropy",
            filename="rho_s_anisotropy_vs_delta0.png",
        ),
        _plot_by_delta(
            data=data,
            figure_dir=figure_dir,
            values=[("offdiag_ratio", data["offdiag_ratio"])],
            ylabel="offdiag ratio",
            title="K_total offdiagonal diagnostic",
            filename="offdiag_ratio_vs_delta0.png",
            yscale="log",
        ),
    ]


def _format_command(args: argparse.Namespace) -> str:
    parts = ["python", "validation/scripts/numerical_stability/diagnose_bdg_static_gauge_closure.py"]
    if args.quick:
        parts.append("--quick")
    else:
        option_values = [
            ("--kinds", args.kinds),
            ("--delta0-list", args.delta0_list),
            ("--omega-list", args.omega_list),
            ("--nk", [args.nk]),
            ("--temperature", [args.temperature]),
            ("--eta", [args.eta]),
            ("--output-prefix", [args.output_prefix]),
        ]
        for option, values in option_values:
            parts.append(option)
            parts.extend(str(value) for value in values)
    return " ".join(shlex.quote(str(part)) for part in parts)


def _summary_path(output_prefix: Path) -> Path:
    if output_prefix.resolve() == DEFAULT_OUTPUT_PREFIX.resolve():
        return SUMMARY_PATH
    return output_prefix.parent / "bdg_static_gauge_closure_summary.md"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_summary(
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    command: str,
    figure_paths: list[Path],
) -> Path:
    lines = [
        "# BdG Static Gauge-Closure Diagnostic",
        "",
        "This is a BdG static gauge-closure diagnostic. It checks whether local",
        "K_para + K_dia cancels in the Delta0 -> 0 BdG normal limit and whether",
        "Delta0 > 0 gives a finite, symmetry-consistent candidate rho_s.",
        "",
        "It is not a final optical conductivity, not a final Casimir input, does",
        "not contain finite momentum response, and does not change n0_policy.",
        "",
        f"run_command = `{command}`",
        f"quick_test_only={bool(args.quick)}",
        "benchmark_only=True",
        "local_response=True",
        "static_gauge_closure_diagnostic=True",
        "not_final_optical_conductivity=True",
        "not_final_Casimir_input=True",
        "",
        "## Parameters",
        f"- kinds={', '.join(args.kinds)}",
        f"- delta0_list={', '.join(f'{value:g}' for value in args.delta0_list)}",
        f"- omega_list={', '.join(f'{value:g}' for value in args.omega_list)}",
        f"- nk={args.nk}",
        f"- temperature_K={args.temperature:g}",
        f"- eta_eV={args.eta:g}",
        "",
        "## Delta0=0 Gauge Residual",
    ]
    zero_mask = np.isclose(data["delta0_eV"], 0.0)
    for kind in sorted(set(str(item) for item in data["kind"])):
        for omega in data["omega_list"]:
            mask = zero_mask & (data["kind"] == kind) & np.isclose(data["omega_eV"], omega)
            if np.any(mask):
                value = float(np.nanmax(data["gauge_residual"][mask]))
                lines.append(f"- {kind}, omega={omega:g}: gauge_residual={value:.6g}")
    lines.extend(["", "## C4 / Offdiag Diagnostics"])
    for kind in sorted(set(str(item) for item in data["kind"])):
        mask = data["kind"] == kind
        anisotropy = float(np.nanmax(np.abs(data["rho_s_anisotropy"][mask])))
        offdiag = float(np.nanmax(data["offdiag_ratio"][mask]))
        lines.append(f"- {kind}: max_abs_rho_s_anisotropy={anisotropy:.6g}, max_offdiag_ratio={offdiag:.6g}")
    lines.extend(["", "## Figures"])
    lines.extend(f"- {_display_path(path)}" for path in figure_paths)
    summary_path = _summary_path(args.output_prefix)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--delta0-list", nargs="+", type=float, default=list(DEFAULT_DELTA0_LIST))
    parser.add_argument("--omega-list", nargs="+", type=float, default=list(DEFAULT_OMEGA_LIST))
    parser.add_argument("--nk", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        for key, value in _quick_overrides().items():
            setattr(args, key, value)
    return args


def main() -> None:
    args = parse_args()
    command = _format_command(args)
    data = run_diagnostic(
        kinds=list(args.kinds),
        delta0_list=list(args.delta0_list),
        omega_list=list(args.omega_list),
        nk=int(args.nk),
        temperature_K=float(args.temperature),
        eta_eV=float(args.eta),
    )
    npz_path, csv_path, figure_paths = save_outputs(data, args.output_prefix)
    summary_path = write_summary(data, args, command, figure_paths)
    print(f"npz_path = {npz_path}")
    print(f"csv_path = {csv_path}")
    print(f"summary_path = {summary_path}")
    print("figure_paths = " + ", ".join(str(path) for path in figure_paths))
    print("note = BdG static gauge-closure diagnostic only; not final optical conductivity or Casimir input.")


if __name__ == "__main__":
    main()
