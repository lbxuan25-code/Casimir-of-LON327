#!/usr/bin/env python3
"""Compare BdG Delta0=0 kernels against normal-state kernel pieces.

This is a diagnostic-only script. It does not modify the BdG response formula,
the Casimir pipeline, or any reflection-matrix logic.
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

from lno327 import KuboConfig, PairingAmplitudes, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.bdg_response import (  # noqa: E402
    bdg_diamagnetic_kernel,
    bdg_paramagnetic_kernel_imag_axis,
    bdg_total_kernel_imag_axis,
)
from lno327.conductivity import conductivity_eigensystem  # noqa: E402
from lno327.model import normal_state_mass_operator  # noqa: E402
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

KINDS = ("spm", "dwave")
DEFAULT_OMEGA_LIST = (0.0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4)
DELTA0_EV = 0.0
EPS = 1e-300
OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "bdg_normal_limit_kernel_decomposition"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "bdg_normal_limit_kernel_decomposition"
SUMMARY_PATH = OUTPUT_ROOT / "bdg_normal_limit_kernel_decomposition_summary.md"

MATRIX_PREFIXES = (
    "bdg_K_para",
    "bdg_K_dia",
    "bdg_K_total",
    "normal_K_para",
    "normal_K_dia",
    "normal_K_total",
)

REQUIRED_COLUMNS = (
    "kind",
    "delta0_eV",
    "omega_eV",
    "nk",
    "temperature_K",
    "eta_eV",
    "bdg_K_para_xx",
    "bdg_K_para_yy",
    "bdg_K_para_xy",
    "bdg_K_para_yx",
    "bdg_K_dia_xx",
    "bdg_K_dia_yy",
    "bdg_K_dia_xy",
    "bdg_K_dia_yx",
    "bdg_K_total_xx",
    "bdg_K_total_yy",
    "bdg_K_total_xy",
    "bdg_K_total_yx",
    "bdg_norm_para",
    "bdg_norm_dia",
    "bdg_norm_total",
    "bdg_gauge_residual",
    "normal_K_para_xx",
    "normal_K_para_yy",
    "normal_K_para_xy",
    "normal_K_para_yx",
    "normal_K_dia_xx",
    "normal_K_dia_yy",
    "normal_K_dia_xy",
    "normal_K_dia_yx",
    "normal_K_total_xx",
    "normal_K_total_yy",
    "normal_K_total_xy",
    "normal_K_total_yx",
    "normal_norm_para",
    "normal_norm_dia",
    "normal_norm_total",
    "normal_gauge_residual",
    "para_ratio_xx",
    "dia_ratio_xx",
    "total_ratio_xx",
    "para_sign_match_xx",
    "dia_sign_match_xx",
    "para_relative_error",
    "dia_relative_error",
    "total_relative_error",
    "bdg_offdiag_ratio",
    "normal_offdiag_ratio",
    "bdg_C4_anisotropy",
    "normal_C4_anisotropy",
    "benchmark_only",
    "local_response",
    "normal_limit_decomposition_diagnostic",
    "not_final_response_formula",
    "not_final_optical_conductivity",
    "not_final_Casimir_input",
)


def _quick_overrides() -> dict[str, object]:
    return {"nk": 6, "omega_list": [0.0, 1e-4]}


def _safe_ratio(numerator: complex | float, denominator: complex | float) -> complex:
    if abs(denominator) <= EPS:
        return complex(np.nan)
    return complex(numerator / denominator)


def _norm_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(float(denominator), EPS))


def _matrix_components(prefix: str, matrix: np.ndarray) -> dict[str, complex]:
    return {
        f"{prefix}_xx": complex(matrix[0, 0]),
        f"{prefix}_yy": complex(matrix[1, 1]),
        f"{prefix}_xy": complex(matrix[0, 1]),
        f"{prefix}_yx": complex(matrix[1, 0]),
    }


def _offdiag_ratio(matrix: np.ndarray) -> float:
    offdiag = float(np.linalg.norm([matrix[0, 1], matrix[1, 0]]))
    diag = float(np.linalg.norm([matrix[0, 0], matrix[1, 1]]))
    return _norm_ratio(offdiag, diag)


def _c4_anisotropy(matrix: np.ndarray) -> float:
    denominator = matrix[0, 0] + matrix[1, 1]
    if abs(denominator) <= EPS:
        return 0.0
    return float(np.real((matrix[0, 0] - matrix[1, 1]) / denominator))


def _sign_match(value_a: complex, value_b: complex) -> bool:
    real_a = float(np.real(value_a))
    real_b = float(np.real(value_b))
    if abs(real_a) <= EPS or abs(real_b) <= EPS:
        return False
    return bool(np.sign(real_a) == np.sign(real_b))


def normal_paramagnetic_kernel_imag_axis(
    k_points: np.ndarray,
    config: KuboConfig,
    weights: np.ndarray,
) -> np.ndarray:
    """Return a normal-state kernel-level current-current bubble.

    This local helper mirrors the BdG kernel-level convention: the m=n term is
    ``-df/dE`` and is not divided by omega. It intentionally does not call
    ``kubo_conductivity_imag_axis``, whose public output is conductivity-like.
    """

    omega = config.omega_eV + config.eta_eV
    kernel = np.zeros((2, 2), dtype=complex)
    for weight, (kx, ky) in zip(weights, k_points, strict=True):
        bands = conductivity_eigensystem(float(kx), float(ky), config)
        velocities = [bands.velocity_x_band, bands.velocity_y_band]
        for m, energy_m in enumerate(bands.energies_eV):
            for n, energy_n in enumerate(bands.energies_eV):
                if m == n:
                    response_factor = bands.negative_fermi_derivative[m]
                else:
                    occupation_diff = bands.occupations[m] - bands.occupations[n]
                    if np.isclose(occupation_diff, 0.0):
                        continue
                    energy_diff = energy_m - energy_n
                    if abs(energy_diff) < config.eta_eV:
                        continue
                    response_factor = -occupation_diff * energy_diff / (energy_diff**2 + omega**2)
                for alpha in range(2):
                    for beta in range(2):
                        kernel[alpha, beta] += (
                            weight
                            * response_factor
                            * velocities[alpha][m, n]
                            * velocities[beta][n, m]
                        )
    return kernel


def normal_diamagnetic_kernel(
    k_points: np.ndarray,
    config: KuboConfig,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the normal-state mass-expectation/contact kernel."""

    kernel = np.zeros((2, 2), dtype=complex)
    directions = ("x", "y")
    for weight, (kx, ky) in zip(weights, k_points, strict=True):
        bands = conductivity_eigensystem(float(kx), float(ky), config)
        for alpha, direction_a in enumerate(directions):
            for beta, direction_b in enumerate(directions):
                vertex = normal_state_mass_operator(float(kx), float(ky), direction_a, direction_b)
                vertex_band = bands.states.conjugate().T @ vertex @ bands.states
                kernel[alpha, beta] += weight * np.sum(bands.occupations * np.diag(vertex_band))
    return kernel


def _row_for_kind_omega(
    *,
    kind: str,
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
    pairing = PairingAmplitudes(delta0_eV=DELTA0_EV)

    bdg_para = bdg_paramagnetic_kernel_imag_axis(mesh, config, kind, pairing, weights)  # type: ignore[arg-type]
    bdg_dia = bdg_diamagnetic_kernel(kind, pairing, mesh, config, weights)  # type: ignore[arg-type]
    bdg_total = bdg_total_kernel_imag_axis(mesh, config, kind, pairing, weights).total  # type: ignore[arg-type]

    normal_para = normal_paramagnetic_kernel_imag_axis(mesh, config, weights)
    normal_dia = normal_diamagnetic_kernel(mesh, config, weights)
    normal_total = normal_dia - normal_para

    bdg_norm_para = float(np.linalg.norm(bdg_para))
    bdg_norm_dia = float(np.linalg.norm(bdg_dia))
    bdg_norm_total = float(np.linalg.norm(bdg_total))
    normal_norm_para = float(np.linalg.norm(normal_para))
    normal_norm_dia = float(np.linalg.norm(normal_dia))
    normal_norm_total = float(np.linalg.norm(normal_total))

    row: dict[str, object] = {
        "kind": kind,
        "delta0_eV": DELTA0_EV,
        "omega_eV": float(omega_eV),
        "nk": int(nk),
        "temperature_K": float(temperature_K),
        "eta_eV": float(eta_eV),
        **_matrix_components("bdg_K_para", bdg_para),
        **_matrix_components("bdg_K_dia", bdg_dia),
        **_matrix_components("bdg_K_total", bdg_total),
        "bdg_norm_para": bdg_norm_para,
        "bdg_norm_dia": bdg_norm_dia,
        "bdg_norm_total": bdg_norm_total,
        "bdg_gauge_residual": _norm_ratio(bdg_norm_total, max(bdg_norm_para, bdg_norm_dia)),
        **_matrix_components("normal_K_para", normal_para),
        **_matrix_components("normal_K_dia", normal_dia),
        **_matrix_components("normal_K_total", normal_total),
        "normal_norm_para": normal_norm_para,
        "normal_norm_dia": normal_norm_dia,
        "normal_norm_total": normal_norm_total,
        "normal_gauge_residual": _norm_ratio(normal_norm_total, max(normal_norm_para, normal_norm_dia)),
        "para_ratio_xx": _safe_ratio(bdg_para[0, 0], normal_para[0, 0]),
        "dia_ratio_xx": _safe_ratio(bdg_dia[0, 0], normal_dia[0, 0]),
        "total_ratio_xx": _safe_ratio(bdg_total[0, 0], normal_total[0, 0]),
        "para_sign_match_xx": _sign_match(bdg_para[0, 0], normal_para[0, 0]),
        "dia_sign_match_xx": _sign_match(bdg_dia[0, 0], normal_dia[0, 0]),
        "para_relative_error": _norm_ratio(float(np.linalg.norm(bdg_para - normal_para)), normal_norm_para),
        "dia_relative_error": _norm_ratio(float(np.linalg.norm(bdg_dia - normal_dia)), normal_norm_dia),
        "total_relative_error": _norm_ratio(float(np.linalg.norm(bdg_total - normal_total)), normal_norm_total),
        "bdg_offdiag_ratio": _offdiag_ratio(bdg_total),
        "normal_offdiag_ratio": _offdiag_ratio(normal_total),
        "bdg_C4_anisotropy": _c4_anisotropy(bdg_total),
        "normal_C4_anisotropy": _c4_anisotropy(normal_total),
        "benchmark_only": True,
        "local_response": True,
        "normal_limit_decomposition_diagnostic": True,
        "not_final_response_formula": True,
        "not_final_optical_conductivity": True,
        "not_final_Casimir_input": True,
    }
    return row


def run_diagnostic(
    *,
    kinds: list[str],
    omega_list: list[float],
    nk: int,
    temperature_K: float,
    eta_eV: float,
) -> dict[str, np.ndarray]:
    if nk <= 0:
        raise ValueError("nk must be positive")
    if eta_eV <= 0.0:
        raise ValueError("eta must be positive")
    if not omega_list:
        raise ValueError("omega-list must not be empty")
    if any(omega < 0.0 for omega in omega_list):
        raise ValueError("omega values must be non-negative")

    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    rows = [
        _row_for_kind_omega(
            kind=kind,
            omega_eV=omega,
            nk=nk,
            temperature_K=temperature_K,
            eta_eV=eta_eV,
            mesh=mesh,
            weights=weights,
        )
        for kind in kinds
        for omega in omega_list
    ]

    data: dict[str, np.ndarray] = {}
    complex_columns = {column for prefix in MATRIX_PREFIXES for column in (
        f"{prefix}_xx",
        f"{prefix}_yy",
        f"{prefix}_xy",
        f"{prefix}_yx",
    )}
    complex_columns |= {"para_ratio_xx", "dia_ratio_xx", "total_ratio_xx"}
    bool_columns = {
        "para_sign_match_xx",
        "dia_sign_match_xx",
        "benchmark_only",
        "local_response",
        "normal_limit_decomposition_diagnostic",
        "not_final_response_formula",
        "not_final_optical_conductivity",
        "not_final_Casimir_input",
    }
    int_columns = {"nk"}

    for column in REQUIRED_COLUMNS:
        values = [row[column] for row in rows]
        if column == "kind":
            data[column] = np.asarray(values, dtype="U16")
        elif column in complex_columns:
            data[column] = np.asarray(values, dtype=complex)
        elif column in bool_columns:
            data[column] = np.asarray(values, dtype=bool)
        elif column in int_columns:
            data[column] = np.asarray(values, dtype=int)
        else:
            data[column] = np.asarray(values, dtype=float)
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


def _plot_series(
    *,
    data: dict[str, np.ndarray],
    figure_dir: Path,
    series: list[tuple[str, np.ndarray]],
    ylabel: str,
    title: str,
    filename: str,
    yscale: str | None = None,
) -> Path:
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    path = figure_dir / filename
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    kinds = sorted(set(str(item) for item in data["kind"]))
    for kind in kinds:
        mask = data["kind"] == kind
        order = np.argsort(data["omega_eV"][mask])
        omega = data["omega_eV"][mask][order]
        for label, values in series:
            ax.plot(omega, np.real(values[mask][order]), marker="o", label=f"{kind} {label}")
    ax.set_xlabel("omega (eV)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yscale is not None:
        ax.set_yscale(yscale)
    style_publication_axis(ax)
    save_publication_figure(fig, path)
    plt.close(fig)
    return path


def save_figures(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    return [
        _plot_series(
            data=data,
            figure_dir=figure_dir,
            series=[("BdG K_para_xx", data["bdg_K_para_xx"]), ("normal K_para_xx", data["normal_K_para_xx"])],
            ylabel="kernel component",
            title="BdG vs normal K_para_xx at Delta0=0",
            filename="bdg_vs_normal_K_para_xx_vs_omega.png",
        ),
        _plot_series(
            data=data,
            figure_dir=figure_dir,
            series=[("BdG K_dia_xx", data["bdg_K_dia_xx"]), ("normal K_dia_xx", data["normal_K_dia_xx"])],
            ylabel="kernel component",
            title="BdG vs normal K_dia_xx at Delta0=0",
            filename="bdg_vs_normal_K_dia_xx_vs_omega.png",
        ),
        _plot_series(
            data=data,
            figure_dir=figure_dir,
            series=[("BdG K_total_xx", data["bdg_K_total_xx"]), ("normal K_total_xx", data["normal_K_total_xx"])],
            ylabel="kernel component",
            title="BdG vs normal K_total_xx at Delta0=0",
            filename="bdg_vs_normal_K_total_xx_vs_omega.png",
        ),
        _plot_series(
            data=data,
            figure_dir=figure_dir,
            series=[("para_ratio_xx", data["para_ratio_xx"])],
            ylabel="BdG / normal",
            title="K_para_xx ratio at Delta0=0",
            filename="para_ratio_xx_vs_omega.png",
        ),
        _plot_series(
            data=data,
            figure_dir=figure_dir,
            series=[("dia_ratio_xx", data["dia_ratio_xx"])],
            ylabel="BdG / normal",
            title="K_dia_xx ratio at Delta0=0",
            filename="dia_ratio_xx_vs_omega.png",
        ),
        _plot_series(
            data=data,
            figure_dir=figure_dir,
            series=[
                ("para_relative_error", data["para_relative_error"]),
                ("dia_relative_error", data["dia_relative_error"]),
                ("total_relative_error", data["total_relative_error"]),
            ],
            ylabel="relative error",
            title="BdG-normal kernel relative errors",
            filename="relative_error_vs_omega.png",
            yscale="log",
        ),
    ]


def _format_command(args: argparse.Namespace) -> str:
    parts = ["python", "validation/scripts/numerical_stability/diagnose_bdg_normal_limit_kernel_decomposition.py"]
    if args.quick:
        parts.append("--quick")
    else:
        option_values = [
            ("--kinds", args.kinds),
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
    return output_prefix.parent / "bdg_normal_limit_kernel_decomposition_summary.md"


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
    lowest_omega = float(np.nanmin(data["omega_eV"]))
    omega_mask = np.isclose(data["omega_eV"], lowest_omega)
    mean_errors = {
        "para_relative_error": float(np.nanmean(data["para_relative_error"][omega_mask])),
        "dia_relative_error": float(np.nanmean(data["dia_relative_error"][omega_mask])),
        "total_relative_error": float(np.nanmean(data["total_relative_error"][omega_mask])),
    }
    most_inconsistent = max(mean_errors, key=mean_errors.get)
    lines = [
        "# BdG Normal-Limit Kernel Decomposition Diagnostic",
        "",
        "This is a BdG normal-limit kernel decomposition diagnostic. It compares",
        "Delta0=0 BdG K_para, K_dia, and K_total against locally constructed",
        "normal-state kernel-level K_para and mass-expectation K_dia on the same",
        "mesh and KuboConfig.",
        "",
        "K_total is interpreted as the Peierls/free-energy validated stiffness",
        "kernel K_dia - K_para in the current positive-bubble convention.",
        "",
        "The purpose is to locate whether static stiffness mismatch is tied to",
        "the paramagnetic bubble, the diamagnetic/contact term, sign convention,",
        "Nambu redundancy, or occupation convention. It is not a final response",
        "formula selection.",
        "",
        "This diagnostic does not modify the formal BdG response formula.",
        "It does not modify Casimir calculations.",
        "It contains no finite momentum response.",
        "It is not a final optical conductivity or Casimir input.",
        "",
        f"run_command = `{command}`",
        f"quick_test_only={bool(args.quick)}",
        "benchmark_only=True",
        "local_response=True",
        "normal_limit_decomposition_diagnostic=True",
        "delta0_eV=0.0",
        "not_final_response_formula=True",
        "not_final_optical_conductivity=True",
        "not_final_Casimir_input=True",
        "",
        "## Parameters",
        f"- kinds={', '.join(args.kinds)}",
        f"- omega_list={', '.join(f'{value:g}' for value in args.omega_list)}",
        f"- nk={args.nk}",
        f"- temperature_K={args.temperature:g}",
        f"- eta_eV={args.eta:g}",
        "",
        f"## Delta0=0 Ratios At Lowest Omega ({lowest_omega:g} eV)",
        "",
        "Ratios use K_total = K_dia - K_para.",
    ]
    for kind in sorted(set(str(item) for item in data["kind"])):
        mask = omega_mask & (data["kind"] == kind)
        if not np.any(mask):
            continue
        index = int(np.flatnonzero(mask)[0])
        lines.append(
            "- "
            f"{kind}: para_ratio_xx={data['para_ratio_xx'][index]:.6g}, "
            f"dia_ratio_xx={data['dia_ratio_xx'][index]:.6g}, "
            f"total_ratio_xx={data['total_ratio_xx'][index]:.6g}, "
            f"para_relative_error={data['para_relative_error'][index]:.6g}, "
            f"dia_relative_error={data['dia_relative_error'][index]:.6g}, "
            f"total_relative_error={data['total_relative_error'][index]:.6g}"
        )
    lines.extend(
        [
            "",
            "## Most Inconsistent Piece",
            f"lowest_omega_mean_para_relative_error={mean_errors['para_relative_error']:.6g}",
            f"lowest_omega_mean_dia_relative_error={mean_errors['dia_relative_error']:.6g}",
            f"lowest_omega_mean_total_relative_error={mean_errors['total_relative_error']:.6g}",
            f"largest_lowest_omega_relative_error={most_inconsistent}",
            "",
            "## Next Step",
            "Use this decomposition to decide which term needs analytic review.",
            "Do not treat any ratio or sign here as a formula fix without a",
            "separate derivation and validation pass.",
            "",
            "## Figures",
        ]
    )
    lines.extend(f"- {_display_path(path)}" for path in figure_paths)
    summary_path = _summary_path(args.output_prefix)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
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
    print("note = BdG normal-limit kernel decomposition diagnostic only; not final optical conductivity or Casimir input.")


if __name__ == "__main__":
    main()
