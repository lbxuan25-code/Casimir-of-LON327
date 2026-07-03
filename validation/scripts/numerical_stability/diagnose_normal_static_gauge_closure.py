#!/usr/bin/env python3
"""Diagnose normal-state static gauge closure with a Peierls-twist baseline.

This script is diagnostic-only. It constructs normal-state kernel-level
paramagnetic and diamagnetic pieces locally, and compares convention candidates
against a finite-difference free-energy stiffness from H(k - A).
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import KuboConfig, k_weights, uniform_bz_mesh  # noqa: E402
from lno327.conductivity import conductivity_eigensystem  # noqa: E402
from lno327.models.lno327_four_orbital.normal import normal_state_hamiltonian  # noqa: E402
from lno327.models.lno327_four_orbital.vertices import normal_state_mass_operator  # noqa: E402
from validation.lib.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

DEFAULT_OMEGA_LIST = (0.0, 1e-6, 1e-5, 1e-4)
DEFAULT_NK_LIST = (8, 12, 16, 24)
DEFAULT_TWIST_LIST = (1e-3, 5e-4, 2e-4)
EPS = 1e-300
OUTPUT_ROOT = ROOT / "validation" / "outputs" / "response" / "normal_static_gauge_closure"
DEFAULT_OUTPUT_PREFIX = OUTPUT_ROOT / "data" / "normal_static_gauge_closure"
SUMMARY_PATH = OUTPUT_ROOT / "normal_static_gauge_closure_summary.md"

CANDIDATES = (
    ("para_plus_dia", 1.0, 1.0),
    ("minus_para_plus_dia", -1.0, 1.0),
    ("para_minus_dia", 1.0, -1.0),
    ("minus_para_minus_dia", -1.0, -1.0),
)

MATRIX_PREFIXES = (
    "K_para_intra",
    "K_para_inter",
    "K_para_total",
    "K_dia",
)

REQUIRED_COLUMNS = (
    "omega_eV",
    "nk",
    "temperature_K",
    "eta_eV",
    "twist_A",
    "K_para_intra_xx",
    "K_para_intra_yy",
    "K_para_intra_xy",
    "K_para_intra_yx",
    "K_para_inter_xx",
    "K_para_inter_yy",
    "K_para_inter_xy",
    "K_para_inter_yx",
    "K_para_total_xx",
    "K_para_total_yy",
    "K_para_total_xy",
    "K_para_total_yx",
    "K_dia_xx",
    "K_dia_yy",
    "K_dia_xy",
    "K_dia_yx",
    "D_fd_xx",
    "D_fd_yy",
    "D_fd_xy",
    "para_plus_dia_xx",
    "minus_para_plus_dia_xx",
    "para_minus_dia_xx",
    "minus_para_minus_dia_xx",
    "best_candidate_name",
    "best_candidate_error",
    "candidate_minus_fd_norm",
    "relative_error_to_fd",
    "C4_anisotropy",
    "offdiag_ratio",
    "norm_para_intra",
    "norm_para_inter",
    "norm_para_total",
    "norm_dia",
    "benchmark_only",
    "local_response",
    "normal_static_gauge_closure_diagnostic",
    "peierls_twist_diagnostic",
    "not_final_response_formula",
    "not_final_optical_conductivity",
    "not_final_Casimir_input",
)


def _quick_overrides() -> dict[str, object]:
    return {
        "nk_list": [6, 8],
        "omega_list": [0.0, 1e-4],
        "twist_list": [1e-3],
    }


def _matrix_components(prefix: str, matrix: np.ndarray) -> dict[str, complex]:
    return {
        f"{prefix}_xx": complex(matrix[0, 0]),
        f"{prefix}_yy": complex(matrix[1, 1]),
        f"{prefix}_xy": complex(matrix[0, 1]),
        f"{prefix}_yx": complex(matrix[1, 0]),
    }


def _norm_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(float(denominator), EPS))


def _offdiag_ratio(matrix: np.ndarray) -> float:
    offdiag = float(np.linalg.norm([matrix[0, 1], matrix[1, 0]]))
    diag = float(np.linalg.norm([matrix[0, 0], matrix[1, 1]]))
    return _norm_ratio(offdiag, diag)


def _c4_anisotropy(matrix: np.ndarray) -> float:
    denominator = matrix[0, 0] + matrix[1, 1]
    if abs(denominator) <= EPS:
        return 0.0
    return float(np.real((matrix[0, 0] - matrix[1, 1]) / denominator))


def normal_paramagnetic_decomposition(
    mesh: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return kernel-level normal K_para_intra and K_para_inter."""

    omega = config.omega_eV + config.eta_eV
    intra = np.zeros((2, 2), dtype=complex)
    inter = np.zeros((2, 2), dtype=complex)
    for weight, (kx, ky) in zip(weights, mesh, strict=True):
        bands = conductivity_eigensystem(float(kx), float(ky), config)
        velocities = [bands.velocity_x_band, bands.velocity_y_band]
        for m, energy_m in enumerate(bands.energies_eV):
            for n, energy_n in enumerate(bands.energies_eV):
                if m == n:
                    response_factor = bands.negative_fermi_derivative[m]
                    target = intra
                else:
                    occupation_diff = bands.occupations[m] - bands.occupations[n]
                    if np.isclose(occupation_diff, 0.0):
                        continue
                    energy_diff = energy_m - energy_n
                    if abs(energy_diff) < config.eta_eV:
                        continue
                    response_factor = -occupation_diff * energy_diff / (energy_diff**2 + omega**2)
                    target = inter
                for alpha in range(2):
                    for beta in range(2):
                        target[alpha, beta] += (
                            weight
                            * response_factor
                            * velocities[alpha][m, n]
                            * velocities[beta][n, m]
                        )
    return intra, inter


def normal_diamagnetic_kernel(mesh: np.ndarray, weights: np.ndarray, config: KuboConfig) -> np.ndarray:
    """Return normal-state mass-expectation/contact kernel."""

    kernel = np.zeros((2, 2), dtype=complex)
    directions = ("x", "y")
    for weight, (kx, ky) in zip(weights, mesh, strict=True):
        bands = conductivity_eigensystem(float(kx), float(ky), config)
        for alpha, direction_a in enumerate(directions):
            for beta, direction_b in enumerate(directions):
                vertex = normal_state_mass_operator(float(kx), float(ky), direction_a, direction_b)
                vertex_band = bands.states.conjugate().T @ vertex @ bands.states
                kernel[alpha, beta] += weight * np.sum(bands.occupations * np.diag(vertex_band))
    return kernel


def grand_potential_density(mesh: np.ndarray, weights: np.ndarray, config: KuboConfig, ax: float, ay: float) -> float:
    """Return weighted normal-state grand potential density for H(k - A)."""

    total = 0.0
    temperature = float(config.temperature_eV)
    for weight, (kx, ky) in zip(weights, mesh, strict=True):
        energies = np.linalg.eigvalsh(normal_state_hamiltonian(float(kx) - ax, float(ky) - ay))
        shifted = energies - config.fermi_level_eV
        if temperature <= 0.0:
            contribution = np.sum(shifted[shifted < 0.0])
        else:
            contribution = -temperature * np.sum(np.logaddexp(0.0, -shifted / temperature))
        total += float(weight * contribution)
    return float(total)


def peierls_stiffness_fd(mesh: np.ndarray, weights: np.ndarray, config: KuboConfig, twist: float) -> np.ndarray:
    """Return finite-difference stiffness matrix from normal-state F(A)."""

    f0 = grand_potential_density(mesh, weights, config, 0.0, 0.0)
    fpx = grand_potential_density(mesh, weights, config, twist, 0.0)
    fmx = grand_potential_density(mesh, weights, config, -twist, 0.0)
    fpy = grand_potential_density(mesh, weights, config, 0.0, twist)
    fmy = grand_potential_density(mesh, weights, config, 0.0, -twist)
    fpp = grand_potential_density(mesh, weights, config, twist, twist)
    fpm = grand_potential_density(mesh, weights, config, twist, -twist)
    fmp = grand_potential_density(mesh, weights, config, -twist, twist)
    fmm = grand_potential_density(mesh, weights, config, -twist, -twist)
    dxx = (fpx + fmx - 2.0 * f0) / (twist**2)
    dyy = (fpy + fmy - 2.0 * f0) / (twist**2)
    dxy = (fpp - fpm - fmp + fmm) / (4.0 * twist**2)
    return np.array([[dxx, dxy], [dxy, dyy]], dtype=complex)


def _row_for_case(
    *,
    omega_eV: float,
    nk: int,
    temperature_K: float,
    eta_eV: float,
    twist: float,
) -> dict[str, object]:
    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    para_intra, para_inter = normal_paramagnetic_decomposition(mesh, weights, config)
    para_total = para_intra + para_inter
    dia = normal_diamagnetic_kernel(mesh, weights, config)
    d_fd = peierls_stiffness_fd(mesh, weights, config, twist)
    candidates = {
        name: para_prefactor * para_total + dia_prefactor * dia
        for name, para_prefactor, dia_prefactor in CANDIDATES
    }
    candidate_errors = {
        name: float(np.linalg.norm(matrix - d_fd))
        for name, matrix in candidates.items()
    }
    best_name = min(candidate_errors, key=candidate_errors.get)
    best_matrix = candidates[best_name]
    best_error = candidate_errors[best_name]
    fd_norm = float(np.linalg.norm(d_fd))
    row: dict[str, object] = {
        "omega_eV": float(omega_eV),
        "nk": int(nk),
        "temperature_K": float(temperature_K),
        "eta_eV": float(eta_eV),
        "twist_A": float(twist),
        **_matrix_components("K_para_intra", para_intra),
        **_matrix_components("K_para_inter", para_inter),
        **_matrix_components("K_para_total", para_total),
        **_matrix_components("K_dia", dia),
        "D_fd_xx": complex(d_fd[0, 0]),
        "D_fd_yy": complex(d_fd[1, 1]),
        "D_fd_xy": complex(d_fd[0, 1]),
        "para_plus_dia_xx": complex(candidates["para_plus_dia"][0, 0]),
        "minus_para_plus_dia_xx": complex(candidates["minus_para_plus_dia"][0, 0]),
        "para_minus_dia_xx": complex(candidates["para_minus_dia"][0, 0]),
        "minus_para_minus_dia_xx": complex(candidates["minus_para_minus_dia"][0, 0]),
        "best_candidate_name": best_name,
        "best_candidate_error": best_error,
        "candidate_minus_fd_norm": best_error,
        "relative_error_to_fd": _norm_ratio(best_error, fd_norm),
        "C4_anisotropy": _c4_anisotropy(best_matrix),
        "offdiag_ratio": _offdiag_ratio(best_matrix),
        "norm_para_intra": float(np.linalg.norm(para_intra)),
        "norm_para_inter": float(np.linalg.norm(para_inter)),
        "norm_para_total": float(np.linalg.norm(para_total)),
        "norm_dia": float(np.linalg.norm(dia)),
        "benchmark_only": True,
        "local_response": True,
        "normal_static_gauge_closure_diagnostic": True,
        "peierls_twist_diagnostic": True,
        "not_final_response_formula": True,
        "not_final_optical_conductivity": True,
        "not_final_Casimir_input": True,
    }
    return row


def run_diagnostic(
    *,
    omega_list: list[float],
    nk_list: list[int],
    temperature_K: float,
    eta_eV: float,
    twist_list: list[float],
) -> dict[str, np.ndarray]:
    if eta_eV <= 0.0:
        raise ValueError("eta must be positive")
    if not omega_list:
        raise ValueError("omega-list must not be empty")
    if not nk_list:
        raise ValueError("nk-list must not be empty")
    if not twist_list:
        raise ValueError("twist-list must not be empty")
    if any(omega < 0.0 for omega in omega_list):
        raise ValueError("omega values must be non-negative")
    if any(nk <= 0 for nk in nk_list):
        raise ValueError("nk values must be positive")
    if any(twist <= 0.0 for twist in twist_list):
        raise ValueError("twist values must be positive")

    rows = [
        _row_for_case(
            omega_eV=omega,
            nk=nk,
            temperature_K=temperature_K,
            eta_eV=eta_eV,
            twist=twist,
        )
        for omega in omega_list
        for nk in nk_list
        for twist in twist_list
    ]

    data: dict[str, np.ndarray] = {}
    complex_columns = {
        column
        for prefix in MATRIX_PREFIXES
        for column in (f"{prefix}_xx", f"{prefix}_yy", f"{prefix}_xy", f"{prefix}_yx")
    }
    complex_columns |= {
        "D_fd_xx",
        "D_fd_yy",
        "D_fd_xy",
        "para_plus_dia_xx",
        "minus_para_plus_dia_xx",
        "para_minus_dia_xx",
        "minus_para_minus_dia_xx",
    }
    bool_columns = {
        "benchmark_only",
        "local_response",
        "normal_static_gauge_closure_diagnostic",
        "peierls_twist_diagnostic",
        "not_final_response_formula",
        "not_final_optical_conductivity",
        "not_final_Casimir_input",
    }
    for column in REQUIRED_COLUMNS:
        values = [row[column] for row in rows]
        if column == "best_candidate_name":
            data[column] = np.asarray(values, dtype="U32")
        elif column in complex_columns:
            data[column] = np.asarray(values, dtype=complex)
        elif column in bool_columns:
            data[column] = np.asarray(values, dtype=bool)
        elif column == "nk":
            data[column] = np.asarray(values, dtype=int)
        else:
            data[column] = np.asarray(values, dtype=float)
    data["omega_list"] = np.asarray(omega_list, dtype=float)
    data["nk_list"] = np.asarray(nk_list, dtype=int)
    data["twist_list"] = np.asarray(twist_list, dtype=float)
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
        for index in range(data["omega_eV"].size):
            writer.writerow({column: _csv_value(data[column][index]) for column in REQUIRED_COLUMNS})
    figure_paths = save_figures(data, figure_dir)
    return npz_path, csv_path, figure_paths


def _plot_vs_nk(
    *,
    data: dict[str, np.ndarray],
    figure_dir: Path,
    values: list[tuple[str, np.ndarray]],
    ylabel: str,
    title: str,
    filename: str,
    omega_filter: float | None = None,
    yscale: str | None = None,
) -> Path:
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    path = figure_dir / filename
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    omega = float(np.nanmin(data["omega_eV"])) if omega_filter is None else float(omega_filter)
    for twist in data["twist_list"]:
        mask = np.isclose(data["omega_eV"], omega) & np.isclose(data["twist_A"], twist)
        order = np.argsort(data["nk"][mask])
        x = data["nk"][mask][order]
        for label, array in values:
            ax.plot(x, np.real(array[mask][order]), marker="o", label=f"{label}, A={twist:g}")
    ax.set_xlabel("nk")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} at omega={omega:g} eV")
    if yscale is not None:
        ax.set_yscale(yscale)
    style_publication_axis(ax)
    save_publication_figure(fig, path)
    plt.close(fig)
    return path


def _plot_best_error_vs_omega(data: dict[str, np.ndarray], figure_dir: Path) -> Path:
    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    path = figure_dir / "best_candidate_error_vs_omega.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for nk in data["nk_list"]:
        for twist in data["twist_list"]:
            mask = (data["nk"] == nk) & np.isclose(data["twist_A"], twist)
            order = np.argsort(data["omega_eV"][mask])
            ax.plot(
                data["omega_eV"][mask][order],
                data["best_candidate_error"][mask][order],
                marker="o",
                label=f"nk={nk}, A={twist:g}",
            )
    ax.set_xlabel("omega (eV)")
    ax.set_ylabel("best candidate error")
    ax.set_title("Best candidate error vs omega")
    ax.set_yscale("log")
    style_publication_axis(ax)
    save_publication_figure(fig, path)
    plt.close(fig)
    return path


def save_figures(data: dict[str, np.ndarray], figure_dir: Path) -> list[Path]:
    return [
        _plot_vs_nk(
            data=data,
            figure_dir=figure_dir,
            values=[("D_fd_xx", data["D_fd_xx"])],
            ylabel="D_fd_xx",
            title="Peierls free-energy stiffness",
            filename="D_fd_xx_vs_nk.png",
        ),
        _plot_vs_nk(
            data=data,
            figure_dir=figure_dir,
            values=[
                ("para_plus_dia", data["para_plus_dia_xx"]),
                ("minus_para_plus_dia", data["minus_para_plus_dia_xx"]),
                ("para_minus_dia", data["para_minus_dia_xx"]),
                ("minus_para_minus_dia", data["minus_para_minus_dia_xx"]),
            ],
            ylabel="candidate K_xx",
            title="Candidate K_xx conventions",
            filename="candidate_K_xx_vs_nk.png",
        ),
        _plot_vs_nk(
            data=data,
            figure_dir=figure_dir,
            values=[("best_candidate_error", data["best_candidate_error"])],
            ylabel="candidate error",
            title="Best candidate error",
            filename="candidate_error_vs_nk.png",
            yscale="log",
        ),
        _plot_vs_nk(
            data=data,
            figure_dir=figure_dir,
            values=[
                ("K_para_intra_xx", data["K_para_intra_xx"]),
                ("K_para_inter_xx", data["K_para_inter_xx"]),
                ("K_dia_xx", data["K_dia_xx"]),
            ],
            ylabel="kernel component",
            title="Normal kernel decomposition",
            filename="intra_inter_dia_decomposition_vs_nk.png",
        ),
        _plot_best_error_vs_omega(data, figure_dir),
    ]


def _format_command(args: argparse.Namespace) -> str:
    parts = ["python", "validation/scripts/numerical_stability/diagnose_normal_static_gauge_closure.py"]
    if args.quick:
        parts.append("--quick")
    else:
        option_values = [
            ("--omega-list", args.omega_list),
            ("--nk-list", args.nk_list),
            ("--temperature", [args.temperature]),
            ("--eta", [args.eta]),
            ("--twist-list", args.twist_list),
            ("--output-prefix", [args.output_prefix]),
        ]
        for option, values in option_values:
            parts.append(option)
            parts.extend(str(value) for value in values)
    return " ".join(shlex.quote(str(part)) for part in parts)


def _summary_path(output_prefix: Path) -> Path:
    if output_prefix.resolve() == DEFAULT_OUTPUT_PREFIX.resolve():
        return SUMMARY_PATH
    return output_prefix.parent / "normal_static_gauge_closure_summary.md"


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
    omega0 = float(np.nanmin(data["omega_eV"]))
    twist0 = float(np.nanmin(data["twist_A"]))
    static_mask = np.isclose(data["omega_eV"], omega0) & np.isclose(data["twist_A"], twist0)
    best_names, counts = np.unique(data["best_candidate_name"][static_mask], return_counts=True)
    dominant_best = str(best_names[int(np.argmax(counts))]) if best_names.size else "none"
    largest_norm_name = "none"
    if np.any(static_mask):
        norms = {
            "K_para_intra": float(np.nanmean(data["norm_para_intra"][static_mask])),
            "K_para_inter": float(np.nanmean(data["norm_para_inter"][static_mask])),
            "K_dia": float(np.nanmean(data["norm_dia"][static_mask])),
        }
        largest_norm_name = max(norms, key=norms.get)
    fd_by_nk = []
    for nk in sorted(set(int(item) for item in data["nk"][static_mask])):
        mask = static_mask & (data["nk"] == nk)
        if np.any(mask):
            fd_by_nk.append((nk, float(np.nanmean(np.real(data["D_fd_xx"][mask])))))
    lines = [
        "# Normal-State Static Gauge Closure Diagnostic",
        "",
        "This is a normal-state static gauge closure diagnostic. It uses a",
        "Peierls-twist finite-difference free-energy stiffness as an independent",
        "baseline for checking normal kernel conventions.",
        "The Peierls baseline is a stiffness reference; this diagnostic does not",
        "assume clean normal-state stiffness must vanish at finite mesh.",
        "",
        "The purpose is not to choose a final response formula. It is to locate",
        "whether static closure failure is tied to normal K_para sign, K_dia",
        "sign/contact convention, the mass operator, or the intra/inter balance.",
        "",
        "This diagnostic does not modify the formal response formula.",
        "It does not modify BdG, Casimir, reflection, or finite-q code.",
        "It is not a final optical conductivity or Casimir input.",
        "",
        f"run_command = `{command}`",
        f"quick_test_only={bool(args.quick)}",
        "benchmark_only=True",
        "local_response=True",
        "normal_static_gauge_closure_diagnostic=True",
        "peierls_twist_diagnostic=True",
        "not_final_response_formula=True",
        "not_final_optical_conductivity=True",
        "not_final_Casimir_input=True",
        "",
        "## Parameters",
        f"- omega_list={', '.join(f'{value:g}' for value in args.omega_list)}",
        f"- nk_list={', '.join(str(value) for value in args.nk_list)}",
        f"- temperature_K={args.temperature:g}",
        f"- eta_eV={args.eta:g}",
        f"- twist_list={', '.join(f'{value:g}' for value in args.twist_list)}",
        "",
        f"## Peierls D_fd Trend At omega={omega0:g}, twist={twist0:g}",
    ]
    lines.extend(f"- nk={nk}: D_fd_xx={value:.6g}" for nk, value in fd_by_nk)
    lines.extend(
        [
            "",
            "## Candidate Convention",
            f"dominant_best_candidate={dominant_best}",
            "minus_para_plus_dia means K_dia - K_para.",
            f"largest_mean_static_component_norm={largest_norm_name}",
            "",
            "## Next Step",
            "Use the Peierls baseline and intra/inter/dia decomposition to decide",
            "which normal-state convention needs analytic review. Do not treat the",
            "best candidate reported here as a formula fix without a derivation.",
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
    parser.add_argument("--omega-list", nargs="+", type=float, default=list(DEFAULT_OMEGA_LIST))
    parser.add_argument("--nk-list", nargs="+", type=int, default=list(DEFAULT_NK_LIST))
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--twist-list", nargs="+", type=float, default=list(DEFAULT_TWIST_LIST))
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
        omega_list=list(args.omega_list),
        nk_list=list(args.nk_list),
        temperature_K=float(args.temperature),
        eta_eV=float(args.eta),
        twist_list=list(args.twist_list),
    )
    npz_path, csv_path, figure_paths = save_outputs(data, args.output_prefix)
    summary_path = write_summary(data, args, command, figure_paths)
    print(f"npz_path = {npz_path}")
    print(f"csv_path = {csv_path}")
    print(f"summary_path = {summary_path}")
    print("figure_paths = " + ", ".join(str(path) for path in figure_paths))
    print("note = normal-state static gauge closure diagnostic only; not final optical conductivity or Casimir input.")


if __name__ == "__main__":
    main()
