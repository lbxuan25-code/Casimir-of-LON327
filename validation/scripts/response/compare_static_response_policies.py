#!/usr/bin/env python3
"""Compare conservative n=0 Matsubara response policies.

This script is diagnostic only. The current local baseline omits the n=0 term;
the extrapolated and static-kernel branches are sensitivity checks, not final
Casimir inputs.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import (  # noqa: E402
    CasimirSetup,
    PairingAmplitudes,
    casimir_energy_integrand,
    casimir_torque_integrand,
    local_response_matsubara_index,
    matrix_symmetry_diagnostics,
    model_response_to_sheet_conductivity,
)
from lno327.constants import SIGMA0  # noqa: E402
from lno327.conductivity import ConductivityTensor  # noqa: E402
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

POLICIES = ("skip", "extrapolate_from_lowest_matsubara", "use_static_kernel")
KINDS = ("normal", "spm", "dwave")

REQUIRED_NPZ_FIELDS = {
    "kind",
    "policy",
    "status",
    "approximate",
    "matrix_finite",
    "response_xx",
    "response_yy",
    "response_xy",
    "response_yx",
    "delta",
    "relative_offdiag",
    "relative_eigen_split",
    "sheet_conductivity_xx",
    "reflection_dimensionless_xx",
    "energy_integrand_n0",
    "torque_integrand_n0",
    "not_used_as_sigma",
    "notes",
}


def _complex_nan() -> complex:
    return complex(np.nan, np.nan)


def _tensor_from_sheet_matrix(matrix: np.ndarray) -> ConductivityTensor:
    return ConductivityTensor(
        xx=matrix[0, 0],
        yy=matrix[1, 1],
        xy=matrix[0, 1],
        yx=matrix[1, 0],
    )


def _notes_for_policy(policy: str, result_notes: tuple[str, ...]) -> tuple[str, ...]:
    if policy == "skip":
        return (
            "n=0 omitted in current local baseline",
            "omission avoids undefined superconducting zero-frequency conductivity",
        )
    if policy == "extrapolate_from_lowest_matsubara":
        return (
            *result_notes,
            "sensitivity estimate only, not final n=0 physics",
        )
    return (
        *result_notes,
        "static kernel is not Sigma_SC(0)",
        "static kernel is not used as sheet conductivity or reflection input",
    )


def compare_static_policies(
    kinds: list[str],
    policies: list[str],
    nk: int,
    temperature_K: float,
    delta0_eV: float,
    eta_eV: float,
    distance_m: float,
    k_parallel: float,
    phi: float,
    theta: float,
) -> dict[str, np.ndarray]:
    """Evaluate n=0 policies and optional diagnostic integrands."""

    if nk <= 0:
        raise ValueError("nk must be positive")
    if temperature_K <= 0.0:
        raise ValueError("temperature must be positive")
    if eta_eV <= 0.0:
        raise ValueError("eta must be positive")
    if distance_m <= 0.0:
        raise ValueError("distance must be positive")
    if k_parallel <= 0.0:
        raise ValueError("k_parallel must be positive for n=0 reflection diagnostics")

    rows = [(kind, policy) for kind in kinds for policy in policies]
    row_count = len(rows)
    data: dict[str, np.ndarray] = {
        "kind": np.empty(row_count, dtype="U16"),
        "policy": np.empty(row_count, dtype="U48"),
        "status": np.empty(row_count, dtype="U48"),
        "approximate": np.zeros(row_count, dtype=bool),
        "matrix_finite": np.zeros(row_count, dtype=bool),
        "response_xx": np.full(row_count, _complex_nan(), dtype=complex),
        "response_yy": np.full(row_count, _complex_nan(), dtype=complex),
        "response_xy": np.full(row_count, _complex_nan(), dtype=complex),
        "response_yx": np.full(row_count, _complex_nan(), dtype=complex),
        "delta": np.full(row_count, _complex_nan(), dtype=complex),
        "relative_offdiag": np.full(row_count, np.nan, dtype=float),
        "relative_eigen_split": np.full(row_count, np.nan, dtype=float),
        "sheet_conductivity_xx": np.full(row_count, _complex_nan(), dtype=complex),
        "reflection_dimensionless_xx": np.full(row_count, _complex_nan(), dtype=complex),
        "energy_integrand_n0": np.full(row_count, _complex_nan(), dtype=complex),
        "torque_integrand_n0": np.full(row_count, _complex_nan(), dtype=complex),
        "not_used_as_sigma": np.zeros(row_count, dtype=bool),
        "notes": np.empty(row_count, dtype=object),
        "nk": np.array(nk),
        "temperature_K": np.array(temperature_K),
        "delta0_eV": np.array(delta0_eV),
        "eta_eV": np.array(eta_eV),
        "distance_m": np.array(distance_m),
        "k_parallel": np.array(k_parallel),
        "phi": np.array(phi),
        "theta": np.array(theta),
    }

    setup = CasimirSetup(temperature=temperature_K, distance=distance_m)
    params = PairingAmplitudes(delta0_eV=delta0_eV)

    for index, (kind, policy) in enumerate(rows):
        result = local_response_matsubara_index(
            kind,  # type: ignore[arg-type]
            0,
            temperature_K,
            policy=policy,  # type: ignore[arg-type]
            nk=nk,
            eta_eV=eta_eV,
            pairing_params=params,
        )

        data["kind"][index] = kind
        data["policy"][index] = policy
        data["status"][index] = result.status
        data["approximate"][index] = result.approximate
        data["not_used_as_sigma"][index] = policy in {"skip", "use_static_kernel"}
        data["notes"][index] = _notes_for_policy(policy, result.notes)

        if result.matrix is None:
            continue

        matrix = np.asarray(result.matrix, dtype=complex)
        finite = bool(np.isfinite(matrix).all())
        data["matrix_finite"][index] = finite
        data["response_xx"][index] = matrix[0, 0]
        data["response_yy"][index] = matrix[1, 1]
        data["response_xy"][index] = matrix[0, 1]
        data["response_yx"][index] = matrix[1, 0]
        diagnostics = matrix_symmetry_diagnostics(matrix)
        data["delta"][index] = complex(diagnostics["delta"])
        data["relative_offdiag"][index] = float(diagnostics["relative_offdiag"])
        data["relative_eigen_split"][index] = float(diagnostics["relative_eigen_split"])

        if policy != "extrapolate_from_lowest_matsubara" or not finite:
            continue

        sheet = model_response_to_sheet_conductivity(matrix)
        sheet_matrix = sheet.tensor.matrix()
        data["sheet_conductivity_xx"][index] = sheet_matrix[0, 0]
        data["reflection_dimensionless_xx"][index] = sheet_matrix[0, 0] / SIGMA0
        tensor = _tensor_from_sheet_matrix(sheet_matrix)
        data["energy_integrand_n0"][index] = casimir_energy_integrand(
            setup,
            0.0,
            k_parallel,
            phi,
            theta,
            tensor,
            tensor,
        )
        data["torque_integrand_n0"][index] = casimir_torque_integrand(
            setup,
            0.0,
            k_parallel,
            phi,
            theta,
            tensor,
            tensor,
        )

    return data


def output_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    resolved_prefix = output_prefix.resolve()
    project_data_root = (ROOT / "validation" / "outputs" / "archive" / "response" / "static_policy_comparison" / "data").resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "validation" / "outputs" / "archive" / "response" / "static_policy_comparison" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return (
        npz_path,
        csv_path,
        figure_dir / f"{output_prefix.name}_response_xx.png",
        figure_dir / f"{output_prefix.name}_anisotropy.png",
        figure_dir / f"{output_prefix.name}_torque_integrand_n0.png",
    )


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    return value


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path, Path, Path, Path | None]:
    npz_path, csv_path, response_plot_path, anisotropy_plot_path, torque_plot_path = output_paths(output_prefix)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    response_plot_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)

    fieldnames = [
        "kind",
        "policy",
        "status",
        "approximate",
        "matrix_finite",
        "response_xx",
        "response_yy",
        "response_xy",
        "response_yx",
        "delta",
        "relative_offdiag",
        "relative_eigen_split",
        "sheet_conductivity_xx",
        "reflection_dimensionless_xx",
        "energy_integrand_n0",
        "torque_integrand_n0",
        "not_used_as_sigma",
        "notes",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(data["kind"].size):
            writer.writerow({name: _csv_value(data[name][index]) for name in fieldnames})

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    kinds = list(dict.fromkeys(str(kind) for kind in data["kind"]))
    x_positions = np.arange(len(kinds), dtype=float)
    width = 0.24

    fig_response, ax_response = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    fig_aniso, ax_aniso = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    fig_torque, ax_torque = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)

    for offset, policy in enumerate(POLICIES):
        values = []
        delta_values = []
        offdiag_values = []
        torque_values = []
        for kind in kinds:
            mask = (data["kind"] == kind) & (data["policy"] == policy)
            values.append(float(np.real(data["response_xx"][mask][0])) if np.any(mask) else np.nan)
            delta_values.append(float(abs(data["delta"][mask][0])) if np.any(mask) else np.nan)
            offdiag_values.append(float(data["relative_offdiag"][mask][0]) if np.any(mask) else np.nan)
            torque_values.append(float(np.real(data["torque_integrand_n0"][mask][0])) if np.any(mask) else np.nan)

        shift = (offset - 1) * width
        ax_response.bar(x_positions + shift, values, width=width, label=policy)
        ax_aniso.plot(x_positions, delta_values, marker="o", label=f"{policy} |delta|")
        ax_aniso.plot(x_positions, offdiag_values, marker="s", linestyle="--", label=f"{policy} offdiag")
        ax_torque.bar(x_positions + shift, torque_values, width=width, label=policy)

    ax_response.set_xticks(x_positions, kinds)
    ax_response.set_ylabel("Re response_xx / static_kernel_xx")
    ax_response.set_title("n=0 policy response diagnostic")
    style_publication_axis(ax_response)
    save_publication_figure(fig_response, response_plot_path)
    plt.close(fig_response)

    ax_aniso.set_xticks(x_positions, kinds)
    ax_aniso.set_ylabel("relative diagnostic")
    aniso_values = np.concatenate(
        [
            np.abs(data["delta"][np.isfinite(np.abs(data["delta"]))]),
            data["relative_offdiag"][np.isfinite(data["relative_offdiag"])],
        ]
    )
    if aniso_values.size > 0 and bool(np.any(aniso_values > 0.0)):
        ax_aniso.set_yscale("log")
    ax_aniso.set_title("n=0 policy anisotropy diagnostics")
    style_publication_axis(ax_aniso)
    save_publication_figure(fig_aniso, anisotropy_plot_path)
    plt.close(fig_aniso)

    finite_torque = np.isfinite(np.real(data["torque_integrand_n0"]))
    saved_torque_path: Path | None = None
    if bool(np.any(finite_torque)):
        ax_torque.set_xticks(x_positions, kinds)
        ax_torque.set_ylabel("Re torque_integrand_n0")
        ax_torque.set_title("n=0 extrapolated torque integrand diagnostic")
        style_publication_axis(ax_torque)
        save_publication_figure(fig_torque, torque_plot_path)
        saved_torque_path = torque_plot_path
    plt.close(fig_torque)

    return npz_path, csv_path, response_plot_path, anisotropy_plot_path, saved_torque_path


def print_summary(data: dict[str, np.ndarray]) -> None:
    for kind in dict.fromkeys(str(item) for item in data["kind"]):
        mask = data["kind"] == kind
        max_delta = float(np.nanmax(np.abs(data["delta"][mask])))
        max_offdiag = float(np.nanmax(data["relative_offdiag"][mask]))
        print(f"kind = {kind}")
        print(f"max_abs_delta = {max_delta}")
        print(f"max_relative_offdiag = {max_offdiag}")
        for policy in POLICIES:
            row = mask & (data["policy"] == policy)
            if not np.any(row):
                continue
            index = int(np.flatnonzero(row)[0])
            print(
                f"  policy = {policy}; status = {data['status'][index]}; "
                f"approximate = {bool(data['approximate'][index])}; "
                f"matrix_finite = {bool(data['matrix_finite'][index])}; "
                f"not_used_as_sigma = {bool(data['not_used_as_sigma'][index])}; "
                f"torque_integrand_n0 = {data['torque_integrand_n0'][index]}"
            )
    print("recommendation = local baseline uses skip; extrapolate/static_kernel are diagnostics only.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--nk", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=30.0, help="Temperature in K.")
    parser.add_argument("--delta0", type=float, default=0.04, help="Pairing amplitude Delta0 in eV.")
    parser.add_argument("--eta", type=float, default=1e-4, help="Imaginary-axis regulator in eV.")
    parser.add_argument("--distance", type=float, default=3e-8, help="Plate distance in meters.")
    parser.add_argument("--k-parallel", type=float, default=1e6, help="Parallel wave vector in 1/m.")
    parser.add_argument("--phi", type=float, default=0.2)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT
        / "outputs"
        / "archive"
        / "response"
        / "static_policy_comparison"
        / "data"
        / "static_policy_comparison",
    )
    args = parser.parse_args()

    data = compare_static_policies(
        args.kinds,
        args.policies,
        args.nk,
        args.temperature,
        args.delta0,
        args.eta,
        args.distance,
        args.k_parallel,
        args.phi,
        args.theta,
    )
    paths = save_outputs(data, args.output_prefix)
    print_summary(data)
    print(f"npz_path = {paths[0]}")
    print(f"csv_path = {paths[1]}")
    print(f"figure_paths = {paths[2]}, {paths[3]}, {paths[4]}")


if __name__ == "__main__":
    main()
