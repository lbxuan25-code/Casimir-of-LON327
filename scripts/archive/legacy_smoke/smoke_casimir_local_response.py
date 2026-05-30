#!/usr/bin/env python3
"""Smoke-test LocalSheetResponse plumbing into Casimir integrands."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327 import (  # noqa: E402
    CasimirSetup,
    ConductivityTensor,
    PairingAmplitudes,
    bosonic_matsubara_energy_eV,
    casimir_energy_integrand,
    casimir_torque_integrand,
    k_weights,
    local_response_imag_axis,
    matrix_symmetry_diagnostics,
    require_sheet_conductivity_for_reflection,
    sheet_conductivity_to_reflection_dimensionless,
    uniform_bz_mesh,
)
from lno327.casimir import matsubara_frequency  # noqa: E402
from lno327.plotting import (  # noqa: E402
    configure_publication_matplotlib,
    save_publication_figure,
    style_publication_axis,
)

REQUIRED_NPZ_FIELDS = {
    "kind",
    "omega_eV",
    "xi_rad_s",
    "response_xx",
    "response_yy",
    "response_xy",
    "response_yx",
    "response_unit_stage",
    "sheet_conductivity_xx",
    "sheet_conductivity_yy",
    "reflection_dimensionless_xx",
    "reflection_dimensionless_yy",
    "unit_conversion_status",
    "response_isotropic_diagnostic",
    "energy_integrand",
    "torque_integrand",
    "toy_anisotropic_torque_integrand",
    "theta_scan",
    "theta_scan_torque",
    "theta_scan_toy_anisotropic_torque",
    "valid_for_casimir_input",
    "delta0_eV",
    "nk",
    "temperature_K",
    "matsubara_index",
    "distance_m",
    "k_parallel_m_inv",
    "phi_rad",
    "theta_rad",
}

NOTE = "smoke test only; unit conversion applied; n=0 and finite-q still unresolved."


def response_isotropic_diagnostic(response_matrix: np.ndarray) -> float:
    diagnostics = matrix_symmetry_diagnostics(response_matrix)
    return float(
        max(
            abs(complex(diagnostics["delta"])),
            float(diagnostics["relative_offdiag"]),
            float(diagnostics["relative_eigen_split"]),
        )
    )


def toy_isotropic_tensor() -> ConductivityTensor:
    return ConductivityTensor(xx=1e-4, yy=1e-4, xy=0.0, yx=0.0)


def toy_anisotropic_tensor() -> ConductivityTensor:
    return ConductivityTensor(xx=2e-4, yy=1e-4, xy=0.0, yx=0.0)


def local_tensor_for_kind(
    kind: str,
    omega_eV: float,
    nk: int,
    temperature_K: float,
    eta_eV: float,
    delta0_eV: float,
) -> tuple[np.ndarray, ConductivityTensor, ConductivityTensor, str, bool]:
    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    response = local_response_imag_axis(
        kind,  # type: ignore[arg-type]
        omega_eV,
        mesh,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        pairing_params=PairingAmplitudes(delta0_eV=delta0_eV),
        k_weights=weights,
    )
    sheet = require_sheet_conductivity_for_reflection(response.matrix)
    reflection_dimensionless = sheet_conductivity_to_reflection_dimensionless(sheet)
    return response.matrix, sheet.tensor, reflection_dimensionless.tensor, sheet.normalization_status, response.valid_for_casimir_input


def evaluate_tensor(
    tensor: ConductivityTensor,
    setup: CasimirSetup,
    xi: float,
    k_parallel: float,
    phi: float,
    theta: float,
) -> tuple[complex, complex]:
    energy = casimir_energy_integrand(setup, xi, k_parallel, phi, theta, tensor, tensor)
    torque = casimir_torque_integrand(setup, xi, k_parallel, phi, theta, tensor, tensor)
    return energy, torque


def theta_scan(
    tensors: dict[str, ConductivityTensor],
    setup: CasimirSetup,
    xi: float,
    k_parallel: float,
    phi: float,
    num_theta: int = 101,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    theta_values = np.linspace(0.0, np.pi, num_theta)
    scans = {
        kind: np.empty(theta_values.size, dtype=complex)
        for kind in tensors
    }
    for index, theta in enumerate(theta_values):
        for kind, tensor in tensors.items():
            scans[kind][index] = casimir_torque_integrand(
                setup,
                xi,
                k_parallel,
                phi,
                float(theta),
                tensor,
                tensor,
            )
    return theta_values, scans


def scan_smoke(
    kinds: list[str],
    delta0_eV: float,
    nk: int,
    temperature_K: float,
    matsubara_index: int,
    eta_eV: float,
    distance_m: float,
    k_parallel_m_inv: float,
    phi_rad: float,
    theta_rad: float,
) -> dict[str, np.ndarray]:
    if matsubara_index < 1:
        raise ValueError("matsubara_index must be >= 1 for this smoke pipeline")
    if nk <= 0:
        raise ValueError("nk must be positive")
    if eta_eV <= 0.0:
        raise ValueError("eta_eV must be positive")
    if distance_m <= 0.0:
        raise ValueError("distance must be positive")
    if k_parallel_m_inv < 0.0:
        raise ValueError("k_parallel must be non-negative")

    omega_eV = bosonic_matsubara_energy_eV(matsubara_index, temperature_K)
    xi = matsubara_frequency(matsubara_index, temperature_K)
    setup = CasimirSetup(temperature=temperature_K, distance=distance_m)
    toy_aniso = toy_anisotropic_tensor()
    toy_energy, toy_torque = evaluate_tensor(
        toy_aniso,
        setup,
        xi,
        k_parallel_m_inv,
        phi_rad,
        theta_rad,
    )

    data: dict[str, np.ndarray] = {
        "kind": np.array(kinds, dtype="U16"),
        "omega_eV": np.full(len(kinds), omega_eV, dtype=float),
        "xi_rad_s": np.full(len(kinds), xi, dtype=float),
        "response_xx": np.empty(len(kinds), dtype=complex),
        "response_yy": np.empty(len(kinds), dtype=complex),
        "response_xy": np.empty(len(kinds), dtype=complex),
        "response_yx": np.empty(len(kinds), dtype=complex),
        "response_unit_stage": np.full(len(kinds), "model_response", dtype="U32"),
        "sheet_conductivity_xx": np.empty(len(kinds), dtype=complex),
        "sheet_conductivity_yy": np.empty(len(kinds), dtype=complex),
        "reflection_dimensionless_xx": np.empty(len(kinds), dtype=complex),
        "reflection_dimensionless_yy": np.empty(len(kinds), dtype=complex),
        "unit_conversion_status": np.empty(len(kinds), dtype="U64"),
        "response_isotropic_diagnostic": np.empty(len(kinds), dtype=float),
        "energy_integrand": np.empty(len(kinds), dtype=complex),
        "torque_integrand": np.empty(len(kinds), dtype=complex),
        "toy_anisotropic_torque_integrand": np.full(len(kinds), toy_torque, dtype=complex),
        "valid_for_casimir_input": np.empty(len(kinds), dtype=bool),
        "delta0_eV": np.array(delta0_eV),
        "nk": np.array(nk),
        "temperature_K": np.array(temperature_K),
        "matsubara_index": np.array(matsubara_index),
        "eta_eV": np.array(eta_eV),
        "distance_m": np.array(distance_m),
        "k_parallel_m_inv": np.array(k_parallel_m_inv),
        "phi_rad": np.array(phi_rad),
        "theta_rad": np.array(theta_rad),
        "toy_anisotropic_energy_integrand": np.array(toy_energy),
    }

    tensors: dict[str, ConductivityTensor] = {}
    for index, kind in enumerate(kinds):
        matrix, tensor, reflection_dimensionless, conversion_status, valid_for_casimir_input = local_tensor_for_kind(
            kind,
            omega_eV,
            nk,
            temperature_K,
            eta_eV,
            delta0_eV,
        )
        energy, torque = evaluate_tensor(
            tensor,
            setup,
            xi,
            k_parallel_m_inv,
            phi_rad,
            theta_rad,
        )
        tensors[kind] = tensor
        data["response_xx"][index] = matrix[0, 0]
        data["response_yy"][index] = matrix[1, 1]
        data["response_xy"][index] = matrix[0, 1]
        data["response_yx"][index] = matrix[1, 0]
        data["sheet_conductivity_xx"][index] = tensor.xx
        data["sheet_conductivity_yy"][index] = tensor.yy
        data["reflection_dimensionless_xx"][index] = reflection_dimensionless.xx
        data["reflection_dimensionless_yy"][index] = reflection_dimensionless.yy
        data["unit_conversion_status"][index] = conversion_status
        data["response_isotropic_diagnostic"][index] = response_isotropic_diagnostic(matrix)
        data["energy_integrand"][index] = energy
        data["torque_integrand"][index] = torque
        data["valid_for_casimir_input"][index] = valid_for_casimir_input

    tensors["toy_anisotropic"] = toy_aniso
    theta_values, scans = theta_scan(tensors, setup, xi, k_parallel_m_inv, phi_rad)
    data["theta_scan"] = theta_values
    data["theta_scan_torque"] = np.vstack([scans[kind] for kind in kinds])
    data["theta_scan_toy_anisotropic_torque"] = scans["toy_anisotropic"]

    return data


def output_paths(output_prefix: Path) -> tuple[Path, Path]:
    npz_path = output_prefix.with_suffix(".npz")
    resolved_prefix = output_prefix.resolve()
    project_data_root = (ROOT / "outputs" / "archive" / "smoke" / "smoke" / "casimir_local_response" / "data").resolve()
    if resolved_prefix.is_relative_to(project_data_root):
        figure_dir = ROOT / "outputs" / "archive" / "smoke" / "smoke" / "casimir_local_response" / "figures"
    else:
        figure_dir = output_prefix.parent / "figures"
    return npz_path, figure_dir / f"{output_prefix.name}_theta_scan.png"


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> tuple[Path, Path]:
    npz_path, figure_path = output_paths(output_prefix)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)

    configure_publication_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    theta_values = data["theta_scan"]
    for index, kind in enumerate(data["kind"]):
        ax.plot(
            theta_values,
            data["theta_scan_torque"][index].real,
            label=str(kind),
        )
    ax.plot(
        theta_values,
        data["theta_scan_toy_anisotropic_torque"].real,
        label="toy anisotropic",
        linestyle="--",
        color="black",
    )
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel("Re torque integrand")
    ax.set_title("Casimir local-response smoke theta scan")
    style_publication_axis(ax)
    save_publication_figure(fig, figure_path)
    plt.close(fig)
    return npz_path, figure_path


def print_summary(data: dict[str, np.ndarray]) -> None:
    for index, kind in enumerate(data["kind"]):
        print(f"kind = {kind}")
        print(f"response_isotropic_diagnostic = {float(data['response_isotropic_diagnostic'][index])}")
        print(f"energy_integrand = {data['energy_integrand'][index]}")
        print(f"torque_integrand = {data['torque_integrand'][index]}")
        print(f"toy_anisotropic_torque_integrand = {data['toy_anisotropic_torque_integrand'][index]}")
        print(f"note = {NOTE}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=("normal", "spm", "dwave"), default=["normal", "spm", "dwave"])
    parser.add_argument("--delta0", type=float, default=0.04, help="Pairing amplitude Delta0 in eV.")
    parser.add_argument("--nk", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=30.0, help="Temperature in K.")
    parser.add_argument("--matsubara-index", type=int, default=1)
    parser.add_argument("--eta", type=float, default=1e-4, help="Imaginary-axis regulator in eV.")
    parser.add_argument("--distance", type=float, default=30e-9, help="Plate separation in meters.")
    parser.add_argument("--k-parallel", type=float, default=1e6, help="In-plane wavevector in 1/m.")
    parser.add_argument("--phi", type=float, default=0.2, help="In-plane integration angle in radians.")
    parser.add_argument("--theta", type=float, default=0.7, help="Relative plate angle in radians.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "outputs" / "archive" / "smoke" / "smoke" / "casimir_local_response" / "data" / "casimir_local_response_smoke",
    )
    args = parser.parse_args()

    data = scan_smoke(
        args.kinds,
        args.delta0,
        args.nk,
        args.temperature,
        args.matsubara_index,
        args.eta,
        args.distance,
        args.k_parallel,
        args.phi,
        args.theta,
    )
    paths = save_outputs(data, args.output_prefix)
    print_summary(data)
    print(f"npz_path = {paths[0]}")
    print(f"figure_path = {paths[1]}")


if __name__ == "__main__":
    main()
