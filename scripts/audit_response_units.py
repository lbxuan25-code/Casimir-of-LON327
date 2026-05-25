#!/usr/bin/env python3
"""Audit response-unit conventions before Casimir calculations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lno327 import (  # noqa: E402
    PairingAmplitudes,
    ResponseUnitConvention,
    bosonic_matsubara_energy_eV,
    k_weights,
    local_response_imag_axis,
    model_response_to_sheet_conductivity,
    sheet_conductivity_to_dimensionless,
    uniform_bz_mesh,
)


def audit_units(
    kinds: list[str],
    delta0_eV: float,
    nk: int,
    temperature_K: float,
    matsubara_index: int,
    eta_eV: float,
) -> dict[str, np.ndarray]:
    if matsubara_index < 1:
        raise ValueError("matsubara_index must be >= 1")
    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    omega_eV = bosonic_matsubara_energy_eV(matsubara_index, temperature_K)
    convention = ResponseUnitConvention(mode="dimensionless_model")

    data: dict[str, np.ndarray] = {
        "kind": np.array(kinds, dtype="U16"),
        "unit_label": np.empty(len(kinds), dtype="U64"),
        "normalization_status": np.empty(len(kinds), dtype="U96"),
        "dimensionless_sigma_t": np.empty(len(kinds), dtype=complex),
        "dimensionless_sigma_t_abs": np.empty(len(kinds), dtype=float),
        "valid_for_casimir_input": np.empty(len(kinds), dtype=bool),
        "notes": np.empty(len(kinds), dtype=object),
        "omega_eV": np.array(omega_eV),
        "delta0_eV": np.array(delta0_eV),
        "nk": np.array(nk),
        "temperature_K": np.array(temperature_K),
        "matsubara_index": np.array(matsubara_index),
        "eta_eV": np.array(eta_eV),
    }

    for index, kind in enumerate(kinds):
        response = local_response_imag_axis(
            kind,  # type: ignore[arg-type]
            omega_eV,
            mesh,
            temperature_K=temperature_K,
            eta_eV=eta_eV,
            pairing_params=PairingAmplitudes(delta0_eV=delta0_eV),
            k_weights=weights,
            unit_convention=convention,
        )
        conversion = model_response_to_sheet_conductivity(response.matrix, convention)
        dimensionless = sheet_conductivity_to_dimensionless(conversion.tensor)
        sigma_t = dimensionless.xx + dimensionless.yy
        data["unit_label"][index] = conversion.unit_label
        data["normalization_status"][index] = conversion.normalization_status
        data["dimensionless_sigma_t"][index] = sigma_t
        data["dimensionless_sigma_t_abs"][index] = abs(sigma_t)
        data["valid_for_casimir_input"][index] = response.valid_for_casimir_input and conversion.valid_for_casimir_input
        data["notes"][index] = response.notes + conversion.notes

    return data


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> Path:
    npz_path = output_prefix.with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)
    return npz_path


def print_summary(data: dict[str, np.ndarray]) -> None:
    for index, kind in enumerate(data["kind"]):
        print(f"kind = {kind}")
        print(f"unit_label = {data['unit_label'][index]}")
        print(f"normalization_status = {data['normalization_status'][index]}")
        print(f"dimensionless_sigma_t_abs = {float(data['dimensionless_sigma_t_abs'][index])}")
        print(f"valid_for_casimir_input = {bool(data['valid_for_casimir_input'][index])}")
        print(f"notes = {data['notes'][index]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=("normal", "spm", "dwave"), default=["normal", "spm", "dwave"])
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--nk", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--matsubara-index", type=int, default=1)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "outputs" / "response" / "unit_audit" / "data" / "response_unit_audit",
    )
    args = parser.parse_args()

    data = audit_units(args.kinds, args.delta0, args.nk, args.temperature, args.matsubara_index, args.eta)
    path = save_outputs(data, args.output_prefix)
    print_summary(data)
    print(f"npz_path = {path}")


if __name__ == "__main__":
    main()
