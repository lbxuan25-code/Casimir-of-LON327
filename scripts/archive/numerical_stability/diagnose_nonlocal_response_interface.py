#!/usr/bin/env python3
"""Diagnose the nonlocal response interface and local fallback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327 import (  # noqa: E402
    PairingAmplitudes,
    bosonic_matsubara_energy_eV,
    k_weights,
    local_response_imag_axis,
    nonlocal_response_imag_axis,
    uniform_bz_mesh,
)


def diagnose_nonlocal(
    kinds: list[str],
    delta0_eV: float,
    nk: int,
    temperature_K: float,
    matsubara_index: int,
    eta_eV: float,
    q_parallel: float,
    phi: float,
) -> dict[str, np.ndarray]:
    if matsubara_index < 1:
        raise ValueError("matsubara_index must be >= 1")
    mesh = uniform_bz_mesh(nk)
    weights = k_weights(mesh)
    omega = bosonic_matsubara_energy_eV(matsubara_index, temperature_K)

    data: dict[str, np.ndarray] = {
        "kind": np.array(kinds, dtype="U16"),
        "q0_matches_local": np.empty(len(kinds), dtype=bool),
        "q_positive_runs": np.empty(len(kinds), dtype=bool),
        "q_positive_nonlocal_resolved": np.empty(len(kinds), dtype=bool),
        "finite_q_placeholder_error": np.empty(len(kinds), dtype="U128"),
        "valid_for_casimir_input": np.empty(len(kinds), dtype=bool),
        "omega_eV": np.array(omega),
        "delta0_eV": np.array(delta0_eV),
        "nk": np.array(nk),
        "temperature_K": np.array(temperature_K),
        "matsubara_index": np.array(matsubara_index),
        "eta_eV": np.array(eta_eV),
        "q_parallel": np.array(q_parallel),
        "phi": np.array(phi),
    }

    for index, kind in enumerate(kinds):
        params = PairingAmplitudes(delta0_eV=delta0_eV)
        local = local_response_imag_axis(
            kind,  # type: ignore[arg-type]
            omega,
            mesh,
            temperature_K=temperature_K,
            eta_eV=eta_eV,
            pairing_params=params,
            k_weights=weights,
        )
        q0 = nonlocal_response_imag_axis(
            kind,  # type: ignore[arg-type]
            omega,
            0.0,
            phi,
            "local_fallback",
            mesh,
            temperature_K=temperature_K,
            eta_eV=eta_eV,
            pairing_params=params,
            k_weights=weights,
        )
        qpos = nonlocal_response_imag_axis(
            kind,  # type: ignore[arg-type]
            omega,
            q_parallel,
            phi,
            "local_fallback",
            mesh,
            temperature_K=temperature_K,
            eta_eV=eta_eV,
            pairing_params=params,
            k_weights=weights,
        )
        try:
            nonlocal_response_imag_axis(
                kind,  # type: ignore[arg-type]
                omega,
                q_parallel,
                phi,
                "finite_q_placeholder",
                mesh,
                temperature_K=temperature_K,
                eta_eV=eta_eV,
                pairing_params=params,
                k_weights=weights,
            )
            placeholder_error = ""
        except NotImplementedError as exc:
            placeholder_error = str(exc)

        data["q0_matches_local"][index] = bool(np.allclose(q0.matrix, local.matrix))
        data["q_positive_runs"][index] = bool(np.isfinite(qpos.matrix).all())
        data["q_positive_nonlocal_resolved"][index] = qpos.nonlocal_resolved
        data["finite_q_placeholder_error"][index] = placeholder_error
        data["valid_for_casimir_input"][index] = qpos.valid_for_casimir_input

    return data


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> Path:
    npz_path = output_prefix.with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)
    return npz_path


def print_summary(data: dict[str, np.ndarray]) -> None:
    for index, kind in enumerate(data["kind"]):
        print(f"kind = {kind}")
        print(f"q0_matches_local = {bool(data['q0_matches_local'][index])}")
        print(f"q_positive_runs = {bool(data['q_positive_runs'][index])}")
        print(f"q_positive_nonlocal_resolved = {bool(data['q_positive_nonlocal_resolved'][index])}")
        print(f"finite_q_placeholder_error = {data['finite_q_placeholder_error'][index]}")
        print(f"valid_for_casimir_input = {bool(data['valid_for_casimir_input'][index])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=("normal", "spm", "dwave"), default=["normal", "spm", "dwave"])
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--nk", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--matsubara-index", type=int, default=1)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--q-parallel", type=float, default=1e6)
    parser.add_argument("--phi", type=float, default=0.2)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "outputs" / "archive" / "response" / "nonlocal_interface" / "data" / "nonlocal_interface_diagnostic",
    )
    args = parser.parse_args()

    data = diagnose_nonlocal(
        args.kinds,
        args.delta0,
        args.nk,
        args.temperature,
        args.matsubara_index,
        args.eta,
        args.q_parallel,
        args.phi,
    )
    path = save_outputs(data, args.output_prefix)
    print_summary(data)
    print(f"npz_path = {path}")


if __name__ == "__main__":
    main()
