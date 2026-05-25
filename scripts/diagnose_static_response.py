#!/usr/bin/env python3
"""Diagnose n=0 Matsubara response policies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lno327 import PairingAmplitudes, local_response_matsubara_index  # noqa: E402


def diagnose_static(
    kinds: list[str],
    policies: list[str],
    delta0_eV: float,
    nk: int,
    temperature_K: float,
    eta_eV: float,
) -> dict[str, np.ndarray]:
    rows = [(kind, policy) for kind in kinds for policy in policies]
    data: dict[str, np.ndarray] = {
        "kind": np.array([row[0] for row in rows], dtype="U16"),
        "policy": np.array([row[1] for row in rows], dtype="U48"),
        "status": np.empty(len(rows), dtype="U48"),
        "approximate": np.empty(len(rows), dtype=bool),
        "matrix_finite": np.empty(len(rows), dtype=bool),
        "unit_label": np.empty(len(rows), dtype="U80"),
        "valid_for_casimir_input": np.empty(len(rows), dtype=bool),
        "notes": np.empty(len(rows), dtype=object),
        "delta0_eV": np.array(delta0_eV),
        "nk": np.array(nk),
        "temperature_K": np.array(temperature_K),
        "eta_eV": np.array(eta_eV),
    }

    for index, (kind, policy) in enumerate(rows):
        result = local_response_matsubara_index(
            kind,  # type: ignore[arg-type]
            0,
            temperature_K,
            policy=policy,  # type: ignore[arg-type]
            nk=nk,
            eta_eV=eta_eV,
            pairing_params=PairingAmplitudes(delta0_eV=delta0_eV),
        )
        data["status"][index] = result.status
        data["approximate"][index] = result.approximate
        data["matrix_finite"][index] = False if result.matrix is None else bool(np.isfinite(result.matrix).all())
        data["unit_label"][index] = result.unit_label
        data["valid_for_casimir_input"][index] = result.valid_for_casimir_input
        data["notes"][index] = result.notes

    return data


def save_outputs(data: dict[str, np.ndarray], output_prefix: Path) -> Path:
    npz_path = output_prefix.with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)
    return npz_path


def print_summary(data: dict[str, np.ndarray]) -> None:
    for index, kind in enumerate(data["kind"]):
        print(f"kind = {kind}")
        print(f"policy = {data['policy'][index]}")
        print(f"status = {data['status'][index]}")
        print(f"approximate = {bool(data['approximate'][index])}")
        print(f"matrix_finite = {bool(data['matrix_finite'][index])}")
        print(f"valid_for_casimir_input = {bool(data['valid_for_casimir_input'][index])}")
        print(f"notes = {data['notes'][index]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=("normal", "spm", "dwave"), default=["normal", "spm", "dwave"])
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("skip", "extrapolate_from_lowest_matsubara", "use_static_kernel"),
        default=["skip", "extrapolate_from_lowest_matsubara", "use_static_kernel"],
    )
    parser.add_argument("--delta0", type=float, default=0.04)
    parser.add_argument("--nk", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=30.0)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "outputs" / "response" / "static_response" / "data" / "static_response_diagnostic",
    )
    args = parser.parse_args()

    data = diagnose_static(args.kinds, args.policies, args.delta0, args.nk, args.temperature, args.eta)
    path = save_outputs(data, args.output_prefix)
    print_summary(data)
    print(f"npz_path = {path}")


if __name__ == "__main__":
    main()
