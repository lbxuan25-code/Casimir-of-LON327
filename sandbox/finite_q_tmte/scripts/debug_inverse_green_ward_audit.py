#!/usr/bin/env python3
"""Debug-only inverse-Green Ward audit CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SANDBOX_ROOT))

from tmte.adapters.model_adapter import available_models  # noqa: E402
from tmte.pipeline.inverse_green_ward_audit import run_and_write_inverse_green_ward_audit  # noqa: E402


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("matsubara index must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only inverse-Green Ward audit; not a production fix.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--kx", type=float, default=0.37)
    parser.add_argument("--ky", type=float, default=-0.41)
    parser.add_argument("--fermionic-energy", type=float, default=0.0)
    parser.add_argument("--nk-for-model", type=_positive_int, default=5)
    parser.add_argument("--delta0", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--current-vertex", default="peierls")
    parser.add_argument(
        "--frequency-convention",
        dest="frequency_conventions",
        action="append",
        choices=("matsubara_i_transfer", "matsubara_minus_i_transfer", "real_transfer_debug"),
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _best_comparison(row: dict[str, object]) -> tuple[str, float, float]:
    best_ref = ""
    best_minus = float("inf")
    best_plus = float("inf")
    for comp in row["comparisons"]:  # type: ignore[index]
        minus = float(comp["combo_minus_reference"]["residual_to_reference_over_reference"])
        plus = float(comp["combo_plus_reference"]["residual_to_reference_over_reference"])
        score = min(minus, plus)
        if score < min(best_minus, best_plus):
            best_ref = str(comp["reference"])
            best_minus = minus
            best_plus = plus
    return best_ref, best_minus, best_plus


def _print_summary(payload: dict[str, object]) -> None:
    print("inverse_green_ward_audit summary")
    print("status:", payload["status"])
    print("point:", payload["point"])
    print("note:", payload["nambu_reference_note"])
    for block in payload["frequency_convention_results"]:  # type: ignore[index]
        freq = block["frequency"]
        print("\nfrequency_convention:", freq["frequency_convention"])
        print("transfer_z_plus_minus_z_minus_eV:", freq["transfer_z_plus_minus_z_minus_eV"])
        print("reference norms:")
        for side in ["source", "observable"]:
            print(" ", side)
            for ref in block["references"][side]:  # type: ignore[index]
                print(f"    {ref['name']}: norm={float(ref['norm']):.6e}")
        print("candidate best comparisons:")
        for side in ["source", "observable"]:
            print(" ", side)
            for row in block["candidate_comparisons"][side]:  # type: ignore[index]
                ref, minus, plus = _best_comparison(row)
                print(f"    {row['candidate']}: best_ref={ref} minus/ref={minus:.6e} plus/ref={plus:.6e}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_inverse_green_ward_audit(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_value=args.q,
        kx=args.kx,
        ky=args.ky,
        fermionic_energy_eV=args.fermionic_energy,
        nk_for_model=args.nk_for_model,
        delta0_eV=args.delta0,
        eta_eV=args.eta,
        current_vertex=args.current_vertex,
        frequency_conventions=tuple(args.frequency_conventions) if args.frequency_conventions else ("matsubara_i_transfer", "matsubara_minus_i_transfer", "real_transfer_debug"),
    )
    print(f"inverse_green_ward_audit.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
