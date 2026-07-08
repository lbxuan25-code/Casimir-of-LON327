#!/usr/bin/env python3
"""Run the finite-q TM/TE sandbox scan."""

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
from tmte.pipeline.scan_runner import run_and_write_scan  # noqa: E402


def _nonnegative_int(value: str) -> int:
    index = int(value)
    if index < 0:
        raise argparse.ArgumentTypeError("matsubara index must be non-negative")
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run direct finite-q TM/TE target-basis sandbox scan.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--q-values", nargs="+", type=float, required=True)
    parser.add_argument("--nk", type=int, required=True)
    parser.add_argument("--delta0", type=float, default=None)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_scan(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        q_values=tuple(args.q_values),
        nk=args.nk,
        delta0_eV=args.delta0,
        temperature_K=args.temperature_K,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
    )
    print(f"tmte_scan.json written to {args.output_dir}")
    print(f"schema_version: {payload['schema_version']}")
    print(f"valid_for_casimir_input: {payload['status']['valid_for_casimir_input']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
