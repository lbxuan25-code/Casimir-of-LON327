#!/usr/bin/env python3
"""Run an nk-sweep diagnostic for the finite-q TM/TE sandbox."""

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
from tmte.pipeline.nk_sweep import run_and_write_nk_sweep  # noqa: E402


def _nonnegative_int(value: str) -> int:
    index = int(value)
    if index < 0:
        raise argparse.ArgumentTypeError("matsubara index must be non-negative")
    return index


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("nk values must be positive integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run finite-q TM/TE nk-sweep diagnostics.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q-values", nargs="+", type=float, required=True)
    parser.add_argument("--nk-values", nargs="+", type=_positive_int, required=True)
    parser.add_argument("--delta0", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_nk_sweep(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_values=tuple(args.q_values),
        nk_values=tuple(args.nk_values),
        delta0_eV=args.delta0,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
    )
    print(f"nk_sweep.json written to {args.output_dir}")
    print(f"schema_version: {payload['schema_version']}")
    print(f"valid_for_casimir_input: {payload['status']['valid_for_casimir_input']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

