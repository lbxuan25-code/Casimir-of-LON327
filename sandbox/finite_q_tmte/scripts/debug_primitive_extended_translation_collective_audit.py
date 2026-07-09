#!/usr/bin/env python3
"""Debug-only primitive translation RHS audit with collective mixed term."""

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
from tmte.pipeline.primitive_extended_translation_collective_audit import (  # noqa: E402
    DEFAULT_CANDIDATE,
    run_and_write_primitive_extended_translation_collective_audit,
)


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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _fmt_complex(value: object) -> str:
    if isinstance(value, dict):
        real = float(value.get("real", 0.0))
        imag = float(value.get("imag", 0.0))
        return f"{real:+.8e}{imag:+.8e}j"
    z = complex(value)  # type: ignore[arg-type]
    return f"{z.real:+.8e}{z.imag:+.8e}j"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only primitive translation RHS audit with collective mixed term.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=_positive_float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--delta0-eV", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--contact-scale", type=float, default=1.0)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_ranked(title: str, rows: list[dict[str, object]]) -> None:
    print(title)
    print("rank name diff/missing fit_alpha fit_res/missing")
    for idx, row in enumerate(rows[:10], start=1):
        fit = row["fit_to_target"]
        print(
            f"{idx:02d} {row['name']} "
            f"{float(row['difference_over_target_norm']):.8e} "
            f"{_fmt_complex(fit['alpha'])} "
            f"{float(fit['residual_over_target_norm']):.8e}"
        )


def _print_summary(payload: dict[str, object]) -> None:
    print("primitive_extended_translation_collective_audit summary")
    print("status:", payload["status"])
    print("model:", payload["model"])
    print("frequency:", payload["frequency"])
    print("debug_parameters:", payload["debug_parameters"])
    print("block_norms:", payload["block_norms"])
    ward = payload["ward_decomposition"]
    for side in ["left", "right"]:
        row = ward[side]
        print(
            f"{side}: "
            f"em={float(row['em_total']['norm']):.8e} "
            f"mixed={float(row['mixed_collective']['norm']):.8e} "
            f"extended={float(row['extended_total']['norm']):.8e} "
            f"extended/em={float(row['em_to_extended_reduction']):.8e}"
        )
    _print_ranked("left translation candidates against extended missing_to_close", payload["left_translation_candidates_ranked"])
    _print_ranked("right translation candidates against extended missing_to_close", payload["right_translation_candidates_ranked"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_primitive_extended_translation_collective_audit(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_value=args.q,
        nk=args.nk,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
        contact_scale=args.contact_scale,
        candidate_name=args.candidate,
    )
    print(f"primitive_extended_translation_collective_audit.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
