#!/usr/bin/env python3
"""Debug-only primitive EM translation RHS audit CLI."""

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
from tmte.pipeline.primitive_em_translation_rhs_audit import DEFAULT_CANDIDATE, run_and_write_primitive_em_translation_rhs_audit  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Debug-only primitive EM translation RHS audit.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=_positive_float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--delta0-eV", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_summary(payload: dict[str, object]) -> None:
    print("primitive_em_translation_rhs_audit summary")
    print("status:", payload["status"])
    print("model:", payload["model"])
    print("frequency:", payload["frequency"])
    print("debug_parameters:", payload["debug_parameters"])
    print("block_norms:", payload["block_norms"])
    ward = payload["ward_decomposition"]
    for side in ["left", "right"]:
        row = ward[side]
        print(f"{side}: total={float(row['total']['norm']):.8e} missing={float(row['missing_to_close']['norm']):.8e}")
    print("ranked primitive EM translation candidates against left missing_to_close")
    print("rank name diff/missing fit_alpha fit_res/missing")
    for idx, row in enumerate(payload["candidate_translation_vectors_ranked"][:12], start=1):
        fit = row["fit_to_target"]
        print(
            f"{idx:02d} {row['name']} "
            f"{float(row['difference_over_target_norm']):.8e} "
            f"{_fmt_complex(fit['alpha'])} "
            f"{float(fit['residual_over_target_norm']):.8e}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_primitive_em_translation_rhs_audit(
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
        candidate_name=args.candidate,
    )
    print(f"primitive_em_translation_rhs_audit.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
