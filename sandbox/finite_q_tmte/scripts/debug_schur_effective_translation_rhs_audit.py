#!/usr/bin/env python3
"""Debug-only Schur-effective translation RHS audit CLI."""

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
from tmte.pipeline.schur_effective_translation_rhs_audit import (  # noqa: E402
    DEFAULT_CANDIDATE,
    run_and_write_schur_effective_translation_rhs_audit,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only Schur-effective translation RHS audit.")
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


def _print_side(side_name: str, side: dict[str, object]) -> None:
    print(
        f"{side_name}: "
        f"S_res/rhs={float(side['s_channel_residual']['norm_over_reference']):.8e} "
        f"eta_norm={float(side['eta_channel_total_C_eta']['norm']):.8e} "
        f"eta_proj/rhs={float(side['eta_projection_over_rhs_s']):.8e} "
        f"Keff_norm={float(side['effective_direct']['norm']):.8e} "
        f"pred_norm={float(side['effective_rhs_predicted']['norm']):.8e} "
        f"eff_res/ref={float(side['effective_residual_over_reference']):.8e}"
    )


def _print_summary(payload: dict[str, object]) -> None:
    print("schur_effective_translation_rhs_audit summary")
    print("status:", payload["status"])
    print("model:", payload["model"])
    print("frequency:", payload["frequency"])
    print("debug_parameters:", payload["debug_parameters"])
    print("block_norms:", payload["block_norms"])
    print("schur_solve_metadata:", payload["schur_solve_metadata"])
    print("summary:", payload["summary"])
    ward = payload["ward_decomposition"]
    _print_side("left", ward["left"])
    _print_side("right", ward["right"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_schur_effective_translation_rhs_audit(
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
    print(f"schur_effective_translation_rhs_audit.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
