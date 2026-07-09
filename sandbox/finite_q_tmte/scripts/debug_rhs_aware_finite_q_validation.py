#!/usr/bin/env python3
"""Debug-only RHS-aware finite-q validation summary CLI."""

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
from tmte.pipeline.rhs_aware_finite_q_validation import (  # noqa: E402
    DEFAULT_CONDITION_MAX,
    DEFAULT_RESIDUAL_TOL,
    run_and_write_rhs_aware_finite_q_validation,
)
from tmte.pipeline.schur_effective_translation_rhs_audit import DEFAULT_CANDIDATE  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Debug-only RHS-aware finite-q validation summary.")
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
    parser.add_argument("--residual-tol", type=float, default=DEFAULT_RESIDUAL_TOL)
    parser.add_argument("--condition-max", type=float, default=DEFAULT_CONDITION_MAX)
    parser.add_argument("--include-raw-schur-audit", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_summary(payload: dict[str, object]) -> None:
    status = payload["status"]
    metrics = payload["metrics"]
    print("rhs_aware_finite_q_validation summary")
    print("status:", status)
    print("model:", payload["model"])
    print("frequency:", payload["frequency"])
    print("debug_parameters:", payload["debug_parameters"])
    print("thresholds:", payload["thresholds"])
    print(
        "metrics: "
        f"max_S_res/rhs={float(metrics['max_s_channel_residual_over_rhs_s']):.8e} "
        f"max_eff_res/ref={float(metrics['max_effective_residual_over_reference']):.8e} "
        f"max_eta_proj/rhs={float(metrics['max_eta_projection_over_rhs_s']):.8e} "
        f"max_legacy_zero_rhs/Keff={float(metrics['max_legacy_zero_rhs_residual_over_k_eff_norm']):.8e} "
        f"Keff_norm={float(metrics['K_eff_norm']):.8e} "
        f"cond_etaeta={float(metrics['K_etaeta_condition_number']):.8e}"
    )
    print("legacy_zero_rhs_check:", payload["legacy_zero_rhs_check"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_rhs_aware_finite_q_validation(
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
        residual_tol=args.residual_tol,
        condition_max=args.condition_max,
        include_raw_schur_audit=args.include_raw_schur_audit,
    )
    print(f"rhs_aware_finite_q_validation.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
