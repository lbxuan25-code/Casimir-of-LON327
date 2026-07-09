#!/usr/bin/env python3
"""Debug-only minimal single-point Casimir path CLI."""

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
from tmte.pipeline.minimal_casimir_path import run_and_write_minimal_casimir_path  # noqa: E402
from tmte.pipeline.schur_effective_translation_rhs_audit import DEFAULT_CANDIDATE  # noqa: E402


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


def _theta_zero(value: str) -> float:
    parsed = float(value)
    if abs(parsed) > 1e-14:
        raise argparse.ArgumentTypeError("minimal Casimir path v1 supports only theta_deg=0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only minimal single-point Casimir path.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_positive_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=_positive_float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--separation-nm", type=_positive_float, required=True)
    parser.add_argument("--delta0-eV", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--theta-deg", type=_theta_zero, default=0.0)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--skip-rhs-aware-validation", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_summary(payload: dict[str, object]) -> None:
    point = payload["minimal_casimir_point"]
    response = point["response"]
    conductivity = point["conductivity"]
    reflection = point["reflection"]
    trace_log = point["trace_log"]
    validation = payload.get("rhs_aware_validation")
    ward_closed = None if validation is None else validation["status"]["rhs_aware_ward_closed"]
    print("minimal_casimir_path summary")
    print("status:", payload["status"])
    print("input:", payload["input"])
    print(
        "metrics: "
        f"ward_closed={ward_closed} "
        f"Keff_norm={float(payload['sandbox_response_source']['K_eff_norm']):.8e} "
        f"spatial_norm={float(response['spatial_response_norm']):.8e} "
        f"sigma_model_norm={float(conductivity['sigma_model_norm']):.8e} "
        f"sigma_tilde_norm={float(conductivity['sigma_tilde_norm']):.8e} "
        f"R_TE_TM_norm={float(reflection['R_TE_TM_norm']):.8e} "
        f"round_trip={float(trace_log['round_trip_factor']):.8e} "
        f"logdet_abs={float(trace_log['logdet_abs']):.8e}"
    )
    print("sanity_checks:", point["sanity_checks"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_minimal_casimir_path(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_value=args.q,
        nk=args.nk,
        separation_nm=args.separation_nm,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
        theta_deg=args.theta_deg,
        candidate_name=args.candidate,
        include_rhs_aware_validation=not args.skip_rhs_aware_validation,
    )
    print(f"minimal_casimir_path.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
