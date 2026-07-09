#!/usr/bin/env python3
"""Debug-only pairing/contact missing-contribution audit CLI."""

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
from tmte.pipeline.pairing_contact_missing_audit import (  # noqa: E402
    DEFAULT_DELTA0_VALUES,
    DEFAULT_PAIRINGS,
    run_and_write_pairing_contact_missing_audit,
)
from tmte.pipeline.primitive_response_closure_suite import DEFAULT_CANDIDATE  # noqa: E402
from tmte.pipeline.primitive_response_ward_audit import primitive_ward_candidate_vectors  # noqa: E402


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


def _candidate_choices() -> tuple[str, ...]:
    return tuple(row["candidate"] for row in primitive_ward_candidate_vectors(0.01, 0.02, 0.1))


def _fmt_float(value: object, digits: int = 8) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{digits}e}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only pairing/contact missing-contribution audit.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairings", nargs="+", default=list(DEFAULT_PAIRINGS))
    parser.add_argument("--delta0-values", nargs="+", type=float, default=list(DEFAULT_DELTA0_VALUES))
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--contact-scale", type=float, default=1.0)
    parser.add_argument("--candidate", choices=_candidate_choices(), default=DEFAULT_CANDIDATE)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_rows(payload: dict[str, object]) -> None:
    print("summary rows")
    print("pairing delta0 status alpha_real missing_fraction overlap proj_resid/required left_right_diff")
    for row in payload["summary_rows"]:
        if row.get("status") != "ok":
            print(f"{row.get('pairing')} {row.get('delta0_eV')} ERROR {row.get('error_type')}: {row.get('error_message')}")
            continue
        print(
            f"{row['pairing']} "
            f"{float(row['delta0_eV']):.6f} "
            f"{row['status']} "
            f"{_fmt_float(row['alpha_real_mean'])} "
            f"{_fmt_float(row['missing_fraction_real'])} "
            f"{_fmt_float(row['parallelism_abs_overlap_mean'])} "
            f"{_fmt_float(row['projection_residual_over_required_mean'])} "
            f"{_fmt_float(row['left_right_alpha_abs_diff'])}"
        )


def _print_trends(payload: dict[str, object]) -> None:
    print("\ntrend by pairing")
    for pairing, info in payload["by_pairing"].items():
        trend = info["trend_alpha_vs_delta0_squared"]
        print(
            f"{pairing}: num_ok={info['num_ok']} "
            f"alpha_delta0_zero={_fmt_float(info['delta0_zero_alpha_real_mean'])} "
            f"max_missing={_fmt_float(info['max_missing_fraction_real'])} "
            f"min_overlap={_fmt_float(info['min_parallelism_abs_overlap'])} "
            f"trend_status={trend['status']}"
        )
        if trend.get("status") == "linear_fit_alpha_vs_delta0_squared":
            print(
                f"  alpha ~= {float(trend['intercept']):.8e} "
                f"+ ({float(trend['slope_per_eV2']):.8e}) * delta0^2; "
                f"max_resid={float(trend['max_abs_residual']):.8e}"
            )


def _print_summary(payload: dict[str, object]) -> None:
    print("pairing_contact_missing_audit summary")
    print("status:", payload["status"])
    print("debug_parameters:", payload["debug_parameters"])
    _print_rows(payload)
    _print_trends(payload)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_pairing_contact_missing_audit(
        args.output_dir,
        model_name=args.model,
        pairings=tuple(args.pairings),
        delta0_values=tuple(args.delta0_values),
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_value=args.q,
        nk=args.nk,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
        contact_scale=args.contact_scale,
        candidate_name=args.candidate,
        fail_fast=bool(args.fail_fast),
    )
    print(f"pairing_contact_missing_audit.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
