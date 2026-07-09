#!/usr/bin/env python3
"""Debug-only Matsubara-index scan over q-scan diagnostics."""

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
from tmte.pipeline.minimal_casimir_n_scan import run_and_write_minimal_casimir_n_scan  # noqa: E402
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only Matsubara-index scan over q-scan diagnostics.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-indices", "--n-values", dest="matsubara_indices", nargs="+", type=_positive_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q-values", nargs="+", type=_positive_float, required=True)
    parser.add_argument("--phi-values", nargs="+", type=float, required=True)
    parser.add_argument("--plate1-theta-deg", type=float, default=0.0)
    parser.add_argument("--plate2-theta-deg", type=float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--separation-nm", type=_positive_float, required=True)
    parser.add_argument("--delta0-eV", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--skip-rhs-aware-validation", action="store_true")
    parser.add_argument("--include-q-scan-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_summary(payload: dict[str, object], output_dir: Path) -> None:
    print("minimal_casimir_n_scan summary")
    print("status:", payload["status"])
    print("input:", payload["input"])
    print("aggregate:", payload["summary"])
    print("n,xi_eV,q_phi_avg_abs,q_phi_avg_real,ratio_to_prev,partial_abs,max_Rdiff,all_finite,kappa_match")
    for row in payload["rows"]:
        ratio = row["ratio_abs_to_previous_n"]
        ratio_text = "None" if ratio is None else f"{float(ratio):.12e}"
        print(
            f"{int(row['matsubara_index'])},"
            f"{float(row['xi_eV']):.12e},"
            f"{float(row['q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic']):.12e},"
            f"{float(row['q_trapezoid_integral_of_q_weighted_phi_average_logdet_real_diagnostic']):.12e},"
            f"{ratio_text},"
            f"{float(row['partial_sum_abs_diagnostic_no_prefactor']):.12e},"
            f"{float(row['max_Rdiff_over_q']):.12e},"
            f"{row['all_finite_logdet']},"
            f"{row['all_kappa_match']}"
        )
    print(f"json written to {output_dir / 'minimal_casimir_n_scan.json'}")
    print(f"csv written to {output_dir / 'minimal_casimir_n_scan.csv'}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_minimal_casimir_n_scan(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_indices=tuple(args.matsubara_indices),
        temperature_K=args.temperature_K,
        q_values=tuple(args.q_values),
        phi_values_deg=tuple(args.phi_values),
        plate1_theta_deg=args.plate1_theta_deg,
        plate2_theta_deg=args.plate2_theta_deg,
        nk=args.nk,
        separation_nm=args.separation_nm,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
        candidate_name=args.candidate,
        include_rhs_aware_validation=not args.skip_rhs_aware_validation,
        include_q_scan_payloads=args.include_q_scan_payloads,
    )
    _print_summary(payload, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
