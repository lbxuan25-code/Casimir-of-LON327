#!/usr/bin/env python3
"""Debug-only phi scan for the minimal Casimir theta diagnostic."""

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
from tmte.pipeline.minimal_casimir_phi_scan import run_and_write_minimal_casimir_phi_scan  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Debug-only phi scan for the minimal Casimir theta diagnostic.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_positive_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=_positive_float, required=True, help="lab q magnitude in model units")
    parser.add_argument("--phi-values", nargs="+", type=float, required=True, help="lab q direction values in degrees; 0 and 360 count as duplicates")
    parser.add_argument("--plate1-theta-deg", type=float, default=0.0)
    parser.add_argument("--plate2-theta-deg", type=float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--separation-nm", type=_positive_float, required=True)
    parser.add_argument("--delta0-eV", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--skip-rhs-aware-validation", action="store_true")
    parser.add_argument("--include-point-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _format_optional(value: object, fmt: str = ".8e") -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    return format(float(value), fmt)


def _print_summary(payload: dict[str, object], output_dir: Path) -> None:
    print("minimal_casimir_phi_scan summary")
    print("status:", payload["status"])
    print("input:", payload["input"])
    print("aggregate:", payload["summary"])
    print("phi_deg,logdet_real,logdet_abs,delta_abs,d_abs_dphi,Rdiff,R1_norm,R2_norm,p1_Keff,p2_Keff,kappa_match")
    for row in payload["rows"]:
        print(
            f"{float(row['phi_mod_deg']):.8g},"
            f"{float(row['logdet_real']):.12e},"
            f"{float(row['logdet_abs']):.12e},"
            f"{float(row['delta_logdet_abs_from_phi0']):.12e},"
            f"{_format_optional(row['d_logdet_abs_dphi_rad_diagnostic'], '.12e')},"
            f"{float(row['Rdiff']):.12e},"
            f"{float(row['R1_norm']):.12e},"
            f"{float(row['R2_norm']):.12e},"
            f"{float(row['p1_Keff_norm']):.12e},"
            f"{float(row['p2_Keff_norm']):.12e},"
            f"{row['kappa_match']}"
        )
    print(f"json written to {output_dir / 'minimal_casimir_phi_scan.json'}")
    print(f"csv written to {output_dir / 'minimal_casimir_phi_scan.csv'}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_minimal_casimir_phi_scan(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_magnitude=args.q,
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
        include_point_payloads=args.include_point_payloads,
    )
    _print_summary(payload, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
