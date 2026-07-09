#!/usr/bin/env python3
"""Debug-only theta scan for the minimal Casimir theta diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SANDBOX_ROOT))

from tmte.adapters.model_adapter import available_models  # noqa: E402
from tmte.pipeline.minimal_casimir_qvec_path import q_model_vector_from_polar  # noqa: E402
from tmte.pipeline.minimal_casimir_theta_scan import run_and_write_minimal_casimir_theta_scan  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Debug-only theta scan for the minimal Casimir theta diagnostic.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_positive_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=_positive_float, default=None, help="lab q magnitude in model units; combine with --phi-deg")
    parser.add_argument("--phi-deg", type=float, default=0.0, help="lab q angle in degrees when using --q")
    parser.add_argument("--qx", type=float, default=None, help="explicit lab qx in model units")
    parser.add_argument("--qy", type=float, default=None, help="explicit lab qy in model units")
    parser.add_argument("--plate1-theta-deg", type=float, default=0.0)
    parser.add_argument("--theta-values", nargs="+", type=float, required=True, help="plate2 theta values in degrees")
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


def q_vector_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> np.ndarray:
    explicit = args.qx is not None or args.qy is not None
    polar = args.q is not None
    if explicit and polar:
        parser.error("use either --q/--phi-deg or --qx/--qy, not both")
    if explicit:
        if args.qx is None or args.qy is None:
            parser.error("explicit lab q-vector input requires both --qx and --qy")
        q = np.asarray([float(args.qx), float(args.qy)], dtype=float)
        if np.linalg.norm(q) <= 1e-14:
            parser.error("explicit lab q-vector must be nonzero")
        return q
    if polar:
        return q_model_vector_from_polar(args.q, args.phi_deg)
    parser.error("provide either --q or --qx/--qy")
    raise AssertionError("unreachable")


def _format_optional(value: object, fmt: str = ".8e") -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    return format(float(value), fmt)


def _print_summary(payload: dict[str, object], output_dir: Path) -> None:
    print("minimal_casimir_theta_scan summary")
    print("status:", payload["status"])
    print("input:", payload["input"])
    print("aggregate:", payload["summary"])
    print("theta_deg,logdet_real,logdet_abs,delta_abs,d_abs_dtheta,Rdiff,R1_norm,R2_norm,p1_Keff,p2_Keff,kappa_match")
    for row in payload["rows"]:
        print(
            f"{float(row['theta_deg']):.8g},"
            f"{float(row['logdet_real']):.12e},"
            f"{float(row['logdet_abs']):.12e},"
            f"{float(row['delta_logdet_abs_from_theta0']):.12e},"
            f"{_format_optional(row['d_logdet_abs_dtheta_rad_diagnostic'], '.12e')},"
            f"{float(row['Rdiff']):.12e},"
            f"{float(row['R1_norm']):.12e},"
            f"{float(row['R2_norm']):.12e},"
            f"{float(row['p1_Keff_norm']):.12e},"
            f"{float(row['p2_Keff_norm']):.12e},"
            f"{row['kappa_match']}"
        )
    print(f"json written to {output_dir / 'minimal_casimir_theta_scan.json'}")
    print(f"csv written to {output_dir / 'minimal_casimir_theta_scan.csv'}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    q_lab_vector = q_vector_from_args(args, parser)
    payload = run_and_write_minimal_casimir_theta_scan(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_lab_vector=q_lab_vector,
        theta_values_deg=tuple(args.theta_values),
        plate1_theta_deg=args.plate1_theta_deg,
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
