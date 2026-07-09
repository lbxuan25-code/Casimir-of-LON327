#!/usr/bin/env python3
"""Debug-only q-vector minimal single-point Casimir path CLI."""

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
from tmte.pipeline.minimal_casimir_qvec_path import (  # noqa: E402
    q_model_vector_from_polar,
    run_and_write_minimal_casimir_qvec_path,
)
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
        raise argparse.ArgumentTypeError("q-vector minimal Casimir path v1 supports only theta_deg=0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only q-vector minimal single-point Casimir path.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_positive_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=_positive_float, default=None, help="q magnitude in model units; combine with --phi-deg")
    parser.add_argument("--phi-deg", type=float, default=0.0, help="q angle in degrees when using --q")
    parser.add_argument("--qx", type=float, default=None, help="explicit qx in model units")
    parser.add_argument("--qy", type=float, default=None, help="explicit qy in model units")
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


def q_vector_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> np.ndarray:
    explicit = args.qx is not None or args.qy is not None
    polar = args.q is not None
    if explicit and polar:
        parser.error("use either --q/--phi-deg or --qx/--qy, not both")
    if explicit:
        if args.qx is None or args.qy is None:
            parser.error("explicit q-vector input requires both --qx and --qy")
        q = np.asarray([float(args.qx), float(args.qy)], dtype=float)
        if np.linalg.norm(q) <= 1e-14:
            parser.error("explicit q-vector must be nonzero")
        return q
    if polar:
        return q_model_vector_from_polar(args.q, args.phi_deg)
    parser.error("provide either --q or --qx/--qy")
    raise AssertionError("unreachable")


def _ward_closed_label(validation: object) -> object:
    if not isinstance(validation, dict):
        return None
    status = validation.get("status")
    if not isinstance(status, dict):
        return None
    return status.get("rhs_aware_ward_closed")


def _print_summary(payload: dict[str, object]) -> None:
    point = payload["minimal_casimir_point"]
    response = point["response"]
    conductivity = point["conductivity"]
    reflection = point["reflection"]
    trace_log = point["trace_log"]
    validation = payload.get("rhs_aware_validation")
    ward_closed = _ward_closed_label(validation)
    print("minimal_casimir_qvec_path summary")
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
    print("q_geometry:", point["q_geometry"])
    print("sanity_checks:", point["sanity_checks"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    q_model_vector = q_vector_from_args(args, parser)
    payload = run_and_write_minimal_casimir_qvec_path(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_model_vector=q_model_vector,
        nk=args.nk,
        separation_nm=args.separation_nm,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
        theta_deg=args.theta_deg,
        candidate_name=args.candidate,
        include_rhs_aware_validation=not args.skip_rhs_aware_validation,
    )
    print(f"minimal_casimir_qvec_path.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
