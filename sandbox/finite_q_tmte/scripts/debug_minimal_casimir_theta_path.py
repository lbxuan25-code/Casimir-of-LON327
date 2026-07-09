#!/usr/bin/env python3
"""Debug-only theta/plate-rotation minimal single-point Casimir path CLI."""

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
from tmte.pipeline.minimal_casimir_theta_path import run_and_write_minimal_casimir_theta_path  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Debug-only theta/plate-rotation minimal single-point Casimir path.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_positive_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=_positive_float, default=None, help="lab q magnitude in model units; combine with --phi-deg")
    parser.add_argument("--phi-deg", type=float, default=0.0, help="lab q angle in degrees when using --q")
    parser.add_argument("--qx", type=float, default=None, help="explicit lab qx in model units")
    parser.add_argument("--qy", type=float, default=None, help="explicit lab qy in model units")
    parser.add_argument("--plate1-theta-deg", type=float, default=0.0)
    parser.add_argument("--plate2-theta-deg", type=float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--separation-nm", type=_positive_float, required=True)
    parser.add_argument("--delta0-eV", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
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
            parser.error("explicit lab q-vector input requires both --qx and --qy")
        q = np.asarray([float(args.qx), float(args.qy)], dtype=float)
        if np.linalg.norm(q) <= 1e-14:
            parser.error("explicit lab q-vector must be nonzero")
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
    mixed = payload["mixed_trace_log"]
    p1 = payload["plate1"]
    p2 = payload["plate2"]
    p1_point = p1["minimal_casimir_point"]
    p2_point = p2["minimal_casimir_point"]
    print("minimal_casimir_theta_path summary")
    print("status:", payload["status"])
    print("input:", payload["input"])
    print(
        "metrics: "
        f"p1_ward={_ward_closed_label(p1.get('rhs_aware_validation'))} "
        f"p2_ward={_ward_closed_label(p2.get('rhs_aware_validation'))} "
        f"p1_Keff={float(p1['sandbox_response_source']['K_eff_norm']):.8e} "
        f"p2_Keff={float(p2['sandbox_response_source']['K_eff_norm']):.8e} "
        f"p1_R={float(p1_point['reflection']['R_TE_TM_norm']):.8e} "
        f"p2_R={float(p2_point['reflection']['R_TE_TM_norm']):.8e} "
        f"Rdiff={float(mixed['R1_minus_R2_norm']):.8e} "
        f"round_trip={float(mixed['round_trip_factor']):.8e} "
        f"mixed_logdet_abs={float(mixed['logdet_abs']):.8e}"
    )
    print("geometry:", payload["geometry"])
    print("sanity_checks:", payload["sanity_checks"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    q_lab_vector = q_vector_from_args(args, parser)
    payload = run_and_write_minimal_casimir_theta_path(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_lab_vector=q_lab_vector,
        plate1_theta_deg=args.plate1_theta_deg,
        plate2_theta_deg=args.plate2_theta_deg,
        nk=args.nk,
        separation_nm=args.separation_nm,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
        candidate_name=args.candidate,
        include_rhs_aware_validation=not args.skip_rhs_aware_validation,
    )
    print(f"minimal_casimir_theta_path.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
