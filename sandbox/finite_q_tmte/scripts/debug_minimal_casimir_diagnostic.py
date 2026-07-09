#!/usr/bin/env python3
"""Unified debug-only entry point for minimal Casimir diagnostics.

This script is a light orchestration layer over the existing sandbox diagnostic
pipelines.  It does not replace the pipeline modules and does not promote any
sandbox result to production Casimir input.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SANDBOX_ROOT))

from tmte.adapters.model_adapter import available_models  # noqa: E402
from tmte.pipeline.minimal_casimir_n_scan import run_and_write_minimal_casimir_n_scan  # noqa: E402
from tmte.pipeline.minimal_casimir_n_tail_fit import (  # noqa: E402
    DEFAULT_QUANTITY_COLUMN,
    SUPPORTED_MODELS,
    run_and_write_minimal_casimir_n_tail_fit,
)
from tmte.pipeline.minimal_casimir_phi_scan import run_and_write_minimal_casimir_phi_scan  # noqa: E402
from tmte.pipeline.minimal_casimir_q_scan import run_and_write_minimal_casimir_q_scan  # noqa: E402
from tmte.pipeline.minimal_casimir_qvec_path import q_model_vector_from_polar  # noqa: E402
from tmte.pipeline.minimal_casimir_shift_scan import (  # noqa: E402
    DEFAULT_R_NORM_WARNING_THRESHOLD,
    run_and_write_minimal_casimir_shift_scan,
)
from tmte.pipeline.minimal_casimir_theta_scan import run_and_write_minimal_casimir_theta_scan  # noqa: E402
from tmte.pipeline.schur_effective_translation_rhs_audit import DEFAULT_CANDIDATE  # noqa: E402

DEFAULT_SHIFT_FRACTIONS = (0.0,)


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


def _add_common_model_args(parser: argparse.ArgumentParser, *, include_plate2_theta: bool = True, include_single_matsubara: bool = True) -> None:
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    if include_single_matsubara:
        parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_positive_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--plate1-theta-deg", type=float, default=0.0)
    if include_plate2_theta:
        parser.add_argument("--plate2-theta-deg", type=float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--separation-nm", type=_positive_float, required=True)
    parser.add_argument("--delta0-eV", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--skip-rhs-aware-validation", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)


def _add_shift_fractions_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shift-fractions",
        nargs="+",
        type=float,
        default=list(DEFAULT_SHIFT_FRACTIONS),
        help="mesh shift fractions; default is no-shift [0.0] for diagnostic scans",
    )


def _q_lab_from_args(args: argparse.Namespace) -> np.ndarray:
    qx = getattr(args, "qx", None)
    qy = getattr(args, "qy", None)
    q = getattr(args, "q", None)
    phi_deg = getattr(args, "phi_deg", None)
    if qx is not None or qy is not None:
        if qx is None or qy is None:
            raise ValueError("--qx and --qy must be provided together")
        if q is not None or phi_deg is not None:
            raise ValueError("use either explicit --qx/--qy or polar --q/--phi-deg, not both")
        return np.asarray([float(qx), float(qy)], dtype=float)
    if q is None or phi_deg is None:
        raise ValueError("provide either --qx/--qy or --q plus --phi-deg")
    return q_model_vector_from_polar(float(q), float(phi_deg))


def _print_payload_summary(kind: str, payload: dict[str, Any], output_dir: Path, rows_limit: int | None) -> None:
    print(f"minimal_casimir_diagnostic {kind} summary")
    print("status:", payload.get("status"))
    print("input:", payload.get("input"))
    print("aggregate:", payload.get("summary"))
    rows = payload.get("rows") or []
    if rows_limit is not None and rows:
        print(f"rows_preview: first {min(rows_limit, len(rows))} / {len(rows)}")
        for row in rows[:rows_limit]:
            print(row)
    print(f"output_dir: {output_dir}")


def _run_theta_scan(args: argparse.Namespace) -> dict[str, Any]:
    q_lab = _q_lab_from_args(args)
    return run_and_write_minimal_casimir_theta_scan(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_lab_vector=q_lab,
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


def _run_phi_scan(args: argparse.Namespace) -> dict[str, Any]:
    return run_and_write_minimal_casimir_phi_scan(
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


def _run_q_scan(args: argparse.Namespace) -> dict[str, Any]:
    return run_and_write_minimal_casimir_q_scan(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
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
        include_phi_scan_payloads=args.include_phi_scan_payloads,
    )


def _run_shift_scan(args: argparse.Namespace) -> dict[str, Any]:
    return run_and_write_minimal_casimir_shift_scan(
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
        shift_values=tuple(args.shift_values),
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta,
        candidate_name=args.candidate,
        include_rhs_aware_validation=not args.skip_rhs_aware_validation,
        include_phi_scan_payloads=args.include_phi_scan_payloads,
        r_norm_warning_threshold=args.r_norm_warning_threshold,
    )


def _run_n_scan(args: argparse.Namespace) -> dict[str, Any]:
    return run_and_write_minimal_casimir_n_scan(
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


def _run_n_tail_fit(args: argparse.Namespace) -> dict[str, Any]:
    return run_and_write_minimal_casimir_n_tail_fit(
        args.output_dir,
        input_csv_path=args.input_csv,
        quantity_column=args.quantity_column,
        models=tuple(args.models),
        fit_min_n=args.fit_min_n,
        fit_max_n=args.fit_max_n,
        tail_start_n_exclusive=args.tail_start_n_exclusive,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified debug-only minimal Casimir diagnostics. Default shift policy is no-shift.",
    )
    parser.add_argument("--rows-preview", type=int, default=0, help="print first N rows after the aggregate summary")
    subparsers = parser.add_subparsers(dest="command", required=True)

    theta = subparsers.add_parser("theta-scan", help="scan plate angle at fixed lab q")
    _add_common_model_args(theta, include_plate2_theta=False)
    _add_shift_fractions_arg(theta)
    theta.add_argument("--q", type=_positive_float, default=None, help="lab q magnitude for polar input")
    theta.add_argument("--phi-deg", type=float, default=None, help="lab q direction in degrees for polar input")
    theta.add_argument("--qx", type=float, default=None, help="explicit lab qx")
    theta.add_argument("--qy", type=float, default=None, help="explicit lab qy")
    theta.add_argument("--theta-values", nargs="+", type=float, required=True)
    theta.add_argument("--include-point-payloads", action="store_true")
    theta.set_defaults(func=_run_theta_scan, kind="theta-scan")

    phi = subparsers.add_parser("phi-scan", help="scan lab q direction at fixed q and plate angle")
    _add_common_model_args(phi)
    _add_shift_fractions_arg(phi)
    phi.add_argument("--q", type=_positive_float, required=True)
    phi.add_argument("--phi-values", nargs="+", type=float, required=True)
    phi.add_argument("--include-point-payloads", action="store_true")
    phi.set_defaults(func=_run_phi_scan, kind="phi-scan")

    qscan = subparsers.add_parser("q-scan", help="scan q magnitude using periodic phi averages")
    _add_common_model_args(qscan)
    _add_shift_fractions_arg(qscan)
    qscan.add_argument("--q-values", nargs="+", type=_positive_float, required=True)
    qscan.add_argument("--phi-values", nargs="+", type=float, required=True)
    qscan.add_argument("--include-phi-scan-payloads", action="store_true")
    qscan.set_defaults(func=_run_q_scan, kind="q-scan")

    shift = subparsers.add_parser("shift-scan", help="scan single-shift phi diagnostics and R-norm guard")
    _add_common_model_args(shift)
    shift.add_argument("--q", type=_positive_float, required=True)
    shift.add_argument("--phi-values", nargs="+", type=float, required=True)
    shift.add_argument("--shift-values", nargs="+", type=float, required=True)
    shift.add_argument("--r-norm-warning-threshold", type=_positive_float, default=DEFAULT_R_NORM_WARNING_THRESHOLD)
    shift.add_argument("--include-phi-scan-payloads", action="store_true")
    shift.set_defaults(func=_run_shift_scan, kind="shift-scan")

    nscan = subparsers.add_parser("n-scan", help="scan positive Matsubara indices using q-scan diagnostics")
    _add_common_model_args(nscan, include_single_matsubara=False)
    _add_shift_fractions_arg(nscan)
    nscan.add_argument("--matsubara-indices", "--n-values", dest="matsubara_indices", nargs="+", type=_positive_int, required=True)
    nscan.add_argument("--q-values", nargs="+", type=_positive_float, required=True)
    nscan.add_argument("--phi-values", nargs="+", type=float, required=True)
    nscan.add_argument("--include-q-scan-payloads", action="store_true")
    nscan.set_defaults(func=_run_n_scan, kind="n-scan")

    tail = subparsers.add_parser("n-tail-fit", help="offline power-law fit for n-scan CSV tail data")
    tail.add_argument("--input-csv", type=Path, required=True)
    tail.add_argument("--quantity-column", default=DEFAULT_QUANTITY_COLUMN)
    tail.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS))
    tail.add_argument("--fit-min-n", type=_positive_int, default=None)
    tail.add_argument("--fit-max-n", type=_positive_int, default=None)
    tail.add_argument("--tail-start-n-exclusive", type=_positive_int, default=None)
    tail.add_argument("--output-dir", type=Path, required=True)
    tail.set_defaults(func=_run_n_tail_fit, kind="n-tail-fit")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = args.func(args)
    rows_limit = None if args.rows_preview is None or args.rows_preview <= 0 else int(args.rows_preview)
    _print_payload_summary(args.kind, payload, args.output_dir, rows_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
