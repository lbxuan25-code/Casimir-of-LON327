#!/usr/bin/env python3
"""Debug-only q-along-x extended Ward-kernel audit."""

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
from tmte.pipeline.extended_ward_kernel import run_and_write_extended_ward_kernel  # noqa: E402


def _nonnegative_int(value: str) -> int:
    index = int(value)
    if index < 0:
        raise argparse.ArgumentTypeError("matsubara index must be non-negative")
    return index


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("nk must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only q-along-x finite-q TM/TE extended Ward-kernel audit.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairing", default="dwave")
    parser.add_argument("--matsubara-index", "--n", dest="matsubara_index", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--nk", type=_positive_int, required=True)
    parser.add_argument("--delta0", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--contact-scale", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _phase_abs(result: dict[str, object], side: str) -> float:
    key = "W_eta_left" if side == "left" else "W_eta_right"
    values = result[key]  # type: ignore[index]
    for item in values:
        if item["label"] == "phase_eta2":
            value = item["value"]
            if isinstance(value, dict):
                return abs(complex(float(value["real"]), float(value["imag"])))
            return abs(complex(value))
    return 0.0


def _print_summary(payload: dict[str, object]) -> None:
    print("candidate left_em_norm left_eta_norm right_em_norm right_eta_norm left_total right_total |W_L_phase| |W_R_phase|")
    for result in payload["candidate_results"]:  # type: ignore[index]
        norms = result["norms"]
        print(
            f"{result['candidate']} "
            f"{float(norms['left_em_norm']):.6e} "
            f"{float(norms['left_collective_norm']):.6e} "
            f"{float(norms['right_em_norm']):.6e} "
            f"{float(norms['right_collective_norm']):.6e} "
            f"{float(norms['left_total_extended_norm']):.6e} "
            f"{float(norms['right_total_extended_norm']):.6e} "
            f"{_phase_abs(result, 'left'):.6e} "
            f"{_phase_abs(result, 'right'):.6e}"
        )
    consistency = payload["schur_consistency"]  # type: ignore[index]
    print("schur_consistency:")
    print(f"fitted_left_em_minus_schur_g_row_norm {float(consistency['fitted_left_em_minus_schur_g_row_norm']):.6e}")
    print(f"fitted_right_em_minus_schur_g_col_norm {float(consistency['fitted_right_em_minus_schur_g_col_norm']):.6e}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_extended_ward_kernel(
        args.output_dir,
        model_name=args.model,
        pairing_name=args.pairing,
        matsubara_index=args.matsubara_index,
        temperature_K=args.temperature_K,
        q_value=args.q,
        nk=args.nk,
        delta0_eV=args.delta0,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
        contact_scale=args.contact_scale,
        tolerance=args.tolerance,
    )
    print(f"extended_ward_kernel.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
