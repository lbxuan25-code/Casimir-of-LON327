#!/usr/bin/env python3
"""Debug-only robustness scan for normal equal-time Ward identity."""

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
from tmte.pipeline.normal_equal_time_robustness_scan import run_and_write_normal_equal_time_robustness_scan  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Debug-only robustness scan for normal equal-time Ward identity.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--matsubara-indices", "--n-values", nargs="+", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q-values", nargs="+", type=_positive_float, required=True)
    parser.add_argument("--nk-values", nargs="+", type=_positive_int, required=True)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--diff-tol", type=float, default=1e-10)
    parser.add_argument("--fit-res-tol", type=float, default=1e-10)
    parser.add_argument("--alpha-tol", type=float, default=1e-10)
    parser.add_argument("--keep-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _fmt_complex(row: dict[str, float]) -> str:
    return f"{float(row['real']):+.8e}{float(row['imag']):+.8e}j"


def _print_summary(payload: dict[str, object]) -> None:
    print("normal_equal_time_robustness_scan summary")
    print("status:", payload["status"])
    print("scan_parameters:", payload["scan_parameters"])
    print("aggregate:", payload["aggregate"])
    print("rows")
    print("pass n q nk top diff/missing fit_res alpha contact_alpha_left")
    for row in payload["summary_rows"]:
        mark = "PASS" if row["passed_translation_identity"] else "FAIL"
        print(
            f"{mark} "
            f"n={int(row['matsubara_index'])} "
            f"q={float(row['q_value']):.6g} "
            f"nk={int(row['nk'])} "
            f"top={row['top_candidate']} "
            f"diff={float(row['top_difference_over_missing']):.3e} "
            f"fit={float(row['top_fit_residual_over_missing']):.3e} "
            f"alpha={_fmt_complex(row['top_fit_alpha'])} "
            f"contact_alpha={_fmt_complex(row['contact_alpha_left'])}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_normal_equal_time_robustness_scan(
        args.output_dir,
        model_name=args.model,
        matsubara_indices=tuple(args.matsubara_indices),
        temperature_K=args.temperature_K,
        q_values=tuple(args.q_values),
        nk_values=tuple(args.nk_values),
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
        diff_tol=args.diff_tol,
        fit_res_tol=args.fit_res_tol,
        alpha_tol=args.alpha_tol,
        keep_payloads=bool(args.keep_payloads),
    )
    print(f"normal_equal_time_robustness_scan.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
