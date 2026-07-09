#!/usr/bin/env python3
"""Debug-only robustness scan for primitive extended translation collective audit."""

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
from tmte.pipeline.primitive_extended_translation_collective_audit import DEFAULT_CANDIDATE  # noqa: E402
from tmte.pipeline.primitive_extended_translation_collective_robustness_scan import (  # noqa: E402
    run_and_write_primitive_extended_translation_collective_robustness_scan,
)


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


def _fmt_complex(row: dict[str, float]) -> str:
    return f"{float(row['real']):+.8e}{float(row['imag']):+.8e}j"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug-only robustness scan for primitive extended translation collective audit.")
    parser.add_argument("--model", choices=available_models(), default="symmetry_bdg_2band")
    parser.add_argument("--pairings", nargs="+", default=["spm", "dwave"])
    parser.add_argument("--matsubara-indices", "--n-values", nargs="+", type=_nonnegative_int, required=True)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--q-values", nargs="+", type=_positive_float, required=True)
    parser.add_argument("--nk-values", nargs="+", type=_positive_int, required=True)
    parser.add_argument("--delta0-eV", type=float, default=None)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--shift-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--contact-scale", type=float, default=1.0)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--diff-tol", type=float, default=1e-9)
    parser.add_argument("--fit-res-tol", type=float, default=1e-9)
    parser.add_argument("--alpha-tol", type=float, default=1e-9)
    parser.add_argument("--keep-payloads", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_summary(payload: dict[str, object]) -> None:
    print("primitive_extended_translation_collective_robustness_scan summary")
    print("status:", payload["status"])
    print("scan_parameters:", payload["scan_parameters"])
    print("aggregate:", payload["aggregate"])
    print("rows")
    print("pass pairing n q nk left_top left_diff right_top right_diff left_alpha right_alpha ext/em")
    for row in payload["summary_rows"]:
        mark = "PASS" if row["passed_translation_identity"] else "FAIL"
        print(
            f"{mark} "
            f"pairing={row['pairing']} "
            f"n={int(row['matsubara_index'])} "
            f"q={float(row['q_value']):.6g} "
            f"nk={int(row['nk'])} "
            f"left_top={row['left']['top_candidate']} "
            f"left_diff={float(row['left']['top_difference_over_missing']):.3e} "
            f"right_top={row['right']['top_candidate']} "
            f"right_diff={float(row['right']['top_difference_over_missing']):.3e} "
            f"left_alpha={_fmt_complex(row['left']['top_fit_alpha'])} "
            f"right_alpha={_fmt_complex(row['right']['top_fit_alpha'])} "
            f"ext/em={float(row['left_extended_over_em']):.3e}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_primitive_extended_translation_collective_robustness_scan(
        args.output_dir,
        model_name=args.model,
        pairings=tuple(args.pairings),
        matsubara_indices=tuple(args.matsubara_indices),
        temperature_K=args.temperature_K,
        q_values=tuple(args.q_values),
        nk_values=tuple(args.nk_values),
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta,
        shift_fractions=tuple(args.shift_fractions),
        contact_scale=args.contact_scale,
        candidate_name=args.candidate,
        diff_tol=args.diff_tol,
        fit_res_tol=args.fit_res_tol,
        alpha_tol=args.alpha_tol,
        keep_payloads=bool(args.keep_payloads),
    )
    print(f"primitive_extended_translation_collective_robustness_scan.json written to {args.output_dir}")
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
