#!/usr/bin/env python3
"""Offline diagnostic tail fit for minimal Casimir n-scan CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SANDBOX_ROOT))

from tmte.pipeline.minimal_casimir_n_tail_fit import (  # noqa: E402
    DEFAULT_QUANTITY_COLUMN,
    SUPPORTED_MODELS,
    run_and_write_minimal_casimir_n_tail_fit,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline diagnostic tail fit for n-scan CSV outputs.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--quantity-column", default=DEFAULT_QUANTITY_COLUMN)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS))
    parser.add_argument("--fit-min-n", type=_positive_int, default=None)
    parser.add_argument("--fit-max-n", type=_positive_int, default=None)
    parser.add_argument("--tail-start-n-exclusive", type=_positive_int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_summary(payload: dict[str, object], output_dir: Path) -> None:
    print("minimal_casimir_n_tail_fit summary")
    print("status:", payload["status"])
    print("input:", payload["input"])
    print("aggregate:", payload["summary"])
    print("model,x_name,p,r2_log,rmse_log,max_rel_err,tail_mid,tail_lower,tail_upper,convergent")
    for row in payload["fit_summaries"]:
        print(
            f"{row['model']},"
            f"{row['x_name']},"
            f"{float(row['p']):.12e},"
            f"{float(row['r2_log_space']):.12e},"
            f"{float(row['rmse_log_space']):.12e},"
            f"{float(row['max_abs_relative_error']):.12e},"
            f"{row['tail_midpoint_estimate_diagnostic']},"
            f"{row['tail_lower_bound_diagnostic']},"
            f"{row['tail_upper_bound_diagnostic']},"
            f"{row['tail_convergent']}"
        )
    print(f"json written to {output_dir / 'minimal_casimir_n_tail_fit.json'}")
    print(f"summary csv written to {output_dir / 'minimal_casimir_n_tail_fit_summary.csv'}")
    print(f"residual csv written to {output_dir / 'minimal_casimir_n_tail_fit_residuals.csv'}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_minimal_casimir_n_tail_fit(
        args.output_dir,
        input_csv_path=args.input_csv,
        quantity_column=args.quantity_column,
        models=tuple(args.models),
        fit_min_n=args.fit_min_n,
        fit_max_n=args.fit_max_n,
        tail_start_n_exclusive=args.tail_start_n_exclusive,
    )
    _print_summary(payload, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
