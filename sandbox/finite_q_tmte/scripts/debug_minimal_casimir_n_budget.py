#!/usr/bin/env python3
"""Offline diagnostic n-budget aggregation for minimal Casimir n scans."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SANDBOX_ROOT))

from tmte.pipeline.minimal_casimir_n_budget import run_and_write_minimal_casimir_n_budget  # noqa: E402
from tmte.pipeline.minimal_casimir_n_tail_fit import DEFAULT_QUANTITY_COLUMN  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline diagnostic n-budget aggregation for n-scan CSV outputs.")
    parser.add_argument("--input-csv", dest="input_csv_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--quantity-column", default=DEFAULT_QUANTITY_COLUMN)
    parser.add_argument("--tail-fit-json", dest="tail_fit_json_paths", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_summary(payload: dict[str, object], output_dir: Path) -> None:
    print("minimal_casimir_n_budget summary")
    print("status:", payload["status"])
    print("input:", payload["input"])
    print("aggregate:", payload["summary"])
    print(f"json written to {output_dir / 'minimal_casimir_n_budget.json'}")
    print(f"terms csv written to {output_dir / 'minimal_casimir_n_budget_terms.csv'}")
    print(f"gaps csv written to {output_dir / 'minimal_casimir_n_budget_gaps.csv'}")
    print(f"tail fits csv written to {output_dir / 'minimal_casimir_n_budget_tail_fits.csv'}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_minimal_casimir_n_budget(
        args.output_dir,
        input_csv_paths=tuple(args.input_csv_paths),
        quantity_column=args.quantity_column,
        tail_fit_json_paths=tuple(args.tail_fit_json_paths),
    )
    _print_summary(payload, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
