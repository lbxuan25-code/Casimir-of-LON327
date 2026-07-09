#!/usr/bin/env python3
"""Offline health report for existing minimal Casimir diagnostic artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SANDBOX_ROOT))

from tmte.pipeline.minimal_casimir_health_report import (  # noqa: E402
    DEFAULT_PHI_RANGE_WARNING_THRESHOLD,
    DEFAULT_RDIFF_WARNING_THRESHOLD,
    DEFAULT_R_NORM_WARNING_THRESHOLD,
    run_and_write_minimal_casimir_health_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline credibility health report for existing minimal Casimir artifacts.")
    parser.add_argument("--input-json", dest="input_json_paths", nargs="*", type=Path, default=[])
    parser.add_argument("--input-csv", dest="input_csv_paths", nargs="*", type=Path, default=[])
    parser.add_argument("--r-norm-warning-threshold", type=float, default=DEFAULT_R_NORM_WARNING_THRESHOLD)
    parser.add_argument("--rdiff-warning-threshold", type=float, default=DEFAULT_RDIFF_WARNING_THRESHOLD)
    parser.add_argument("--phi-range-warning-threshold", type=float, default=DEFAULT_PHI_RANGE_WARNING_THRESHOLD)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _print_summary(payload: dict[str, object], output_dir: Path) -> None:
    print("minimal_casimir_health_report summary")
    print("status:", payload["status"])
    print("input:", payload["input"])
    print("aggregate:", payload["summary"])
    print(f"json written to {output_dir / 'minimal_casimir_health_report.json'}")
    print(f"csv written to {output_dir / 'minimal_casimir_health_report_findings.csv'}")
    print(f"markdown written to {output_dir / 'minimal_casimir_health_report.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_and_write_minimal_casimir_health_report(
        args.output_dir,
        input_json_paths=tuple(args.input_json_paths),
        input_csv_paths=tuple(args.input_csv_paths),
        r_norm_warning_threshold=args.r_norm_warning_threshold,
        rdiff_warning_threshold=args.rdiff_warning_threshold,
        phi_range_warning_threshold=args.phi_range_warning_threshold,
    )
    _print_summary(payload, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
