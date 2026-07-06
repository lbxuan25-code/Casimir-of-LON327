#!/usr/bin/env python3
"""Print a short finite-q BdG validation status summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总 finite-q BdG validation 状态。")
    parser.add_argument("--report-json", type=Path, default=Path("validation/outputs/finite_q_ward/report.json"))
    args = parser.parse_args(argv)
    report = _read_json(args.report_json)
    finite_q_status = report.get("finite_q_status", {})
    print("finite-q BdG validation summary")
    print(f"q0_precondition_status: {report.get('q0_precondition_status', {})}")
    print(f"diagnostic_run_completed: {finite_q_status.get('diagnostic_run_completed', False)}")
    print(f"ward_identity_closed: {finite_q_status.get('ward_identity_closed', False)}")
    print("valid_for_casimir_input: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
