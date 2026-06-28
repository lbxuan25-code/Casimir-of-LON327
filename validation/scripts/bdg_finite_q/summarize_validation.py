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
    parser.add_argument("--q0-status-json", type=Path, default=Path("validation/outputs/bdg_finite_q/q0_status.json"))
    parser.add_argument("--ward-status-json", type=Path, default=Path("validation/outputs/bdg_finite_q/ward_scan_status.json"))
    args = parser.parse_args(argv)
    q0 = _read_json(args.q0_status_json)
    ward = _read_json(args.ward_status_json)
    print("finite-q BdG validation summary")
    print(f"q0_precondition_status: {q0.get('status_by_pairing', {})}")
    print(f"diagnostic_run_completed: {ward.get('diagnostic_run_completed', False)}")
    print(f"ward_identity_closed: {ward.get('ward_identity_closed', False)}")
    print("valid_for_casimir_input: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
