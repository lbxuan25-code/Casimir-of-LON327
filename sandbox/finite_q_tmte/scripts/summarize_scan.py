#!/usr/bin/env python3
"""Summarize an existing finite-q TM/TE sandbox JSON output."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from tmte.io.readers import read_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize tmte_scan.json.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    payload = read_json(args.path)
    scan = payload.get("scan_parameters", {})
    print(f"schema_version: {payload.get('schema_version')}")
    print(f"model: {payload.get('model', {}).get('name')} pairing={payload.get('model', {}).get('pairing')}")
    print(f"valid_for_casimir_input: {payload.get('status', {}).get('valid_for_casimir_input')}")
    print(f"result_count: {scan.get('result_count')} q_count={scan.get('q_count')}")
    print(f"shifted_average: {scan.get('shifted_mesh_average', {}).get('average_order')}")
    for row in payload.get("results", []):
        diag = row.get("diagnostics", {})
        schur = row.get("schur", {})
        print(
            "q={q} physical_norm={phys} gauge_row={grow} gauge_col={gcol} cond={cond} schur={method} suspect={suspect}".format(
                q=row.get("q_model"),
                phys=diag.get("physical_matrix_norm"),
                grow=diag.get("gauge_row_norm"),
                gcol=diag.get("gauge_col_norm"),
                cond=diag.get("etaeta_condition_number"),
                method=schur.get("solve_method"),
                suspect=schur.get("numerically_suspect"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
