#!/usr/bin/env python3
"""Summarize a finite-q TM/TE nk_sweep.json output."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT))

from tmte.io.readers import read_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize nk_sweep.json.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    payload = read_json(args.path)
    frequency = payload.get("frequency", {})
    print(f"schema_version: {payload.get('schema_version')}")
    print(f"model: {payload.get('model', {}).get('name')} pairing={payload.get('model', {}).get('pairing')}")
    print(
        "matsubara_index={n} temperature_K={temp} xi_eV={xi} zero_matsubara_mode={zero}".format(
            n=frequency.get("matsubara_index"),
            temp=frequency.get("temperature_K"),
            xi=frequency.get("xi_eV"),
            zero=frequency.get("zero_matsubara_mode"),
        )
    )
    for row in payload.get("nk_results", []):
        diag = row.get("diagnostics", {})
        ratios = row.get("ratios", {})
        schur = row.get("schur", {})
        print(
            "nk={nk} q={q} gauge_row={grow} gauge_col={gcol} gauge_gg={gg} physical={phys} "
            "gauge_over_physical={gphys} gauge_over_tm_abs={gtm} gauge_gg_over_tm_abs={ggtm} "
            "cond={cond} schur={method} suspect={suspect}".format(
                nk=row.get("nk"),
                q=row.get("q_model"),
                grow=diag.get("gauge_row_norm"),
                gcol=diag.get("gauge_col_norm"),
                gg=diag.get("gauge_gg_norm"),
                phys=diag.get("physical_matrix_norm"),
                gphys=ratios.get("gauge_over_physical"),
                gtm=ratios.get("gauge_over_tm_abs"),
                ggtm=ratios.get("gauge_gg_over_tm_abs"),
                cond=diag.get("etaeta_condition_number"),
                method=schur.get("solve_method"),
                suspect=schur.get("numerically_suspect"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

