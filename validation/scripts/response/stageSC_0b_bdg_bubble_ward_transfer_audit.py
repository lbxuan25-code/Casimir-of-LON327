#!/usr/bin/env python3
"""Audit how the BdG operator Ward identity transfers into bubble band sums."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from bdg_bubble_ward_transfer_common import (  # noqa: E402
    PAIRINGS,
    StageSC0bInputs,
    audit_pairing,
    concise_metric,
    convention_summary,
    overall_summary,
)


OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "stageSC_0b_bdg_bubble_ward_transfer_audit.json"
    md_path = OUTPUT_DIR / "stageSC_0b_bdg_bubble_ward_transfer_audit.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# stageSC_0b_bdg_bubble_ward_transfer_audit",
        "",
        f"- status: {payload['status']}",
        f"- quick: {payload['quick']}",
        f"- cases: {len(payload['cases'])}",
        f"- candidate: {payload['convention']['candidate']}",
        f"- candidate_ordering: {payload['convention']['candidate_ordering']}",
        f"- qV_sign: {payload['convention']['qV_sign']}",
        f"- C_eta2: {_format_complex(payload['convention']['C_eta2'])}",
        "",
        "| pairing | band-pair identity | bubble transfer | right vertex orientation | bubble+direct Ward | dominant failure |",
        "| ------- | -----------------: | --------------: | -----------------------: | -----------------: | ---------------- |",
    ]
    for row in payload["summary_table"]:
        lines.append(
            f"| {row['pairing']} | {row['band_pair_identity']:.6g} | {row['bubble_transfer']:.6g} | "
            f"{row['right_vertex_orientation']:.6g} | {row['bubble_plus_direct_Ward']:.6g} | "
            f"{row['dominant_failure']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def _format_complex(value: complex | dict[str, float]) -> str:
    if isinstance(value, dict):
        real = float(value.get("real", 0.0))
        imag = float(value.get("imag", 0.0))
    else:
        real = float(np.real(value))
        imag = float(np.imag(value))
    if abs(real) < 1e-15:
        return f"{imag:.6g}i"
    return f"{real:.6g}{imag:+.6g}i"


def build_payload(quick: bool) -> dict[str, Any]:
    delta0 = 0.04
    cases = [audit_pairing(StageSC0bInputs(pairing=pairing, delta0_eV=delta0)) for pairing in PAIRINGS]
    summary = overall_summary(cases)
    return {
        "status": summary["status"],
        "quick": bool(quick),
        "diagnostic_only": True,
        "writes_production_casimir_outputs": False,
        "convention": convention_summary(delta0),
        "summary": summary,
        "summary_table": [concise_metric(case) for case in cases],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    payload = build_payload(args.quick)
    _write_report(payload)


if __name__ == "__main__":
    main()

