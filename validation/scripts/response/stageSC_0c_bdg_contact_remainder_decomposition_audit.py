#!/usr/bin/env python3
"""Decompose the BdG contact/equal-time Ward remainder by source channel."""

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
    audit_contact_remainder_pairing,
    convention_summary,
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


def _format_complex(value: complex | dict[str, float]) -> str:
    if isinstance(value, dict):
        real = float(value.get("real", 0.0))
        imag = float(value.get("imag", 0.0))
    else:
        real = float(np.real(value))
        imag = float(np.imag(value))
    if abs(imag) < 1e-15:
        return f"{real:.6g}"
    if abs(real) < 1e-15:
        return f"{imag:.6g}i"
    return f"{real:.6g}{imag:+.6g}i"


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_pairing = {case["pairing"]: case for case in cases}
    return {
        "status": by_pairing["onsite_s"]["status"],
        "max_contact_closure_abs_by_pairing": {
            pairing: by_pairing[pairing]["max_contact_closure_abs"] for pairing in PAIRINGS
        },
        "dominant_source_channel_by_pairing": {
            pairing: by_pairing[pairing]["dominant_source_channel"] for pairing in PAIRINGS
        },
        "dominant_missing_block_guess_by_pairing": {
            pairing: by_pairing[pairing]["dominant_missing_block_guess"] for pairing in PAIRINGS
        },
        "unavailable_contact_block_present_by_pairing": {
            pairing: by_pairing[pairing]["unavailable_contact_block_present"] for pairing in PAIRINGS
        },
        "formal_casimir_ran": False,
        "diagnostic_only": True,
    }


def build_payload(quick: bool) -> dict[str, Any]:
    delta0 = 0.04
    cases = [audit_contact_remainder_pairing(StageSC0bInputs(pairing=pairing, delta0_eV=delta0)) for pairing in PAIRINGS]
    summary = _summary(cases)
    return {
        "status": summary["status"],
        "quick": bool(quick),
        "diagnostic_only": True,
        "writes_production_casimir_outputs": False,
        "formal_casimir_ran": False,
        "convention": convention_summary(delta0),
        "summary": summary,
        "cases": cases,
    }


def _write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "stageSC_0c_bdg_contact_remainder_decomposition_audit.json"
    md_path = OUTPUT_DIR / "stageSC_0c_bdg_contact_remainder_decomposition_audit.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# stageSC_0c_bdg_contact_remainder_decomposition_audit",
        "",
        f"- status: {payload['status']}",
        f"- quick: {payload['quick']}",
        f"- cases: {len(payload['cases'])}",
        f"- candidate: {payload['convention']['candidate']}",
        f"- candidate_ordering: {payload['convention']['candidate_ordering']}",
        f"- qV_sign: {payload['convention']['qV_sign']}",
        f"- C_eta2: {_format_complex(payload['convention']['C_eta2'])}",
        f"- formal Casimir ran: {payload['formal_casimir_ran']}",
        "",
        "| pairing | max contact closure | dominant B | dominant missing guess | status |",
        "| ------- | ------------------: | ---------- | ---------------------- | ------ |",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['pairing']} | {case['max_contact_closure_abs']:.6g} | "
            f"{case['dominant_source_channel']} | {case['dominant_missing_block_guess']} | {case['status']} |"
        )
    lines.extend(
        [
            "",
            "| pairing | D_eta2_eta2 needed | Goldstone Cg existing | difference | comment |",
            "| ------- | -----------------: | --------------------: | ---------: | ------- |",
        ]
    )
    for case in payload["cases"]:
        eta2 = case["representative_channel_rows"]["eta2"]
        needed = eta2["D_eta2_eta2_needed"]
        cg = eta2["goldstone_counterterm_Cg_existing"]
        diff = eta2["needed_minus_goldstone_counterterm"]
        comment = "diff large" if abs(diff) >= 1e-8 else "matches"
        lines.append(
            f"| {case['pairing']} | {_format_complex(needed)} | {_format_complex(cg)} | "
            f"{_format_complex(diff)} | {comment} |"
        )
    lines.extend(
        [
            "",
            "| pairing | B | E_B | existing direct projection | residual |",
            "| ------- | - | --: | -------------------------: | -------: |",
        ]
    )
    for case in payload["cases"]:
        for channel, row in case["representative_channel_rows"].items():
            lines.append(
                f"| {case['pairing']} | {channel} | {_format_complex(row['E_B'])} | "
                f"{_format_complex(row['D_projection_existing'])} | "
                f"{_format_complex(row['contact_closure_residual'])} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    payload = build_payload(args.quick)
    _write_report(payload)


if __name__ == "__main__":
    main()

