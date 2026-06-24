#!/usr/bin/env python3
"""Audit BdG shifted-grid and direct/contact response-assembly consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bdg_shifted_grid_assembly_common import (
    BAND_SHIFTED_MONITOR,
    BAND_SHIFTED_PASS,
    PAIRINGS,
    SHIFTED_DIRECT_PASS,
    audit_shifted_grid_assembly,
    commensurate_grid_spec,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
N_LIST_QUICK = (12, 18, 24, 36)
N_LIST_FULL = (24, 36, 48, 72, 96)
FIXED_Q_N_LIST = (24, 36, 48, 72)


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


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "N": int(rows[0]["N"]),
        "q_model": list(rows[0]["q_model"]),
        "E_shifted_plus_qD_abs": max(float(row["E_shifted_plus_qD_abs"]) for row in rows),
        "E_band_minus_E_shifted_abs": max(float(row["E_band_minus_E_shifted_abs"]) for row in rows),
        "E_band_plus_qD_abs": max(float(row["E_band_plus_qD_abs"]) for row in rows),
    }


def _grouped(rows: list[dict[str, Any]], pairing: str, grid_type: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["pairing"] == pairing and row["grid_type"] == grid_type]
    keys = sorted({(int(row["N"]), tuple(row["q_model"])) for row in selected})
    return [
        _aggregate(
            [
                row
                for row in selected
                if int(row["N"]) == n_grid and tuple(row["q_model"]) == q_model
            ]
        )
        for n_grid, q_model in keys
    ]


def _best(records: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        records,
        key=lambda row: (
            float(row["E_band_plus_qD_abs"]),
            float(row["E_band_minus_E_shifted_abs"]),
            float(row["E_shifted_plus_qD_abs"]),
        ),
    )


def _pairing_interpretation(
    grid_step: list[dict[str, Any]],
    half_step: list[dict[str, Any]],
    fixed_q: list[dict[str, Any]],
) -> tuple[str, str]:
    grid_direct = max(float(row["E_shifted_plus_qD_abs"]) for row in grid_step)
    grid_band = max(float(row["E_band_minus_E_shifted_abs"]) for row in grid_step)
    grid_closure = max(float(row["E_band_plus_qD_abs"]) for row in grid_step)
    if grid_direct >= SHIFTED_DIRECT_PASS:
        return (
            "direct_expectation_mismatch",
            "shifted-trace/contact cancellation fails; inspect direct expectation prefactor, sign, and trace assembly",
        )
    if grid_band >= BAND_SHIFTED_PASS or grid_closure >= BAND_SHIFTED_PASS:
        return (
            "band_vs_shifted_remainder",
            "direct identity passes but grid-step band and shifted trace disagree",
        )
    ordinary = half_step + fixed_q
    if max(float(row["E_band_plus_qD_abs"]) for row in ordinary) >= BAND_SHIFTED_PASS:
        return (
            "ordinary_quadrature_convergence",
            "grid-step assembly closes; remaining half-step/fixed-q residual is a shifted-grid quadrature error",
        )
    return "none", "all tested shifted-trace, direct, and band closures pass"


def build_payload(quick: bool = True) -> dict[str, Any]:
    n_list = N_LIST_QUICK if quick else N_LIST_FULL
    rows: list[dict[str, Any]] = []
    for pairing in PAIRINGS:
        for n_grid in n_list:
            for grid_type in ("grid_step_commensurate", "half_step_commensurate"):
                for diagonal in (False, True):
                    spec = commensurate_grid_spec(n_grid, grid_type, diagonal=diagonal)
                    rows.extend(
                        audit_shifted_grid_assembly(
                            pairing,
                            n_grid,
                            spec["q_model"],
                            grid_type,
                        )
                    )
        for n_grid in FIXED_Q_N_LIST:
            rows.extend(audit_shifted_grid_assembly(pairing, n_grid, (0.01, 0.0), "fixed_q"))

    summary: dict[str, Any] = {
        "formal_casimir_ran": False,
        "diagnostic_only": True,
    }
    pairing_dominant: dict[str, str] = {}
    pairing_interpretations: dict[str, str] = {}
    for pairing in PAIRINGS:
        grid_step = _grouped(rows, pairing, "grid_step_commensurate")
        half_step = _grouped(rows, pairing, "half_step_commensurate")
        fixed_q = _grouped(rows, pairing, "fixed_q")
        dominant, interpretation = _pairing_interpretation(grid_step, half_step, fixed_q)
        pairing_dominant[pairing] = dominant
        pairing_interpretations[pairing] = interpretation
        summary[pairing] = {
            "grid_step_commensurate_best": _best(grid_step),
            "half_step_commensurate_best": _best(half_step),
            "fixed_q_best": _best(fixed_q),
            "dominant_failure": dominant,
            "interpretation": interpretation,
        }

    onsite_grid_rows = [
        row
        for row in rows
        if row["pairing"] == "onsite_s" and row["grid_type"] == "grid_step_commensurate"
    ]
    onsite_direct = max(float(row["E_shifted_plus_qD_abs"]) for row in onsite_grid_rows)
    onsite_band = max(float(row["E_band_minus_E_shifted_abs"]) for row in onsite_grid_rows)
    onsite_closure = max(float(row["E_band_plus_qD_abs"]) for row in onsite_grid_rows)
    if (
        onsite_direct < SHIFTED_DIRECT_PASS
        and onsite_band < BAND_SHIFTED_PASS
        and onsite_closure < BAND_SHIFTED_PASS
    ):
        status = "PASSED"
    elif (
        onsite_direct < SHIFTED_DIRECT_PASS
        and onsite_band < BAND_SHIFTED_MONITOR
        and onsite_closure < BAND_SHIFTED_MONITOR
    ):
        status = "MONITOR"
    else:
        status = "FAILED"
    dominant_failure = pairing_dominant["onsite_s"]
    best_interpretation = pairing_interpretations["onsite_s"]
    summary.update(
        {
            "status": status,
            "dominant_failure": dominant_failure,
            "best_interpretation": best_interpretation,
        }
    )
    return {
        **summary,
        "quick": bool(quick),
        "candidate": "A",
        "candidate_ordering": "rho_Hp_minus_Hm_rho",
        "qV_sign": -1,
        "C_eta2": 2j * 0.04,
        "source_channels": ["Vx", "Vy"],
        "summary": summary,
        "cases": rows,
    }


def _write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_0e_bdg_shifted_grid_response_assembly_audit"
    json_path = OUTPUT_DIR / f"{stem}.json"
    md_path = OUTPUT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {stem}",
        "",
        f"- status: {payload['status']}",
        f"- formal Casimir ran: {payload['formal_casimir_ran']}",
        f"- diagnostic only: {payload['diagnostic_only']}",
        f"- dominant failure: {payload['dominant_failure']}",
        f"- best interpretation: {payload['best_interpretation']}",
    ]
    for grid_type, title in (
        ("grid_step_commensurate", "Grid-step commensurate"),
        ("half_step_commensurate", "Half-step commensurate"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| pairing | N | q | B | E_band-E_shifted | E_shifted+qD | E_band+qD | status |",
                "| ------- | -: | - | - | ----------------: | -----------: | --------: | ------ |",
            ]
        )
        for row in payload["cases"]:
            if row["grid_type"] != grid_type:
                continue
            q_label = f"({row['q_model'][0]:.7g},{row['q_model'][1]:.7g})"
            lines.append(
                f"| {row['pairing']} | {row['N']} | {q_label} | {row['source_channel']} | "
                f"{row['E_band_minus_E_shifted_abs']:.8g} | {row['E_shifted_plus_qD_abs']:.8g} | "
                f"{row['E_band_plus_qD_abs']:.8g} | {row['status']} |"
            )
    lines.extend(
        [
            "",
            "## Fixed-q convergence: onsite_s",
            "",
            "| N | B | E_band-E_shifted | E_shifted+qD | E_band+qD |",
            "| -: | - | ----------------: | -----------: | --------: |",
        ]
    )
    for row in payload["cases"]:
        if row["grid_type"] == "fixed_q" and row["pairing"] == "onsite_s":
            lines.append(
                f"| {row['N']} | {row['source_channel']} | {row['E_band_minus_E_shifted_abs']:.8g} | "
                f"{row['E_shifted_plus_qD_abs']:.8g} | {row['E_band_plus_qD_abs']:.8g} |"
            )
    lines.extend(
        [
            "",
            "## Dominant interpretation",
            "",
            "| pairing | dominant failure | interpretation |",
            "| ------- | ---------------- | -------------- |",
        ]
    )
    for pairing in PAIRINGS:
        lines.append(
            f"| {pairing} | {payload[pairing]['dominant_failure']} | "
            f"{payload[pairing]['interpretation']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", dest="quick", action="store_true", default=True)
    mode.add_argument("--full", dest="quick", action="store_false")
    args = parser.parse_args()
    payload = build_payload(args.quick)
    _write_report(payload)


if __name__ == "__main__":
    main()
