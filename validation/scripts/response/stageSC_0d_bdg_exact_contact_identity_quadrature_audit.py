#!/usr/bin/env python3
"""Audit exact Peierls contact identities and shift-invariant quadrature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bdg_contact_identity_common import (
    CONTACT_IDENTITY_PASS,
    PAIRINGS,
    SPATIAL_CLOSURE_PASS,
    assess_stageSC_0d,
    audit_pointwise_contact_identity,
    spatial_contact_closure,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
PART_B_N = 24
CONVERGENCE_N = (12, 18, 24, 36)
FIXED_Q = (0.01, 0.0)


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
        value = complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    if abs(value.imag) < 1e-14:
        return f"{value.real:.8g}"
    return f"{value.real:.8g}{value.imag:+.3g}i"


def build_payload(quick: bool = False) -> dict[str, Any]:
    """Run the complete StageSC-0d audit; quick preserves the required grids."""

    part_a = [audit_pointwise_contact_identity(pairing) for pairing in PAIRINGS]
    commensurate_q = (
        (2.0 * np.pi / PART_B_N, 0.0),
        (2.0 * np.pi / PART_B_N, 2.0 * np.pi / PART_B_N),
    )
    part_b = [
        spatial_contact_closure(pairing, PART_B_N, q_model)
        for pairing in PAIRINGS
        for q_model in commensurate_q
    ]
    grid_convergence = [
        spatial_contact_closure("onsite_s", n_grid, (2.0 * np.pi / n_grid, 0.0))
        for n_grid in CONVERGENCE_N
    ]
    fixed_q_convergence = {
        pairing: [spatial_contact_closure(pairing, n_grid, FIXED_Q) for n_grid in CONVERGENCE_N]
        for pairing in PAIRINGS
    }

    part_a_max = max(float(case["max_contact_identity_abs"]) for case in part_a)
    part_b_max = max(float(case["max_spatial_contact_closure_abs"]) for case in part_b)
    fixed_finest_max = max(
        float(fixed_q_convergence[pairing][-1]["max_spatial_contact_closure_abs"])
        for pairing in PAIRINGS
    )
    status, dominant_failure, best_interpretation = assess_stageSC_0d(
        part_a_max,
        part_b_max,
        fixed_finest_max,
    )
    by_pairing_a = {case["pairing"]: case for case in part_a}
    summary: dict[str, Any] = {
        "status": status,
        "formal_casimir_ran": False,
        "partA_contact_identity_passed": part_a_max < CONTACT_IDENTITY_PASS,
        "partB_shift_invariant_closure_passed": part_b_max < SPATIAL_CLOSURE_PASS,
        "dominant_failure": dominant_failure,
        "best_interpretation": best_interpretation,
    }
    for pairing in PAIRINGS:
        pairing_b = [case for case in part_b if case["pairing"] == pairing]
        summary[pairing] = {
            "partA_bdg_contact_identity_max_abs": by_pairing_a[pairing]["bdg_contact_identity_max_abs"],
            "partB_spatial_closure_max_abs": max(
                float(case["max_spatial_contact_closure_abs"]) for case in pairing_b
            ),
            "fixed_q_convergence": fixed_q_convergence[pairing],
            "fixed_q_endpoint_decreased": (
                fixed_q_convergence[pairing][-1]["max_spatial_contact_closure_abs"]
                < fixed_q_convergence[pairing][0]["max_spatial_contact_closure_abs"]
            ),
            "fixed_q_monotonic_decrease": all(
                right["max_spatial_contact_closure_abs"] <= left["max_spatial_contact_closure_abs"]
                for left, right in zip(
                    fixed_q_convergence[pairing],
                    fixed_q_convergence[pairing][1:],
                )
            ),
        }
    return {
        **summary,
        "status": status,
        "quick": bool(quick),
        "diagnostic_only": True,
        "formal_casimir_ran": False,
        "writes_production_casimir_outputs": False,
        "joint_convergence_note": (
            "q_model = 2*pi/N, so this is a joint small-q/finer-grid diagnostic, "
            "not fixed-q convergence."
        ),
        "summary": summary,
        "partA_pointwise_contact_identity": part_a,
        "partB_shift_invariant_contact_closure": part_b,
        "grid_convergence": grid_convergence,
        "fixed_q_convergence_by_pairing": fixed_q_convergence,
    }


def _write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_0d_bdg_exact_contact_identity_quadrature_audit"
    json_path = OUTPUT_DIR / f"{stem}.json"
    md_path = OUTPUT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# {stem}",
        "",
        f"- total status: {payload['status']}",
        f"- formal Casimir ran: {payload['formal_casimir_ran']}",
        f"- diagnostic only: {payload['diagnostic_only']}",
        f"- dominant failure: {payload['summary']['dominant_failure']}",
        f"- best interpretation: {payload['summary']['best_interpretation']}",
        "",
        "## Part A: pointwise exact contact identity",
        "",
        "| pairing | j | normal identity max | BdG identity max | status |",
        "| ------- | - | ------------------: | ----------------: | ------ |",
    ]
    for case in payload["partA_pointwise_contact_identity"]:
        for direction_j in ("x", "y"):
            row = case["by_direction"][direction_j]
            lines.append(
                f"| {case['pairing']} | {direction_j} | {row['normal_contact_identity_max_abs']:.8g} | "
                f"{row['bdg_contact_identity_max_abs']:.8g} | {row['status']} |"
            )
    lines.extend(
        [
            "",
            "## Part B: shift-invariant contact closure",
            "",
            "| pairing | q | Vx closure | Vy closure | max | status |",
            "| ------- | - | ---------: | ---------: | --: | ------ |",
        ]
    )
    for case in payload["partB_shift_invariant_contact_closure"]:
        q_label = f"({case['q_model'][0]:.8g}, {case['q_model'][1]:.8g})"
        lines.append(
            f"| {case['pairing']} | {q_label} | {_format_complex(case['Vx']['closure_residual'])} | "
            f"{_format_complex(case['Vy']['closure_residual'])} | "
            f"{case['max_spatial_contact_closure_abs']:.8g} | {case['status']} |"
        )
    lines.extend(
        [
            "",
            "## Part C: joint small-q/finer-grid diagnostic",
            "",
            payload["joint_convergence_note"],
            "",
            "| N | q=2pi/N | onsite_s max spatial closure |",
            "| -: | ------: | ---------------------------: |",
        ]
    )
    for case in payload["grid_convergence"]:
        lines.append(
            f"| {case['N']} | {case['q_model'][0]:.8g} | {case['max_spatial_contact_closure_abs']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Part D: fixed-q quadrature convergence",
            "",
            "| N | fixed q=0.01 | onsite_s max spatial closure |",
            "| -: | -----------: | ---------------------------: |",
        ]
    )
    for case in payload["fixed_q_convergence_by_pairing"]["onsite_s"]:
        lines.append(f"| {case['N']} | 0.01 | {case['max_spatial_contact_closure_abs']:.8g} |")
    lines.extend(
        [
            "",
            "### Fixed-q results for all pairings",
            "",
            "| pairing | N | max spatial closure | status |",
            "| ------- | -: | ------------------: | ------ |",
        ]
    )
    for pairing in PAIRINGS:
        for case in payload["fixed_q_convergence_by_pairing"][pairing]:
            lines.append(
                f"| {pairing} | {case['N']} | {case['max_spatial_contact_closure_abs']:.8g} | {case['status']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The pointwise contact formula passes, while one or more integrated closures fail. "
                "This does not establish a missing contact term; quadrature/contact closure implementation "
                "and shifted-grid/response-assembly consistency remain unresolved."
                if payload["summary"]["partA_contact_identity_passed"]
                and not payload["summary"]["partB_shift_invariant_closure_passed"]
                else payload["summary"]["best_interpretation"]
            ),
        ]
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
