#!/usr/bin/env python3
"""Audit amplitude-phase BdG Ward closure on grid-step commensurate q."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bdg_commensurate_q_common import (
    build_dwave_decomposition,
    commensurate_case_status,
    commensurate_q_spec,
    overall_stageSC_2bC_status,
)
from bdg_quadrature_strategy_common import (
    composite_uniform_quadrature,
    compute_bdg_components_for_composite_grid,
)
from lno327.conductivity import KuboConfig


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
N_LIST_QUICK = (24, 36, 48)
N_LIST_FULL = (24, 36, 48, 72, 96)
M_CASES_QUICK = ((1, 0), (1, 1))
M_CASES_FULL = ((1, 0), (1, 1), (2, 0), (2, 1))
PHASE_OPTIONS = {
    "onsite_s": ("midpoint",),
    "spm": ("midpoint",),
    "dwave": ("symmetric_kpm", "midpoint"),
}


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


def _ward_components(values: np.ndarray) -> dict[str, float]:
    return {
        label: float(abs(value))
        for label, value in zip(("rho", "x", "y"), values, strict=True)
    }


def _case_row(
    pairing: str,
    spec: dict[str, Any],
    phase_vertex: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    contact = max(
        float(result["contact_closure"][channel]["E_band_plus_qD_abs"])
        for channel in ("Vx", "Vy")
    )
    amplitude_left = np.asarray(result["amplitude_phase_left_ward"])
    amplitude_right = np.asarray(result["amplitude_phase_right_ward"])
    conductivity_values = (
        result["sigma_diag_min_real"],
        result["sigma_offdiag_rel"],
        result["sigma_xx_yy_anisotropy"],
        result["max_abs_sigma_tilde"],
    )
    row = {
        "pairing": pairing,
        "N": int(spec["N"]),
        "m_case": list(spec["m_case"]),
        "q_model": np.asarray(spec["q_model"]).tolist(),
        "q_half_in_grid_steps": np.asarray(spec["q_half_in_grid_steps"]).tolist(),
        "q_half_lands_on_grid": bool(spec["q_half_lands_on_grid"]),
        "phase_vertex": phase_vertex,
        "contact_closure_max_abs": contact,
        "bare_total_ward_max_abs": float(result["bare_total_ward_max_abs"]),
        "amplitude_phase_ward_max_abs": float(result["amplitude_phase_ward_max_abs"]),
        "left_ward_max_abs": float(np.max(np.abs(amplitude_left))),
        "right_ward_max_abs": float(np.max(np.abs(amplitude_right))),
        "left_ward_components": _ward_components(amplitude_left),
        "right_ward_components": _ward_components(amplitude_right),
        "collective_condition_number": float(result["collective_condition_number"]),
        "collective_inverse_method": result["collective_inverse_method"],
        "collective_total_det_abs": float(result["collective_total_det_abs"]),
        "sigma_diag_min_real": float(result["sigma_diag_min_real"]),
        "sigma_offdiag_rel": float(result["sigma_offdiag_rel"]),
        "sigma_xx_yy_anisotropy": float(result["sigma_xx_yy_anisotropy"]),
        "max_abs_sigma_tilde": float(result["max_abs_sigma_tilde"]),
        "bare_ward_monitor_only": True,
    }
    row["status"], row["dominant_failure"] = commensurate_case_status(
        pairing,
        contact,
        row["amplitude_phase_ward_max_abs"],
        row["bare_total_ward_max_abs"],
        row["collective_condition_number"],
        conductivity_finite=all(np.isfinite(float(value)) for value in conductivity_values),
    )
    return row


def build_payload(quick: bool = True) -> dict[str, Any]:
    n_list = N_LIST_QUICK if quick else N_LIST_FULL
    m_cases = M_CASES_QUICK if quick else M_CASES_FULL
    cfg = KuboConfig.from_kelvin(
        omega_eV=0.01,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    cases: list[dict[str, Any]] = []
    for pairing in ("onsite_s", "spm", "dwave"):
        for n_grid in n_list:
            points, weights = composite_uniform_quadrature(n_grid, [(0.0, 0.0)])
            for m_case in m_cases:
                spec = commensurate_q_spec(n_grid, m_case)
                q = np.asarray(spec["q_model"], dtype=float)
                for phase_vertex in PHASE_OPTIONS[pairing]:
                    result = compute_bdg_components_for_composite_grid(
                        pairing,
                        0.01,
                        q,
                        points,
                        weights,
                        cfg,
                        delta0_eV=0.04,
                        phase_vertex=phase_vertex,
                    )
                    cases.append(_case_row(pairing, spec, phase_vertex, result))

    formal_casimir_ran = False
    status = overall_stageSC_2bC_status(cases, formal_casimir_ran)
    dwave_decomposition = build_dwave_decomposition(cases)
    if status == "PASSED":
        interpretation = "All tested commensurate-q amplitude-phase responses pass."
        next_step = "Proceed to a commensurate-q response-table/interpolation audit; do not run formal Casimir yet."
    elif status == "PARTIAL_PASS_MATERIAL_DWAVE_BLOCKED":
        interpretation = "onsite_s and spm pass; momentum-dependent dwave collective form-factor closure remains blocked."
        next_step = "Derive/audit the gauge-covariant dwave collective vertex and counterterm."
    else:
        interpretation = "A validation pairing or commensurate contact closure failed."
        next_step = "Return to amplitude-phase Schur/counterterm assembly before downstream validation."
    by_pairing: dict[str, Any] = {}
    for pairing in ("onsite_s", "spm", "dwave"):
        selected = [case for case in cases if case["pairing"] == pairing]
        best = min(selected, key=lambda case: float(case["amplitude_phase_ward_max_abs"]))
        by_pairing[pairing] = {
            "max_amplitude_phase_ward_abs": max(
                float(case["amplitude_phase_ward_max_abs"]) for case in selected
            ),
            "max_contact_closure_abs": max(float(case["contact_closure_max_abs"]) for case in selected),
            "best_case": best,
            "all_passed": all(case["status"] == "PASSED" for case in selected),
        }
    summary = {
        "status": status,
        "formal_casimir_ran": formal_casimir_ran,
        "diagnostic_only": True,
        "bare_ward_monitor_only": True,
        "interpretation": interpretation,
        "next_step": next_step,
        **by_pairing,
    }
    return {
        **summary,
        "quick": bool(quick),
        "production_default_modified": False,
        "collective_mode": "amplitude_phase",
        "collective_counterterm": "goldstone_gap_equation",
        "summary": summary,
        "cases": cases,
        "dwave_decomposition": dwave_decomposition,
    }


def _q_label(case: dict[str, Any]) -> str:
    return f"({case['q_model'][0]:.7g},{case['q_model'][1]:.7g})"


def _write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_2bC_bdg_amplitude_phase_commensurate_q_audit"
    json_path = OUTPUT_DIR / f"{stem}.json"
    md_path = OUTPUT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {stem}",
        "",
        f"- overall status: {payload['status']}",
        f"- formal Casimir ran: {payload['formal_casimir_ran']}",
        "- bare_total_ward_max_abs is monitor-only for superconducting gauge-restored response.",
        f"- interpretation: {payload['interpretation']}",
        "",
        "## Commensurate-q AP Ward summary",
        "",
        "| pairing | N | q | phase vertex | contact closure | bare Ward monitor | AP Ward | status |",
        "| ------- | -: | - | ------------ | --------------: | ----------------: | ------: | ------ |",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['pairing']} | {case['N']} | {_q_label(case)} | {case['phase_vertex']} | "
            f"{case['contact_closure_max_abs']:.8g} | {case['bare_total_ward_max_abs']:.8g} | "
            f"{case['amplitude_phase_ward_max_abs']:.8g} | {case['status']} |"
        )
    lines.extend(
        [
            "",
            "## onsite_s/spm pass summary",
            "",
            "| pairing | best N | best q | best AP Ward | contact closure | status |",
            "| ------- | -----: | ------ | -----------: | --------------: | ------ |",
        ]
    )
    for pairing in ("onsite_s", "spm"):
        summary = payload[pairing]
        best = summary["best_case"]
        lines.append(
            f"| {pairing} | {best['N']} | {_q_label(best)} | {best['amplitude_phase_ward_max_abs']:.8g} | "
            f"{summary['max_contact_closure_abs']:.8g} | {'PASSED' if summary['all_passed'] else 'FAILED'} |"
        )
    lines.extend(
        [
            "",
            "## dwave form-factor comparison",
            "",
            "| N | q | phase vertex | AP Ward | contact closure | condition number | status |",
            "| -: | - | ------------ | ------: | --------------: | ---------------: | ------ |",
        ]
    )
    for case in payload["cases"]:
        if case["pairing"] != "dwave":
            continue
        lines.append(
            f"| {case['N']} | {_q_label(case)} | {case['phase_vertex']} | "
            f"{case['amplitude_phase_ward_max_abs']:.8g} | {case['contact_closure_max_abs']:.8g} | "
            f"{case['collective_condition_number']:.8g} | {case['status']} |"
        )
    lines.extend(
        [
            "",
            "## Overall conclusion",
            "",
            "| overall status | interpretation | next step |",
            "| -------------- | -------------- | --------- |",
            f"| {payload['status']} | {payload['interpretation']} | {payload['next_step']} |",
        ]
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
