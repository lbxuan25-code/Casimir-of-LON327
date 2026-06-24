"""Diagnostic-only status helpers for the StageSC-2bC commensurate-q audit."""

from __future__ import annotations

from typing import Any

import numpy as np


def commensurate_q_spec(n_grid: int, m_case: tuple[int, int]) -> dict[str, Any]:
    if n_grid <= 0:
        raise ValueError("n_grid must be positive")
    mx, my = (int(value) for value in m_case)
    spacing = 2.0 * np.pi / float(n_grid)
    q = np.array([2.0 * spacing * mx, 2.0 * spacing * my], dtype=float)
    half_steps = q / (2.0 * spacing)
    return {
        "N": int(n_grid),
        "m_case": [mx, my],
        "q_model": q,
        "grid_spacing": spacing,
        "q_half_in_grid_steps": half_steps,
        "q_half_lands_on_grid": bool(np.allclose(half_steps, np.rint(half_steps), atol=1e-12, rtol=0.0)),
    }


def commensurate_case_status(
    pairing: str,
    contact_closure_abs: float,
    amplitude_phase_ward_abs: float,
    bare_ward_monitor_abs: float,
    collective_condition_number: float,
    conductivity_finite: bool = True,
) -> tuple[str, str]:
    """Classify a case while deliberately excluding bare Ward from the gate."""

    del bare_ward_monitor_abs
    if contact_closure_abs >= 1e-10:
        return "FAILED", "contact_closure"
    if not np.isfinite(collective_condition_number) or collective_condition_number > 1e12:
        return "FAILED", "collective_singular"
    if not conductivity_finite:
        return "FAILED", "conductivity_sanity"
    if pairing in {"onsite_s", "spm"}:
        if amplitude_phase_ward_abs < 1e-8:
            return "PASSED", "none"
        if amplitude_phase_ward_abs < 1e-6:
            return "MONITOR", "amplitude_phase_ward"
        return "FAILED", "amplitude_phase_ward"
    if pairing == "dwave":
        if amplitude_phase_ward_abs < 1e-6:
            return "PASSED", "none"
        if amplitude_phase_ward_abs < 1e-4:
            return "MONITOR", "dwave_form_factor"
        return "FAILED", "dwave_form_factor"
    raise ValueError("unknown pairing")


def overall_stageSC_2bC_status(cases: list[dict[str, Any]], formal_casimir_ran: bool = False) -> str:
    if formal_casimir_ran:
        return "FAILED"
    if any(float(case["contact_closure_max_abs"]) >= 1e-10 for case in cases):
        return "FAILED"
    for pairing in ("onsite_s", "spm"):
        selected = [case for case in cases if case["pairing"] == pairing]
        if not selected or any(case["status"] != "PASSED" for case in selected):
            return "FAILED"
    dwave = [case for case in cases if case["pairing"] == "dwave"]
    if dwave and all(case["status"] == "PASSED" for case in dwave):
        return "PASSED"
    if dwave and all(case["dominant_failure"] in {"none", "amplitude_phase_ward", "dwave_form_factor"} for case in dwave):
        return "PARTIAL_PASS_MATERIAL_DWAVE_BLOCKED"
    return "FAILED"


def build_dwave_decomposition(cases: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    maxima: dict[str, float] = {}
    for phase_vertex in ("symmetric_kpm", "midpoint"):
        selected = [
            case for case in cases if case["pairing"] == "dwave" and case["phase_vertex"] == phase_vertex
        ]
        if not selected:
            continue
        worst = max(selected, key=lambda case: float(case["amplitude_phase_ward_max_abs"]))
        maximum = max(float(case["amplitude_phase_ward_max_abs"]) for case in selected)
        maxima[phase_vertex] = maximum
        output[phase_vertex] = {
            "amplitude_phase_ward_max_abs": maximum,
            "worst_N": int(worst["N"]),
            "worst_q_model": list(worst["q_model"]),
            "left_ward_components": worst["left_ward_components"],
            "right_ward_components": worst["right_ward_components"],
            "collective_condition_number": max(
                float(case["collective_condition_number"]) for case in selected
            ),
            "contact_closure_max_abs": max(float(case["contact_closure_max_abs"]) for case in selected),
        }
    if set(maxima) == {"symmetric_kpm", "midpoint"}:
        output["best_phase_vertex"] = min(maxima, key=maxima.get)  # type: ignore[arg-type]
        output["residual_ratio_midpoint_over_symmetric"] = float(
            maxima["midpoint"] / max(maxima["symmetric_kpm"], 1e-300)
        )
    return output

