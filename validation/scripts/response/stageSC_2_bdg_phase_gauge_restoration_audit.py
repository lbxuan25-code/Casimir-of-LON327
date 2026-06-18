#!/usr/bin/env python3
"""Audit global phase Schur restoration over phase-kernel conventions."""

from __future__ import annotations

import numpy as np

from bdg_finite_q_audit_common import parser, response_case, status_from_failures, ward_norms, write_report


PAIRINGS = ("onsite_s", "spm", "dwave")
PHASE_VERTICES = ("midpoint", "symmetric_kpm")
DIRECT_SWITCHES = (False, True)
DIRECT_CONVENTIONS = ("plus", "minus")


def _case_status(selected_ward: float, improvement: float, *, direct_included: bool) -> str:
    if not direct_included:
        return "FAILED"
    if selected_ward < 1e-6 and improvement > 10.0:
        return "PASSED"
    if selected_ward < 1e-5 and improvement > 2.0:
        return "MONITOR"
    return "FAILED"


def main() -> None:
    args = parser(__doc__).parse_args()
    q = np.array([0.01, 0.01])
    omega = 0.01
    cases = []
    for pairing in PAIRINGS:
        for phase_vertex in PHASE_VERTICES:
            for direct_included in DIRECT_SWITCHES:
                conventions = DIRECT_CONVENTIONS if direct_included else ("plus",)
                for direct_convention in conventions:
                    result = response_case(
                        pairing,
                        omega,
                        q,
                        0.04,
                        args.quick,
                        phase=True,
                        phase_vertex=phase_vertex,
                        include_phase_phase_direct=direct_included,
                        phase_phase_direct_convention=direct_convention,
                        collective_mode="phase_only",
                    )
                    bare = ward_norms(result.bare_total, omega, q)
                    minus = ward_norms(result.minus_schur, omega, q)
                    plus = ward_norms(result.plus_schur, omega, q)
                    selected = ward_norms(result.gauge_restored, omega, q)
                    improvement = bare["max_norm"] / max(selected["max_norm"], 1e-300)
                    case_status = _case_status(selected["max_norm"], improvement, direct_included=direct_included)
                    cases.append(
                        {
                            "pairing": pairing,
                            "omega_eV": omega,
                            "q_model": q,
                            "phase_vertex": phase_vertex,
                            "phase_phase_direct_included": direct_included,
                            "phase_phase_direct_convention": direct_convention,
                            "status": case_status,
                            "bare_total_ward": bare,
                            "minus_schur_ward": minus,
                            "plus_schur_ward": plus,
                            "gauge_restored_ward": selected,
                            "max_bare_Ward": bare["max_norm"],
                            "max_minus_schur_Ward": minus["max_norm"],
                            "max_plus_schur_Ward": plus["max_norm"],
                            "selected_gauge_restored_Ward": selected["max_norm"],
                            "best_Ward_this_case": min(minus["max_norm"], plus["max_norm"]),
                            "best_schur_sign_this_case": "minus" if minus["max_norm"] <= plus["max_norm"] else "plus",
                            "improvement_factor": float(improvement),
                            "phase_phase_bubble": result.phase_phase_bubble,
                            "phase_phase_direct": result.phase_phase_direct,
                            "phase_phase_total": result.phase_phase_total,
                            "phase_phase_bubble_abs": float(abs(result.phase_phase_bubble)),
                            "phase_phase_direct_abs": float(abs(result.phase_phase_direct)),
                            "phase_phase_total_abs": float(abs(result.phase_phase_total)),
                            "phase_phase_abs": result.metadata["phase_phase_abs"],
                            "phase_correction_status": result.metadata["phase_correction_status"],
                            "finite_q_current_vertex_status": result.metadata["finite_q_current_vertex_status"],
                            "phase_kernel_status": result.metadata["phase_kernel_status"],
                            "metadata": result.metadata,
                        }
                    )

    best_by_pairing = {}
    for pairing in PAIRINGS:
        pairing_cases = [case for case in cases if case["pairing"] == pairing and case["phase_phase_direct_included"]]
        best = min(pairing_cases, key=lambda case: case["best_Ward_this_case"])
        best_by_pairing[pairing] = best

    material_failures = [
        f"{pairing} material Ward failed"
        for pairing in ("spm", "dwave")
        if best_by_pairing[pairing]["status"] != "PASSED"
    ]
    onsite_failures = [] if best_by_pairing["onsite_s"]["status"] == "PASSED" else ["onsite_s toy Ward did not close"]
    failures = onsite_failures + material_failures
    monitors = [
        f"{pairing} material Ward monitor"
        for pairing in ("spm", "dwave")
        if best_by_pairing[pairing]["status"] == "MONITOR"
    ]
    selected = {
        "phase_vertex": "symmetric_kpm",
        "phase_phase_direct_included": True,
        "phase_phase_direct_convention": "plus",
        "schur_sign": "minus",
        "reason": "default derived convention; pass/fail judged separately",
    }
    write_report(
        "stageSC_2_bdg_phase_gauge_restoration_audit",
        {
            "status": status_from_failures(failures, monitors),
            "quick": bool(args.quick),
            "summary": {
                "best_onsite_s_Ward": best_by_pairing["onsite_s"]["best_Ward_this_case"],
                "best_spm_Ward": best_by_pairing["spm"]["best_Ward_this_case"],
                "best_dwave_Ward": best_by_pairing["dwave"]["best_Ward_this_case"],
                "best_phase_vertex_by_pairing": {
                    pairing: best_by_pairing[pairing]["phase_vertex"] for pairing in PAIRINGS
                },
                "best_phase_phase_direct_convention_by_pairing": {
                    pairing: best_by_pairing[pairing]["phase_phase_direct_convention"] for pairing in PAIRINGS
                },
                "best_schur_sign_by_pairing": {
                    pairing: best_by_pairing[pairing]["best_schur_sign_this_case"] for pairing in PAIRINGS
                },
                "selected_convention": selected,
                "phase_only_is_diagnostic_not_validation_final": True,
                "selected_gauge_restored_Ward": [
                    case["selected_gauge_restored_Ward"]
                    for case in cases
                    if case["phase_vertex"] == selected["phase_vertex"]
                    and case["phase_phase_direct_included"] == selected["phase_phase_direct_included"]
                    and case["phase_phase_direct_convention"] == selected["phase_phase_direct_convention"]
                ],
                "finite_q_current_vertex_status": sorted({case["finite_q_current_vertex_status"] for case in cases}),
            },
            "failures": failures,
            "monitors": monitors,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
