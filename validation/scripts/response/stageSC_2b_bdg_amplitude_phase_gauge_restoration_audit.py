#!/usr/bin/env python3
"""Audit amplitude/phase 2x2 collective Schur gauge restoration."""

from __future__ import annotations

import numpy as np

from bdg_finite_q_audit_common import parser, response_case, status_from_failures, ward_norms, write_report


PAIRINGS = ("onsite_s", "spm", "dwave")
PHASE_VERTICES = ("midpoint", "symmetric_kpm")


def _extended_amp_phase_best(result, omega: float, q: np.ndarray, delta0: float) -> tuple[float, complex]:
    q_left = np.array([1j * omega, q[0], q[1]], dtype=complex)
    best_value = float("inf")
    best_c = 0.0 + 0.0j
    for c_eta2 in (2j * delta0, -2j * delta0):
        left = q_left @ result.bare_total + c_eta2 * result.collective_em_right[1]
        collective = q_left @ result.em_collective_left + c_eta2 * result.collective_total[1]
        value = float(max(np.linalg.norm(left), np.linalg.norm(collective)))
        if value < best_value:
            best_value = value
            best_c = c_eta2
    return best_value, best_c


def _case_status(ward: float, improvement: float, condition: float | None) -> str:
    if condition is None or not np.isfinite(condition) or condition > 1e12:
        return "FAILED"
    if ward < 1e-6 and improvement > 10.0:
        return "PASSED"
    return "FAILED"


def main() -> None:
    args = parser(__doc__).parse_args()
    q = np.array([0.01, 0.01])
    omega = 0.01
    delta0 = 0.04
    cases = []
    for pairing in PAIRINGS:
        for phase_vertex in PHASE_VERTICES:
            phase_only = response_case(
                pairing,
                omega,
                q,
                delta0,
                args.quick,
                phase=True,
                phase_vertex=phase_vertex,
                include_phase_phase_direct=True,
                phase_phase_direct_convention="plus",
                collective_mode="phase_only",
            )
            amp_phase = response_case(
                pairing,
                omega,
                q,
                delta0,
                args.quick,
                phase=True,
                phase_vertex=phase_vertex,
                include_phase_phase_direct=True,
                phase_phase_direct_convention="plus",
                collective_mode="amplitude_phase",
                collective_counterterm="goldstone_gap_equation",
            )
            bare = ward_norms(amp_phase.bare_total, omega, q)
            phase_only_minus = ward_norms(phase_only.minus_schur, omega, q)
            phase_only_plus = ward_norms(phase_only.plus_schur, omega, q)
            amp_ward = ward_norms(amp_phase.amplitude_phase_schur, omega, q)
            improvement = bare["max_norm"] / max(amp_ward["max_norm"], 1e-300)
            condition = amp_phase.metadata["collective_total_condition_number"]
            extended_best, best_c = _extended_amp_phase_best(amp_phase, omega, q, delta0)
            status = _case_status(amp_ward["max_norm"], improvement, condition)
            cases.append(
                {
                    "pairing": pairing,
                    "phase_vertex": phase_vertex,
                    "bare_Ward": bare["max_norm"],
                    "phase_only_best_Ward": min(phase_only_minus["max_norm"], phase_only_plus["max_norm"]),
                    "amplitude_phase_Ward": amp_ward["max_norm"],
                    "improvement_over_bare": improvement,
                    "collective_bubble": amp_phase.collective_bubble,
                    "collective_counterterm": amp_phase.collective_counterterm,
                    "collective_total": amp_phase.collective_total,
                    "collective_total_condition_number": condition,
                    "goldstone_counterterm_Cg": amp_phase.metadata["goldstone_counterterm_Cg"],
                    "C_eta2_candidates": [2j * delta0, -2j * delta0],
                    "best_C_eta2": best_c,
                    "extended_Ward_best": extended_best,
                    "status": status,
                    "metadata": amp_phase.metadata,
                }
            )
    best_by_pairing = {
        pairing: min((case for case in cases if case["pairing"] == pairing), key=lambda case: case["amplitude_phase_Ward"])
        for pairing in PAIRINGS
    }
    failures = []
    if best_by_pairing["onsite_s"]["status"] != "PASSED":
        failures.append("onsite_s amplitude-phase benchmark failed")
    for pairing in ("spm", "dwave"):
        if best_by_pairing[pairing]["status"] != "PASSED":
            failures.append(f"{pairing} material amplitude-phase Ward failed")
    write_report(
        "stageSC_2b_bdg_amplitude_phase_gauge_restoration_audit",
        {
            "status": status_from_failures(failures),
            "quick": bool(args.quick),
            "summary": {
                "onsite_s_passed": best_by_pairing["onsite_s"]["status"] == "PASSED",
                "material_pairings_passed": all(best_by_pairing[pairing]["status"] == "PASSED" for pairing in ("spm", "dwave")),
                "best_phase_vertex_by_pairing": {pairing: best_by_pairing[pairing]["phase_vertex"] for pairing in PAIRINGS},
                "bare_Ward_by_pairing": {pairing: best_by_pairing[pairing]["bare_Ward"] for pairing in PAIRINGS},
                "phase_only_best_Ward_by_pairing": {pairing: best_by_pairing[pairing]["phase_only_best_Ward"] for pairing in PAIRINGS},
                "amplitude_phase_Ward_by_pairing": {pairing: best_by_pairing[pairing]["amplitude_phase_Ward"] for pairing in PAIRINGS},
                "improvement_by_pairing": {pairing: best_by_pairing[pairing]["improvement_over_bare"] for pairing in PAIRINGS},
                "condition_number_by_pairing": {pairing: best_by_pairing[pairing]["collective_total_condition_number"] for pairing in PAIRINGS},
                "best_C_eta2_by_pairing": {pairing: best_by_pairing[pairing]["best_C_eta2"] for pairing in PAIRINGS},
            },
            "failures": failures,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
