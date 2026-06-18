#!/usr/bin/env python3
"""Audit global phase correction against Ward residuals."""

from __future__ import annotations

import numpy as np

from bdg_finite_q_audit_common import parser, response_case, status_from_failures, ward_norms, write_report


def main() -> None:
    args = parser(__doc__).parse_args()
    cases = []
    failures: list[str] = []
    monitors: list[str] = []
    for pairing in ("spm", "dwave"):
        q = np.array([0.01, 0.01])
        omega = 0.01
        result = response_case(pairing, omega, q, 0.04, args.quick, phase=True)
        bare = ward_norms(result.bare_total, omega, q)
        minus = result.metadata["ward_residual_minus_schur"]
        plus = result.metadata["ward_residual_plus_schur"]
        restored = ward_norms(result.gauge_restored, omega, q)
        improvement = bare["max_norm"] / max(restored["max_norm"], 1e-300)
        if restored["max_norm"] < 1e-6 and improvement > 10.0:
            case_status = "PASSED"
        elif restored["max_norm"] < 1e-5 and improvement > 2.0:
            case_status = "MONITOR"
            monitors.append(f"{pairing} Ward monitor")
        else:
            case_status = "FAILED"
            failures.append(f"{pairing} Ward failed selected Schur-minus criterion")
        cases.append(
            {
                "pairing": pairing,
                "omega_eV": omega,
                "q_model": q,
                "status": case_status,
                "bare_total_ward": bare,
                "minus_schur_ward": minus,
                "plus_schur_ward": plus,
                "gauge_restored_ward": restored,
                "max_bare_Ward": bare["max_norm"],
                "max_minus_schur_Ward": minus["max_norm"],
                "max_plus_schur_Ward": plus["max_norm"],
                "selected_gauge_restored_Ward": restored["max_norm"],
                "improvement_factor": float(improvement),
                "phase_phase_abs": result.metadata["phase_phase_abs"],
                "phase_correction_status": result.metadata["phase_correction_status"],
                "finite_q_current_vertex_status": result.metadata["finite_q_current_vertex_status"],
                "phase_correction_on_metadata": result.metadata,
                "phase_correction_off_ward": ward_norms(response_case(pairing, omega, q, 0.04, args.quick, phase=False).gauge_restored, omega, q),
            }
        )
    selected = [case["selected_gauge_restored_Ward"] for case in cases]
    write_report(
        "stageSC_2_bdg_phase_gauge_restoration_audit",
        {
            "status": status_from_failures(failures, monitors),
            "quick": bool(args.quick),
            "summary": {
                "max_bare_Ward": max(case["max_bare_Ward"] for case in cases),
                "max_minus_schur_Ward": max(case["max_minus_schur_Ward"] for case in cases),
                "max_plus_schur_Ward": max(case["max_plus_schur_Ward"] for case in cases),
                "selected_gauge_restored_Ward": max(selected),
                "min_improvement_factor": min(case["improvement_factor"] for case in cases),
                "phase_correction_status": sorted({case["phase_correction_status"] for case in cases}),
                "finite_q_current_vertex_status": sorted({case["finite_q_current_vertex_status"] for case in cases}),
            },
            "failures": failures,
            "monitors": monitors,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
