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
        restored = ward_norms(result.gauge_restored, omega, q)
        improvement = bare["max_norm"] / max(restored["max_norm"], 1e-300)
        if restored["max_norm"] >= 1e-5 or improvement <= 1.0:
            failures.append(f"{pairing} Ward not improved below monitor threshold")
        elif restored["max_norm"] >= 1e-6:
            monitors.append(f"{pairing} Ward residual monitor")
        cases.append(
            {
                "pairing": pairing,
                "omega_eV": omega,
                "q_model": q,
                "bare_total_ward": bare,
                "gauge_restored_ward": restored,
                "improvement_factor": float(improvement),
                "phase_correction_on_metadata": result.metadata,
                "phase_correction_off_ward": ward_norms(response_case(pairing, omega, q, 0.04, args.quick, phase=False).gauge_restored, omega, q),
            }
        )
    write_report(
        "stageSC_2_bdg_phase_gauge_restoration_audit",
        {
            "status": status_from_failures(failures, monitors),
            "quick": bool(args.quick),
            "failures": failures,
            "monitors": monitors,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
