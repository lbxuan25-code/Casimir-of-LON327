#!/usr/bin/env python3
"""Audit bare finite-q BdG bubble and contact structure."""

from __future__ import annotations

import numpy as np

from bdg_finite_q_audit_common import matrix_diagnostics, parser, response_case, status_from_failures, write_report


def main() -> None:
    args = parser(__doc__).parse_args()
    pairings = ("spm", "dwave")
    omegas = (0.005, 0.01)
    qs = (np.array([0.01, 0.0]), np.array([0.01, 0.01]))
    cases = []
    failures: list[str] = []
    for pairing in pairings:
        for omega in omegas:
            for q in qs:
                result = response_case(pairing, omega, q, 0.04, args.quick, phase=False)
                diag = {
                    "bare_bubble": matrix_diagnostics(result.bare_bubble),
                    "direct": matrix_diagnostics(result.direct),
                    "bare_total": matrix_diagnostics(result.bare_total),
                }
                if not all(item["all_finite"] for item in diag.values()):
                    failures.append(f"{pairing} omega={omega} q={q.tolist()} nonfinite")
                cases.append(
                    {
                        "pairing": pairing,
                        "omega_eV": omega,
                        "q_model": q,
                        "bare_bubble": result.bare_bubble,
                        "direct": result.direct,
                        "bare_total": result.bare_total,
                        "diagnostics": diag,
                        "metadata": result.metadata,
                    }
                )
    write_report(
        "stageSC_1_bdg_finite_q_bare_kernel_audit",
        {
            "status": status_from_failures(failures),
            "quick": bool(args.quick),
            "failures": failures,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
