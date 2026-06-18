#!/usr/bin/env python3
"""Audit Delta->0 against the existing normal finite-q response backend."""

from __future__ import annotations

import numpy as np

from bdg_finite_q_audit_common import normal_response, parser, response_case, status_from_failures, write_report


def main() -> None:
    args = parser(__doc__).parse_args()
    q = np.array([0.01, 0.0])
    omega = 0.01
    normal = normal_response(omega, q, args.quick)
    cases = []
    failures: list[str] = []
    max_diffs_by_pairing = {}
    for pairing in ("onsite_s", "spm", "dwave"):
        max_diffs = []
        for delta0 in (0.04, 0.01, 0.003, 0.001, 0.0):
            result = response_case(pairing, omega, q, delta0, args.quick, phase=True)
            diff = result.gauge_restored - normal
            max_abs = float(np.max(np.abs(diff)))
            rel = float(max_abs / max(np.max(np.abs(normal)), 1e-300))
            max_diffs.append((delta0, max_abs))
            if delta0 == 0.0 and max_abs >= 1e-8:
                failures.append(f"{pairing} true BdG delta0=0 did not match existing normal-state backend")
            cases.append(
                {
                    "pairing": pairing,
                    "delta0_eV": delta0,
                    "absolute_difference": float(np.linalg.norm(diff)),
                    "relative_difference": rel,
                    "max_component_difference": max_abs,
                    "used_normal_backend_shortcut": False,
                    "metadata": result.metadata,
                }
            )
        max_diffs_by_pairing[pairing] = max_diffs
    shortcut = response_case(
        "spm",
        omega,
        q,
        0.0,
        args.quick,
        phase=True,
        use_normal_backend_in_delta0_limit=True,
    )
    shortcut_diff = float(np.max(np.abs(shortcut.gauge_restored - normal)))
    zero_cases = {case["pairing"]: case for case in cases if case["delta0_eV"] == 0.0}
    write_report(
        "stageSC_3_bdg_normal_limit_audit",
        {
            "status": status_from_failures(failures),
            "quick": bool(args.quick),
            "summary": {
                "delta0_eV_list": [0.04, 0.01, 0.003, 0.001, 0.0],
                "true_BdG_delta0_0_abs_diff_to_normal": {
                    pairing: zero_cases[pairing]["max_component_difference"] for pairing in zero_cases
                },
                "true_BdG_delta0_0_rel_diff_to_normal": {
                    pairing: zero_cases[pairing]["relative_difference"] for pairing in zero_cases
                },
                "small_delta_trend": {
                    pairing: [
                        {"delta0_eV": delta0, "max_component_difference": diff}
                        for delta0, diff in max_diffs
                        if delta0 > 0.0
                    ]
                    for pairing, max_diffs in max_diffs_by_pairing.items()
                },
                "normal_backend_reference_used_only_for_comparison": True,
                "shortcut_reference_diff_not_used_for_pass": shortcut_diff,
            },
            "failures": failures,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
