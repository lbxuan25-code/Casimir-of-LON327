#!/usr/bin/env python3
"""Audit small-q BdG response against existing local BdG convention."""

from __future__ import annotations

import numpy as np

from bdg_finite_q_audit_common import local_bdg_response_as_pi, parser, response_case, status_from_failures, write_report


def main() -> None:
    args = parser(__doc__).parse_args()
    omega = 0.01
    local = local_bdg_response_as_pi("spm", omega, 0.04, args.quick)
    cases = []
    monitors: list[str] = []
    for qmag in (0.05, 0.02, 0.01, 0.005):
        q = np.array([qmag, 0.0])
        result = response_case("spm", omega, q, 0.04, args.quick, phase=True)
        scale = max(np.linalg.norm(local[1:3, 1:3]), 1e-300)
        bare_diff = float(np.linalg.norm(result.bare_total[1:3, 1:3] - local[1:3, 1:3]))
        phase_diff = float(np.linalg.norm(result.minus_schur[1:3, 1:3] - local[1:3, 1:3]))
        amp_diff = float(np.linalg.norm(result.amplitude_phase_schur[1:3, 1:3] - local[1:3, 1:3]))
        cases.append(
            {
                "q_model_magnitude": qmag,
                "local_comparison_abs": amp_diff,
                "local_comparison_relative": float(amp_diff / scale),
                "bare_relative": float(bare_diff / scale),
                "phase_only_relative": float(phase_diff / scale),
                "amplitude_phase_relative": float(amp_diff / scale),
                "phase_vertex": result.metadata["phase_vertex"],
                "phase_phase_direct_included": result.metadata["phase_phase_direct_included"],
                "phase_phase_direct_convention": result.metadata["phase_phase_direct_convention"],
                "sign_prefactor_diagnostic": "finite-q Pi spatial block compared to local Pi=-omega*sigma_like_response",
            }
        )
    if cases[-1]["local_comparison_abs"] > cases[0]["local_comparison_abs"]:
        monitors.append("smallest q is not closer to local BdG than largest q")
    if any(cases[index + 1]["local_comparison_abs"] > cases[index]["local_comparison_abs"] for index in range(len(cases) - 1)):
        monitors.append("q-scaling is not monotonic decreasing")
    write_report(
        "stageSC_4_bdg_q0_limit_audit",
        {
            "status": status_from_failures([], monitors),
            "quick": bool(args.quick),
            "summary": {
                "q_scaling_table": [
                    {
                        "q_model_magnitude": case["q_model_magnitude"],
                        "local_comparison_abs": case["local_comparison_abs"],
                        "local_comparison_relative": case["local_comparison_relative"],
                        "bare_relative": case["bare_relative"],
                        "phase_only_relative": case["phase_only_relative"],
                        "amplitude_phase_relative": case["amplitude_phase_relative"],
                    }
                    for case in cases
                ],
                "smallest_q_abs": cases[-1]["local_comparison_abs"],
                "largest_q_abs": cases[0]["local_comparison_abs"],
                "monotonic_decreasing": not monitors,
                "phase_vertex": cases[0]["phase_vertex"],
                "phase_phase_direct_included": cases[0]["phase_phase_direct_included"],
                "phase_phase_direct_convention": cases[0]["phase_phase_direct_convention"],
            },
            "monitors": monitors,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
