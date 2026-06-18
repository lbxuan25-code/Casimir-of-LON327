#!/usr/bin/env python3
"""Audit the extended electromagnetic-plus-phase Ward identities."""

from __future__ import annotations

import numpy as np

from bdg_finite_q_audit_common import parser, response_case, status_from_failures, ward_norms, write_report


PAIRINGS = ("onsite_s", "spm", "dwave")
PHASE_VERTICES = ("midpoint", "symmetric_kpm")
DIRECT_CONVENTIONS = ("plus", "minus")
C_THETA_CANDIDATES = (1j, -1j, 2j, -2j)


def _extended_residuals(result, omega: float, q: np.ndarray, c_theta: complex) -> tuple[float, float]:
    q_left = np.array([1j * omega, q[0], q[1]], dtype=complex)
    left = q_left @ result.bare_total + c_theta * result.phase_coupling_right
    theta = q_left @ result.phase_coupling_left + c_theta * result.phase_phase_total
    return float(np.linalg.norm(left)), float(abs(theta))


def main() -> None:
    args = parser(__doc__).parse_args()
    q = np.array([0.01, 0.01])
    omega = 0.01
    cases = []
    for pairing in PAIRINGS:
        for phase_vertex in PHASE_VERTICES:
            for direct_convention in DIRECT_CONVENTIONS:
                result = response_case(
                    pairing,
                    omega,
                    q,
                    0.04,
                    args.quick,
                    phase=True,
                    phase_vertex=phase_vertex,
                    include_phase_phase_direct=True,
                    phase_phase_direct_convention=direct_convention,
                )
                bare = ward_norms(result.bare_total, omega, q)
                schur = ward_norms(result.gauge_restored, omega, q)
                for c_theta in C_THETA_CANDIDATES:
                    left_residual, theta_residual = _extended_residuals(result, omega, q, c_theta)
                    extended_max = max(left_residual, theta_residual)
                    cases.append(
                        {
                            "pairing": pairing,
                            "phase_vertex": phase_vertex,
                            "phase_phase_direct_convention": direct_convention,
                            "C_theta": c_theta,
                            "extended_left_residual_max": left_residual,
                            "extended_theta_residual_max": theta_residual,
                            "extended_residual_max": extended_max,
                            "schur_selected_ward": schur["max_norm"],
                            "bare_ward": bare["max_norm"],
                            "improvement_over_bare": bare["max_norm"] / max(extended_max, 1e-300),
                            "phase_phase_bubble": result.phase_phase_bubble,
                            "phase_phase_direct": result.phase_phase_direct,
                            "phase_phase_total": result.phase_phase_total,
                            "phase_phase_bubble_abs": float(abs(result.phase_phase_bubble)),
                            "phase_phase_direct_abs": float(abs(result.phase_phase_direct)),
                            "phase_phase_total_abs": float(abs(result.phase_phase_total)),
                            "metadata": result.metadata,
                        }
                    )

    best_by_pairing = {}
    for pairing in PAIRINGS:
        best_by_pairing[pairing] = min(
            (case for case in cases if case["pairing"] == pairing),
            key=lambda case: case["extended_residual_max"],
        )

    failures = []
    for pairing, best in best_by_pairing.items():
        if not (best["extended_residual_max"] < best["bare_ward"] / 10.0):
            failures.append(f"{pairing} extended Ward residual not clearly below bare Ward")
    material_failures = [
        f"{pairing} material extended Ward failed"
        for pairing in ("spm", "dwave")
        if best_by_pairing[pairing]["extended_residual_max"] >= 1e-6
    ]
    failures.extend(material_failures)
    write_report(
        "stageSC_2a_bdg_extended_ward_identity_audit",
        {
            "status": status_from_failures(failures),
            "quick": bool(args.quick),
            "summary": {
                "best_C_theta_by_pairing": {
                    pairing: best_by_pairing[pairing]["C_theta"] for pairing in PAIRINGS
                },
                "best_extended_residual_by_pairing": {
                    pairing: best_by_pairing[pairing]["extended_residual_max"] for pairing in PAIRINGS
                },
                "best_phase_vertex_by_pairing": {
                    pairing: best_by_pairing[pairing]["phase_vertex"] for pairing in PAIRINGS
                },
                "best_phase_phase_direct_convention_by_pairing": {
                    pairing: best_by_pairing[pairing]["phase_phase_direct_convention"] for pairing in PAIRINGS
                },
                "best_schur_selected_ward_by_pairing": {
                    pairing: best_by_pairing[pairing]["schur_selected_ward"] for pairing in PAIRINGS
                },
                "onsite_s_passed": best_by_pairing["onsite_s"]["extended_residual_max"] < 1e-6,
                "material_pairings_passed": all(
                    best_by_pairing[pairing]["extended_residual_max"] < 1e-6 for pairing in ("spm", "dwave")
                ),
            },
            "failures": failures,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
