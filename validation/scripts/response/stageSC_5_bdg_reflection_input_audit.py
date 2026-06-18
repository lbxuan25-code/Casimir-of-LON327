#!/usr/bin/env python3
"""Audit BdG finite-q response as reflection input, without Casimir energy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bdg_finite_q_audit_common import OUTPUT_DIR, parser, reflection_package, response_case, status_from_failures, write_report


def _prior_stages_allow() -> bool:
    required = [
        "stageSC_1_bdg_finite_q_bare_kernel_audit.json",
        "stageSC_2_bdg_phase_gauge_restoration_audit.json",
        "stageSC_3_bdg_normal_limit_audit.json",
        "stageSC_4_bdg_q0_limit_audit.json",
    ]
    for name in required:
        path = OUTPUT_DIR / name
        if not path.exists():
            return False
        status = json.loads(path.read_text(encoding="utf-8")).get("status")
        if status not in {"PASSED", "MONITOR"}:
            return False
    return True


def main() -> None:
    args = parser(__doc__).parse_args()
    failures: list[str] = []
    if not _prior_stages_allow():
        failures.append("prior StageSC reports are missing or failed")
    q = np.array([0.01, 0.01])
    omega = 0.01
    result = response_case("spm", omega, q, 0.04, args.quick, phase=True)
    package = reflection_package(result.gauge_restored, omega, q)
    sigma = package["sigma_model"]
    refl = package["reflection_TE_TM"]
    if not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(refl)):
        failures.append("nonfinite sigma/reflection")
    q_flip = response_case("spm", omega, -q, 0.04, args.quick, phase=True)
    q_sign_diff = float(np.max(np.abs(result.gauge_restored - q_flip.gauge_restored)))
    cases = [
        {
            "omega_eV": omega,
            "q_model": q,
            "sigma_diagonal_real": [float(sigma[0, 0].real), float(sigma[1, 1].real)],
            "sigma_diagonal_positive_sanity": bool(sigma[0, 0].real >= -1e-10 and sigma[1, 1].real >= -1e-10),
            "offdiag": package["offdiag"],
            "q_sign_check_max_abs": q_sign_diff,
            "reflection_finite": bool(np.all(np.isfinite(refl))),
            "max_abs_R": float(np.max(np.abs(refl))),
            "no_casimir_energy_force_torque": True,
        }
    ]
    write_report(
        "stageSC_5_bdg_reflection_input_audit",
        {"status": status_from_failures(failures), "quick": bool(args.quick), "failures": failures, "cases": cases},
    )


if __name__ == "__main__":
    main()
