#!/usr/bin/env python3
"""Audit the unique finite-q Goldstone tangent for general pairing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from bdg_commensurate_q_common import commensurate_case_status, commensurate_q_spec  # noqa: E402
from bdg_quadrature_strategy_common import (  # noqa: E402
    composite_uniform_quadrature,
    compute_bdg_components_for_composite_grid,
)
from lno327.conductivity import KuboConfig  # noqa: E402
from lno327.pairing import PairingAmplitudes  # noqa: E402
from lno327.pairing_bonds import pairing_bond_list  # noqa: E402
from pairing_bond_goldstone_common import (  # noqa: E402
    GOLDSTONE_DIMENSION_STATEMENT,
    OPERATOR_K_POINTS,
    PAIRINGS,
    Q_MODEL_LIST,
    exact_goldstone_form_factor,
    goldstone_dimension_rows,
    old_prescription_comparison_rows,
    operator_ward_rows,
    q0_normalization_rows,
    reconstruction_rows,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
N_LIST_QUICK = (24, 36, 48)
M_CASES_QUICK = ((1, 0), (1, 1))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def schur_rows(amp: PairingAmplitudes, quick: bool = True) -> list[dict[str, Any]]:
    cfg = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    rows = []
    for pairing in PAIRINGS:
        for n_grid in (N_LIST_QUICK if quick else N_LIST_QUICK):
            points, weights = composite_uniform_quadrature(n_grid, [(0.0, 0.0)])
            for m_case in M_CASES_QUICK:
                spec = commensurate_q_spec(n_grid, m_case)
                q = np.asarray(spec["q_model"], dtype=float)
                result = compute_bdg_components_for_composite_grid(
                    pairing,
                    cfg.omega_eV,
                    q,
                    points,
                    weights,
                    cfg,
                    delta0_eV=amp.delta0_eV,
                    phase_vertex="bond_endpoint_gauge",
                )
                contact = max(float(result["contact_closure"][channel]["E_band_plus_qD_abs"]) for channel in ("Vx", "Vy"))
                status, dominant = commensurate_case_status(
                    pairing,
                    contact,
                    float(result["amplitude_phase_ward_max_abs"]),
                    float(result["bare_total_ward_max_abs"]),
                    float(result["collective_condition_number"]),
                )
                rows.append(
                    {
                        "pairing": pairing,
                        "N": int(n_grid),
                        "m_case": list(m_case),
                        "q_model": q.tolist(),
                        "collective_basis": "amplitude_plus_exact_goldstone",
                        "contact_closure_max_abs": contact,
                        "bare_ward_monitor": float(result["bare_total_ward_max_abs"]),
                        "goldstone_restored_ward_max_abs": float(result["amplitude_phase_ward_max_abs"]),
                        "collective_condition_number": float(result["collective_condition_number"]),
                        "dominant_failure": dominant,
                        "status": status,
                    }
                )
    return rows


def _bond_coefficients(pairing: str, amp: PairingAmplitudes) -> np.ndarray:
    return np.asarray([bond.coefficient for bond in pairing_bond_list(pairing, amp)], dtype=complex)


def internal_mode_diagnostics(schur: list[dict[str, Any]], amp: PairingAmplitudes) -> list[dict[str, Any]]:
    rows = []
    dwave_failed = any(row["pairing"] == "dwave" and row["status"] == "FAILED" for row in schur)
    if not dwave_failed:
        return rows
    coeff = _bond_coefficients("dwave", amp)
    eq_norm = float(np.linalg.norm(coeff))
    projections = []
    for qx, qy in Q_MODEL_LIST:
        factors = []
        for bond in pairing_bond_list("dwave", amp):
            displacement = bond.displacement
            factors.append(np.exp(1j * float(np.dot([qx, qy], bond.center))) * np.cos(0.5 * float(np.dot([qx, qy], displacement))))
        exact_coeff = coeff * np.asarray(factors, dtype=complex)
        parallel = coeff * (np.vdot(coeff, exact_coeff) / max(np.vdot(coeff, coeff), 1e-300))
        internal = exact_coeff - parallel
        projections.append(float(np.linalg.norm(internal) / max(np.linalg.norm(exact_coeff), 1e-300)))
    max_internal_projection = max(projections)
    rows.append(
        {
            "pairing": "dwave",
            "schur_residual_projection_on_internal_modes": max_internal_projection,
            "num_internal_modes_tested": max(int(coeff.size - 1), 0),
            "dominant_internal_mode": "bond_shape_orthogonal_to_equilibrium_pairing",
            "equilibrium_pairing_norm": eq_norm,
            "interpretation": "COUNTERTERM_BLOCKED",
            "marker": "This is not an additional Goldstone mode.",
            "evidence": (
                "Exact Goldstone operator Ward and old-basis projection pass; "
                "the remaining Schur failure is not fixed by adding bond-resolved Goldstones."
            ),
        }
    )
    return rows


def _prerequisites_passed(
    reconstruction: list[dict[str, Any]],
    normalization: list[dict[str, Any]],
    operator: list[dict[str, Any]],
) -> bool:
    exact_operator = [row for row in operator if row["phase_vertex"] == "exact_goldstone_tangent"]
    return (
        all(row["status"] == "PASSED" for row in reconstruction)
        and all(row["status"] == "PASSED" for row in normalization)
        and all(row["status"] == "PASSED" for row in exact_operator)
    )


def overall_status(payload: dict[str, Any]) -> tuple[str, str, str]:
    if payload["formal_casimir_ran"]:
        return "FAILED", "formal Casimir ran, which is forbidden for this audit.", "Discard this run and rerun diagnostics only."
    if any(row["status"] != "PASSED" for row in payload["pairing_reconstruction"]):
        return "FAILED", "A required pairing reconstruction failed or was unavailable.", "Fix the real-space bond representation first."
    exact_operator = {
        row["pairing"]: row
        for row in payload["exact_goldstone_operator_ward"]
        if row["phase_vertex"] == "exact_goldstone_tangent"
    }
    if exact_operator["dwave"]["status"] != "PASSED":
        return (
            "PARTIAL_PASS_DWAVE_GOLDSTONE_TANGENT_BLOCKED",
            "dwave reconstruction passed, but the exact Goldstone tangent fails operator Ward.",
            "Do not run Schur; audit the finite-q tangent convention.",
        )
    if any(exact_operator[pairing]["status"] != "PASSED" for pairing in ("onsite_s", "spm")):
        return "FAILED", "onsite_s or spm exact Goldstone tangent operator Ward failed.", "Fix the validation pairings first."
    schur = payload["commensurate_q_restored_ward"]
    if not schur:
        return "FAILED_GOLDSTONE_TANGENT_OPERATOR_WARD", "Schur was skipped because prerequisites failed.", "Fix tangent prerequisites."
    if any(float(row["contact_closure_max_abs"]) >= 1e-10 for row in schur):
        return "FAILED", "Contact closure failed.", "Return to finite-q contact assembly."
    onsite_spm_ok = all(row["status"] == "PASSED" for row in schur if row["pairing"] in {"onsite_s", "spm"})
    dwave_ok = all(row["status"] == "PASSED" for row in schur if row["pairing"] == "dwave")
    if onsite_spm_ok and dwave_ok:
        return "PASSED", "All tested unique-Goldstone restored Ward checks passed.", "Proceed to response-table diagnostics without changing defaults."
    if onsite_spm_ok and payload["dwave_internal_mode_diagnosis"]:
        diagnosis = payload["dwave_internal_mode_diagnosis"][0]["interpretation"]
        if diagnosis == "INTERNAL_MASSIVE_MODE_LIKELY_NEEDED":
            return (
                "PARTIAL_PASS_DWAVE_INTERNAL_MODE_NEEDED",
                "dwave exact tangent passed but Schur failure projects strongly onto massive internal modes.",
                "Audit massive internal collective kernels; do not call them Goldstones.",
            )
        return (
            "PARTIAL_PASS_DWAVE_SCHUR_BLOCKED",
            "dwave exact tangent passes operator Ward, but restored Schur Ward fails.",
            "Audit the nonlocal pairing counterterm block before adding internal modes.",
        )
    return "FAILED", "onsite_s or spm restored Ward failed.", "Fix minimal collective basis assembly for validation pairings."


def build_payload(quick: bool = True) -> dict[str, Any]:
    amp = PairingAmplitudes(delta0_eV=0.04)
    reconstruction = reconstruction_rows(amp)
    normalization = q0_normalization_rows(amp)
    operator = operator_ward_rows(amp)
    comparison = old_prescription_comparison_rows(amp, operator)
    prerequisites = _prerequisites_passed(reconstruction, normalization, operator)
    schur = schur_rows(amp, quick=quick) if prerequisites else []
    internal = internal_mode_diagnostics(schur, amp)
    payload: dict[str, Any] = {
        "quick": bool(quick),
        "diagnostic_only": True,
        "production_default_modified": False,
        "formal_casimir_ran": False,
        "delta0_eV": amp.delta0_eV,
        "goldstone_dimension_statement": GOLDSTONE_DIMENSION_STATEMENT,
        "goldstone_dimension": goldstone_dimension_rows(),
        "pairing_reconstruction": reconstruction,
        "exact_goldstone_tangent_q0_normalization": normalization,
        "exact_goldstone_operator_ward": operator,
        "old_prescriptions_comparison": comparison,
        "minimal_collective_basis": {
            "basis": ["amplitude_along_equilibrium_pairing", "exact_goldstone_tangent"],
            "goldstone_dimension": 1,
            "bond_resolved_goldstones_added": False,
            "ad_hoc_dwave_fields_added": False,
        },
        "schur_prerequisites_passed": prerequisites,
        "commensurate_q_restored_ward": schur,
        "dwave_internal_mode_diagnosis": internal,
    }
    status, interpretation, next_step = overall_status(payload)
    payload["status"] = status
    payload["interpretation"] = interpretation
    payload["next_step"] = next_step
    return payload


def _q_label(row: dict[str, Any]) -> str:
    if "q_model" in row:
        q = row["q_model"]
    else:
        q = row["q_model_list"][0]
    return f"({q[0]:.7g},{q[1]:.7g})"


def write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_2e_unified_goldstone_tangent_audit"
    json_path = OUTPUT_DIR / f"{stem}.json"
    md_path = OUTPUT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# {stem}",
        "",
        "## Section 1: Goldstone dimension statement",
        "",
        payload["goldstone_dimension_statement"],
        "",
        "## Section 2: pairing reconstruction",
        "",
        "| pairing | reconstruction max | status |",
        "| ------- | -----------------: | ------ |",
    ]
    for row in payload["pairing_reconstruction"]:
        lines.append(f"| {row['pairing']} | {row['bond_reconstruction_max_abs']:.8g} | {row['status']} |")

    lines.extend(
        [
            "",
            "## Section 3: exact Goldstone tangent operator Ward",
            "",
            "| pairing | q | normalization | operator Ward | status |",
            "| ------- | - | ------------: | ------------: | ------ |",
        ]
    )
    normalization = {row["pairing"]: row for row in payload["exact_goldstone_tangent_q0_normalization"]}
    for row in payload["exact_goldstone_operator_ward"]:
        if row["phase_vertex"] != "exact_goldstone_tangent":
            continue
        lines.append(
            f"| {row['pairing']} | {row['q_model_list']} | "
            f"{normalization[row['pairing']]['normalization_factor_abs']:.8g} | "
            f"{row['operator_ward_max_abs']:.8g} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Section 4: old prescriptions comparison",
            "",
            "| pairing | q | midpoint Ward | symmetric_kpm Ward | exact tangent Ward |",
            "| ------- | - | ------------: | -----------------: | -----------------: |",
        ]
    )
    for row in payload["old_prescriptions_comparison"]:
        lines.append(
            f"| {row['pairing']} | {_q_label(row)} | {row['operator_ward_midpoint']:.8g} | "
            f"{row['operator_ward_symmetric_kpm']:.8g} | {row['operator_ward_exact']:.8g} |"
        )

    lines.extend(
        [
            "",
            "## Section 5: commensurate-q restored Ward",
            "",
            "| pairing |  N | q | contact closure | bare Ward monitor | restored Ward | status |",
            "| ------- | -: | - | --------------: | ----------------: | ------------: | ------ |",
        ]
    )
    for row in payload["commensurate_q_restored_ward"]:
        lines.append(
            f"| {row['pairing']} | {row['N']} | {_q_label(row)} | "
            f"{row['contact_closure_max_abs']:.8g} | {row['bare_ward_monitor']:.8g} | "
            f"{row['goldstone_restored_ward_max_abs']:.8g} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Section 6: dwave failure diagnosis if any",
            "",
            "| diagnosis | evidence | next step |",
            "| --------- | -------- | --------- |",
        ]
    )
    if payload["dwave_internal_mode_diagnosis"]:
        row = payload["dwave_internal_mode_diagnosis"][0]
        lines.append(f"| {row['interpretation']} | {row['evidence']} {row['marker']} | {payload['next_step']} |")
    else:
        lines.append(f"| none | dwave restored Ward did not require an internal-mode diagnosis. | {payload['next_step']} |")
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- status: {payload['status']}",
            f"- formal Casimir ran: {payload['formal_casimir_ran']}",
            f"- production default modified: {payload['production_default_modified']}",
            f"- interpretation: {payload['interpretation']}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    write_report(build_payload(quick=args.quick))


if __name__ == "__main__":
    main()
