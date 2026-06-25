#!/usr/bin/env python3
"""Audit real-space pairing bonds and bond-derived collective vertices."""

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
from lno327.bdg_finite_q_response import (  # noqa: E402
    _eta2_phase_vertex,
    bdg_finite_q_vector_vertex,
    collective_form_factor,
)
from lno327.conductivity import KuboConfig  # noqa: E402
from lno327.pairing import PairingAmplitudes, bdg_hamiltonian, pairing_matrix  # noqa: E402
from lno327.pairing_bonds import (  # noqa: E402
    bond_endpoint_gauge_form_factor,
    pairing_bond_list,
    pairing_from_bonds,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
PAIRINGS = ("onsite_s", "spm", "dwave")
TEST_K_POINTS = ((0.0, 0.0), (0.13, 0.27), (0.41, -0.22), (1.11, 0.73), (-0.64, 1.37))
OPERATOR_K_POINTS = TEST_K_POINTS[1:]
Q_MODEL_LIST = ((0.01, 0.0), (0.01, 0.01))
PHASE_VERTICES = ("midpoint", "symmetric_kpm", "bond_endpoint_gauge")
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


def _pairing_delta(pairing: str, kx: float, ky: float, amp: PairingAmplitudes) -> np.ndarray:
    if pairing == "onsite_s":
        return amp.delta0_eV * np.eye(4, dtype=complex)
    return pairing_matrix(pairing, kx, ky, amp)  # type: ignore[arg-type]


def _rho_vertex() -> np.ndarray:
    eye = np.eye(4, dtype=complex)
    zero = np.zeros((4, 4), dtype=complex)
    return np.block([[eye, zero], [zero, -eye]])


def _status_pass_monitor(value: float, passed: float, monitor: float | None = None) -> str:
    if value < passed:
        return "PASSED"
    if monitor is not None and value < monitor:
        return "MONITOR"
    return "FAILED"


def reconstruction_rows(amp: PairingAmplitudes) -> list[dict[str, Any]]:
    rows = []
    for pairing in PAIRINGS:
        if not pairing_bond_list(pairing, amp):
            rows.append(
                {
                    "pairing": pairing,
                    "bond_reconstruction_max_abs": None,
                    "bond_reconstruction_fro": None,
                    "status": "PAIRING_BOND_REPRESENTATION_UNAVAILABLE",
                }
            )
            continue
        residuals = [
            pairing_from_bonds(pairing, kx, ky, amp) - _pairing_delta(pairing, kx, ky, amp)
            for kx, ky in TEST_K_POINTS
        ]
        max_abs = max(float(np.max(np.abs(item))) for item in residuals)
        fro = float(np.sqrt(sum(float(np.linalg.norm(item)) ** 2 for item in residuals)))
        rows.append(
            {
                "pairing": pairing,
                "bond_reconstruction_max_abs": max_abs,
                "bond_reconstruction_fro": fro,
                "status": "PASSED" if max_abs < 1e-12 else "FAILED",
            }
        )
    return rows


def vertex_comparison_rows(amp: PairingAmplitudes) -> list[dict[str, Any]]:
    rows = []
    for pairing in PAIRINGS:
        for qx, qy in Q_MODEL_LIST:
            diffs_mid = []
            diffs_sym = []
            for kx, ky in OPERATOR_K_POINTS:
                exact = bond_endpoint_gauge_form_factor(pairing, kx, ky, qx, qy, amp)
                midpoint = collective_form_factor(pairing, kx, ky, qx, qy, amp, "midpoint")  # type: ignore[arg-type]
                symmetric = collective_form_factor(pairing, kx, ky, qx, qy, amp, "symmetric_kpm")  # type: ignore[arg-type]
                diffs_mid.append(float(np.max(np.abs(exact - midpoint))))
                diffs_sym.append(float(np.max(np.abs(exact - symmetric))))
            diff_mid = max(diffs_mid)
            diff_sym = max(diffs_sym)
            rows.append(
                {
                    "pairing": pairing,
                    "q_model": [float(qx), float(qy)],
                    "phase_vertex": "bond_endpoint_gauge",
                    "diff_to_midpoint": diff_mid,
                    "diff_to_symmetric_kpm": diff_sym,
                    "diff_to_existing_best": min(diff_mid, diff_sym),
                    "center_convention": "bond centers are retained explicitly; current bonds have C_b=(0,0) or endpoint-origin convention",
                }
            )
    return rows


def operator_ward_rows(amp: PairingAmplitudes) -> list[dict[str, Any]]:
    rho = _rho_vertex()
    rows = []
    for pairing in PAIRINGS:
        for phase_vertex in PHASE_VERTICES:
            residuals = []
            for kx, ky in OPERATOR_K_POINTS:
                for qx, qy in Q_MODEL_LIST:
                    k_minus = (kx - 0.5 * qx, ky - 0.5 * qy)
                    k_plus = (kx + 0.5 * qx, ky + 0.5 * qy)
                    h_minus = bdg_hamiltonian(*k_minus, _pairing_delta(pairing, *k_minus, amp))
                    h_plus = bdg_hamiltonian(*k_plus, _pairing_delta(pairing, *k_plus, amp))
                    qv = qx * bdg_finite_q_vector_vertex(kx, ky, qx, qy, "x") + qy * bdg_finite_q_vector_vertex(
                        kx, ky, qx, qy, "y"
                    )
                    phi = collective_form_factor(pairing, kx, ky, qx, qy, amp, phase_vertex)  # type: ignore[arg-type]
                    residuals.append(rho @ h_plus - h_minus @ rho - qv + 2j * amp.delta0_eV * _eta2_phase_vertex(phi))
            max_abs = max(float(np.max(np.abs(item))) for item in residuals)
            fro = float(np.sqrt(sum(float(np.linalg.norm(item)) ** 2 for item in residuals)))
            rows.append(
                {
                    "pairing": pairing,
                    "phase_vertex": phase_vertex,
                    "candidate": "A",
                    "candidate_ordering": "rho_Hp_minus_Hm_rho",
                    "qV_sign": -1,
                    "C_eta2": 2j * amp.delta0_eV,
                    "operator_ward_max_abs": max_abs,
                    "operator_ward_fro": fro,
                    "status": _status_pass_monitor(max_abs, 1e-12, 1e-10),
                }
            )
    return rows


def projection_rows(amp: PairingAmplitudes) -> list[dict[str, Any]]:
    rows = []
    for pairing in PAIRINGS:
        basis_vertex = "symmetric_kpm" if pairing == "dwave" else "midpoint"
        lhs_blocks = []
        rhs_blocks = []
        for kx, ky in OPERATOR_K_POINTS:
            for qx, qy in Q_MODEL_LIST:
                lhs_blocks.append(bond_endpoint_gauge_form_factor(pairing, kx, ky, qx, qy, amp).reshape(-1))
                rhs_blocks.append(collective_form_factor(pairing, kx, ky, qx, qy, amp, basis_vertex).reshape(-1))  # type: ignore[arg-type]
        lhs = np.concatenate(lhs_blocks)
        matrix = np.concatenate(rhs_blocks)[:, None]
        coeffs, *_ = np.linalg.lstsq(matrix, lhs, rcond=None)
        residual = lhs - matrix @ coeffs
        abs_res = float(np.linalg.norm(residual))
        rel_res = float(abs_res / max(float(np.linalg.norm(lhs)), 1e-300))
        rows.append(
            {
                "pairing": pairing,
                "basis_phase_vertices": [basis_vertex],
                "collective_basis_projection_residual": abs_res,
                "collective_basis_projection_relative_residual": rel_res,
                "num_phase_channels": 1,
                "status": _status_pass_monitor(rel_res, 1e-10, 1e-8),
            }
        )
    return rows


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
                        "phase_vertex": "bond_endpoint_gauge",
                        "N": int(n_grid),
                        "m_case": list(m_case),
                        "q_model": q.tolist(),
                        "contact_closure_max_abs": contact,
                        "bare_ward_monitor": float(result["bare_total_ward_max_abs"]),
                        "amplitude_phase_ward_max_abs": float(result["amplitude_phase_ward_max_abs"]),
                        "collective_condition_number": float(result["collective_condition_number"]),
                        "dominant_failure": dominant,
                        "status": status,
                    }
                )
    return rows


def overall_status(payload: dict[str, Any]) -> tuple[str, str, str]:
    recon = {row["pairing"]: row for row in payload["pairing_reconstruction"]}
    projection = {row["pairing"]: row for row in payload["collective_basis_projection"]}
    operator = {
        row["pairing"]: row
        for row in payload["operator_ward"]
        if row["phase_vertex"] == "bond_endpoint_gauge"
    }
    schur = payload["commensurate_q_schur_ward"]
    if payload["formal_casimir_ran"] or any(row["status"] != "PASSED" for row in recon.values()):
        return "FAILED", "A required pairing bond reconstruction failed or formal Casimir ran.", "Fix bond reconstruction before Schur diagnostics."
    if any(operator[pairing]["status"] != "PASSED" for pairing in PAIRINGS):
        return "FAILED", "Bond endpoint gauge vertex failed the operator-level pairing Ward identity.", "Do not run Schur with this phase vertex."
    if any(projection[pairing]["status"] == "FAILED" for pairing in PAIRINGS):
        return (
            "PARTIAL_PASS_DWAVE_BASIS_INCOMPLETE" if projection["dwave"]["status"] == "FAILED" else "FAILED",
            "An exact gauge-phase vertex is outside the current collective phase basis.",
            "Report COLLECTIVE_BASIS_INCOMPLETE instead of forcing Schur closure.",
        )
    if not schur:
        return "FAILED", "Schur audit was skipped because prerequisites were not satisfied.", "Resolve prerequisite failures before Schur diagnostics."
    if any(float(row["contact_closure_max_abs"]) >= 1e-10 for row in schur):
        return "FAILED", "Commensurate contact closure failed.", "Return to finite-q contact assembly."
    if all(row["status"] == "PASSED" for row in schur):
        return "PASSED", "All tested pairing-bond collective vertex audits passed.", "Proceed to response-table diagnostics; keep production defaults unchanged."
    onsite_spm_ok = all(row["status"] == "PASSED" for row in schur if row["pairing"] in {"onsite_s", "spm"})
    if onsite_spm_ok and projection["dwave"]["status"] == "FAILED":
        return (
            "PARTIAL_PASS_DWAVE_BASIS_INCOMPLETE",
            "onsite_s and spm passed, but dwave exact gauge variation is outside the current collective basis.",
            "Report COLLECTIVE_BASIS_INCOMPLETE instead of forcing Schur closure.",
        )
    if onsite_spm_ok:
        return (
            "PARTIAL_PASS_DWAVE_COUNTERTERM_BLOCKED",
            "dwave reconstruction/operator/projection pass, but the amplitude-phase Ward Schur residual remains above threshold.",
            "Audit the nonlocal pairing counterterm block before changing production defaults.",
        )
    return "FAILED", "onsite_s or spm failed the Schur audit.", "Fix validation pairings before material pairing diagnostics."


def build_payload(quick: bool = True) -> dict[str, Any]:
    amp = PairingAmplitudes(delta0_eV=0.04)
    reconstruction = reconstruction_rows(amp)
    comparison = vertex_comparison_rows(amp)
    operator = operator_ward_rows(amp)
    projection = projection_rows(amp)
    prerequisites_pass = (
        all(row["status"] == "PASSED" for row in reconstruction)
        and all(
            row["status"] == "PASSED"
            for row in operator
            if row["phase_vertex"] == "bond_endpoint_gauge"
        )
        and all(row["status"] in {"PASSED", "MONITOR"} for row in projection)
    )
    payload: dict[str, Any] = {
        "quick": bool(quick),
        "diagnostic_only": True,
        "production_default_modified": False,
        "formal_casimir_ran": False,
        "delta0_eV": amp.delta0_eV,
        "pairing_reconstruction": reconstruction,
        "phase_vertex_comparison": comparison,
        "operator_ward": operator,
        "collective_basis_projection": projection,
        "schur_prerequisites_passed": prerequisites_pass,
        "commensurate_q_schur_ward": schur_rows(amp, quick=quick) if prerequisites_pass else [],
    }
    status, interpretation, next_step = overall_status(payload)
    payload["status"] = status
    payload["interpretation"] = interpretation
    payload["next_step"] = next_step
    if status == "PARTIAL_PASS_DWAVE_BASIS_INCOMPLETE":
        payload["blocked_marker"] = "COLLECTIVE_BASIS_INCOMPLETE"
    return payload


def _q_label(row: dict[str, Any]) -> str:
    q = row["q_model"]
    return f"({q[0]:.7g},{q[1]:.7g})"


def write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_2d_pairing_bond_collective_vertex_audit"
    json_path = OUTPUT_DIR / f"{stem}.json"
    md_path = OUTPUT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {stem}",
        "",
        "## Table 1: pairing reconstruction",
        "",
        "| pairing | reconstruction max | status |",
        "| ------- | -----------------: | ------ |",
    ]
    for row in payload["pairing_reconstruction"]:
        value = row["bond_reconstruction_max_abs"]
        lines.append(f"| {row['pairing']} | {value:.8g} | {row['status']} |")
    lines.extend(["", "## Table 2: operator Ward by phase vertex", "", "| pairing | phase vertex | operator Ward max | status |", "| ------- | ------------ | ----------------: | ------ |"])
    for row in payload["operator_ward"]:
        lines.append(f"| {row['pairing']} | {row['phase_vertex']} | {row['operator_ward_max_abs']:.8g} | {row['status']} |")
    lines.extend(["", "## Table 3: collective basis projection", "", "| pairing | num channels | projection rel residual | status |", "| ------- | -----------: | ----------------------: | ------ |"])
    for row in payload["collective_basis_projection"]:
        lines.append(
            f"| {row['pairing']} | {row['num_phase_channels']} | {row['collective_basis_projection_relative_residual']:.8g} | {row['status']} |"
        )
    lines.extend(["", "## Table 4: commensurate-q Schur Ward", "", "| pairing | phase vertex |  N | q | contact closure | AP Ward | status |", "| ------- | ------------ | -: | - | --------------: | ------: | ------ |"])
    for row in payload["commensurate_q_schur_ward"]:
        lines.append(
            f"| {row['pairing']} | {row['phase_vertex']} | {row['N']} | {_q_label(row)} | "
            f"{row['contact_closure_max_abs']:.8g} | {row['amplitude_phase_ward_max_abs']:.8g} | {row['status']} |"
        )
    lines.extend(["", "## Table 5: overall conclusion", "", "| status | interpretation | next step |", "| ------ | -------------- | --------- |", f"| {payload['status']} | {payload['interpretation']} | {payload['next_step']} |"])
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
