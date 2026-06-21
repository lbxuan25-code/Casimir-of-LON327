#!/usr/bin/env python3
"""Audit finite-q BdG operator-level Ward vertex identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.bdg_finite_q_response import bdg_finite_q_vector_vertex, collective_form_factor  # noqa: E402
from lno327.pairing import PairingAmplitudes, bdg_hamiltonian, pairing_matrix  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
PAIRINGS = ("onsite_s", "spm", "dwave")
PHASE_VERTICES = ("midpoint", "symmetric_kpm")
K_POINTS = (
    (0.13, 0.27),
    (0.41, -0.22),
    (1.11, 0.73),
    (-0.64, 1.37),
)
Q_MODEL_LIST = (
    (0.01, 0.0),
    (0.01, 0.01),
)
CANDIDATE_SPECS = {
    "A": ("rho_Hp_minus_Hm_rho", 1.0),
    "B": ("Hp_rho_minus_rho_Hm", 1.0),
    "C": ("rho_Hp_minus_Hm_rho", -1.0),
    "D": ("Hp_rho_minus_rho_Hm", -1.0),
}


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


def _eta2_vertex(phi: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi)
    return np.block([[zero, 1j * phi], [-1j * phi.conjugate().T, zero]]).astype(complex)


def _block_metrics(matrix: np.ndarray) -> dict[str, float]:
    particle = matrix[:4, :4]
    hole = matrix[4:, 4:]
    upper_pairing = matrix[:4, 4:]
    lower_pairing = matrix[4:, :4]
    particle_abs = float(np.max(np.abs(particle)))
    hole_abs = float(np.max(np.abs(hole)))
    pairing_abs = float(max(np.max(np.abs(upper_pairing)), np.max(np.abs(lower_pairing))))
    return {
        "normal_block_residual_max_abs": float(max(particle_abs, hole_abs)),
        "pairing_block_residual_max_abs": pairing_abs,
        "particle_block_residual_max_abs": particle_abs,
        "hole_block_residual_max_abs": hole_abs,
    }


def _base_ordering(name: str, rho: np.ndarray, h_minus: np.ndarray, h_plus: np.ndarray) -> np.ndarray:
    if name == "rho_Hp_minus_Hm_rho":
        return rho @ h_plus - h_minus @ rho
    if name == "Hp_rho_minus_rho_Hm":
        return h_plus @ rho - rho @ h_minus
    raise ValueError(f"unknown ordering {name}")


def _case_rows(quick: bool) -> list[dict[str, Any]]:
    _ = quick
    amp = PairingAmplitudes(delta0_eV=0.04)
    rho = _rho_vertex()
    c_eta2_candidates = [2.0 * amp.delta0_eV, -2.0 * amp.delta0_eV, 2.0j * amp.delta0_eV, -2.0j * amp.delta0_eV]
    rows: list[dict[str, Any]] = []
    for pairing in PAIRINGS:
        for kx, ky in K_POINTS:
            for qx, qy in Q_MODEL_LIST:
                kx_minus, ky_minus = kx - 0.5 * qx, ky - 0.5 * qy
                kx_plus, ky_plus = kx + 0.5 * qx, ky + 0.5 * qy
                h_minus = bdg_hamiltonian(kx_minus, ky_minus, _pairing_delta(pairing, kx_minus, ky_minus, amp))
                h_plus = bdg_hamiltonian(kx_plus, ky_plus, _pairing_delta(pairing, kx_plus, ky_plus, amp))
                qv = qx * bdg_finite_q_vector_vertex(kx, ky, qx, qy, "x") + qy * bdg_finite_q_vector_vertex(kx, ky, qx, qy, "y")
                for phase_vertex in PHASE_VERTICES:
                    phi = collective_form_factor(pairing, kx, ky, qx, qy, amp, phase_vertex)  # type: ignore[arg-type]
                    gamma_eta2 = _eta2_vertex(phi)
                    for candidate_name, (ordering, collective_sign) in CANDIDATE_SPECS.items():
                        base = _base_ordering(ordering, rho, h_minus, h_plus)
                        for qv_sign in (1, -1):
                            for c_eta2 in c_eta2_candidates:
                                residual = base + float(qv_sign) * qv + collective_sign * c_eta2 * gamma_eta2
                                metrics = _block_metrics(residual)
                                rows.append(
                                    {
                                        "pairing": pairing,
                                        "k": [float(kx), float(ky)],
                                        "q_model": [float(qx), float(qy)],
                                        "phase_vertex": phase_vertex,
                                        "candidate_name": candidate_name,
                                        "candidate_ordering": ordering,
                                        "qV_sign": int(qv_sign),
                                        "C_eta2": c_eta2,
                                        "operator_residual_fro": float(np.linalg.norm(residual)),
                                        "operator_residual_max_abs": float(np.max(np.abs(residual))),
                                        **metrics,
                                    }
                                )
    return rows


def _best_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_by_pairing: dict[str, Any] = {}
    for pairing in PAIRINGS:
        best = min((row for row in rows if row["pairing"] == pairing), key=lambda row: row["operator_residual_max_abs"])
        residual = float(best["operator_residual_max_abs"])
        if residual < 1e-10:
            status = "PASSED"
        elif residual < 1e-8:
            status = "MONITOR"
        else:
            status = "FAILED"
        best_by_pairing[pairing] = {**best, "status": status}
    onsite_status = best_by_pairing["onsite_s"]["status"]
    material_passed = all(best_by_pairing[pairing]["status"] == "PASSED" for pairing in ("spm", "dwave"))
    if onsite_status == "PASSED" and material_passed:
        status = "PASSED"
        reason = "onsite_s and material pairings operator identities passed"
    elif onsite_status == "PASSED":
        status = "MONITOR"
        reason = "onsite_s operator identity passed, material pairing identity still unresolved"
    else:
        status = "FAILED"
        reason = "onsite_s operator identity failed; return to Nambu density/current/source convention"
    return {
        "best_by_pairing": best_by_pairing,
        "onsite_s_operator_identity_passed": onsite_status == "PASSED",
        "material_pairings_operator_identity_passed": material_passed,
        "status": status,
        "failure_reason": reason,
        "diagnostic_only": True,
    }


def _format_c(value: complex | dict[str, float]) -> str:
    if isinstance(value, dict):
        real = value.get("real", 0.0)
        imag = value.get("imag", 0.0)
    else:
        real = float(np.real(value))
        imag = float(np.imag(value))
    if abs(imag) < 1e-15:
        return f"{real:.6g}"
    if abs(real) < 1e-15:
        return f"{imag:.6g}i"
    return f"{real:.6g}{imag:+.6g}i"


def _write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "stageSC_0_bdg_operator_ward_vertex_audit.json"
    md_path = OUTPUT_DIR / "stageSC_0_bdg_operator_ward_vertex_audit.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    best = payload["summary"]["best_by_pairing"]
    lines = [
        "# stageSC_0_bdg_operator_ward_vertex_audit",
        "",
        f"- status: {payload['status']}",
        f"- quick: {payload['quick']}",
        f"- cases: {len(payload['cases'])}",
        f"- failure_reason: {payload['summary']['failure_reason']}",
        "",
        "| pairing | best residual | candidate | phase vertex | qV sign | C_eta2 | normal block | pairing block | status |",
        "| ------- | ------------: | --------- | ------------ | ------: | ------ | -----------: | ------------: | ------ |",
    ]
    for pairing in PAIRINGS:
        row = best[pairing]
        lines.append(
            f"| {pairing} | {row['operator_residual_max_abs']:.6g} | {row['candidate_name']} | "
            f"{row['phase_vertex']} | {row['qV_sign']} | {_format_c(row['C_eta2'])} | "
            f"{row['normal_block_residual_max_abs']:.6g} | {row['pairing_block_residual_max_abs']:.6g} | "
            f"{row['status']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    rows = _case_rows(args.quick)
    summary = _best_summary(rows)
    payload = {
        "status": summary["status"],
        "quick": bool(args.quick),
        "summary": summary,
        "cases": rows,
    }
    _write_report(payload)


if __name__ == "__main__":
    main()
