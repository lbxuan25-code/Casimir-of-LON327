"""Diagnostic-only helpers for unified Goldstone tangent audits."""

from __future__ import annotations

from typing import Any

import numpy as np

from lno327.bdg_finite_q_response import (
    _eta2_phase_vertex,
    bdg_finite_q_vector_vertex,
    collective_form_factor,
)
from lno327.pairing import PairingAmplitudes, bdg_hamiltonian, pairing_matrix
from lno327.pairing_bonds import bond_endpoint_gauge_form_factor, pairing_bond_list, pairing_from_bonds

PAIRINGS = ("onsite_s", "spm", "dwave")
TEST_K_POINTS = ((0.0, 0.0), (0.13, 0.27), (0.41, -0.22), (1.11, 0.73), (-0.64, 1.37))
OPERATOR_K_POINTS = TEST_K_POINTS[1:]
Q_MODEL_LIST = ((0.01, 0.0), (0.01, 0.01))
PHASE_VERTICES = ("midpoint", "symmetric_kpm", "exact_goldstone_tangent")
GOLDSTONE_DIMENSION_STATEMENT = (
    "For all tested spm/dwave/onsite_s pairings, if only total charge U(1) is "
    "spontaneously broken, the Goldstone manifold dimension is one. "
    "Bond-resolved internal modes are not additional Goldstone modes."
)


def goldstone_dimension_rows() -> list[dict[str, Any]]:
    return [
        {
            "pairing": pairing,
            "goldstone_dimension": 1,
            "symmetry_assumption": "only_total_charge_U1_spontaneously_broken",
            "bond_resolved_internal_modes_are_goldstone": False,
            "status": "PASSED",
        }
        for pairing in PAIRINGS
    ]


def pairing_delta(pairing: str, kx: float, ky: float, amp: PairingAmplitudes) -> np.ndarray:
    if pairing == "onsite_s":
        return amp.delta0_eV * np.eye(4, dtype=complex)
    return pairing_matrix(pairing, kx, ky, amp)  # type: ignore[arg-type]


def rho_vertex() -> np.ndarray:
    eye = np.eye(4, dtype=complex)
    zero = np.zeros((4, 4), dtype=complex)
    return np.block([[eye, zero], [zero, -eye]])


def status_pass_monitor(value: float, passed: float, monitor: float | None = None) -> str:
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
            pairing_from_bonds(pairing, kx, ky, amp) - pairing_delta(pairing, kx, ky, amp)
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


def exact_goldstone_form_factor(
    pairing: str,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    amp: PairingAmplitudes,
) -> np.ndarray:
    return bond_endpoint_gauge_form_factor(pairing, kx, ky, qx, qy, amp)


def phase_form_factor(
    pairing: str,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    amp: PairingAmplitudes,
    phase_vertex: str,
) -> np.ndarray:
    if phase_vertex == "exact_goldstone_tangent":
        return exact_goldstone_form_factor(pairing, kx, ky, qx, qy, amp)
    return collective_form_factor(pairing, kx, ky, qx, qy, amp, phase_vertex)  # type: ignore[arg-type]


def q0_normalization_rows(amp: PairingAmplitudes) -> list[dict[str, Any]]:
    rows = []
    for pairing in PAIRINGS:
        diffs = []
        factors = []
        mismatch = False
        for kx, ky in TEST_K_POINTS:
            exact = exact_goldstone_form_factor(pairing, kx, ky, 0.0, 0.0, amp)
            existing = collective_form_factor(pairing, kx, ky, 0.0, 0.0, amp, "midpoint")  # type: ignore[arg-type]
            denom = np.vdot(existing.reshape(-1), existing.reshape(-1))
            factor = 1.0 + 0.0j if abs(denom) == 0.0 else np.vdot(existing.reshape(-1), exact.reshape(-1)) / denom
            factors.append(factor)
            residual = exact - factor * existing
            diffs.append(float(np.max(np.abs(residual))))
            if np.max(np.abs(residual)) >= 1e-12:
                mismatch = True
        norm = complex(np.mean(factors))
        diff = max(diffs)
        rows.append(
            {
                "pairing": pairing,
                "goldstone_dimension": 1,
                "goldstone_tangent_source": "local_U1_endpoint_transformation",
                "q0_diff_to_existing_phase_vertex": diff,
                "normalization_factor": norm,
                "normalization_factor_abs": float(abs(norm)),
                "status": "Q0_PHASE_VERTEX_SHAPE_MISMATCH" if mismatch else "PASSED",
            }
        )
    return rows


def operator_ward_rows(amp: PairingAmplitudes) -> list[dict[str, Any]]:
    rho = rho_vertex()
    rows = []
    for pairing in PAIRINGS:
        for phase_vertex in PHASE_VERTICES:
            residuals = []
            q_values = []
            for kx, ky in OPERATOR_K_POINTS:
                for qx, qy in Q_MODEL_LIST:
                    k_minus = (kx - 0.5 * qx, ky - 0.5 * qy)
                    k_plus = (kx + 0.5 * qx, ky + 0.5 * qy)
                    h_minus = bdg_hamiltonian(*k_minus, pairing_delta(pairing, *k_minus, amp))
                    h_plus = bdg_hamiltonian(*k_plus, pairing_delta(pairing, *k_plus, amp))
                    qv = qx * bdg_finite_q_vector_vertex(kx, ky, qx, qy, "x") + qy * bdg_finite_q_vector_vertex(
                        kx, ky, qx, qy, "y"
                    )
                    phi = phase_form_factor(pairing, kx, ky, qx, qy, amp, phase_vertex)
                    residuals.append(rho @ h_plus - h_minus @ rho - qv + 2j * amp.delta0_eV * _eta2_phase_vertex(phi))
                    q_values.append((float(qx), float(qy)))
            max_abs = max(float(np.max(np.abs(item))) for item in residuals)
            fro = float(np.sqrt(sum(float(np.linalg.norm(item)) ** 2 for item in residuals)))
            rows.append(
                {
                    "pairing": pairing,
                    "phase_vertex": phase_vertex,
                    "q_model_list": [list(q) for q in sorted(set(q_values))],
                    "candidate": "A",
                    "identity": "rho_Hp_minus_Hm_rho",
                    "qV_sign": -1,
                    "C_theta": 2j * amp.delta0_eV,
                    "normalization_factor": 1.0,
                    "operator_ward_max_abs": max_abs,
                    "operator_ward_fro": fro,
                    "status": status_pass_monitor(max_abs, 1e-12, 1e-10),
                }
            )
    return rows


def old_prescription_comparison_rows(amp: PairingAmplitudes, operator_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["pairing"], row["phase_vertex"]): row for row in operator_rows}
    rows = []
    for pairing in PAIRINGS:
        for qx, qy in Q_MODEL_LIST:
            diff_mid = []
            diff_sym = []
            for kx, ky in OPERATOR_K_POINTS:
                exact = exact_goldstone_form_factor(pairing, kx, ky, qx, qy, amp)
                midpoint = collective_form_factor(pairing, kx, ky, qx, qy, amp, "midpoint")  # type: ignore[arg-type]
                symmetric = collective_form_factor(pairing, kx, ky, qx, qy, amp, "symmetric_kpm")  # type: ignore[arg-type]
                diff_mid.append(float(np.max(np.abs(exact - midpoint))))
                diff_sym.append(float(np.max(np.abs(exact - symmetric))))
            rows.append(
                {
                    "pairing": pairing,
                    "q_model": [float(qx), float(qy)],
                    "diff_exact_to_midpoint": max(diff_mid),
                    "diff_exact_to_symmetric_kpm": max(diff_sym),
                    "operator_ward_midpoint": by_key[(pairing, "midpoint")]["operator_ward_max_abs"],
                    "operator_ward_symmetric_kpm": by_key[(pairing, "symmetric_kpm")]["operator_ward_max_abs"],
                    "operator_ward_exact": by_key[(pairing, "exact_goldstone_tangent")]["operator_ward_max_abs"],
                }
            )
    return rows

