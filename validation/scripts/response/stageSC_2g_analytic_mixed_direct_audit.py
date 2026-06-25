#!/usr/bin/env python3
"""Diagnostic-only audit of analytic bond-level EM-eta mixed direct blocks."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from bdg_commensurate_q_common import commensurate_q_spec  # noqa: E402
from bdg_quadrature_strategy_common import (  # noqa: E402
    NAMBU_PREFACTOR,
    _bdg_hamiltonian_batch,
    _fermi_matrix,
    composite_uniform_quadrature,
    compute_bdg_components_for_composite_grid,
    single_composite_schur,
)
from lno327.conductivity import KuboConfig, fermi_function  # noqa: E402
from lno327.pairing import PairingAmplitudes  # noqa: E402
from lno327.pairing_bonds import PairingBond, bond_endpoint_gauge_form_factor, pairing_bond_list  # noqa: E402
from lno327.ward_response import physical_ward_residuals  # noqa: E402
from pairing_bond_goldstone_common import operator_ward_rows  # noqa: E402
from stageSC_2f_schur_block_assembly_audit import (  # noqa: E402
    _candidate_responses as stage2f_candidate_responses,
    _etaeta_metric_factor,
    _five_channel_residual,
    _block_matrix,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
PAIRINGS = ("onsite_s", "spm", "dwave", "dwave_const_form")
N_LIST_QUICK = (24, 36, 48)
M_CASES_QUICK = ((1, 0), (1, 1))
ANALYTIC_CANDIDATES = (
    "analytic_right_only_plus",
    "analytic_right_only_minus",
    "analytic_left_only_plus",
    "analytic_left_only_minus",
    "analytic_both_same_sign_plus",
    "analytic_both_same_sign_minus",
    "analytic_both_physical_current_convention",
    "analytic_both_opposite_convention",
)


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


def _sinc(value: float) -> float:
    if abs(value) < 1e-14:
        return 1.0
    return float(np.sin(value) / value)


def _const_dwave_bonds(amp: PairingAmplitudes) -> list[PairingBond]:
    delta0 = complex(amp.delta0_eV)
    return [
        PairingBond("dwave_const_form", left, right, (0.0, 0.0), (0.0, 0.0), delta0, "validation_const_form")
        for left, right in ((0, 1), (1, 0), (2, 3), (3, 2))
    ]


def _diagnostic_bonds(pairing: str, amp: PairingAmplitudes) -> list[PairingBond]:
    if pairing == "dwave_const_form":
        return _const_dwave_bonds(amp)
    return pairing_bond_list(pairing, amp)


def bond_mixed_direct_form_factor(
    pairing: str,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    amp: PairingAmplitudes,
) -> np.ndarray:
    axis = {"x": 0, "y": 1}[direction]
    delta0 = float(amp.delta0_eV)
    if delta0 == 0.0:
        raise ValueError("mixed direct form factor is undefined for delta0=0")
    output = np.zeros((4, 4), dtype=complex)
    k = np.asarray([kx, ky], dtype=float)
    q = np.asarray([qx, qy], dtype=float)
    for bond in _diagnostic_bonds(pairing, amp):
        displacement = bond.displacement
        q_dot_d_half = 0.5 * float(np.dot(q, displacement))
        output[bond.left_orbital, bond.right_orbital] += (
            bond.coefficient
            / delta0
            * np.exp(-1j * float(np.dot(k, displacement)))
            * displacement[axis]
            * np.cos(q_dot_d_half)
            * _sinc(q_dot_d_half)
        )
    return output


def _gamma_eta1(matrix: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(matrix)
    return np.block([[zero, matrix], [matrix.conjugate().T, zero]]).astype(complex)


def _gamma_eta2(matrix: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(matrix)
    return np.block([[zero, 1j * matrix], [-1j * matrix.conjugate().T, zero]]).astype(complex)


def bond_mixed_direct_vertex(
    pairing: str,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    eta_channel: str,
    amp: PairingAmplitudes,
) -> np.ndarray:
    psi = bond_mixed_direct_form_factor(pairing, kx, ky, qx, qy, direction, amp)
    if eta_channel == "eta1":
        return _gamma_eta2(psi)
    if eta_channel == "eta2":
        return -_gamma_eta1(psi)
    raise ValueError("eta_channel must be eta1 or eta2")


def _ward_parts(response: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, float]:
    left, right = physical_ward_residuals(response, omega_eV, q)
    return {
        "left_ward": float(np.max(np.abs(left))),
        "right_ward": float(np.max(np.abs(right))),
        "max_ward": float(max(np.max(np.abs(left)), np.max(np.abs(right)))),
    }


def _direct_expectation(points: np.ndarray, weights: np.ndarray, cfg: KuboConfig, vertices: np.ndarray, pairing: str, amp: PairingAmplitudes) -> np.ndarray:
    output = np.zeros(vertices.shape[:2], dtype=complex)
    chunk_size = 512
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        p = points[start:stop]
        w = weights[start:stop]
        h = _bdg_hamiltonian_batch(pairing, p, amp.delta0_eV)
        energies, states = np.linalg.eigh(h)
        occupations = fermi_function(energies, cfg.fermi_level_eV, cfg.temperature_eV)
        f_mid = _fermi_matrix(states, occupations)
        v = vertices[:, :, start:stop]
        expectation = NAMBU_PREFACTOR * np.einsum("bij,stbji->stb", f_mid, v, optimize=True)
        output += np.einsum("b,stb->st", w, expectation, optimize=True)
    return output


def _mixed_vertices_for_points(pairing: str, points: np.ndarray, q: np.ndarray, amp: PairingAmplitudes, orientation: str) -> np.ndarray:
    vertices = np.zeros((2, 2, points.shape[0], 8, 8), dtype=complex)
    q_use = -q if orientation == "reverse_q_hermitian" else q
    for bidx, (kx, ky) in enumerate(points):
        for i, direction in enumerate(("x", "y")):
            for a, eta in enumerate(("eta1", "eta2")):
                vertex = bond_mixed_direct_vertex(
                    pairing,
                    float(kx),
                    float(ky),
                    float(q_use[0]),
                    float(q_use[1]),
                    direction,
                    eta,
                    amp,
                )
                vertices[i, a, bidx] = vertex.conjugate().T if orientation == "reverse_q_hermitian" else vertex
    return vertices


def analytic_mixed_direct_blocks(
    pairing: str,
    points: np.ndarray,
    weights: np.ndarray,
    q: np.ndarray,
    cfg: KuboConfig,
    amp: PairingAmplitudes,
    right_orientation: str,
) -> tuple[np.ndarray, np.ndarray]:
    same_q_vertices = _mixed_vertices_for_points(pairing, points, q, amp, "same_q_naive")
    reverse_vertices = _mixed_vertices_for_points(pairing, points, q, amp, right_orientation)
    left_raw = _direct_expectation(points, weights, cfg, same_q_vertices, pairing, amp)
    right_raw = _direct_expectation(points, weights, cfg, reverse_vertices, pairing, amp)
    left = np.zeros((3, 2), dtype=complex)
    right = np.zeros((2, 3), dtype=complex)
    left[1:, :] = left_raw
    right[:, 1:] = right_raw.T
    return left, right


def _schur(bare: np.ndarray, left: np.ndarray, collective: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, float]:
    response, condition, _ = single_composite_schur(bare, left, collective, right)
    return response, condition


def _candidate_response_map(
    result: dict[str, Any],
    left_direct: np.ndarray,
    right_direct: np.ndarray,
    pairing: str,
    q: np.ndarray,
    amp: PairingAmplitudes,
) -> dict[str, tuple[np.ndarray, float]]:
    bare = result["bare_total"]
    left = result["em_collective_left"]
    right = result["collective_em_right"]
    collective = result["collective_total"]
    etaeta = collective.copy()
    etaeta[1, 1] *= _etaeta_metric_factor(pairing, q, amp)
    lsq = stage2f_candidate_responses(result, pairing, q, amp)["mixed_only"]
    candidates: dict[str, tuple[np.ndarray, float]] = {
        "baseline": (result["amplitude_phase_schur"], float(result["collective_condition_number"])),
        "stage2f_lsq_mixed_only": (lsq, float(result["collective_condition_number"])),
        "etaeta_only": _schur(bare, left, etaeta, right),
    }
    sign_specs = {
        "analytic_right_only_plus": (0.0, 1.0),
        "analytic_right_only_minus": (0.0, -1.0),
        "analytic_left_only_plus": (1.0, 0.0),
        "analytic_left_only_minus": (-1.0, 0.0),
        "analytic_both_same_sign_plus": (1.0, 1.0),
        "analytic_both_same_sign_minus": (-1.0, -1.0),
        "analytic_both_physical_current_convention": (-1.0, 1.0),
        "analytic_both_opposite_convention": (1.0, -1.0),
    }
    for name, (left_sign, right_sign) in sign_specs.items():
        response, condition = _schur(bare, left + left_sign * left_direct, collective, right + right_sign * right_direct)
        candidates[name] = (response, condition)
    analytic_best = min(ANALYTIC_CANDIDATES, key=lambda name: _ward_parts(candidates[name][0], 0.01, q)["max_ward"])
    left_sign, right_sign = sign_specs[analytic_best]
    candidates["analytic_best_plus_etaeta"] = _schur(
        bare,
        left + left_sign * left_direct,
        etaeta,
        right + right_sign * right_direct,
    )
    return candidates


def _residual_by_right_channel(response: np.ndarray, result: dict[str, Any], omega_eV: float, q: np.ndarray, delta0_eV: float) -> dict[str, float]:
    block = _block_matrix(result, bare=response, collective=result["collective_total"])
    return _five_channel_residual(block, omega_eV, q, delta0_eV)


def build_cases(quick: bool = True, right_orientation: str = "reverse_q_hermitian") -> list[dict[str, Any]]:
    amp = PairingAmplitudes(delta0_eV=0.04)
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
                left_direct, right_direct = analytic_mixed_direct_blocks(pairing, points, weights, q, cfg, amp, right_orientation)
                candidate_map = _candidate_response_map(result, left_direct, right_direct, pairing, q, amp)
                analytic_candidates = {
                    name: _ward_parts(candidate_map[name][0], cfg.omega_eV, q)
                    for name in ANALYTIC_CANDIDATES
                }
                best_name = min(ANALYTIC_CANDIDATES, key=lambda name: analytic_candidates[name]["max_ward"])
                baseline_parts = _ward_parts(candidate_map["baseline"][0], cfg.omega_eV, q)
                lsq_parts = _ward_parts(candidate_map["stage2f_lsq_mixed_only"][0], cfg.omega_eV, q)
                best_parts = analytic_candidates[best_name]
                etaeta_parts = _ward_parts(candidate_map["etaeta_only"][0], cfg.omega_eV, q)
                best_plus_etaeta_parts = _ward_parts(candidate_map["analytic_best_plus_etaeta"][0], cfg.omega_eV, q)
                contact = max(float(result["contact_closure"][channel]["E_band_plus_qD_abs"]) for channel in ("Vx", "Vy"))
                rows.append(
                    {
                        "pairing": pairing,
                        "N": int(n_grid),
                        "m_case": list(m_case),
                        "q_model": q.tolist(),
                        "right_orientation": right_orientation,
                        "contact_closure_max_abs": contact,
                        "bare_ward_monitor": float(result["bare_total_ward_max_abs"]),
                        "baseline_restored_ward": baseline_parts["max_ward"],
                        "stage2f_lsq_mixed_only_restored_ward": lsq_parts["max_ward"],
                        "etaeta_only": etaeta_parts,
                        "analytic_candidates": analytic_candidates,
                        "analytic_best_plus_etaeta": best_plus_etaeta_parts,
                        "best_analytic_candidate": best_name,
                        "best_analytic_restored_ward": best_parts["max_ward"],
                        "analytic_improvement_factor": float(
                            baseline_parts["max_ward"] / max(best_parts["max_ward"], 1e-300)
                        ),
                        "analytic_vs_lsq_gap": float(best_parts["max_ward"] / max(lsq_parts["max_ward"], 1e-300)),
                        "mixed_direct_norms": {
                            "left_direct_norm": float(np.linalg.norm(left_direct)),
                            "right_direct_norm": float(np.linalg.norm(right_direct)),
                        },
                        "residual_by_right_channel_baseline": _residual_by_right_channel(
                            candidate_map["baseline"][0], result, cfg.omega_eV, q, amp.delta0_eV
                        ),
                        "residual_by_right_channel_best_analytic": _residual_by_right_channel(
                            candidate_map[best_name][0], result, cfg.omega_eV, q, amp.delta0_eV
                        ),
                    }
                )
    return rows


def _status(cases: list[dict[str, Any]]) -> str:
    onsite_spm_ok = all(
        min(
            [case["baseline_restored_ward"]]
            + [candidate["max_ward"] for candidate in case["analytic_candidates"].values()]
        )
        < 1e-6
        for case in cases
        if case["pairing"] in {"onsite_s", "spm"}
    )
    if not onsite_spm_ok:
        return "FAILED_STAGE2G_REGRESSION_ON_ONSITE_OR_SPM"
    dwave = [case for case in cases if case["pairing"] == "dwave"]
    max_best = max(float(case["best_analytic_restored_ward"]) for case in dwave)
    max_baseline = max(float(case["baseline_restored_ward"]) for case in dwave)
    max_lsq = max(float(case["stage2f_lsq_mixed_only_restored_ward"]) for case in dwave)
    if max_best < 1e-8:
        expected = {"analytic_both_physical_current_convention"}
        best_names = {case["best_analytic_candidate"] for case in dwave}
        if best_names <= expected:
            return "PASSED_STAGE2G_ANALYTIC_MIXED_DIRECT_CLOSED_DWAVE"
        return "PASSED_STAGE2G_ANALYTIC_MIXED_DIRECT_IDENTIFIED_BUT_SIGN_NEEDS_REVIEW"
    if max_best < 0.1 * max_baseline:
        return "PARTIAL_STAGE2G_ANALYTIC_IMPROVES_BUT_NOT_CLOSED"
    if max_lsq < 1e-8:
        return "PARTIAL_STAGE2G_LSQ_CLOSES_BUT_ANALYTIC_DOES_NOT"
    return "PARTIAL_STAGE2G_ANALYTIC_IMPROVES_BUT_NOT_CLOSED"


@lru_cache(maxsize=2)
def build_payload(quick: bool = True) -> dict[str, Any]:
    cases = build_cases(quick=quick, right_orientation="reverse_q_hermitian")
    same_q_cases = build_cases(quick=quick, right_orientation="same_q_naive")
    status = _status(cases)
    dwave = [case for case in cases if case["pairing"] == "dwave"]
    best_names = sorted({case["best_analytic_candidate"] for case in dwave})
    payload = {
        "status": status,
        "quick": bool(quick),
        "diagnostic_only": True,
        "production_default_modified": False,
        "formal_casimir_ran": False,
        "analytic_mixed_direct_tested": True,
        "least_squares_candidate_used_as_production_formula": False,
        "right_orientation_primary": "reverse_q_hermitian",
        "right_orientation_negative_control": "same_q_naive",
        "best_analytic_candidates_for_dwave": best_names,
        "max_dwave_best_analytic_ward": max(float(case["best_analytic_restored_ward"]) for case in dwave),
        "max_dwave_lsq_reference_ward": max(float(case["stage2f_lsq_mixed_only_restored_ward"]) for case in dwave),
        "exact_goldstone_operator_ward": [
            row for row in operator_ward_rows(PairingAmplitudes(delta0_eV=0.04)) if row["phase_vertex"] == "exact_goldstone_tangent"
        ],
        "cases": cases,
        "same_q_naive_orientation_cases": same_q_cases,
    }
    return payload


def _q_label(case: dict[str, Any]) -> str:
    q = case["q_model"]
    return f"({q[0]:.7g},{q[1]:.7g})"


def write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_2g_analytic_mixed_direct_audit"
    json_path = OUTPUT_DIR / f"{stem}.json"
    md_path = OUTPUT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {stem}",
        "",
        f"- status: {payload['status']}",
        f"- diagnostic only: {payload['diagnostic_only']}",
        f"- formal Casimir ran: {payload['formal_casimir_ran']}",
        f"- production default modified: {payload['production_default_modified']}",
        f"- least-squares candidate used as production formula: {payload['least_squares_candidate_used_as_production_formula']}",
        f"- best analytic candidates for dwave: {payload['best_analytic_candidates_for_dwave']}",
        "",
        "## Candidate comparison",
        "",
        "| pairing | N | q | baseline | lsq ref | best analytic | best name | etaeta | analytic+etaeta |",
        "| ------- | -: | - | -------: | ------: | ------------: | --------- | -----: | ---------------: |",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['pairing']} | {case['N']} | {_q_label(case)} | {case['baseline_restored_ward']:.8g} | "
            f"{case['stage2f_lsq_mixed_only_restored_ward']:.8g} | {case['best_analytic_restored_ward']:.8g} | "
            f"{case['best_analytic_candidate']} | {case['etaeta_only']['max_ward']:.8g} | "
            f"{case['analytic_best_plus_etaeta']['max_ward']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Human-readable conclusion",
            "",
            "The analytic mixed direct formula was tested against the StageSC-2f least-squares reference.",
            f"The best dwave analytic candidates are {payload['best_analytic_candidates_for_dwave']}.",
            f"Max dwave best analytic Ward: {payload['max_dwave_best_analytic_ward']:.8g}.",
            f"Max dwave LSQ reference Ward: {payload['max_dwave_lsq_reference_ward']:.8g}.",
            "Left and right Ward components are stored separately for every analytic candidate in JSON.",
            "onsite_s, spm, and dwave_const_form remain validation controls.",
            "etaeta is retained as a secondary comparison and is not the primary tested block.",
            "Formal Casimir input remains forbidden.",
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
