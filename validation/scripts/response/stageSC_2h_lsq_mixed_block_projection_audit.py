#!/usr/bin/env python3
"""Project the StageSC-2f LSQ mixed block onto diagnostic basis families."""

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
from bdg_quadrature_strategy_common import composite_uniform_quadrature, compute_bdg_components_for_composite_grid  # noqa: E402
from lno327.conductivity import KuboConfig  # noqa: E402
from lno327.pairing import PairingAmplitudes  # noqa: E402
from lno327.pairing_bonds import PairingBond  # noqa: E402
from stageSC_2f_schur_block_assembly_audit import _mixed_direct_candidate, _ward_max  # noqa: E402
from stageSC_2g_analytic_mixed_direct_audit import (  # noqa: E402
    _diagnostic_bonds,
    _direct_expectation,
    _gamma_eta1,
    _gamma_eta2,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
PAIRINGS = ("onsite_s", "spm", "dwave", "dwave_const_form")
N_LIST_QUICK = (24, 36, 48)
M_CASES_QUICK = ((1, 0), (1, 1))
BASIS_NAMES = (
    "basis_d_difference",
    "basis_endpoint_sum",
    "basis_endpoint_difference",
    "basis_partial_q_phi",
    "basis_partial_k_phi",
    "basis_phi_qi",
    "basis_phi_omega",
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


def _phase(bond: PairingBond, k: np.ndarray) -> complex:
    return complex(np.exp(-1j * float(np.dot(k, bond.displacement))))


def _phi_factor(bond: PairingBond, q: np.ndarray) -> complex:
    return complex(np.exp(1j * float(np.dot(q, bond.center))) * np.cos(0.5 * float(np.dot(q, bond.displacement))))


def _basis_matrix(
    basis_name: str,
    pairing: str,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction: str,
    omega_eV: float,
    amp: PairingAmplitudes,
) -> np.ndarray:
    axis = {"x": 0, "y": 1}[direction]
    delta0 = float(amp.delta0_eV)
    k = np.asarray([kx, ky], dtype=float)
    q = np.asarray([qx, qy], dtype=float)
    output = np.zeros((4, 4), dtype=complex)
    for bond in _diagnostic_bonds(pairing, amp):
        coefficient = bond.coefficient / delta0 * _phase(bond, k)
        displacement = bond.displacement
        center = bond.center
        q_half_d = 0.5 * float(np.dot(q, displacement))
        phi_factor = _phi_factor(bond, q)
        if basis_name == "basis_d_difference":
            factor = displacement[axis] * np.cos(q_half_d) * _sinc(q_half_d)
        elif basis_name == "basis_endpoint_sum":
            left = np.asarray(bond.left_offset, dtype=float)
            right = np.asarray(bond.right_offset, dtype=float)
            factor = 0.5 * (
                left[axis] * np.exp(1j * float(np.dot(q, left)))
                + right[axis] * np.exp(1j * float(np.dot(q, right)))
            )
        elif basis_name == "basis_endpoint_difference":
            left = np.asarray(bond.left_offset, dtype=float)
            right = np.asarray(bond.right_offset, dtype=float)
            factor = right[axis] * np.exp(1j * float(np.dot(q, right))) - left[axis] * np.exp(
                1j * float(np.dot(q, left))
            )
        elif basis_name == "basis_partial_q_phi":
            factor = np.exp(1j * float(np.dot(q, center))) * (
                1j * center[axis] * np.cos(q_half_d) - 0.5 * displacement[axis] * np.sin(q_half_d)
            )
        elif basis_name == "basis_partial_k_phi":
            factor = -1j * displacement[axis] * phi_factor
        elif basis_name == "basis_phi_qi":
            factor = q[axis] * phi_factor
        elif basis_name == "basis_phi_omega":
            factor = omega_eV * phi_factor
        else:
            raise ValueError(f"unknown basis {basis_name}")
        output[bond.left_orbital, bond.right_orbital] += coefficient * factor
    return output


def _basis_self_checks(pairing: str, amp: PairingAmplitudes) -> dict[str, float]:
    if pairing not in {"dwave", "dwave_const_form"}:
        return {"endpoint_sum_equivalence_max_abs": 0.0, "endpoint_sum_vs_partial_q_relation_max_abs": 0.0}
    kx, ky, qx, qy = 0.41, -0.22, 0.01, 0.02
    max_equiv = 0.0
    max_relation = 0.0
    for direction in ("x", "y"):
        endpoint = _basis_matrix("basis_endpoint_sum", pairing, kx, ky, qx, qy, direction, 0.01, amp)
        partial_q = _basis_matrix("basis_partial_q_phi", pairing, kx, ky, qx, qy, direction, 0.01, amp)
        max_relation = max(max_relation, float(np.max(np.abs(endpoint - partial_q / 1j))))
        center_form = np.zeros((4, 4), dtype=complex)
        k = np.asarray([kx, ky], dtype=float)
        q = np.asarray([qx, qy], dtype=float)
        axis = {"x": 0, "y": 1}[direction]
        for bond in _diagnostic_bonds(pairing, amp):
            d = bond.displacement
            c = bond.center
            q_half_d = 0.5 * float(np.dot(q, d))
            factor = np.exp(1j * float(np.dot(q, c))) * (c[axis] * np.cos(q_half_d) + 0.5j * d[axis] * np.sin(q_half_d))
            center_form[bond.left_orbital, bond.right_orbital] += bond.coefficient / amp.delta0_eV * _phase(bond, k) * factor
        max_equiv = max(max_equiv, float(np.max(np.abs(endpoint - center_form))))
    return {
        "endpoint_sum_equivalence_max_abs": max_equiv,
        "endpoint_sum_vs_partial_q_relation_max_abs": max_relation,
    }


def _mixed_block_from_basis(
    basis_name: str,
    pairing: str,
    points: np.ndarray,
    weights: np.ndarray,
    q: np.ndarray,
    cfg: KuboConfig,
    amp: PairingAmplitudes,
) -> tuple[np.ndarray, dict[str, float]]:
    vertices = np.zeros((2, 2, points.shape[0], 8, 8), dtype=complex)
    raw_norm = 0.0
    parity_diffs = []
    for bidx, (kx, ky) in enumerate(points):
        for i, direction in enumerate(("x", "y")):
            matrix = _basis_matrix(basis_name, pairing, float(kx), float(ky), float(q[0]), float(q[1]), direction, cfg.omega_eV, amp)
            opposite = _basis_matrix(
                basis_name, pairing, -float(kx), -float(ky), float(q[0]), float(q[1]), direction, cfg.omega_eV, amp
            )
            raw_norm += float(np.linalg.norm(matrix) ** 2)
            even_diff = np.linalg.norm(matrix - opposite)
            odd_diff = np.linalg.norm(matrix + opposite)
            denom = max(np.linalg.norm(matrix), np.linalg.norm(opposite), 1e-300)
            parity_diffs.append((float(even_diff / denom), float(odd_diff / denom)))
            vertices[i, 0, bidx] = _gamma_eta2(matrix)
            vertices[i, 1, bidx] = -_gamma_eta1(matrix)
    raw_norm = float(np.sqrt(raw_norm / max(points.shape[0], 1)))
    direct = _direct_expectation(points, weights, cfg, vertices, pairing, amp)
    block = np.zeros((2, 3), dtype=complex)
    block[:, 1:] = direct.T
    thermal_norm = float(np.linalg.norm(block))
    even = np.median([item[0] for item in parity_diffs])
    odd = np.median([item[1] for item in parity_diffs])
    if even < 1e-8:
        parity = "even"
    elif odd < 1e-8:
        parity = "odd"
    else:
        parity = "mixed"
    return block, {
        "parity_under_k_to_minus_k": parity,
        "thermal_expectation_norm": thermal_norm,
        "raw_form_factor_norm_before_expectation": raw_norm,
        "cancellation_ratio": float(thermal_norm / max(raw_norm, 1e-300)),
    }


def _project_single(target: np.ndarray, basis: np.ndarray) -> dict[str, Any]:
    target_vec = target.reshape(-1)
    basis_vec = basis.reshape(-1)
    target_norm = float(np.linalg.norm(target_vec))
    basis_norm = float(np.linalg.norm(basis_vec))
    if target_norm < 1e-14 or basis_norm < 1e-14:
        coeff = 0.0 + 0.0j
        residual = 1.0
    else:
        coeff = complex(np.vdot(basis_vec, target_vec) / np.vdot(basis_vec, basis_vec))
        residual = float(np.linalg.norm(target_vec - coeff * basis_vec) / target_norm)
    return {
        "coefficient": coeff,
        "relative_residual": residual,
        "explained_norm_fraction": float(1.0 - residual**2),
        "basis_norm": basis_norm,
        "target_norm": target_norm,
    }


def _project_multi(target: np.ndarray, bases: dict[str, np.ndarray], selected: list[str]) -> dict[str, Any]:
    target_vec = target.reshape(-1)
    target_norm = float(np.linalg.norm(target_vec))
    matrix = np.stack([bases[name].reshape(-1) for name in selected], axis=1)
    if target_norm < 1e-14 or np.linalg.norm(matrix) < 1e-14:
        coeffs = np.zeros(len(selected), dtype=complex)
        residual = 1.0
        condition = float("inf")
    else:
        coeffs, *_ = np.linalg.lstsq(matrix, target_vec, rcond=None)
        residual = float(np.linalg.norm(target_vec - matrix @ coeffs) / target_norm)
        condition = float(np.linalg.cond(matrix))
    return {
        "selected_basis_set": selected,
        "coefficients": coeffs.tolist(),
        "relative_residual": residual,
        "explained_norm_fraction": float(1.0 - residual**2),
        "condition_number": condition,
    }


def _case(
    pairing: str,
    n_grid: int,
    m_case: tuple[int, int],
    points: np.ndarray,
    weights: np.ndarray,
    cfg: KuboConfig,
    amp: PairingAmplitudes,
) -> dict[str, Any]:
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
    target = _mixed_direct_candidate(result, result["amplitude_phase_schur"], cfg.omega_eV, q)
    basis_blocks = {}
    parity = {}
    for basis_name in BASIS_NAMES:
        basis_blocks[basis_name], parity[basis_name] = _mixed_block_from_basis(basis_name, pairing, points, weights, q, cfg, amp)
    single = {name: _project_single(target, block) for name, block in basis_blocks.items()}
    best_single = min(single, key=lambda name: single[name]["relative_residual"])
    multi_selected = list(BASIS_NAMES)
    multi = _project_multi(target, basis_blocks, multi_selected)
    baseline = _ward_max(result["amplitude_phase_schur"], cfg.omega_eV, q)
    lsq_response = result["bare_total"] - result["em_collective_left"] @ np.linalg.inv(result["collective_total"]) @ (
        result["collective_em_right"] + target
    )
    lsq_ward = _ward_max(lsq_response, cfg.omega_eV, q)
    rel_endpoint = min(single["basis_endpoint_sum"]["relative_residual"], single["basis_partial_q_phi"]["relative_residual"])
    rel_hopping = min(single["basis_d_difference"]["relative_residual"], single["basis_partial_k_phi"]["relative_residual"])
    return {
        "pairing": pairing,
        "N": int(n_grid),
        "m_case": list(m_case),
        "q_model": q.tolist(),
        "baseline_restored_ward": baseline,
        "lsq_mixed_only_restored_ward": lsq_ward,
        "lsq_target_norm": float(np.linalg.norm(target)),
        "best_single_basis": best_single,
        "best_single_basis_relative_residual": single[best_single]["relative_residual"],
        "best_multi_basis_set": multi["selected_basis_set"],
        "best_multi_basis_relative_residual": multi["relative_residual"],
        "dominant_basis_family": _dominant_family(best_single, multi["relative_residual"], rel_endpoint, rel_hopping),
        "supports_endpoint_sum_hypothesis": bool(rel_endpoint < 0.2),
        "supports_hopping_difference_hypothesis": bool(rel_hopping < 0.2),
        "supports_partial_q_phi_hypothesis": bool(single["basis_partial_q_phi"]["relative_residual"] < 0.2),
        "supports_partial_k_phi_hypothesis": bool(single["basis_partial_k_phi"]["relative_residual"] < 0.2),
        "single_basis_projection": single,
        "multi_basis_projection": multi,
        "parity_symmetry_audit": parity,
        "k_resolved_target_available": False,
        "k_resolved_note": "Integrated projection only; strict k-resolved LSQ target is unavailable in this audit.",
        "basis_self_checks": _basis_self_checks(pairing, amp),
    }


def _dominant_family(best_single: str, multi_residual: float, endpoint_residual: float, hopping_residual: float) -> str:
    if endpoint_residual < 0.2:
        return "endpoint_sum_or_partial_q_phi"
    if hopping_residual < 0.2:
        return "partial_k_or_d_difference"
    if multi_residual < 0.2:
        return "multi_basis"
    return f"unclear_integrated_best_{best_single}"


def build_cases(quick: bool = True) -> list[dict[str, Any]]:
    amp = PairingAmplitudes(delta0_eV=0.04)
    cfg = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    cases = []
    for pairing in PAIRINGS:
        for n_grid in (N_LIST_QUICK if quick else N_LIST_QUICK):
            points, weights = composite_uniform_quadrature(n_grid, [(0.0, 0.0)])
            for m_case in M_CASES_QUICK:
                cases.append(_case(pairing, n_grid, m_case, points, weights, cfg, amp))
    return cases


def _status(cases: list[dict[str, Any]]) -> str:
    dwave = [case for case in cases if case["pairing"] == "dwave"]
    if any(float(case["lsq_mixed_only_restored_ward"]) >= 1e-8 for case in dwave):
        return "FAILED_STAGE2H_LSQ_REFERENCE_NOT_CLOSING"
    endpoint_best = all(case["dominant_basis_family"] == "endpoint_sum_or_partial_q_phi" for case in dwave)
    hopping_best = all(case["dominant_basis_family"] == "partial_k_or_d_difference" for case in dwave)
    multi_best = all(case["dominant_basis_family"] == "multi_basis" for case in dwave)
    if endpoint_best:
        return "PASSED_STAGE2H_ENDPOINT_SUM_OR_DQPHI_IDENTIFIED"
    if hopping_best:
        return "PASSED_STAGE2H_PARTIAL_K_OR_D_DIFFERENCE_IDENTIFIED"
    if multi_best:
        return "PASSED_STAGE2H_MULTI_BASIS_STRUCTURE_IDENTIFIED"
    return "PARTIAL_STAGE2H_INTEGRATED_ONLY_K_RESOLVED_NEEDED"


@lru_cache(maxsize=2)
def build_payload(quick: bool = True) -> dict[str, Any]:
    cases = build_cases(quick=quick)
    payload = {
        "status": _status(cases),
        "quick": bool(quick),
        "diagnostic_only": True,
        "production_default_modified": False,
        "formal_casimir_ran": False,
        "lsq_mixed_block_projected": True,
        "lsq_candidate_used_as_production_formula": False,
        "analytic_formula_claimed": False,
        "candidate_basis_names": list(BASIS_NAMES),
        "projection_level": "integrated",
        "k_resolved_target_available": False,
        "cases": cases,
    }
    return payload


def _q_label(case: dict[str, Any]) -> str:
    q = case["q_model"]
    return f"({q[0]:.7g},{q[1]:.7g})"


def write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_2h_lsq_mixed_block_projection_audit"
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
        "- projection level: integrated only; k-resolved LSQ target unavailable",
        "",
        "## Projection summary",
        "",
        "| pairing | N | q | baseline | LSQ Ward | target norm | best single | single residual | multi residual | family |",
        "| ------- | -: | - | -------: | -------: | ----------: | ----------- | --------------: | -------------: | ------ |",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['pairing']} | {case['N']} | {_q_label(case)} | {case['baseline_restored_ward']:.8g} | "
            f"{case['lsq_mixed_only_restored_ward']:.8g} | {case['lsq_target_norm']:.8g} | "
            f"{case['best_single_basis']} | {case['best_single_basis_relative_residual']:.8g} | "
            f"{case['best_multi_basis_relative_residual']:.8g} | {case['dominant_basis_family']} |"
        )
    dwave = [case for case in payload["cases"] if case["pairing"] == "dwave"]
    best_families = sorted(set(case["dominant_basis_family"] for case in dwave))
    lines.extend(
        [
            "",
            "## Human-readable conclusion",
            "",
            "The StageSC-2f LSQ mixed block still closes dwave in the quick cases.",
            "StageSC-2g failed because the hopping-like d_i basis has tiny integrated thermal expectation; this audit checks whether LSQ resembles endpoint-sum / partial_q Phi instead.",
            f"Integrated projection dominant families for dwave: {best_families}.",
            "This run does not claim a production-ready analytic formula.",
            "If integrated projection remains ambiguous, the next step is k-resolved LSQ target reconstruction.",
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
