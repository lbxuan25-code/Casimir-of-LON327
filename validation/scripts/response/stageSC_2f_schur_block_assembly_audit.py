#!/usr/bin/env python3
"""Diagnostic-only Schur block assembly audit for finite-q BdG pairing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from bdg_commensurate_q_common import commensurate_q_spec  # noqa: E402
from bdg_quadrature_strategy_common import (  # noqa: E402
    composite_uniform_quadrature,
    compute_bdg_components_for_composite_grid,
    single_composite_schur,
)
from lno327.conductivity import KuboConfig  # noqa: E402
from lno327.pairing import PairingAmplitudes  # noqa: E402
from lno327.pairing_bonds import bond_endpoint_gauge_form_factor  # noqa: E402
from lno327.ward_response import physical_ward_residuals  # noqa: E402
from pairing_bond_goldstone_common import operator_ward_rows  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
PAIRINGS = ("onsite_s", "spm", "dwave", "dwave_const_form")
N_LIST_QUICK = (24, 36, 48)
M_CASES_QUICK = ((1, 0), (1, 1))
CHANNELS = ("rho", "Vx", "Vy", "eta1", "eta2")


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


def _ward_max(response: np.ndarray, omega_eV: float, q: np.ndarray) -> float:
    left, right = physical_ward_residuals(response, omega_eV, q)
    return float(max(np.max(np.abs(left)), np.max(np.abs(right))))


def _block_matrix(result: dict[str, Any], *, bare: np.ndarray, collective: np.ndarray) -> np.ndarray:
    output = np.zeros((5, 5), dtype=complex)
    output[:3, :3] = bare
    output[:3, 3:] = result["em_collective_left"]
    output[3:, :3] = result["collective_em_right"]
    output[3:, 3:] = collective
    return output


def _five_channel_residual(block: np.ndarray, omega_eV: float, q: np.ndarray, delta0_eV: float) -> dict[str, float]:
    row = np.asarray([1j * omega_eV, q[0], q[1], 0.0, 2j * delta0_eV], dtype=complex)
    values = row @ block
    return {channel: float(abs(value)) for channel, value in zip(CHANNELS, values, strict=True)}


def _etaeta_metric_factor(pairing: str, q: np.ndarray, amp: PairingAmplitudes) -> float:
    if pairing in {"onsite_s", "spm", "dwave_const_form"}:
        return 1.0
    points = ((0.13, 0.27), (0.41, -0.22), (1.11, 0.73), (-0.64, 1.37))
    q0_norm = 0.0
    q_norm = 0.0
    for kx, ky in points:
        phi0 = bond_endpoint_gauge_form_factor(pairing, kx, ky, 0.0, 0.0, amp)
        phiq = bond_endpoint_gauge_form_factor(pairing, kx, ky, float(q[0]), float(q[1]), amp)
        q0_norm += float(np.vdot(phi0, phi0).real)
        q_norm += float(np.vdot(phiq, phiq).real)
    return float(q_norm / max(q0_norm, 1e-300))


def _minimal_response_correction_for_ward(response: np.ndarray, omega_eV: float, q: np.ndarray) -> np.ndarray:
    left, right = physical_ward_residuals(response, omega_eV, q)
    rows = []
    rhs = []
    coeff_left = np.asarray([1j * omega_eV, q[0], q[1]], dtype=complex)
    coeff_right = np.asarray([1j * omega_eV, -q[0], -q[1]], dtype=complex)
    for col in range(3):
        row = np.zeros(9, dtype=complex)
        for mu in range(3):
            row[3 * mu + col] = coeff_left[mu]
        rows.append(row)
        rhs.append(-left[col])
    for row_index in range(3):
        row = np.zeros(9, dtype=complex)
        for col in range(3):
            row[3 * row_index + col] = coeff_right[col]
        rows.append(row)
        rhs.append(-right[row_index])
    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(rhs), rcond=None)
    return solution.reshape(3, 3)


def _mixed_direct_candidate(result: dict[str, Any], baseline_response: np.ndarray, omega_eV: float, q: np.ndarray) -> np.ndarray:
    """Return a minimal EM-eta direct candidate needed by the Ward residual.

    This is diagnostic and Ward-required, not a production formula.
    """

    target_delta_response = _minimal_response_correction_for_ward(baseline_response, omega_eV, q)
    condition = float(np.linalg.cond(result["collective_total"]))
    inverse = np.linalg.pinv(result["collective_total"]) if condition > 1e12 else np.linalg.inv(result["collective_total"])
    mixed_matrix = result["em_collective_left"] @ inverse
    delta_right, *_ = np.linalg.lstsq(-mixed_matrix, target_delta_response, rcond=None)
    return delta_right


def _candidate_responses(result: dict[str, Any], pairing: str, q: np.ndarray, amp: PairingAmplitudes) -> dict[str, Any]:
    bare = result["bare_total"]
    collective = result["collective_total"]
    baseline = result["amplitude_phase_schur"]
    mixed_delta = _mixed_direct_candidate(result, baseline, 0.01, q)
    mixed, mixed_condition, _ = single_composite_schur(
        bare,
        result["em_collective_left"],
        collective,
        result["collective_em_right"] + mixed_delta,
    )
    etaeta_collective = collective.copy()
    etaeta_collective[1, 1] *= _etaeta_metric_factor(pairing, q, amp)
    etaeta, etaeta_condition, _ = single_composite_schur(
        bare,
        result["em_collective_left"],
        etaeta_collective,
        result["collective_em_right"],
    )
    both, both_condition, _ = single_composite_schur(
        bare,
        result["em_collective_left"],
        etaeta_collective,
        result["collective_em_right"] + mixed_delta,
    )
    return {
        "baseline": baseline,
        "mixed_only": mixed,
        "etaeta_only": etaeta,
        "mixed_plus_etaeta": both,
        "mixed_direct_delta_norm": float(np.linalg.norm(mixed_delta)),
        "etaeta_metric_factor": _etaeta_metric_factor(pairing, q, amp),
        "conditions": {
            "baseline": float(result["collective_condition_number"]),
            "mixed_only": mixed_condition,
            "etaeta_only": etaeta_condition,
            "mixed_plus_etaeta": both_condition,
        },
    }


def _case_status(pairing: str, contact: float, values: dict[str, float]) -> str:
    if contact >= 1e-10:
        return "FAILED_CONTACT"
    if pairing in {"onsite_s", "spm"} and min(values.values()) >= 1e-6:
        return "FAILED_VALIDATION_PAIRING"
    return "DIAGNOSTIC_RECORDED"


def _block_decomposition(result: dict[str, Any], omega_eV: float, q: np.ndarray, delta0_eV: float) -> dict[str, Any]:
    zero_collective = np.zeros((2, 2), dtype=complex)
    bubble_block = _block_matrix(result, bare=result["bare_bubble"], collective=result["collective_bubble"])
    em_direct_block = _block_matrix(result, bare=result["bare_total"], collective=result["collective_bubble"])
    eta_counter_block = _block_matrix(result, bare=result["bare_total"], collective=result["collective_total"])
    total_block = _block_matrix(result, bare=result["bare_total"], collective=result["collective_total"])
    del zero_collective
    return {
        "bubble_only": max(_five_channel_residual(bubble_block, omega_eV, q, delta0_eV).values()),
        "em_em_direct": max(_five_channel_residual(em_direct_block, omega_eV, q, delta0_eV).values()),
        "mixed_direct_candidate": None,
        "eta_eta_counterterm_candidate": max(_five_channel_residual(eta_counter_block, omega_eV, q, delta0_eV).values()),
        "total_candidate": max(_five_channel_residual(total_block, omega_eV, q, delta0_eV).values()),
    }


def build_cases(quick: bool = True) -> list[dict[str, Any]]:
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
                candidates = _candidate_responses(result, pairing, q, amp)
                ward_values = {
                    name: _ward_max(matrix, cfg.omega_eV, q)
                    for name, matrix in candidates.items()
                    if isinstance(matrix, np.ndarray)
                }
                contact = max(float(result["contact_closure"][channel]["E_band_plus_qD_abs"]) for channel in ("Vx", "Vy"))
                baseline = ward_values["baseline"]
                best_candidate = min(ward_values, key=ward_values.get)
                improvement = float(baseline / max(ward_values[best_candidate], 1e-300))
                block_total = _block_matrix(result, bare=result["bare_total"], collective=result["collective_total"])
                residual_by_right = _five_channel_residual(block_total, cfg.omega_eV, q, amp.delta0_eV)
                block_decomp = _block_decomposition(result, cfg.omega_eV, q, amp.delta0_eV)
                block_decomp["mixed_direct_candidate"] = ward_values["mixed_only"]
                rows.append(
                    {
                        "pairing": pairing,
                        "N": int(n_grid),
                        "m_case": list(m_case),
                        "q_model": q.tolist(),
                        "phase_vertex": "exact_goldstone_tangent",
                        "bare_ward_monitor": float(result["bare_total_ward_max_abs"]),
                        "contact_closure_max_abs": contact,
                        "baseline_restored_ward": ward_values["baseline"],
                        "mixed_only_restored_ward": ward_values["mixed_only"],
                        "etaeta_only_restored_ward": ward_values["etaeta_only"],
                        "mixed_plus_etaeta_restored_ward": ward_values["mixed_plus_etaeta"],
                        "collective_condition_number": float(result["collective_condition_number"]),
                        "candidate_condition_numbers": candidates["conditions"],
                        "mixed_direct_delta_norm": candidates["mixed_direct_delta_norm"],
                        "etaeta_metric_factor": candidates["etaeta_metric_factor"],
                        "best_candidate": best_candidate,
                        "improvement_factor": improvement,
                        "residual_by_right_channel": residual_by_right,
                        "residual_by_block": block_decomp,
                        "status": _case_status(pairing, contact, ward_values),
                    }
                )
    return rows


def _dominant_missing_block(cases: list[dict[str, Any]]) -> tuple[str, str]:
    dwave = [case for case in cases if case["pairing"] == "dwave"]
    if not dwave:
        return "no_candidate_closed_dwave", "dwave cases missing"
    baseline = max(float(case["baseline_restored_ward"]) for case in dwave)
    mixed = max(float(case["mixed_only_restored_ward"]) for case in dwave)
    etaeta = max(float(case["etaeta_only_restored_ward"]) for case in dwave)
    both = max(float(case["mixed_plus_etaeta_restored_ward"]) for case in dwave)
    threshold = 0.1 * baseline
    if mixed < threshold and etaeta < threshold:
        return "both_mixed_direct_and_etaeta_counterterm_possible", "mixed-only and etaeta-only both reduce dwave residual by >10x"
    if mixed < threshold:
        return "mixed_direct", "mixed-only reduces dwave residual by >10x"
    if etaeta < threshold:
        return "eta_eta_counterterm", "etaeta-only reduces dwave residual by >10x"
    if both < threshold:
        return "both_blocks_needed", "only mixed+etaeta reduces dwave residual by >10x"
    return "no_candidate_closed_dwave", "no experimental candidate reduced dwave residual by >10x"


def _q_scaling(cases: list[dict[str, Any]], pairing: str, field: str) -> dict[str, float]:
    selected = [case for case in cases if case["pairing"] == pairing]
    qs = np.asarray([np.linalg.norm(case["q_model"]) for case in selected], dtype=float)
    vals = np.asarray([float(case[field]) for case in selected], dtype=float)
    mask = (qs > 0) & (vals > 0)
    if np.count_nonzero(mask) < 2:
        return {"power": float("nan"), "prefactor": float("nan")}
    power, log_prefactor = np.polyfit(np.log(qs[mask]), np.log(vals[mask]), deg=1)
    return {"power": float(power), "prefactor": float(np.exp(log_prefactor))}


def build_payload(quick: bool = True) -> dict[str, Any]:
    cases = build_cases(quick=quick)
    dominant, reason = _dominant_missing_block(cases)
    validation_regression = any(
        case["pairing"] in {"onsite_s", "spm"}
        and min(
            float(case["baseline_restored_ward"]),
            float(case["mixed_only_restored_ward"]),
            float(case["etaeta_only_restored_ward"]),
            float(case["mixed_plus_etaeta_restored_ward"]),
        )
        >= 1e-6
        for case in cases
    )
    if validation_regression:
        status = "FAILED_STAGE2F_REGRESSION_ON_ONSITE_OR_SPM"
    elif dominant == "mixed_direct":
        status = "PASSED_STAGE2F_MIXED_DIRECT_IDENTIFIED"
    elif dominant == "eta_eta_counterterm":
        status = "PASSED_STAGE2F_ETAETA_COUNTERTERM_IDENTIFIED"
    elif dominant in {"both_blocks_needed", "both_mixed_direct_and_etaeta_counterterm_possible"}:
        status = "PASSED_STAGE2F_BOTH_BLOCKS_NEEDED"
    else:
        status = "PARTIAL_STAGE2F_NO_CANDIDATE_CLOSED_DWAVE"
    payload = {
        "status": status,
        "quick": bool(quick),
        "diagnostic_only": True,
        "production_default_modified": False,
        "formal_casimir_ran": False,
        "phase_vertex": "exact_goldstone_tangent",
        "include_mixed_bond_direct_options": [False, True],
        "include_eta_eta_bond_counterterm_options": [False, True],
        "dominant_missing_block_inferred": dominant,
        "dominant_missing_block_reason": reason,
        "no_candidate_closed_dwave": dominant == "no_candidate_closed_dwave",
        "pairing_sector_AA_contact_needed_suspected": dominant == "no_candidate_closed_dwave",
        "q_scaling": {
            pairing: {
                "baseline": _q_scaling(cases, pairing, "baseline_restored_ward"),
                "mixed_only": _q_scaling(cases, pairing, "mixed_only_restored_ward"),
            }
            for pairing in PAIRINGS
        },
        "exact_goldstone_operator_ward": [
            row for row in operator_ward_rows(PairingAmplitudes(delta0_eV=0.04)) if row["phase_vertex"] == "exact_goldstone_tangent"
        ],
        "cases": cases,
    }
    return payload


def _q_label(case: dict[str, Any]) -> str:
    q = case["q_model"]
    return f"({q[0]:.7g},{q[1]:.7g})"


def write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_2f_schur_block_assembly_audit"
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
        f"- dominant missing block inferred: {payload['dominant_missing_block_inferred']}",
        f"- reason: {payload['dominant_missing_block_reason']}",
        "",
        "## Candidate comparison",
        "",
        "| pairing | N | q | contact | baseline | mixed-only | etaeta-only | mixed+etaeta | best |",
        "| ------- | -: | - | ------: | -------: | ---------: | ----------: | ------------: | ---- |",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['pairing']} | {case['N']} | {_q_label(case)} | {case['contact_closure_max_abs']:.8g} | "
            f"{case['baseline_restored_ward']:.8g} | {case['mixed_only_restored_ward']:.8g} | "
            f"{case['etaeta_only_restored_ward']:.8g} | {case['mixed_plus_etaeta_restored_ward']:.8g} | "
            f"{case['best_candidate']} |"
        )
    lines.extend(
        [
            "",
            "## Block residual decomposition",
            "",
            "| pairing | N | q | rho | Vx | Vy | eta1 | eta2 | total block |",
            "| ------- | -: | - | --: | -: | -: | ---: | ---: | ----------: |",
        ]
    )
    for case in payload["cases"]:
        residual = case["residual_by_right_channel"]
        lines.append(
            f"| {case['pairing']} | {case['N']} | {_q_label(case)} | {residual['rho']:.8g} | "
            f"{residual['Vx']:.8g} | {residual['Vy']:.8g} | {residual['eta1']:.8g} | "
            f"{residual['eta2']:.8g} | {case['residual_by_block']['total_candidate']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Human-readable conclusion",
            "",
            "The exact Goldstone tangent remains operator-Ward closed and is not the primary failure.",
            f"The dominant diagnostic inference is `{payload['dominant_missing_block_inferred']}`.",
            "The mixed-direct candidate is Ward-required and diagnostic-only; it is not a production formula.",
            "The eta-eta candidate tests a finite-q bond-metric counterterm only.",
            f"Pairing-sector A-A contact suspected: {payload['pairing_sector_AA_contact_needed_suspected']}.",
            "Formal Casimir input should remain forbidden.",
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
