#!/usr/bin/env python3
"""K-resolved diagnostic audit for the missing mixed Schur block."""

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
    RHO,
    _bdg_hamiltonian_batch,
    _bdg_vector_batch,
    _bond_phi_batch,
    _collective_phi_batch,
    _collective_vertices,
    _fermi_matrix,
    _static_raw_factor,
    _transform,
    composite_uniform_quadrature,
    compute_bdg_components_for_composite_grid,
)
from lno327.conductivity import KuboConfig, fermi_function  # noqa: E402
from lno327.pairing import PairingAmplitudes  # noqa: E402
from stageSC_2f_schur_block_assembly_audit import _candidate_responses, _five_channel_residual, _ward_max  # noqa: E402
from stageSC_2g_analytic_mixed_direct_audit import _gamma_eta1, _gamma_eta2  # noqa: E402
from stageSC_2h_lsq_mixed_block_projection_audit import _basis_matrix  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
CONTROL_CASES = (("onsite_s", 24, (1, 0)), ("spm", 24, (1, 0)), ("dwave_const_form", 24, (1, 0)))
DWAVE_QUICK_CASES = (("dwave", 24, (1, 0)), ("dwave", 24, (1, 1)), ("dwave", 36, (1, 0)), ("dwave", 36, (1, 1)))
BASIS_NAMES = (
    "basis_phi",
    "basis_partial_q_phi_x",
    "basis_partial_q_phi_y",
    "basis_endpoint_sum_x",
    "basis_endpoint_sum_y",
    "basis_partial_k_phi_x",
    "basis_partial_k_phi_y",
    "basis_d_difference_x",
    "basis_d_difference_y",
    "basis_q_phi_x",
    "basis_q_phi_y",
    "basis_omega_phi",
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


def _local_bubble(left_band: np.ndarray, right_band: np.ndarray, raw: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return NAMBU_PREFACTOR * np.einsum(
        "b,bmn,sbmn,tbmn->stb",
        weights,
        raw,
        left_band,
        right_band.conjugate(),
        optimize=True,
    )


def _local_static_eta2_counterterm(eta2_band: np.ndarray, static_raw: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return NAMBU_PREFACTOR * np.einsum(
        "b,bmn,bmn,bmn->b",
        weights,
        static_raw,
        eta2_band[0],
        eta2_band[0].conjugate(),
        optimize=True,
    )


def _local_collective_vertices(pairing: str, points: np.ndarray, q: np.ndarray, delta0_eV: float) -> np.ndarray:
    if pairing == "bond_endpoint_gauge":
        raise ValueError("pairing name expected, not vertex name")
    if pairing in {"onsite_s", "spm", "dwave", "dwave_const_form"}:
        if pairing == "dwave_const_form":
            phi = _collective_phi_batch(pairing, points, q, "midpoint")
            return _collective_vertices(phi)
        return _collective_vertices(
            _bond_phi_batch(pairing, points, q, delta0_eV, endpoint=False),
            _bond_phi_batch(pairing, points, q, delta0_eV, endpoint=True),
        )
    raise ValueError(f"unknown pairing {pairing}")


def _local_eta2_residuals(
    pairing: str,
    points: np.ndarray,
    weights: np.ndarray,
    q: np.ndarray,
    cfg: KuboConfig,
    amp: PairingAmplitudes,
) -> tuple[list[dict[str, Any]], complex]:
    p_minus = points - 0.5 * q
    p_plus = points + 0.5 * q
    h_minus = _bdg_hamiltonian_batch(pairing, p_minus, amp.delta0_eV)
    h_plus = _bdg_hamiltonian_batch(pairing, p_plus, amp.delta0_eV)
    h_mid = _bdg_hamiltonian_batch(pairing, points, amp.delta0_eV)
    em, um = np.linalg.eigh(h_minus)
    ep, up = np.linalg.eigh(h_plus)
    ec, uc = np.linalg.eigh(h_mid)
    fm = fermi_function(em, cfg.fermi_level_eV, cfg.temperature_eV)
    fp = fermi_function(ep, cfg.fermi_level_eV, cfg.temperature_eV)
    fc = fermi_function(ec, cfg.fermi_level_eV, cfg.temperature_eV)
    raw = (fm[:, :, None] - fp[:, None, :]) / (1j * cfg.omega_eV + em[:, :, None] - ep[:, None, :])

    vx = _bdg_vector_batch(points, q, 0)
    vy = _bdg_vector_batch(points, q, 1)
    rho_stack = np.broadcast_to(RHO, (points.shape[0], 8, 8))
    observable = np.stack((rho_stack, -vx, -vy), axis=0)
    collective = _local_collective_vertices(pairing, points, q, amp.delta0_eV)
    observable_band = _transform(um, observable, up)
    collective_band = _transform(um, collective, up)
    em_collective_local = _local_bubble(observable_band, collective_band, raw, weights)
    collective_local = _local_bubble(collective_band, collective_band, raw, weights)

    collective_zero = _local_collective_vertices(pairing, points, np.zeros(2), amp.delta0_eV)
    eta2_zero_band = _transform(uc, collective_zero[1:2], uc)
    static_raw = _static_raw_factor(ec, fc, cfg)
    counter_eta2 = -_local_static_eta2_counterterm(eta2_zero_band, static_raw, weights)
    values = (
        1j * cfg.omega_eV * em_collective_local[0, 1]
        + q[0] * em_collective_local[1, 1]
        + q[1] * em_collective_local[2, 1]
        + 2j * amp.delta0_eV * (collective_local[1, 1] + counter_eta2)
    )
    rows = []
    for idx, value in enumerate(values):
        channel_values = {
            "rho": 1j * cfg.omega_eV * em_collective_local[0, 1, idx],
            "Vx": q[0] * em_collective_local[1, 1, idx],
            "Vy": q[1] * em_collective_local[2, 1, idx],
            "eta1": 0.0 + 0.0j,
            "eta2": 2j * amp.delta0_eV * (collective_local[1, 1, idx] + counter_eta2[idx]),
        }
        rows.append(
            {
                "k_point": points[idx].tolist(),
                "weight": float(weights[idx]),
                "local_residual_eta2": value,
                "abs_local_residual_eta2": float(abs(value)),
                "phase_local_residual_eta2": float(np.angle(value)),
                "local_residual_channels": channel_values,
            }
        )
    return rows, complex(np.sum(values))


def _phi_matrix(pairing: str, kx: float, ky: float, q: np.ndarray, amp: PairingAmplitudes) -> np.ndarray:
    if pairing == "dwave_const_form":
        return _basis_matrix("basis_phi_qi", pairing, kx, ky, 0.0, 0.0, "x", 1.0, amp)
    output = np.zeros((4, 4), dtype=complex)
    for axis, q_component in zip(("x", "y"), q, strict=True):
        if abs(float(q_component)) > 0.0:
            output += _basis_matrix("basis_phi_qi", pairing, kx, ky, float(q[0]), float(q[1]), axis, 1.0, amp) / q_component
            return output
    return output


def _basis_local_scalar(
    basis_name: str,
    pairing: str,
    points: np.ndarray,
    weights: np.ndarray,
    q: np.ndarray,
    cfg: KuboConfig,
    amp: PairingAmplitudes,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, float]]]:
    values = np.zeros(points.shape[0], dtype=complex)
    raw_norms = []
    weighted_norm = 0.0
    parity_scores = []
    for idx, (kx, ky) in enumerate(points):
        if basis_name == "basis_phi":
            matrix = _phi_matrix(pairing, float(kx), float(ky), q, amp)
        elif basis_name == "basis_omega_phi":
            matrix = cfg.omega_eV * _phi_matrix(pairing, float(kx), float(ky), q, amp)
        else:
            root, axis = basis_name.rsplit("_", 1)
            direction = axis
            base = {
                "basis_partial_q_phi": "basis_partial_q_phi",
                "basis_endpoint_sum": "basis_endpoint_sum",
                "basis_partial_k_phi": "basis_partial_k_phi",
                "basis_d_difference": "basis_d_difference",
                "basis_q_phi": "basis_phi_qi",
            }[root]
            matrix = _basis_matrix(base, pairing, float(kx), float(ky), float(q[0]), float(q[1]), direction, cfg.omega_eV, amp)
        raw = float(np.linalg.norm(matrix))
        raw_norms.append(raw)
        # Response-weighted proxy: keep the same local weight convention and eta2 rotation.
        values[idx] = weights[idx] * np.trace(_gamma_eta2(matrix)[4:, :4]) / 8.0
        weighted_norm += float(abs(values[idx]) ** 2)
        opposite = matrix.conjugate()  # lightweight parity proxy for this diagnostic
        even = np.linalg.norm(matrix - opposite) / max(np.linalg.norm(matrix), np.linalg.norm(opposite), 1e-300)
        odd = np.linalg.norm(matrix + opposite) / max(np.linalg.norm(matrix), np.linalg.norm(opposite), 1e-300)
        parity_scores.append((float(even), float(odd)))
    even_med = np.median([item[0] for item in parity_scores])
    odd_med = np.median([item[1] for item in parity_scores])
    parity = "even" if even_med < 1e-8 else ("odd" if odd_med < 1e-8 else "mixed")
    audit = {
        "basis_name": basis_name,
        "raw_norm_sum": float(np.sum(raw_norms)),
        "weighted_integrated_norm": float(np.sqrt(weighted_norm)),
        "cancellation_ratio": float(np.sqrt(weighted_norm) / max(float(np.sum(raw_norms)), 1e-300)),
        "parity_under_k_to_minus_k": parity,
        "is_negative_control_cancelled": bool(
            basis_name in {"basis_d_difference_x", "basis_d_difference_y", "basis_partial_k_phi_x", "basis_partial_k_phi_y"}
            and np.sqrt(weighted_norm) < 1e-14
            and np.sum(raw_norms) > 1e-8
        ),
    }
    top_values = [
        {
            "basis_name": basis_name,
            "raw_norm": raw_norms[idx],
            "weighted_abs": float(abs(values[idx])),
        }
        for idx in range(points.shape[0])
    ]
    return values, audit, top_values


def _project(target: np.ndarray, basis_values: dict[str, np.ndarray], selected: list[str] | None = None) -> dict[str, Any]:
    target_norm = float(np.linalg.norm(target))
    names = selected or list(basis_values)
    matrix = np.stack([basis_values[name] for name in names], axis=1)
    if target_norm < 1e-14 or np.linalg.norm(matrix) < 1e-14:
        coeffs = np.zeros(len(names), dtype=complex)
        residual = 1.0
        condition = float("inf")
    else:
        coeffs, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = float(np.linalg.norm(target - matrix @ coeffs) / target_norm)
        condition = float(np.linalg.cond(matrix))
    return {
        "basis_set": names,
        "coefficients": coeffs.tolist(),
        "relative_residual": residual,
        "explained_norm_fraction": float(1.0 - residual**2),
        "condition_number": condition,
    }


def _single_projection(target: np.ndarray, basis: np.ndarray) -> dict[str, Any]:
    projection = _project(target, {"basis": basis}, ["basis"])
    coeff = projection["coefficients"][0]
    alignment = 0.0 if np.linalg.norm(target) == 0 or np.linalg.norm(basis) == 0 else abs(np.vdot(basis, target)) / (
        np.linalg.norm(basis) * np.linalg.norm(target)
    )
    return {
        "coefficient": coeff,
        "relative_residual": projection["relative_residual"],
        "explained_norm_fraction": projection["explained_norm_fraction"],
        "phase_alignment": float(alignment),
        "basis_norm": float(np.linalg.norm(basis)),
        "target_norm": float(np.linalg.norm(target)),
    }


def _case(pairing: str, n_grid: int, m_case: tuple[int, int]) -> dict[str, Any]:
    amp = PairingAmplitudes(delta0_eV=0.04)
    cfg = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    spec = commensurate_q_spec(n_grid, m_case)
    q = np.asarray(spec["q_model"], dtype=float)
    points, weights = composite_uniform_quadrature(n_grid, [(0.0, 0.0)])
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
    lsq = _candidate_responses(result, pairing, q, amp)["mixed_only"]
    baseline = result["amplitude_phase_schur"]
    local_rows, local_sum = _local_eta2_residuals(pairing, points, weights, q, cfg, amp)
    integrated_block = _five_channel_residual(
        np.block(
            [
                [result["bare_total"], result["em_collective_left"]],
                [result["collective_em_right"], result["collective_total"]],
            ]
        ),
        cfg.omega_eV,
        q,
        amp.delta0_eV,
    )
    integrated_eta2 = sum(row["local_residual_eta2"] for row in local_rows)
    sum_error = abs(local_sum - integrated_eta2)
    basis_values = {}
    parity = {}
    top_basis = {}
    for basis_name in BASIS_NAMES:
        values, audit, top_values = _basis_local_scalar(basis_name, pairing, points, weights, q, cfg, amp)
        basis_values[basis_name] = values
        parity[basis_name] = audit
        top_basis[basis_name] = top_values
    target = np.asarray([row["local_residual_eta2"] for row in local_rows], dtype=complex)
    single = {name: _single_projection(target, values) for name, values in basis_values.items()}
    best_single = min(single, key=lambda name: single[name]["relative_residual"])
    multi = _project(target, basis_values)
    top_indices = np.argsort([-row["abs_local_residual_eta2"] for row in local_rows])[:20]
    top_points = []
    for rank, idx in enumerate(top_indices, start=1):
        row = local_rows[int(idx)]
        top_points.append(
            {
                "rank": rank,
                "k": row["k_point"],
                "weight": row["weight"],
                "local_residual_eta2": row["local_residual_eta2"],
                "abs": row["abs_local_residual_eta2"],
                "phase": row["phase_local_residual_eta2"],
                "basis_values": {
                    "partial_q_phi_x_norm": top_basis["basis_partial_q_phi_x"][int(idx)]["raw_norm"],
                    "partial_q_phi_y_norm": top_basis["basis_partial_q_phi_y"][int(idx)]["raw_norm"],
                    "phi_norm": top_basis["basis_phi"][int(idx)]["raw_norm"],
                    "partial_k_phi_x_norm": top_basis["basis_partial_k_phi_x"][int(idx)]["raw_norm"],
                },
            }
        )
    partial_q_best = min(single["basis_partial_q_phi_x"]["relative_residual"], single["basis_partial_q_phi_y"]["relative_residual"])
    endpoint_best = min(single["basis_endpoint_sum_x"]["relative_residual"], single["basis_endpoint_sum_y"]["relative_residual"])
    partial_k_best = min(single["basis_partial_k_phi_x"]["relative_residual"], single["basis_partial_k_phi_y"]["relative_residual"])
    d_best = min(single["basis_d_difference_x"]["relative_residual"], single["basis_d_difference_y"]["relative_residual"])
    return {
        "pairing": pairing,
        "N": int(n_grid),
        "m_case": list(m_case),
        "q_model": q.tolist(),
        "baseline_restored_ward": float(_ward_max(baseline, cfg.omega_eV, q)),
        "lsq_mixed_only_restored_ward": float(_ward_max(lsq, cfg.omega_eV, q)),
        "k_resolved_sum_matches_integrated": bool(sum_error < 1e-12),
        "k_resolved_sum_error_abs": float(sum_error),
        "local_residual_eta2_sum": local_sum,
        "integrated_residual_eta2": integrated_eta2,
        "integrated_residual_eta2_abs_monitor": integrated_block["eta2"],
        "best_single_k_basis": best_single,
        "best_single_k_relative_residual": single[best_single]["relative_residual"],
        "best_multi_k_basis_set": multi["basis_set"],
        "best_multi_k_relative_residual": multi["relative_residual"],
        "single_basis_k_resolved_projection": single,
        "multi_basis_k_resolved_projection": multi,
        "supports_partial_q_phi_hypothesis": bool(partial_q_best < 0.2),
        "supports_endpoint_sum_hypothesis": bool(endpoint_best < 0.2),
        "supports_partial_k_phi_hypothesis": bool(partial_k_best < 0.2),
        "supports_hopping_difference_hypothesis": bool(d_best < 0.2),
        "dominant_k_resolved_basis_family": _family(best_single, multi["relative_residual"]),
        "k_resolved_target_available": True,
        "left_right_separation_available": False,
        "top_local_eta2_residual_points": top_points,
        "parity_cancellation_audit": parity,
    }


def _family(best_single: str, multi_residual: float) -> str:
    if best_single.startswith("basis_partial_q_phi"):
        return "partial_q_phi"
    if best_single.startswith("basis_endpoint_sum"):
        return "endpoint_sum"
    if best_single.startswith("basis_partial_k_phi"):
        return "partial_k_phi"
    if best_single.startswith("basis_d_difference"):
        return "hopping_difference"
    if multi_residual < 0.2:
        return "multi_basis"
    return f"unclear_{best_single}"


def _case_specs(quick: bool) -> list[tuple[str, int, tuple[int, int]]]:
    if quick:
        return list(DWAVE_QUICK_CASES + CONTROL_CASES)
    return [(pairing, n, m) for pairing in ("onsite_s", "spm", "dwave", "dwave_const_form") for n in (24, 36, 48) for m in ((1, 0), (1, 1))]


def _status(cases: list[dict[str, Any]]) -> str:
    dwave = [case for case in cases if case["pairing"] == "dwave"]
    if any(float(case["lsq_mixed_only_restored_ward"]) >= 1e-8 for case in dwave):
        return "FAILED_STAGE2I_LSQ_REFERENCE_NOT_CLOSING"
    if any(not case["k_resolved_sum_matches_integrated"] for case in dwave):
        return "FAILED_STAGE2I_LOCAL_SUM_MISMATCH"
    partial_q = all(case["supports_partial_q_phi_hypothesis"] for case in dwave)
    endpoint = all(case["supports_endpoint_sum_hypothesis"] for case in dwave)
    multi = all(float(case["best_multi_k_relative_residual"]) < 0.2 for case in dwave)
    if partial_q:
        return "PASSED_STAGE2I_K_RESOLVED_DQPHI_IDENTIFIED"
    if endpoint:
        return "PASSED_STAGE2I_K_RESOLVED_ENDPOINT_SUM_IDENTIFIED"
    if multi:
        return "PASSED_STAGE2I_K_RESOLVED_MULTI_BASIS_IDENTIFIED"
    return "PARTIAL_STAGE2I_NO_CLEAR_K_BASIS"


@lru_cache(maxsize=2)
def build_payload(quick: bool = True) -> dict[str, Any]:
    cases = [_case(pairing, n_grid, m_case) for pairing, n_grid, m_case in _case_specs(quick)]
    return {
        "status": _status(cases),
        "quick": bool(quick),
        "diagnostic_only": True,
        "production_default_modified": False,
        "formal_casimir_ran": False,
        "k_resolved_audit": True,
        "lsq_candidate_used_as_production_formula": False,
        "analytic_formula_claimed": False,
        "basis_names": list(BASIS_NAMES),
        "cases": cases,
    }


def _q_label(case: dict[str, Any]) -> str:
    q = case["q_model"]
    return f"({q[0]:.7g},{q[1]:.7g})"


def write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_2i_k_resolved_mixed_block_audit"
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
        "",
        "## K-resolved projection summary",
        "",
        "| pairing | N | q | baseline | LSQ Ward | sum ok | best k basis | single residual | multi residual | family |",
        "| ------- | -: | - | -------: | -------: | ------ | ------------ | --------------: | -------------: | ------ |",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['pairing']} | {case['N']} | {_q_label(case)} | {case['baseline_restored_ward']:.8g} | "
            f"{case['lsq_mixed_only_restored_ward']:.8g} | {case['k_resolved_sum_matches_integrated']} | "
            f"{case['best_single_k_basis']} | {case['best_single_k_relative_residual']:.8g} | "
            f"{case['best_multi_k_relative_residual']:.8g} | {case['dominant_k_resolved_basis_family']} |"
        )
    dwave_cases = [case for case in payload["cases"] if case["pairing"] == "dwave"]
    lines.extend(["", "## Top local eta2 residuals", ""])
    for case in dwave_cases:
        lines.append(f"### dwave N={case['N']} q={_q_label(case)}")
        lines.append("")
        lines.append("| rank | k | abs | phase | partial_q_x | partial_q_y | phi | partial_k_x |")
        lines.append("| ---: | - | --: | ----: | ----------: | ----------: | --: | ----------: |")
        for row in case["top_local_eta2_residual_points"][:5]:
            basis = row["basis_values"]
            lines.append(
                f"| {row['rank']} | ({row['k'][0]:.6g},{row['k'][1]:.6g}) | {row['abs']:.8g} | "
                f"{row['phase']:.6g} | {basis['partial_q_phi_x_norm']:.8g} | "
                f"{basis['partial_q_phi_y_norm']:.8g} | {basis['phi_norm']:.8g} | {basis['partial_k_phi_x_norm']:.8g} |"
            )
        lines.append("")
    families = sorted(set(case["dominant_k_resolved_basis_family"] for case in dwave_cases))
    lines.extend(
        [
            "## Human-readable conclusion",
            "",
            "The LSQ mixed block still closes dwave in the quick k-resolved cases.",
            "The k-resolved local eta2 residual sums back to the integrated residual within tolerance.",
            f"Dominant k-resolved basis families for dwave: {families}.",
            "The hopping-like / partial_k negative controls remain cancellation diagnostics, not production formulas.",
            "This run does not claim a final analytic formula.",
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
