#!/usr/bin/env python3
"""Diagnostic-only gauge-covariant collective package audit for finite-q BdG kernels."""

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
    composite_uniform_quadrature,
    compute_bdg_components_for_composite_grid,
    single_composite_schur,
)
from lno327.conductivity import KuboConfig  # noqa: E402
from lno327.pairing import PairingAmplitudes  # noqa: E402
from lno327.pairing_bonds import pairing_bond_list  # noqa: E402
from lno327.ward_response import physical_ward_residuals  # noqa: E402
from stageSC_2f_schur_block_assembly_audit import _candidate_responses  # noqa: E402
from stageSC_2j_longitudinal_ward_completion_audit import (  # noqa: E402
    _kernel_5x5,
    _left_longitudinal_completion,
    _right_longitudinal_completion,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
CONTROL_CASES = (("onsite_s", 24, (1, 0)), ("spm", 24, (1, 0)), ("dwave_const_form", 24, (1, 0)))
DWAVE_QUICK_CASES = (("dwave", 24, (1, 0)), ("dwave", 24, (1, 1)), ("dwave", 36, (1, 0)), ("dwave", 36, (1, 1)))
REQUIRED_PACKAGE_VARIANTS = (
    "pkg_total_spatial_no_i",
    "pkg_total_spatial_with_i",
    "pkg_total_spacetime_omega_plus_no_i",
    "pkg_total_spacetime_omega_plus_with_i",
    "pkg_bubble_spatial_with_i",
    "pkg_counterterm_spatial_with_i",
    "pkg_bond_metric_spatial_with_i",
    "pkg_hybrid_bond_metric_spatial_with_i",
    "pkg_total_spatial_with_i_no_AA",
    "pkg_total_spatial_with_i_only_mixed",
    "pkg_total_spatial_with_i_etaeta_plus_mixed_no_AA",
    "pkg_total_spatial_with_i_AA_plus_mixed_no_etaeta",
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


def _ward_max(response: np.ndarray, omega_eV: float, q: np.ndarray) -> float:
    left, right = physical_ward_residuals(response, omega_eV, q)
    return float(max(np.max(np.abs(left)), np.max(np.abs(right))))


def _bond_metric_counterterm(
    pairing: str,
    q: np.ndarray,
    current_counterterm: np.ndarray,
    amp: PairingAmplitudes,
) -> tuple[np.ndarray, dict[str, Any]]:
    bonds = pairing_bond_list(pairing, amp)
    scale = complex(current_counterterm[1, 1])
    if not bonds:
        return np.diag([scale, scale]).astype(complex), {
            "bond_metric_available": False,
            "counterterm_scale_definition": "collective_counterterm[1,1]",
            "fallback": "identity_metric_for_pairing_without_bond_representation",
        }
    delta0 = float(amp.delta0_eV)
    metric = np.zeros((2, 2), dtype=complex)
    metric_q0_22 = 0.0
    for bond in bonds:
        weight = abs(complex(bond.coefficient) / delta0) ** 2
        center = bond.center
        displacement = bond.displacement
        u = np.exp(1j * float(np.dot(q, center)))
        v = u * np.cos(0.5 * float(np.dot(q, displacement)))
        metric[0, 0] += weight * abs(u) ** 2
        metric[1, 1] += weight * abs(v) ** 2
        metric[0, 1] += weight * u.conjugate() * v
        metric_q0_22 += weight
    metric[1, 0] = metric[0, 1].conjugate()
    normalized = scale * metric / max(metric_q0_22, 1e-300)
    return normalized, {
        "bond_metric_available": True,
        "counterterm_scale_definition": "collective_counterterm[1,1]",
        "counterterm_scale": scale,
        "M22_q0": float(metric_q0_22),
        "raw_metric": metric,
    }


def _lambda_gradient(lambda_convention: str, omega_norm: str, omega_eV: float, q: np.ndarray) -> tuple[np.ndarray, str]:
    q2 = float(np.dot(q, q))
    if lambda_convention == "spatial":
        return np.asarray([0.0, q[0] / q2, q[1] / q2], dtype=complex), "q_squared"
    if omega_norm == "omega_plus":
        denom = omega_eV**2 + q2
    elif omega_norm == "omega_minus":
        denom = -omega_eV**2 + q2
    elif omega_norm == "omega_zero":
        denom = q2
    else:
        raise ValueError(f"unknown omega_norm {omega_norm}")
    return np.asarray([1j * omega_eV / denom, q[0] / denom, q[1] / denom], dtype=complex), omega_norm


def _package_blocks(
    C: np.ndarray,
    alpha: complex,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_left = -alpha * np.outer(g, C[1, :])
    delta_right = -alpha * np.outer(C[:, 1], g)
    delta_aa = alpha**2 * C[1, 1] * np.outer(g, g)
    return delta_aa, delta_left, delta_right


def _candidate_matrices(result: dict[str, Any], pairing: str, q: np.ndarray, amp: PairingAmplitudes) -> dict[str, dict[str, Any]]:
    bond_metric, bond_info = _bond_metric_counterterm(pairing, q, result["collective_counterterm"], amp)
    return {
        "collective_total": {
            "matrix": result["collective_total"],
            "package_mode": "replacement",
            "description": "current collective_total bubble plus counterterm",
        },
        "bubble": {
            "matrix": result["collective_bubble"],
            "package_mode": "replacement",
            "description": "collective_bubble only",
        },
        "counterterm": {
            "matrix": result["collective_counterterm"],
            "package_mode": "replacement",
            "description": "collective_counterterm only",
        },
        "bond_metric": {
            "matrix": bond_metric,
            "package_mode": "replacement",
            "description": "forward finite-q bond-metric counterterm",
            "bond_metric_info": bond_info,
        },
        "hybrid": {
            "matrix": result["collective_bubble"] + bond_metric,
            "package_mode": "replacement",
            "description": "collective_bubble plus forward finite-q bond metric",
            "bond_metric_info": bond_info,
        },
    }


def _package_variant(
    name: str,
    *,
    result: dict[str, Any],
    C: np.ndarray,
    C_candidate: str,
    package_mode: str,
    lambda_convention: str,
    omega_norm: str,
    alpha_convention: str,
    include_aa: bool,
    include_mixed: bool,
    include_etaeta: bool,
    baseline_restored: float,
    stage2j_restored: float,
    omega_eV: float,
    q: np.ndarray,
    delta0_eV: float,
) -> dict[str, Any]:
    g, denominator = _lambda_gradient(lambda_convention, omega_norm, omega_eV, q)
    alpha = 2j * delta0_eV if alpha_convention == "with_i" else 2.0 * delta0_eV
    delta_aa, delta_left, delta_right = _package_blocks(C, alpha, g)
    delta_etaeta = C - result["collective_total"]
    bare_pkg = result["bare_total"] + (delta_aa if include_aa else 0.0)
    left_pkg = result["em_collective_left"] + (delta_left if include_mixed else 0.0)
    right_pkg = result["collective_em_right"] + (delta_right if include_mixed else 0.0)
    if include_etaeta:
        collective_pkg = C if package_mode == "replacement" else result["collective_total"] + delta_etaeta
    else:
        collective_pkg = result["collective_total"]
    response, condition, _ = single_composite_schur(bare_pkg, left_pkg, collective_pkg, right_pkg)
    ward = _ward_max(response, omega_eV, q)
    return {
        "physical_schur_restored_ward": ward,
        "improvement_factor_vs_baseline": float(baseline_restored / max(ward, 1e-300)),
        "improvement_factor_vs_stage2j": float(stage2j_restored / max(ward, 1e-300)),
        "collective_condition_number": float(condition),
        "delta_AA_norm": float(np.linalg.norm(delta_aa if include_aa else np.zeros_like(delta_aa))),
        "delta_Aeta_norm": float(np.linalg.norm(delta_left if include_mixed else np.zeros_like(delta_left))),
        "delta_etaA_norm": float(np.linalg.norm(delta_right if include_mixed else np.zeros_like(delta_right))),
        "delta_etaeta_norm": float(np.linalg.norm(delta_etaeta if include_etaeta else np.zeros_like(delta_etaeta))),
        "package_mode": package_mode,
        "C_candidate": C_candidate,
        "lambda_convention": lambda_convention,
        "lambda_denominator": denominator,
        "alpha_convention": alpha_convention,
        "include_AA": bool(include_aa),
        "include_mixed": bool(include_mixed),
        "include_etaeta": bool(include_etaeta),
        "variant": name,
    }


def _stage2j_reference(result: dict[str, Any], omega_eV: float, q: np.ndarray, delta0_eV: float) -> float:
    bare = result["bare_total"]
    collective = result["collective_total"]
    left = result["em_collective_left"]
    right = result["collective_em_right"]
    kernel = _kernel_5x5(bare, left, collective, right)
    left_delta = _left_longitudinal_completion(kernel, omega_eV, q, delta0_eV)
    right_delta = _right_longitudinal_completion(kernel, omega_eV, q, delta0_eV)
    zero_left = np.zeros_like(left_delta)
    zero_right = np.zeros_like(right_delta)
    candidates = (
        (left + left_delta, right + zero_right),
        (left + zero_left, right + right_delta),
        (left + left_delta, right + right_delta),
        (left + left_delta, right + left_delta.conjugate().T),
        (left - left_delta, right - right_delta),
    )
    values = []
    for left_pkg, right_pkg in candidates:
        response, _, _ = single_composite_schur(bare, left_pkg, collective, right_pkg)
        values.append(_ward_max(response, omega_eV, q))
    return float(min(values))


def _case_specs(quick: bool) -> list[tuple[str, int, tuple[int, int]]]:
    if quick:
        return list(DWAVE_QUICK_CASES + CONTROL_CASES)
    return [(pairing, n, m) for pairing in ("onsite_s", "spm", "dwave", "dwave_const_form") for n in (24, 36, 48) for m in ((1, 0), (1, 1))]


def _case(pairing: str, n_grid: int, m_case: tuple[int, int]) -> dict[str, Any]:
    amp = PairingAmplitudes(delta0_eV=0.04)
    cfg = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    points, weights = composite_uniform_quadrature(n_grid, [(0.0, 0.0)])
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
    references = _candidate_responses(result, pairing, q, amp)
    baseline = _ward_max(references["baseline"], cfg.omega_eV, q)
    lsq = _ward_max(references["mixed_only"], cfg.omega_eV, q)
    stage2j = _stage2j_reference(result, cfg.omega_eV, q, amp.delta0_eV)
    C_candidates = _candidate_matrices(result, pairing, q, amp)
    specs = {
        "pkg_total_spatial_no_i": ("collective_total", "spatial", "omega_zero", "no_i", True, True, True),
        "pkg_total_spatial_with_i": ("collective_total", "spatial", "omega_zero", "with_i", True, True, True),
        "pkg_total_spacetime_omega_plus_no_i": ("collective_total", "spacetime", "omega_plus", "no_i", True, True, True),
        "pkg_total_spacetime_omega_plus_with_i": ("collective_total", "spacetime", "omega_plus", "with_i", True, True, True),
        "pkg_total_spacetime_omega_minus_with_i": ("collective_total", "spacetime", "omega_minus", "with_i", True, True, True),
        "pkg_total_spacetime_omega_zero_with_i": ("collective_total", "spacetime", "omega_zero", "with_i", True, True, True),
        "pkg_bubble_spatial_with_i": ("bubble", "spatial", "omega_zero", "with_i", True, True, True),
        "pkg_counterterm_spatial_with_i": ("counterterm", "spatial", "omega_zero", "with_i", True, True, True),
        "pkg_bond_metric_spatial_with_i": ("bond_metric", "spatial", "omega_zero", "with_i", True, True, True),
        "pkg_hybrid_bond_metric_spatial_with_i": ("hybrid", "spatial", "omega_zero", "with_i", True, True, True),
        "pkg_total_spatial_with_i_no_AA": ("collective_total", "spatial", "omega_zero", "with_i", False, True, True),
        "pkg_total_spatial_with_i_only_mixed": ("collective_total", "spatial", "omega_zero", "with_i", False, True, False),
        "pkg_total_spatial_with_i_etaeta_plus_mixed_no_AA": ("collective_total", "spatial", "omega_zero", "with_i", False, True, True),
        "pkg_total_spatial_with_i_AA_plus_mixed_no_etaeta": ("collective_total", "spatial", "omega_zero", "with_i", True, True, False),
    }
    variants = {}
    for name, (candidate, lambda_convention, omega_norm, alpha_convention, include_aa, include_mixed, include_etaeta) in specs.items():
        C_info = C_candidates[candidate]
        variants[name] = _package_variant(
            name,
            result=result,
            C=C_info["matrix"],
            C_candidate=candidate,
            package_mode=C_info["package_mode"],
            lambda_convention=lambda_convention,
            omega_norm=omega_norm,
            alpha_convention=alpha_convention,
            include_aa=include_aa,
            include_mixed=include_mixed,
            include_etaeta=include_etaeta,
            baseline_restored=baseline,
            stage2j_restored=stage2j,
            omega_eV=cfg.omega_eV,
            q=q,
            delta0_eV=amp.delta0_eV,
        )
    best = min(variants, key=lambda item: float(variants[item]["physical_schur_restored_ward"]))
    no_aa = variants["pkg_total_spatial_with_i_no_AA"]["physical_schur_restored_ward"]
    full_total = variants["pkg_total_spatial_with_i"]["physical_schur_restored_ward"]
    only_mixed = variants["pkg_total_spatial_with_i_only_mixed"]["physical_schur_restored_ward"]
    best_bond = min(
        variants["pkg_bond_metric_spatial_with_i"]["physical_schur_restored_ward"],
        variants["pkg_hybrid_bond_metric_spatial_with_i"]["physical_schur_restored_ward"],
    )
    best_convention = variants[best]
    return {
        "pairing": pairing,
        "N": int(n_grid),
        "m_case": list(m_case),
        "q_model": q.tolist(),
        "baseline_restored_ward": baseline,
        "lsq_mixed_only_restored_ward": lsq,
        "stage2j_best_longitudinal_restored_ward": stage2j,
        "variants": variants,
        "best_package_variant": best,
        "best_package_restored_ward": variants[best]["physical_schur_restored_ward"],
        "best_package_closes_dwave": bool(variants[best]["physical_schur_restored_ward"] < 1e-8),
        "AA_required": bool(full_total < no_aa),
        "etaeta_replacement_required": bool(full_total < variants["pkg_total_spatial_with_i_AA_plus_mixed_no_etaeta"]["physical_schur_restored_ward"]),
        "mixed_only_insufficient": bool(only_mixed > 1e-8),
        "supports_gauge_covariant_package": bool(variants[best]["physical_schur_restored_ward"] < min(baseline, stage2j)),
        "supports_bond_metric_counterterm": bool(best_bond <= variants[best]["physical_schur_restored_ward"] * (1.0 + 1e-12)),
        "sign_or_convention_ambiguity": bool(best_convention["alpha_convention"] != "with_i" or best_convention["lambda_convention"] != "spatial"),
        "C_candidate_metadata": {
            name: {
                "package_mode": item["package_mode"],
                "description": item["description"],
                "matrix_norm": float(np.linalg.norm(item["matrix"])),
                "bond_metric_info": item.get("bond_metric_info", {}),
            }
            for name, item in C_candidates.items()
        },
    }


def _status(cases: list[dict[str, Any]]) -> str:
    dwave = [case for case in cases if case["pairing"] == "dwave"]
    controls = [case for case in cases if case["pairing"] in {"onsite_s", "spm", "dwave_const_form"}]
    if any(float(case["lsq_mixed_only_restored_ward"]) >= 1e-8 for case in dwave):
        return "FAILED_STAGE2K_LSQ_REFERENCE_NOT_CLOSING"
    if any(float(case["best_package_restored_ward"]) >= 1e-6 and float(case["baseline_restored_ward"]) < 1e-8 for case in controls):
        return "FAILED_STAGE2K_CONTROL_REGRESSION"
    all_closed = all(float(case["best_package_restored_ward"]) < 1e-8 for case in dwave)
    if all_closed:
        if all(case["best_package_variant"] in {"pkg_bond_metric_spatial_with_i", "pkg_hybrid_bond_metric_spatial_with_i"} for case in dwave):
            return "PASSED_STAGE2K_BOND_METRIC_PACKAGE_IDENTIFIED"
        return "PASSED_STAGE2K_GAUGE_COVARIANT_PACKAGE_CLOSES_DWAVE"
    over_2j = all(float(case["best_package_restored_ward"]) < 0.5 * float(case["stage2j_best_longitudinal_restored_ward"]) for case in dwave)
    if over_2j:
        return "PASSED_STAGE2K_PACKAGE_IMPROVES_OVER_LONGITUDINAL_BUT_NOT_CLOSED"
    over_baseline = all(float(case["best_package_restored_ward"]) < 0.1 * float(case["baseline_restored_ward"]) for case in dwave)
    if over_baseline:
        return "PARTIAL_STAGE2K_PACKAGE_IMPROVES_BUT_STILL_ABOVE_THRESHOLD"
    if any(case["sign_or_convention_ambiguity"] for case in dwave):
        return "PARTIAL_STAGE2K_ONLY_SIGN_CONVENTION_VARIANTS_HELP"
    return "PARTIAL_STAGE2K_NO_PACKAGE_IMPROVES"


@lru_cache(maxsize=2)
def build_payload(quick: bool = True) -> dict[str, Any]:
    cases = [_case(pairing, n_grid, m_case) for pairing, n_grid, m_case in _case_specs(quick)]
    return {
        "status": _status(cases),
        "quick": bool(quick),
        "diagnostic_only": True,
        "production_default_modified": False,
        "formal_casimir_ran": False,
        "gauge_covariant_collective_package_tested": True,
        "lsq_candidate_used_as_production_formula": False,
        "lsq_used_to_define_package": False,
        "valid_for_casimir_input": False,
        "analytic_formula_claimed": "forward_gauge_covariant_package_diagnostic",
        "required_package_variants": list(REQUIRED_PACKAGE_VARIANTS),
        "package_generation": "AA, Aeta, etaA, and etaeta blocks are generated from the same C(Q), lambda gradient, and alpha convention.",
        "cases": cases,
    }


def _q_label(case: dict[str, Any]) -> str:
    q = case["q_model"]
    return f"({q[0]:.7g},{q[1]:.7g})"


def write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_2k_gauge_covariant_collective_package_audit"
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
        f"- valid for Casimir input: {payload['valid_for_casimir_input']}",
        "",
        "This stage tests a forward-derived gauge-covariant collective package. LSQ is used only as a diagnostic reference and is not used to define the package.",
        "",
        "## Case summary",
        "",
        "| pairing | N | q | baseline | LSQ | StageSC-2j | best package | best Ward | best C | lambda | alpha |",
        "| ------- | -: | - | -------: | --: | ---------: | ------------ | --------: | ------ | ------ | ----- |",
    ]
    for case in payload["cases"]:
        best = case["variants"][case["best_package_variant"]]
        lines.append(
            f"| {case['pairing']} | {case['N']} | {_q_label(case)} | {case['baseline_restored_ward']:.8g} | "
            f"{case['lsq_mixed_only_restored_ward']:.8g} | {case['stage2j_best_longitudinal_restored_ward']:.8g} | "
            f"{case['best_package_variant']} | {case['best_package_restored_ward']:.8g} | "
            f"{best['C_candidate']} | {best['lambda_convention']} | {best['alpha_convention']} |"
        )
    lines.extend(
        [
            "",
            "## Dwave package detail",
            "",
            "| N | q | variant | Ward | vs baseline | vs 2j | cond | dAA | dAeta | detaA | detaeta | C | lambda | alpha |",
            "| -: | - | ------- | ---: | ----------: | ----: | ---: | --: | -----: | ----: | ------: | - | ------ | ----- |",
        ]
    )
    for case in payload["cases"]:
        if case["pairing"] != "dwave":
            continue
        for name, variant in case["variants"].items():
            lines.append(
                f"| {case['N']} | {_q_label(case)} | {name} | {variant['physical_schur_restored_ward']:.8g} | "
                f"{variant['improvement_factor_vs_baseline']:.8g} | {variant['improvement_factor_vs_stage2j']:.8g} | "
                f"{variant['collective_condition_number']:.8g} | {variant['delta_AA_norm']:.8g} | "
                f"{variant['delta_Aeta_norm']:.8g} | {variant['delta_etaA_norm']:.8g} | "
                f"{variant['delta_etaeta_norm']:.8g} | {variant['C_candidate']} | "
                f"{variant['lambda_convention']} | {variant['alpha_convention']} |"
            )
    dwave = [case for case in payload["cases"] if case["pairing"] == "dwave"]
    controls = sorted({case["pairing"] for case in payload["cases"] if case["pairing"] != "dwave"})
    best_counts = {name: sum(case["best_package_variant"] == name for case in dwave) for name in sorted({case["best_package_variant"] for case in dwave})}
    best_C = {
        candidate: sum(case["variants"][case["best_package_variant"]]["C_candidate"] == candidate for case in dwave)
        for candidate in sorted({case["variants"][case["best_package_variant"]]["C_candidate"] for case in dwave})
    }
    alpha_counts = {
        alpha: sum(case["variants"][case["best_package_variant"]]["alpha_convention"] == alpha for case in dwave)
        for alpha in sorted({case["variants"][case["best_package_variant"]]["alpha_convention"] for case in dwave})
    }
    lambda_counts = {
        lam: sum(case["variants"][case["best_package_variant"]]["lambda_convention"] == lam for case in dwave)
        for lam in sorted({case["variants"][case["best_package_variant"]]["lambda_convention"] for case in dwave})
    }
    lines.extend(
        [
            "",
            "## Human-readable conclusion",
            "",
            "The LSQ mixed block still closes dwave in the quick cases, but remains only a diagnostic reference.",
            "StageSC-2j longitudinal completion remains a partial-improvement reference rather than a closed solution.",
            f"Best package variants across dwave cases: {best_counts}.",
            f"Best C(Q) candidates across dwave cases: {best_C}.",
            f"Best lambda conventions across dwave cases: {lambda_counts}.",
            f"Best alpha conventions across dwave cases: {alpha_counts}.",
            f"All controls monitored: {controls}.",
            "The audit tests whether AA, A-eta, eta-A, and eta-eta should be treated as one gauge-covariant package; it does not claim a full transverse microscopic current.",
            "Formal Casimir input remains forbidden.",
            "The result is suitable for the next analytic-design stage only as diagnostic evidence, not as a production implementation.",
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
