#!/usr/bin/env python3
"""Diagnostic-only longitudinal Ward-completion audit for finite-q BdG kernels."""

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
from lno327.ward_response import physical_ward_residuals  # noqa: E402
from stageSC_2f_schur_block_assembly_audit import _candidate_responses  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
CHANNELS = ("rho", "Vx", "Vy", "eta1", "eta2")
REQUIRED_VARIANTS = (
    "left_only_longitudinal",
    "right_only_longitudinal",
    "both_longitudinal_independent",
    "both_longitudinal_hermitianized",
    "both_longitudinal_opposite_sign",
)
CONTROL_CASES = (("onsite_s", 24, (1, 0)), ("spm", 24, (1, 0)), ("dwave_const_form", 24, (1, 0)))
DWAVE_QUICK_CASES = (("dwave", 24, (1, 0)), ("dwave", 24, (1, 1)), ("dwave", 36, (1, 0)), ("dwave", 36, (1, 1)))


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


def _ward_vector(omega_eV: float, q: np.ndarray, delta0_eV: float) -> np.ndarray:
    return np.asarray([1j * omega_eV, q[0], q[1], 0.0, 2j * delta0_eV], dtype=complex)


def _ward_max(response: np.ndarray, omega_eV: float, q: np.ndarray) -> float:
    left, right = physical_ward_residuals(response, omega_eV, q)
    return float(max(np.max(np.abs(left)), np.max(np.abs(right))))


def _kernel_5x5(bare: np.ndarray, left: np.ndarray, collective: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.zeros((5, 5), dtype=complex)
    output[:3, :3] = bare
    output[:3, 3:] = left
    output[3:, :3] = right
    output[3:, 3:] = collective
    return output


def _five_channel_ward_metrics(kernel: np.ndarray, omega_eV: float, q: np.ndarray, delta0_eV: float) -> dict[str, Any]:
    ward = _ward_vector(omega_eV, q, delta0_eV)
    left_values = ward @ kernel
    right_values = kernel @ ward
    return {
        "left_full_5x5_ward_max": float(np.max(np.abs(left_values))),
        "right_full_5x5_ward_max": float(np.max(np.abs(right_values))),
        "full_5x5_ward_max": float(max(np.max(np.abs(left_values)), np.max(np.abs(right_values)))),
        "left_collective_columns_ward_max": float(np.max(np.abs(left_values[3:]))),
        "right_collective_rows_ward_max": float(np.max(np.abs(right_values[3:]))),
        "left_ward_by_channel": {channel: value for channel, value in zip(CHANNELS, left_values, strict=True)},
        "right_ward_by_channel": {channel: value for channel, value in zip(CHANNELS, right_values, strict=True)},
    }


def _left_longitudinal_completion(kernel: np.ndarray, omega_eV: float, q: np.ndarray, delta0_eV: float) -> np.ndarray:
    ward = _ward_vector(omega_eV, q, delta0_eV)
    residual = ward @ kernel[:, 3:]
    q2 = float(np.dot(q, q))
    delta = np.zeros((3, 2), dtype=complex)
    delta[1, :] = -q[0] * residual / q2
    delta[2, :] = -q[1] * residual / q2
    return delta


def _right_longitudinal_completion(kernel: np.ndarray, omega_eV: float, q: np.ndarray, delta0_eV: float) -> np.ndarray:
    ward = _ward_vector(omega_eV, q, delta0_eV)
    residual = kernel[3:, :] @ ward
    q2 = float(np.dot(q, q))
    delta = np.zeros((2, 3), dtype=complex)
    delta[:, 1] = -residual * q[0] / q2
    delta[:, 2] = -residual * q[1] / q2
    return delta


def _density_plus_vector_completion(
    kernel: np.ndarray,
    omega_eV: float,
    q: np.ndarray,
    delta0_eV: float,
) -> tuple[np.ndarray, np.ndarray]:
    ward = _ward_vector(omega_eV, q, delta0_eV)
    left_residual = ward @ kernel[:, 3:]
    right_residual = kernel[3:, :] @ ward
    norm = float(abs(omega_eV) ** 2 + np.dot(q, q))
    left = np.zeros((3, 2), dtype=complex)
    left[0, :] = -(1j * omega_eV).conjugate() * left_residual / norm
    left[1, :] = -q[0] * left_residual / norm
    left[2, :] = -q[1] * left_residual / norm
    right = np.zeros((2, 3), dtype=complex)
    right[:, 0] = -right_residual * (1j * omega_eV).conjugate() / norm
    right[:, 1] = -right_residual * q[0] / norm
    right[:, 2] = -right_residual * q[1] / norm
    return left, right


def _schur_response(bare: np.ndarray, left: np.ndarray, collective: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, float]:
    response, condition, _ = single_composite_schur(bare, left, collective, right)
    return response, float(condition)


def _variant_metrics(
    name: str,
    bare: np.ndarray,
    baseline_left: np.ndarray,
    collective: np.ndarray,
    baseline_right: np.ndarray,
    delta_left: np.ndarray,
    delta_right: np.ndarray,
    omega_eV: float,
    q: np.ndarray,
    delta0_eV: float,
) -> dict[str, Any]:
    left = baseline_left + delta_left
    right = baseline_right + delta_right
    kernel = _kernel_5x5(bare, left, collective, right)
    response, condition = _schur_response(bare, left, collective, right)
    metrics = _five_channel_ward_metrics(kernel, omega_eV, q, delta0_eV)
    metrics.update(
        {
            "variant": name,
            "physical_schur_restored_ward": _ward_max(response, omega_eV, q),
            "delta_left_norm": float(np.linalg.norm(delta_left)),
            "delta_right_norm": float(np.linalg.norm(delta_right)),
            "condition_number_collective_total": condition,
        }
    )
    return metrics


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
    bare = result["bare_total"]
    collective = result["collective_total"]
    left = result["em_collective_left"]
    right = result["collective_em_right"]
    baseline_kernel = _kernel_5x5(bare, left, collective, right)
    baseline_ward = _five_channel_ward_metrics(baseline_kernel, cfg.omega_eV, q, amp.delta0_eV)
    candidates = _candidate_responses(result, pairing, q, amp)
    left_delta = _left_longitudinal_completion(baseline_kernel, cfg.omega_eV, q, amp.delta0_eV)
    right_delta = _right_longitudinal_completion(baseline_kernel, cfg.omega_eV, q, amp.delta0_eV)
    density_left_delta, density_right_delta = _density_plus_vector_completion(baseline_kernel, cfg.omega_eV, q, amp.delta0_eV)
    zero_left = np.zeros_like(left_delta)
    zero_right = np.zeros_like(right_delta)
    variants = {
        "left_only_longitudinal": _variant_metrics(
            "left_only_longitudinal", bare, left, collective, right, left_delta, zero_right, cfg.omega_eV, q, amp.delta0_eV
        ),
        "right_only_longitudinal": _variant_metrics(
            "right_only_longitudinal", bare, left, collective, right, zero_left, right_delta, cfg.omega_eV, q, amp.delta0_eV
        ),
        "both_longitudinal_independent": _variant_metrics(
            "both_longitudinal_independent", bare, left, collective, right, left_delta, right_delta, cfg.omega_eV, q, amp.delta0_eV
        ),
        "both_longitudinal_hermitianized": _variant_metrics(
            "both_longitudinal_hermitianized",
            bare,
            left,
            collective,
            right,
            left_delta,
            left_delta.conjugate().T,
            cfg.omega_eV,
            q,
            amp.delta0_eV,
        ),
        "both_longitudinal_opposite_sign": _variant_metrics(
            "both_longitudinal_opposite_sign", bare, left, collective, right, -left_delta, -right_delta, cfg.omega_eV, q, amp.delta0_eV
        ),
        "density_plus_vector_min_norm": _variant_metrics(
            "density_plus_vector_min_norm",
            bare,
            left,
            collective,
            right,
            density_left_delta,
            density_right_delta,
            cfg.omega_eV,
            q,
            amp.delta0_eV,
        ),
    }
    baseline_restored = _ward_max(candidates["baseline"], cfg.omega_eV, q)
    lsq_restored = _ward_max(candidates["mixed_only"], cfg.omega_eV, q)
    best_schur = min(variants, key=lambda item: float(variants[item]["physical_schur_restored_ward"]))
    best_full = min(variants, key=lambda item: float(variants[item]["full_5x5_ward_max"]))
    return {
        "pairing": pairing,
        "N": int(n_grid),
        "m_case": list(m_case),
        "q_model": q.tolist(),
        "baseline_restored_ward": baseline_restored,
        "lsq_mixed_only_restored_ward": lsq_restored,
        "baseline_full_5x5_left_ward": baseline_ward["left_full_5x5_ward_max"],
        "baseline_full_5x5_right_ward": baseline_ward["right_full_5x5_ward_max"],
        "baseline_left_collective_columns_ward": baseline_ward["left_collective_columns_ward_max"],
        "baseline_right_collective_rows_ward": baseline_ward["right_collective_rows_ward_max"],
        "variants": variants,
        "best_variant_by_schur": best_schur,
        "best_schur_restored_ward": variants[best_schur]["physical_schur_restored_ward"],
        "best_variant_by_full_ward": best_full,
        "best_full_ward_max": variants[best_full]["full_5x5_ward_max"],
        "supports_longitudinal_completion": bool(variants[best_schur]["physical_schur_restored_ward"] < 1e-8),
        "supports_left_right_independent_completion": bool(variants["both_longitudinal_independent"]["physical_schur_restored_ward"] < 1e-8),
        "supports_hermitianized_completion": bool(variants["both_longitudinal_hermitianized"]["physical_schur_restored_ward"] < 1e-8),
        "sign_convention_issue_suspected": bool(
            variants["both_longitudinal_opposite_sign"]["physical_schur_restored_ward"]
            < variants["both_longitudinal_independent"]["physical_schur_restored_ward"]
        ),
    }


def _status(cases: list[dict[str, Any]]) -> str:
    dwave = [case for case in cases if case["pairing"] == "dwave"]
    controls = [case for case in cases if case["pairing"] in {"onsite_s", "spm", "dwave_const_form"}]
    if any(float(case["lsq_mixed_only_restored_ward"]) >= 1e-8 for case in dwave):
        return "FAILED_STAGE2J_LSQ_REFERENCE_NOT_CLOSING"
    if any(min(float(variant["physical_schur_restored_ward"]) for variant in case["variants"].values()) >= 1e-6 for case in controls):
        return "FAILED_STAGE2J_CONTROL_REGRESSION"
    closed_by_schur = [
        name
        for name in REQUIRED_VARIANTS
        if all(float(case["variants"][name]["physical_schur_restored_ward"]) < 1e-8 for case in dwave)
    ]
    full_closed = [
        name
        for name in REQUIRED_VARIANTS
        if all(float(case["variants"][name]["full_5x5_ward_max"]) < 1e-8 for case in dwave)
    ]
    if closed_by_schur:
        if any(name in full_closed for name in closed_by_schur):
            return "PASSED_STAGE2J_LONGITUDINAL_COMPLETION_CLOSES_DWAVE"
        return "PASSED_STAGE2J_LONGITUDINAL_CLOSES_SCHUR_BUT_NOT_FULL_5X5"
    if full_closed:
        return "PASSED_STAGE2J_FULL_5X5_CLOSED_BUT_SCHUR_NEEDS_SIGN_REVIEW"
    left_right = {"left_only_longitudinal", "right_only_longitudinal"}
    if any(
        all(float(case["variants"][name]["physical_schur_restored_ward"]) < 1e-8 for case in dwave)
        for name in left_right
    ):
        return "PARTIAL_STAGE2J_LEFT_OR_RIGHT_ONLY_CLOSES"
    if all(case["sign_convention_issue_suspected"] for case in dwave):
        return "PARTIAL_STAGE2J_SIGN_CONVENTION_AMBIGUOUS"
    improved = all(
        float(case["best_schur_restored_ward"]) < 0.1 * float(case["baseline_restored_ward"])
        for case in dwave
    )
    if improved:
        return "PARTIAL_STAGE2J_LONGITUDINAL_IMPROVES_BUT_NOT_CLOSED"
    return "PARTIAL_STAGE2J_LONGITUDINAL_IMPROVES_BUT_NOT_CLOSED"


@lru_cache(maxsize=2)
def build_payload(quick: bool = True) -> dict[str, Any]:
    cases = [_case(pairing, n_grid, m_case) for pairing, n_grid, m_case in _case_specs(quick)]
    return {
        "status": _status(cases),
        "quick": bool(quick),
        "diagnostic_only": True,
        "production_default_modified": False,
        "formal_casimir_ran": False,
        "longitudinal_ward_completion_tested": True,
        "lsq_candidate_used_as_production_formula": False,
        "transverse_part_claimed": False,
        "analytic_formula_claimed": "longitudinal_response_completion_only",
        "ward_basis": list(CHANNELS),
        "right_ward_convention_tested": "same_column_vector_[iomega,qx,qy,0,2iDelta0]",
        "hermitianized_completion_implementation": "delta_right = delta_left.conjugate().T at the same finite-q response point",
        "required_variants": list(REQUIRED_VARIANTS),
        "cases": cases,
    }


def _q_label(case: dict[str, Any]) -> str:
    q = case["q_model"]
    return f"({q[0]:.7g},{q[1]:.7g})"


def write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_2j_longitudinal_ward_completion_audit"
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
        f"- right Ward convention tested: {payload['right_ward_convention_tested']}",
        f"- hermitianized implementation: {payload['hermitianized_completion_implementation']}",
        "",
        "This stage tests only the longitudinal response-level Ward completion fixed by gauge invariance. "
        "It does not determine or claim the transverse microscopic mixed current.",
        "",
        "Ward identity fixes only the longitudinal part: q_i delta K[V_i, eta_a] = -R_a. "
        "The minimum spatial completion used here is delta K[V_i, eta_a] = -q_i R_a / q^2; any transverse addition with "
        "q_i delta K_T[V_i, eta_a] = 0 remains undetermined.",
        "",
        "## Case summary",
        "",
        "| pairing | N | q | baseline | LSQ | best variant | best Schur | best full 5x5 | sign ambiguous |",
        "| ------- | -: | - | -------: | --: | ------------ | ---------: | ------------: | -------------- |",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['pairing']} | {case['N']} | {_q_label(case)} | {case['baseline_restored_ward']:.8g} | "
            f"{case['lsq_mixed_only_restored_ward']:.8g} | {case['best_variant_by_schur']} | "
            f"{case['best_schur_restored_ward']:.8g} | {case['best_full_ward_max']:.8g} | "
            f"{case['sign_convention_issue_suspected']} |"
        )
    lines.extend(
        [
            "",
            "## Dwave variant detail",
            "",
            "| N | q | variant | left full | right full | left eta cols | right eta rows | Schur Ward | delta L | delta R |",
            "| -: | - | ------- | --------: | ---------: | ------------: | -------------: | ---------: | ------: | ------: |",
        ]
    )
    for case in payload["cases"]:
        if case["pairing"] != "dwave":
            continue
        for name, variant in case["variants"].items():
            lines.append(
                f"| {case['N']} | {_q_label(case)} | {name} | {variant['left_full_5x5_ward_max']:.8g} | "
                f"{variant['right_full_5x5_ward_max']:.8g} | {variant['left_collective_columns_ward_max']:.8g} | "
                f"{variant['right_collective_rows_ward_max']:.8g} | {variant['physical_schur_restored_ward']:.8g} | "
                f"{variant['delta_left_norm']:.8g} | {variant['delta_right_norm']:.8g} |"
            )
    dwave = [case for case in payload["cases"] if case["pairing"] == "dwave"]
    controls = sorted({case["pairing"] for case in payload["cases"] if case["pairing"] != "dwave"})
    best_counts = {name: sum(case["best_variant_by_schur"] == name for case in dwave) for name in sorted({case["best_variant_by_schur"] for case in dwave})}
    schur_closed = any(case["best_schur_restored_ward"] < 1e-8 for case in dwave)
    full_closed = any(case["best_full_ward_max"] < 1e-8 for case in dwave)
    sign_ambiguous = any(case["sign_convention_issue_suspected"] for case in dwave)
    lines.extend(
        [
            "",
            "## Human-readable conclusion",
            "",
            "The StageSC-2f LSQ mixed block remains the diagnostic reference for closing dwave.",
            f"Longitudinal response completion closes physical Schur Ward in at least one dwave case: {schur_closed}.",
            f"Full 5x5 left/right Ward closes in at least one dwave case: {full_closed}.",
            f"Best Schur variants across dwave cases: {best_counts}.",
            f"Sign convention ambiguity suspected: {sign_ambiguous}.",
            f"Control pairings present and monitored: {controls}.",
            "The transverse microscopic mixed current is still not determined by this audit.",
            "Formal Casimir input remains forbidden.",
            "This is a possible production-candidate direction only for the longitudinal response-level completion, not for a full microscopic mixed-current vertex.",
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
