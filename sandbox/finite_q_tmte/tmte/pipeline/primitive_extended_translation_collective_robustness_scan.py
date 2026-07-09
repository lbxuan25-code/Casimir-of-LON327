"""Diagnostic robustness scan for primitive extended translation/collective Ward audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..io.writers import write_json
from .primitive_extended_translation_collective_audit import run_primitive_extended_translation_collective_audit

SCHEMA_VERSION = "finite_q_tmte_primitive_extended_translation_collective_robustness_scan_v1"
EXPECTED_TOP = {"minus_translation_plus_qM"}


def _complex_value(value: complex) -> dict[str, float]:
    z = complex(value)
    return {"real": float(z.real), "imag": float(z.imag), "abs": float(abs(z))}


def _extract_alpha(fit: dict[str, Any]) -> complex:
    value = fit["alpha"]
    if isinstance(value, dict):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    return complex(value)


def _side_summary(side_name: str, rows: list[dict[str, Any]], *, diff_tol: float, fit_res_tol: float, alpha_tol: float) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"{side_name} has no ranked translation candidates")
    top = rows[0]
    fit = top["fit_to_target"]
    alpha = _extract_alpha(fit)
    diff = float(top["difference_over_target_norm"])
    fit_res = float(fit["residual_over_target_norm"])
    passed = str(top["name"]) in EXPECTED_TOP and diff <= diff_tol and fit_res <= fit_res_tol and abs(alpha - 1.0) <= alpha_tol
    return {
        "top_candidate": str(top["name"]),
        "top_difference_over_missing": diff,
        "top_fit_alpha": _complex_value(alpha),
        "top_fit_residual_over_missing": fit_res,
        "passed_translation_identity": bool(passed),
        "valid_for_casimir_input": False,
    }


def summarize_payload(payload: dict[str, Any], *, diff_tol: float, fit_res_tol: float, alpha_tol: float) -> dict[str, Any]:
    left = _side_summary("left", payload["left_translation_candidates_ranked"], diff_tol=diff_tol, fit_res_tol=fit_res_tol, alpha_tol=alpha_tol)
    right = _side_summary("right", payload["right_translation_candidates_ranked"], diff_tol=diff_tol, fit_res_tol=fit_res_tol, alpha_tol=alpha_tol)
    ward = payload["ward_decomposition"]
    row = {
        "pairing": str(payload["model"]["pairing"]),
        "delta0_eV": float(payload["model"]["delta0_eV"]),
        "q_value": float(payload["debug_parameters"]["q_value"]),
        "matsubara_index": int(payload["frequency"]["matsubara_index"]),
        "temperature_K": float(payload["frequency"]["temperature_K"]),
        "nk": int(payload["debug_parameters"]["nk"]),
        "left_em_norm": float(ward["left"]["em_total"]["norm"]),
        "left_mixed_norm": float(ward["left"]["mixed_collective"]["norm"]),
        "left_extended_norm": float(ward["left"]["extended_total"]["norm"]),
        "left_extended_over_em": float(ward["left"]["em_to_extended_reduction"]),
        "right_em_norm": float(ward["right"]["em_total"]["norm"]),
        "right_mixed_norm": float(ward["right"]["mixed_collective"]["norm"]),
        "right_extended_norm": float(ward["right"]["extended_total"]["norm"]),
        "right_extended_over_em": float(ward["right"]["em_to_extended_reduction"]),
        "left": left,
        "right": right,
        "passed_translation_identity": bool(left["passed_translation_identity"] and right["passed_translation_identity"]),
        "valid_for_casimir_input": False,
    }
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_rows = [row for row in rows if row["passed_translation_identity"]]
    left_diffs = [float(row["left"]["top_difference_over_missing"]) for row in rows]
    right_diffs = [float(row["right"]["top_difference_over_missing"]) for row in rows]
    left_fit = [float(row["left"]["top_fit_residual_over_missing"]) for row in rows]
    right_fit = [float(row["right"]["top_fit_residual_over_missing"]) for row in rows]
    alpha_errors = []
    for row in rows:
        for side in ("left", "right"):
            alpha = complex(row[side]["top_fit_alpha"]["real"], row[side]["top_fit_alpha"]["imag"])
            alpha_errors.append(abs(alpha - 1.0))
    pairings = sorted({str(row["pairing"]) for row in rows})
    return {
        "num_rows": len(rows),
        "num_passed": len(pass_rows),
        "all_passed": len(pass_rows) == len(rows),
        "max_left_difference_over_missing": max(left_diffs) if left_diffs else None,
        "max_right_difference_over_missing": max(right_diffs) if right_diffs else None,
        "max_left_fit_residual_over_missing": max(left_fit) if left_fit else None,
        "max_right_fit_residual_over_missing": max(right_fit) if right_fit else None,
        "max_abs_fit_alpha_minus_one": max(alpha_errors) if alpha_errors else None,
        "pairing_pass_counts": {pairing: sum(1 for row in rows if row["pairing"] == pairing and row["passed_translation_identity"]) for pairing in pairings},
        "pairing_total_counts": {pairing: sum(1 for row in rows if row["pairing"] == pairing) for pairing in pairings},
        "left_top_candidate_counts": {name: sum(1 for row in rows if row["left"]["top_candidate"] == name) for name in sorted({row["left"]["top_candidate"] for row in rows})},
        "right_top_candidate_counts": {name: sum(1 for row in rows if row["right"]["top_candidate"] == name) for name in sorted({row["right"]["top_candidate"] for row in rows})},
        "valid_for_casimir_input": False,
    }


def run_primitive_extended_translation_collective_robustness_scan(
    *,
    model_name: str,
    pairings: Sequence[str],
    matsubara_indices: Sequence[int],
    temperature_K: float,
    q_values: Sequence[float],
    nk_values: Sequence[int],
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    contact_scale: float = 1.0,
    candidate_name: str = "matrix_inferred_matsubara_i_asymmetric",
    diff_tol: float = 1e-9,
    fit_res_tol: float = 1e-9,
    alpha_tol: float = 1e-9,
    keep_payloads: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for pairing in pairings:
        for n in matsubara_indices:
            if int(n) < 0:
                raise ValueError("matsubara indices must be non-negative")
            for q in q_values:
                if float(q) <= 0.0:
                    raise ValueError("q values must be positive")
                for nk in nk_values:
                    if int(nk) <= 0:
                        raise ValueError("nk values must be positive")
                    payload = run_primitive_extended_translation_collective_audit(
                        model_name=model_name,
                        pairing_name=str(pairing),
                        matsubara_index=int(n),
                        temperature_K=float(temperature_K),
                        q_value=float(q),
                        nk=int(nk),
                        delta0_eV=delta0_eV,
                        eta_eV=float(eta_eV),
                        shift_fractions=tuple(float(value) for value in shift_fractions),
                        contact_scale=float(contact_scale),
                        candidate_name=candidate_name,
                    )
                    rows.append(summarize_payload(payload, diff_tol=diff_tol, fit_res_tol=fit_res_tol, alpha_tol=alpha_tol))
                    if keep_payloads:
                        payloads.append(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {"diagnostic_run_completed": True, "diagnostic_only_not_a_fix": True, "accepted_convention": False, "valid_for_casimir_input": False, "reason": "primitive_extended_translation_collective_robustness_scan_not_production_convention"},
        "model": {"name": model_name, "valid_for_casimir_input": False},
        "scan_parameters": {"pairings": [str(p) for p in pairings], "matsubara_indices": [int(n) for n in matsubara_indices], "temperature_K": float(temperature_K), "q_values": [float(q) for q in q_values], "nk_values": [int(nk) for nk in nk_values], "delta0_eV": delta0_eV, "eta_eV": float(eta_eV), "shift_fractions": [float(v) for v in shift_fractions], "contact_scale": float(contact_scale), "candidate_name": candidate_name, "diff_tol": float(diff_tol), "fit_res_tol": float(fit_res_tol), "alpha_tol": float(alpha_tol), "expected_top_candidates": sorted(EXPECTED_TOP), "valid_for_casimir_input": False},
        "aggregate": aggregate_rows(rows),
        "summary_rows": rows,
        "detailed_payloads": payloads if keep_payloads else [],
        "interpretation_guardrails": {"not_a_production_fix": True, "if_all_passed": "primitive extended BdG translation/contact RHS is numerically robust over this scan window", "if_failures_exist": "inspect failed rows before writing analytic derivation or Schur-level audit", "valid_for_casimir_input": False},
        "valid_for_casimir_input": False,
    }


def run_and_write_primitive_extended_translation_collective_robustness_scan(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_primitive_extended_translation_collective_robustness_scan(**kwargs)
    write_json(Path(output_dir) / "primitive_extended_translation_collective_robustness_scan.json", payload)
    return payload
