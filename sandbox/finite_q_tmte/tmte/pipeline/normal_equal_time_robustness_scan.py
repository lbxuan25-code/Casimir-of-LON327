"""Diagnostic-only robustness scan for the normal equal-time Ward term."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..io.writers import write_json
from .normal_equal_time_ward_audit import run_normal_equal_time_ward_audit

SCHEMA_VERSION = "finite_q_tmte_normal_equal_time_robustness_scan_v1"
EXPECTED_TOP = {"minus_translation_forward", "minus_translation_direct"}


def _complex_value(value: complex) -> dict[str, float]:
    z = complex(value)
    return {"real": float(z.real), "imag": float(z.imag), "abs": float(abs(z))}


def _extract_alpha(fit: dict[str, Any]) -> complex:
    value = fit["alpha"]
    if isinstance(value, dict):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    return complex(value)


def summarize_equal_time_payload(payload: dict[str, Any], *, diff_tol: float, fit_res_tol: float, alpha_tol: float) -> dict[str, Any]:
    ranked = payload["candidate_equal_time_vectors_ranked"]
    if not ranked:
        raise ValueError("payload has no ranked equal-time candidates")
    top = ranked[0]
    fit = top["fit_to_target"]
    alpha = _extract_alpha(fit)
    diff = float(top["difference_over_target_norm"])
    fit_res = float(fit["residual_over_target_norm"])
    left = payload["ward_decomposition"]["left"]
    right = payload["ward_decomposition"]["right"]
    contact_alpha_left = _extract_alpha(left["contact_required_over_current"])
    contact_alpha_right = _extract_alpha(right["contact_required_over_current"])
    pass_flag = (
        str(top["name"]) in EXPECTED_TOP
        and diff <= diff_tol
        and fit_res <= fit_res_tol
        and abs(alpha - 1.0) <= alpha_tol
    )
    return {
        "q_value": float(payload["debug_parameters"]["q_value"]),
        "matsubara_index": int(payload["frequency"]["matsubara_index"]),
        "temperature_K": float(payload["frequency"]["temperature_K"]),
        "nk": int(payload["debug_parameters"]["nk"]),
        "top_candidate": str(top["name"]),
        "top_difference_over_missing": diff,
        "top_fit_alpha": _complex_value(alpha),
        "top_fit_residual_over_missing": fit_res,
        "left_total_norm": float(left["total"]["norm"]),
        "right_total_norm": float(right["total"]["norm"]),
        "contact_alpha_left": _complex_value(contact_alpha_left),
        "contact_alpha_right": _complex_value(contact_alpha_right),
        "contact_alpha_left_right_abs_diff": float(abs(contact_alpha_left - contact_alpha_right)),
        "vertex_max_abs_error": float(payload["vertex_identity"]["max_abs_error_over_shifted_meshes"]),
        "passed_translation_identity": bool(pass_flag),
        "valid_for_casimir_input": False,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row["passed_translation_identity"]]
    diffs = [float(row["top_difference_over_missing"]) for row in rows]
    fit_res = [float(row["top_fit_residual_over_missing"]) for row in rows]
    alpha_err = [abs(complex(row["top_fit_alpha"]["real"], row["top_fit_alpha"]["imag"]) - 1.0) for row in rows]
    return {
        "num_rows": len(rows),
        "num_passed": len(ok),
        "all_passed": len(ok) == len(rows),
        "max_top_difference_over_missing": max(diffs) if diffs else None,
        "max_top_fit_residual_over_missing": max(fit_res) if fit_res else None,
        "max_abs_top_fit_alpha_minus_one": max(alpha_err) if alpha_err else None,
        "top_candidate_counts": {name: sum(1 for row in rows if row["top_candidate"] == name) for name in sorted({row["top_candidate"] for row in rows})},
        "valid_for_casimir_input": False,
    }


def run_normal_equal_time_robustness_scan(
    *,
    model_name: str,
    matsubara_indices: Sequence[int],
    temperature_K: float,
    q_values: Sequence[float],
    nk_values: Sequence[int],
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    diff_tol: float = 1e-10,
    fit_res_tol: float = 1e-10,
    alpha_tol: float = 1e-10,
    keep_payloads: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for n in matsubara_indices:
        if int(n) < 0:
            raise ValueError("matsubara indices must be non-negative")
        for q in q_values:
            if float(q) <= 0.0:
                raise ValueError("q values must be positive")
            for nk in nk_values:
                if int(nk) <= 0:
                    raise ValueError("nk values must be positive")
                payload = run_normal_equal_time_ward_audit(
                    model_name=model_name,
                    matsubara_index=int(n),
                    temperature_K=float(temperature_K),
                    q_value=float(q),
                    nk=int(nk),
                    eta_eV=float(eta_eV),
                    shift_fractions=tuple(float(value) for value in shift_fractions),
                )
                rows.append(summarize_equal_time_payload(payload, diff_tol=diff_tol, fit_res_tol=fit_res_tol, alpha_tol=alpha_tol))
                if keep_payloads:
                    payloads.append(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "valid_for_casimir_input": False,
            "reason": "normal_equal_time_robustness_scan_not_production_convention",
        },
        "model": {"name": model_name, "valid_for_casimir_input": False},
        "scan_parameters": {
            "matsubara_indices": [int(value) for value in matsubara_indices],
            "temperature_K": float(temperature_K),
            "q_values": [float(value) for value in q_values],
            "nk_values": [int(value) for value in nk_values],
            "eta_eV": float(eta_eV),
            "shift_fractions": [float(value) for value in shift_fractions],
            "diff_tol": float(diff_tol),
            "fit_res_tol": float(fit_res_tol),
            "alpha_tol": float(alpha_tol),
            "expected_top_candidates": sorted(EXPECTED_TOP),
            "valid_for_casimir_input": False,
        },
        "aggregate": aggregate_rows(rows),
        "summary_rows": rows,
        "detailed_payloads": payloads if keep_payloads else [],
        "interpretation_guardrails": {
            "not_a_production_fix": True,
            "if_all_passed": "normal translation/equal-time RHS is numerically robust over this scan window",
            "if_failures_exist": "inspect failed rows before writing analytic derivation",
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_and_write_normal_equal_time_robustness_scan(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_normal_equal_time_robustness_scan(**kwargs)
    write_json(Path(output_dir) / "normal_equal_time_robustness_scan.json", payload)
    return payload
