"""Diagnostic-only pairing/contact missing-contribution audit.

This audit reuses the contact formula audit for a small pairing and delta0
sweep. It tests whether the contact mismatch is tied to superconducting pairing
and, in particular, whether it appears for momentum-dependent/bond pairing but
not for normal/delta0=0 or simple s-wave-like cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..io.writers import write_json
from .contact_formula_audit import run_contact_formula_audit
from .primitive_response_closure_suite import DEFAULT_CANDIDATE

SCHEMA_VERSION = "finite_q_tmte_pairing_contact_missing_audit_v1"
DEFAULT_PAIRINGS = ("dwave", "spm")
DEFAULT_DELTA0_VALUES = (0.0, 0.05, 0.1, 0.15)


def _as_complex(value: object) -> complex:
    if isinstance(value, dict):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    return complex(value)  # type: ignore[arg-type]


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-30) -> float:
    return float(numerator) / max(float(denominator), float(eps))


def _component_ratio(side: dict[str, Any], label: str = "L") -> dict[str, Any] | None:
    for row in side["componentwise_required_over_current"]:
        if row["label"] == label:
            return row
    return None


def summarize_contact_formula_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract compact scalar/parallelism diagnostics from contact audit payload."""

    analysis = payload["contact_formula_analysis"]
    left = analysis["left"]
    right = analysis["right"]
    left_scalar = left["required_over_current_scalar_projection"]
    right_scalar = right["required_over_current_scalar_projection"]
    alpha_left = _as_complex(left_scalar["alpha_required_over_current"])
    alpha_right = _as_complex(right_scalar["alpha_required_over_current"])
    alpha_mean = 0.5 * (alpha_left + alpha_right)
    l_left = _component_ratio(left, "L")
    l_right = _component_ratio(right, "L")
    l_ratio_left = _as_complex(l_left["required_over_current"]) if l_left is not None and l_left.get("ratio_defined") else 0.0 + 0.0j
    l_ratio_right = _as_complex(l_right["required_over_current"]) if l_right is not None and l_right.get("ratio_defined") else 0.0 + 0.0j
    required_norm = 0.5 * (float(left["contact_required"]["norm"]) + float(right["contact_required"]["norm"]))
    current_norm = 0.5 * (float(left["contact_current"]["norm"]) + float(right["contact_current"]["norm"]))
    residual_norm = 0.5 * (
        float(left["ward_residual_with_current_contact"]["norm"]) + float(right["ward_residual_with_current_contact"]["norm"])
    )
    projection_residual = 0.5 * (float(left_scalar["residual_norm"]) + float(right_scalar["residual_norm"]))
    projection_over_required = 0.5 * (
        float(left_scalar["residual_over_required_norm"]) + float(right_scalar["residual_over_required_norm"])
    )
    abs_overlap = 0.5 * (float(left["parallelism"]["abs_overlap"]) + float(right["parallelism"]["abs_overlap"]))
    return {
        "pairing": payload["model"]["pairing"],
        "delta0_eV": float(payload["debug_parameters"].get("delta0_eV", payload.get("delta0_eV", 0.0))),
        "alpha_left": alpha_left,
        "alpha_right": alpha_right,
        "alpha_mean": alpha_mean,
        "alpha_real_mean": float(np.real(alpha_mean)),
        "alpha_imag_mean": float(np.imag(alpha_mean)),
        "missing_fraction_real": float(1.0 - np.real(alpha_mean)),
        "left_right_alpha_abs_diff": float(abs(alpha_left - alpha_right)),
        "L_component_ratio_left": l_ratio_left,
        "L_component_ratio_right": l_ratio_right,
        "required_norm_mean": required_norm,
        "current_norm_mean": current_norm,
        "ward_residual_current_contact_norm_mean": residual_norm,
        "projection_residual_norm_mean": projection_residual,
        "projection_residual_over_required_mean": projection_over_required,
        "parallelism_abs_overlap_mean": abs_overlap,
        "required_over_current_norm_ratio": _safe_ratio(required_norm, current_norm),
        "valid_for_casimir_input": False,
    }


def trend_by_delta0(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    if len(valid) < 2:
        return {"status": "insufficient_points", "valid_for_casimir_input": False}
    x = np.asarray([float(row["delta0_eV"]) ** 2 for row in valid], dtype=float)
    y = np.asarray([float(row["alpha_real_mean"]) for row in valid], dtype=float)
    if np.ptp(x) <= 1e-30:
        return {"status": "degenerate_delta0_grid", "valid_for_casimir_input": False}
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residual = y - predicted
    return {
        "status": "linear_fit_alpha_vs_delta0_squared",
        "fit_model": "alpha_real_mean ~= intercept + slope * delta0_eV^2",
        "intercept": float(intercept),
        "slope_per_eV2": float(slope),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "alpha_at_delta0_zero_extrapolated": float(intercept),
        "num_points": len(valid),
        "valid_for_casimir_input": False,
    }


def run_pairing_contact_missing_audit(
    *,
    model_name: str,
    pairings: Sequence[str] = DEFAULT_PAIRINGS,
    delta0_values: Sequence[float] = DEFAULT_DELTA0_VALUES,
    matsubara_index: int,
    temperature_K: float,
    q_value: float,
    nk: int,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    contact_scale: float = 1.0,
    candidate_name: str = DEFAULT_CANDIDATE,
    fail_fast: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    detailed_payloads: list[dict[str, Any]] = []
    for pairing in pairings:
        for delta0 in delta0_values:
            try:
                payload = run_contact_formula_audit(
                    model_name=model_name,
                    pairing_name=str(pairing),
                    matsubara_index=matsubara_index,
                    temperature_K=temperature_K,
                    q_value=q_value,
                    nk=nk,
                    delta0_eV=float(delta0),
                    eta_eV=eta_eV,
                    shift_fractions=shift_fractions,
                    contact_scale=contact_scale,
                    candidate_name=candidate_name,
                )
                # Keep the sweep-level delta0 explicit even if lower-level debug parameters omit it.
                payload["delta0_eV"] = float(delta0)
                payload["debug_parameters"] = {**payload["debug_parameters"], "delta0_eV": float(delta0)}
                summary = summarize_contact_formula_payload(payload)
                rows.append({"status": "ok", **summary})
                detailed_payloads.append(payload)
            except Exception as exc:  # pragma: no cover - exercised by real model availability, not unit tests.
                if fail_fast:
                    raise
                rows.append(
                    {
                        "status": "error",
                        "pairing": str(pairing),
                        "delta0_eV": float(delta0),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "valid_for_casimir_input": False,
                    }
                )
    by_pairing: dict[str, Any] = {}
    for pairing in pairings:
        pairing_rows = [row for row in rows if row.get("pairing") == pairing]
        ok_rows = [row for row in pairing_rows if row.get("status") == "ok"]
        by_pairing[str(pairing)] = {
            "num_rows": len(pairing_rows),
            "num_ok": len(ok_rows),
            "trend_alpha_vs_delta0_squared": trend_by_delta0(ok_rows),
            "delta0_zero_alpha_real_mean": next((row["alpha_real_mean"] for row in ok_rows if abs(float(row["delta0_eV"])) <= 1e-15), None),
            "max_missing_fraction_real": max((float(row["missing_fraction_real"]) for row in ok_rows), default=None),
            "min_parallelism_abs_overlap": min((float(row["parallelism_abs_overlap_mean"]) for row in ok_rows), default=None),
            "valid_for_casimir_input": False,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "requires_analytic_derivation_before_convention_change": True,
            "valid_for_casimir_input": False,
            "reason": "pairing_contact_missing_audit_not_production_convention",
        },
        "model": {"name": model_name, "valid_for_casimir_input": False},
        "debug_parameters": {
            "pairings": [str(value) for value in pairings],
            "delta0_values_eV": [float(value) for value in delta0_values],
            "matsubara_index": int(matsubara_index),
            "temperature_K": float(temperature_K),
            "q_value": float(q_value),
            "nk": int(nk),
            "eta_eV": float(eta_eV),
            "shift_fractions": [float(value) for value in shift_fractions],
            "contact_scale": float(contact_scale),
            "candidate_name": candidate_name,
            "fail_fast": bool(fail_fast),
            "valid_for_casimir_input": False,
        },
        "summary_rows": rows,
        "by_pairing": by_pairing,
        "detailed_contact_formula_payloads": detailed_payloads,
        "interpretation_guardrails": {
            "not_a_fit_fix": True,
            "production_contact_coefficient_must_be_derived_not_fitted": True,
            "delta0_zero_alpha_near_one": "normal-state Peierls contact likely consistent; mismatch tied to superconducting sector",
            "spm_alpha_near_one_but_dwave_not": "momentum-dependent or bond-pairing gauge contact is a strong suspect",
            "alpha_deviation_scales_with_delta0_squared": "missing contribution is likely pairing-amplitude controlled",
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_and_write_pairing_contact_missing_audit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_pairing_contact_missing_audit(**kwargs)
    write_json(Path(output_dir) / "pairing_contact_missing_audit.json", payload)
    return payload
