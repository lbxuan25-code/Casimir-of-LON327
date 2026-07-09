"""Diagnostic-only RHS-aware finite-q validation summary.

This module intentionally lives in sandbox/finite_q_tmte.  It wraps the
Schur-effective RHS audit into a production-style summary without changing the
main validation flow or declaring Casimir readiness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..io.writers import write_json
from .schur_effective_translation_rhs_audit import (
    DEFAULT_CANDIDATE,
    run_schur_effective_translation_rhs_audit,
)

SCHEMA_VERSION = "finite_q_tmte_rhs_aware_finite_q_validation_v1"
DEFAULT_RESIDUAL_TOL = 1e-9
DEFAULT_CONDITION_MAX = 1e12


def _finite_float(value: Any) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        return float("inf")
    return parsed


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-30) -> float:
    return float(numerator) / max(abs(float(denominator)), eps)


def _side_metrics(side: dict[str, Any], k_eff_norm: float) -> dict[str, Any]:
    r_eff_norm = _finite_float(side["effective_rhs_predicted"]["norm"])
    k_contract_norm = _finite_float(side["effective_direct"]["norm"])
    return {
        "s_channel_residual_over_rhs_s": _finite_float(side["s_channel_residual"]["norm_over_reference"]),
        "effective_residual_over_reference": _finite_float(side["effective_residual_over_reference"]),
        "eta_projection_over_rhs_s": _finite_float(side["eta_projection_over_rhs_s"]),
        "eta_channel_norm": _finite_float(side["eta_channel_total_C_eta"]["norm"]),
        "r_eff_norm": r_eff_norm,
        "k_eff_contraction_norm": k_contract_norm,
        "r_eff_over_k_eff_norm": _safe_ratio(r_eff_norm, k_eff_norm),
        "legacy_zero_rhs_residual_over_k_eff_norm": _safe_ratio(k_contract_norm, k_eff_norm),
        "valid_for_casimir_input": False,
    }


def summarize_schur_audit(
    schur_audit: dict[str, Any],
    *,
    residual_tol: float = DEFAULT_RESIDUAL_TOL,
    condition_max: float = DEFAULT_CONDITION_MAX,
) -> dict[str, Any]:
    """Build a compact RHS-aware validation summary from a Schur audit payload."""

    block_norms = schur_audit["block_norms"]
    k_eff_norm = _finite_float(block_norms["K_eff_norm"])
    summary = schur_audit["summary"]
    solve_meta = schur_audit["schur_solve_metadata"]
    condition = _finite_float(solve_meta["etaeta_condition_number"])
    left = _side_metrics(schur_audit["ward_decomposition"]["left"], k_eff_norm)
    right = _side_metrics(schur_audit["ward_decomposition"]["right"], k_eff_norm)
    max_s_residual = max(left["s_channel_residual_over_rhs_s"], right["s_channel_residual_over_rhs_s"])
    max_effective_residual = max(left["effective_residual_over_reference"], right["effective_residual_over_reference"])
    max_eta_projection = max(left["eta_projection_over_rhs_s"], right["eta_projection_over_rhs_s"])
    max_legacy_zero_rhs = max(left["legacy_zero_rhs_residual_over_k_eff_norm"], right["legacy_zero_rhs_residual_over_k_eff_norm"])
    primitive_s_channel_closed = bool(max_s_residual <= float(residual_tol))
    schur_effective_closed = bool(max_effective_residual <= float(residual_tol))
    condition_ok = bool(condition <= float(condition_max) and not bool(solve_meta.get("numerically_suspect", False)))
    rhs_aware_ward_closed = bool(primitive_s_channel_closed and schur_effective_closed and condition_ok)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "rhs_aware_ward_closed": rhs_aware_ward_closed,
            "primitive_s_channel_closed": primitive_s_channel_closed,
            "schur_effective_closed": schur_effective_closed,
            "condition_ok": condition_ok,
            "zero_rhs_target_invalid_at_finite_q": True,
            "valid_for_casimir_input": False,
            "reason": "rhs_aware_validation_summary_diagnostic_only",
        },
        "model": schur_audit["model"],
        "frequency": schur_audit["frequency"],
        "debug_parameters": schur_audit["debug_parameters"],
        "thresholds": {
            "residual_tol": float(residual_tol),
            "condition_max": float(condition_max),
            "valid_for_casimir_input": False,
        },
        "metrics": {
            "left": left,
            "right": right,
            "max_s_channel_residual_over_rhs_s": max_s_residual,
            "max_effective_residual_over_reference": max_effective_residual,
            "max_eta_projection_over_rhs_s": max_eta_projection,
            "max_legacy_zero_rhs_residual_over_k_eff_norm": max_legacy_zero_rhs,
            "K_eff_norm": k_eff_norm,
            "K_SS_norm": _finite_float(block_norms["K_SS_norm"]),
            "Schur_correction_norm": _finite_float(block_norms["Schur_correction_norm"]),
            "K_etaeta_condition_number": condition,
            "valid_for_casimir_input": False,
        },
        "legacy_zero_rhs_check": {
            "status": "invalid_target_at_finite_q",
            "left_zero_rhs_residual_over_k_eff_norm": left["legacy_zero_rhs_residual_over_k_eff_norm"],
            "right_zero_rhs_residual_over_k_eff_norm": right["legacy_zero_rhs_residual_over_k_eff_norm"],
            "valid_for_casimir_input": False,
        },
        "interpretation_guardrails": {
            "does_not_modify_main_validation": True,
            "does_not_define_casimir_readiness": True,
            "contact_scale_is_diagnostic_only": True,
            "requires_separate_nk_shift_convergence": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_rhs_aware_finite_q_validation(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_value: float,
    nk: int,
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    contact_scale: float = 1.0,
    candidate_name: str = DEFAULT_CANDIDATE,
    residual_tol: float = DEFAULT_RESIDUAL_TOL,
    condition_max: float = DEFAULT_CONDITION_MAX,
    include_raw_schur_audit: bool = False,
) -> dict[str, Any]:
    schur = run_schur_effective_translation_rhs_audit(
        model_name=model_name,
        pairing_name=pairing_name,
        matsubara_index=matsubara_index,
        temperature_K=temperature_K,
        q_value=q_value,
        nk=nk,
        delta0_eV=delta0_eV,
        eta_eV=eta_eV,
        shift_fractions=shift_fractions,
        contact_scale=contact_scale,
        candidate_name=candidate_name,
    )
    payload = summarize_schur_audit(schur, residual_tol=residual_tol, condition_max=condition_max)
    if include_raw_schur_audit:
        payload["raw_schur_effective_translation_rhs_audit"] = schur
    return payload


def run_and_write_rhs_aware_finite_q_validation(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_rhs_aware_finite_q_validation(**kwargs)
    write_json(Path(output_dir) / "rhs_aware_finite_q_validation.json", payload)
    return payload
