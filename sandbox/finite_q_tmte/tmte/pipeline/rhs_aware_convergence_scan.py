"""Diagnostic-only RHS-aware finite-q convergence scan."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..io.writers import write_json
from .rhs_aware_finite_q_validation import (
    DEFAULT_CONDITION_MAX,
    DEFAULT_RESIDUAL_TOL,
    run_rhs_aware_finite_q_validation,
)
from .schur_effective_translation_rhs_audit import DEFAULT_CANDIDATE

SCHEMA_VERSION = "finite_q_tmte_rhs_aware_convergence_scan_v1"
SHIFT_MODE_FRACTIONS: dict[str, tuple[float, ...]] = {
    "noshift": (0.0,),
    "shifted2": (0.0, 0.5),
    "shifted5": (0.0, 0.2, 0.4, 0.6, 0.8),
}
RELATIVE_EPS = 1e-30


def shift_fractions_for_mode(mode: str) -> tuple[float, ...]:
    try:
        return SHIFT_MODE_FRACTIONS[str(mode)]
    except KeyError as exc:
        raise ValueError(f"unknown shift mode {mode!r}; expected one of {sorted(SHIFT_MODE_FRACTIONS)}") from exc


def _relative_change(new: float, old: float, eps: float = RELATIVE_EPS) -> float:
    return float(abs(float(new) - float(old)) / max(abs(float(new)), abs(float(old)), eps))


def _row_from_validation(payload: dict[str, Any], *, pairing: str, n: int, q: float, nk: int, shift_mode: str) -> dict[str, Any]:
    metrics = payload["metrics"]
    status = payload["status"]
    left = metrics["left"]
    right = metrics["right"]
    return {
        "pairing": str(pairing),
        "matsubara_index": int(n),
        "q_value": float(q),
        "nk": int(nk),
        "shift_mode": str(shift_mode),
        "num_shifted_meshes": int(payload["debug_parameters"]["shifted_mesh_average"]["num_shifted_meshes"]),
        "rhs_aware_ward_closed": bool(status["rhs_aware_ward_closed"]),
        "primitive_s_channel_closed": bool(status["primitive_s_channel_closed"]),
        "schur_effective_closed": bool(status["schur_effective_closed"]),
        "condition_ok": bool(status["condition_ok"]),
        "max_s_channel_residual_over_rhs_s": float(metrics["max_s_channel_residual_over_rhs_s"]),
        "max_effective_residual_over_reference": float(metrics["max_effective_residual_over_reference"]),
        "max_eta_projection_over_rhs_s": float(metrics["max_eta_projection_over_rhs_s"]),
        "max_legacy_zero_rhs_residual_over_k_eff_norm": float(metrics["max_legacy_zero_rhs_residual_over_k_eff_norm"]),
        "K_eff_norm": float(metrics["K_eff_norm"]),
        "K_SS_norm": float(metrics["K_SS_norm"]),
        "Schur_correction_norm": float(metrics["Schur_correction_norm"]),
        "K_etaeta_condition_number": float(metrics["K_etaeta_condition_number"]),
        "left_R_eff_norm": float(left["r_eff_norm"]),
        "right_R_eff_norm": float(right["r_eff_norm"]),
        "left_R_eff_over_K_eff_norm": float(left["r_eff_over_k_eff_norm"]),
        "right_R_eff_over_K_eff_norm": float(right["r_eff_over_k_eff_norm"]),
        "left_eta_projection_over_rhs_s": float(left["eta_projection_over_rhs_s"]),
        "right_eta_projection_over_rhs_s": float(right["eta_projection_over_rhs_s"]),
        "valid_for_casimir_input": False,
    }


def _row_key(row: dict[str, Any], omit: Iterable[str]) -> tuple[Any, ...]:
    omitted = set(omit)
    return tuple((key, row[key]) for key in sorted(row) if key not in omitted and key in {"pairing", "matsubara_index", "q_value", "nk", "shift_mode"})


def compute_nk_convergence(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["pairing"], row["matsubara_index"], row["q_value"], row["shift_mode"])
        grouped[key].append(row)
    comparisons: list[dict[str, Any]] = []
    for (pairing, n, q, shift_mode), group in grouped.items():
        ordered = sorted(group, key=lambda item: int(item["nk"]))
        for old, new in zip(ordered, ordered[1:]):
            comparisons.append(
                {
                    "pairing": pairing,
                    "matsubara_index": int(n),
                    "q_value": float(q),
                    "shift_mode": shift_mode,
                    "nk_from": int(old["nk"]),
                    "nk_to": int(new["nk"]),
                    "relative_change_K_eff_norm": _relative_change(new["K_eff_norm"], old["K_eff_norm"]),
                    "relative_change_left_R_eff_norm": _relative_change(new["left_R_eff_norm"], old["left_R_eff_norm"]),
                    "relative_change_right_R_eff_norm": _relative_change(new["right_R_eff_norm"], old["right_R_eff_norm"]),
                    "relative_change_eta_projection_over_rhs_s": _relative_change(new["max_eta_projection_over_rhs_s"], old["max_eta_projection_over_rhs_s"]),
                    "relative_change_condition_number": _relative_change(new["K_etaeta_condition_number"], old["K_etaeta_condition_number"]),
                    "valid_for_casimir_input": False,
                }
            )
    return comparisons


def compute_shift_convergence(rows: Sequence[dict[str, Any]], *, reference_shift_mode: str = "noshift") -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["pairing"], row["matsubara_index"], row["q_value"], row["nk"])
        grouped[key][str(row["shift_mode"])] = row
    comparisons: list[dict[str, Any]] = []
    for (pairing, n, q, nk), by_shift in grouped.items():
        if reference_shift_mode not in by_shift:
            continue
        ref = by_shift[reference_shift_mode]
        for mode, row in sorted(by_shift.items()):
            if mode == reference_shift_mode:
                continue
            comparisons.append(
                {
                    "pairing": pairing,
                    "matsubara_index": int(n),
                    "q_value": float(q),
                    "nk": int(nk),
                    "shift_from": reference_shift_mode,
                    "shift_to": mode,
                    "relative_change_K_eff_norm": _relative_change(row["K_eff_norm"], ref["K_eff_norm"]),
                    "relative_change_left_R_eff_norm": _relative_change(row["left_R_eff_norm"], ref["left_R_eff_norm"]),
                    "relative_change_right_R_eff_norm": _relative_change(row["right_R_eff_norm"], ref["right_R_eff_norm"]),
                    "relative_change_eta_projection_over_rhs_s": _relative_change(row["max_eta_projection_over_rhs_s"], ref["max_eta_projection_over_rhs_s"]),
                    "relative_change_condition_number": _relative_change(row["K_etaeta_condition_number"], ref["K_etaeta_condition_number"]),
                    "valid_for_casimir_input": False,
                }
            )
    return comparisons


def aggregate_rows(rows: Sequence[dict[str, Any]], nk_convergence: Sequence[dict[str, Any]], shift_convergence: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_rows": 0,
            "num_rhs_aware_closed": 0,
            "all_rhs_aware_closed": False,
            "valid_for_casimir_input": False,
        }
    return {
        "num_rows": len(rows),
        "num_rhs_aware_closed": sum(1 for row in rows if bool(row["rhs_aware_ward_closed"])),
        "all_rhs_aware_closed": all(bool(row["rhs_aware_ward_closed"]) for row in rows),
        "max_s_channel_residual_over_rhs_s": max(float(row["max_s_channel_residual_over_rhs_s"]) for row in rows),
        "max_effective_residual_over_reference": max(float(row["max_effective_residual_over_reference"]) for row in rows),
        "max_eta_projection_over_rhs_s": max(float(row["max_eta_projection_over_rhs_s"]) for row in rows),
        "max_legacy_zero_rhs_residual_over_k_eff_norm": max(float(row["max_legacy_zero_rhs_residual_over_k_eff_norm"]) for row in rows),
        "max_K_etaeta_condition_number": max(float(row["K_etaeta_condition_number"]) for row in rows),
        "pairing_counts": dict(Counter(str(row["pairing"]) for row in rows)),
        "shift_mode_counts": dict(Counter(str(row["shift_mode"]) for row in rows)),
        "num_nk_convergence_pairs": len(nk_convergence),
        "max_nk_relative_change_K_eff_norm": max((float(row["relative_change_K_eff_norm"]) for row in nk_convergence), default=0.0),
        "max_nk_relative_change_R_eff_norm": max((max(float(row["relative_change_left_R_eff_norm"]), float(row["relative_change_right_R_eff_norm"])) for row in nk_convergence), default=0.0),
        "max_nk_relative_change_eta_projection_over_rhs_s": max((float(row["relative_change_eta_projection_over_rhs_s"]) for row in nk_convergence), default=0.0),
        "num_shift_convergence_pairs": len(shift_convergence),
        "max_shift_relative_change_K_eff_norm": max((float(row["relative_change_K_eff_norm"]) for row in shift_convergence), default=0.0),
        "max_shift_relative_change_R_eff_norm": max((max(float(row["relative_change_left_R_eff_norm"]), float(row["relative_change_right_R_eff_norm"])) for row in shift_convergence), default=0.0),
        "max_shift_relative_change_eta_projection_over_rhs_s": max((float(row["relative_change_eta_projection_over_rhs_s"]) for row in shift_convergence), default=0.0),
        "valid_for_casimir_input": False,
    }


def run_rhs_aware_convergence_scan(
    *,
    model_name: str,
    pairings: Sequence[str],
    matsubara_indices: Sequence[int],
    temperature_K: float,
    q_values: Sequence[float],
    nk_values: Sequence[int],
    shift_modes: Sequence[str] = ("noshift",),
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    contact_scale: float = 1.0,
    candidate_name: str = DEFAULT_CANDIDATE,
    residual_tol: float = DEFAULT_RESIDUAL_TOL,
    condition_max: float = DEFAULT_CONDITION_MAX,
    include_validation_payloads: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    validation_payloads: list[dict[str, Any]] = []
    for pairing in pairings:
        for n in matsubara_indices:
            for q in q_values:
                for nk in nk_values:
                    for shift_mode in shift_modes:
                        validation = run_rhs_aware_finite_q_validation(
                            model_name=model_name,
                            pairing_name=str(pairing),
                            matsubara_index=int(n),
                            temperature_K=temperature_K,
                            q_value=float(q),
                            nk=int(nk),
                            delta0_eV=delta0_eV,
                            eta_eV=eta_eV,
                            shift_fractions=shift_fractions_for_mode(str(shift_mode)),
                            contact_scale=contact_scale,
                            candidate_name=candidate_name,
                            residual_tol=residual_tol,
                            condition_max=condition_max,
                            include_raw_schur_audit=False,
                        )
                        rows.append(_row_from_validation(validation, pairing=str(pairing), n=int(n), q=float(q), nk=int(nk), shift_mode=str(shift_mode)))
                        if include_validation_payloads:
                            validation_payloads.append(validation)
    nk_convergence = compute_nk_convergence(rows)
    shift_convergence = compute_shift_convergence(rows)
    aggregate = aggregate_rows(rows, nk_convergence, shift_convergence)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "does_not_modify_main_validation": True,
            "valid_for_casimir_input": False,
            "reason": "rhs_aware_convergence_scan_diagnostic_only",
        },
        "scan_parameters": {
            "model_name": model_name,
            "pairings": [str(item) for item in pairings],
            "matsubara_indices": [int(item) for item in matsubara_indices],
            "temperature_K": float(temperature_K),
            "q_values": [float(item) for item in q_values],
            "nk_values": [int(item) for item in nk_values],
            "shift_modes": [str(item) for item in shift_modes],
            "shift_mode_fractions": {str(mode): [float(value) for value in shift_fractions_for_mode(str(mode))] for mode in shift_modes},
            "delta0_eV": None if delta0_eV is None else float(delta0_eV),
            "eta_eV": float(eta_eV),
            "contact_scale": float(contact_scale),
            "candidate_name": candidate_name,
            "residual_tol": float(residual_tol),
            "condition_max": float(condition_max),
            "valid_for_casimir_input": False,
        },
        "aggregate": aggregate,
        "rows": rows,
        "nk_convergence": nk_convergence,
        "shift_convergence": shift_convergence,
        "interpretation_guardrails": {
            "convergence_metrics_are_norm_level_diagnostics": True,
            "not_a_final_casimir_error_budget": True,
            "zero_rhs_target_invalid_at_finite_q": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }
    if include_validation_payloads:
        payload["validation_payloads"] = validation_payloads
    return payload


def run_and_write_rhs_aware_convergence_scan(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_rhs_aware_convergence_scan(**kwargs)
    write_json(Path(output_dir) / "rhs_aware_convergence_scan.json", payload)
    return payload
