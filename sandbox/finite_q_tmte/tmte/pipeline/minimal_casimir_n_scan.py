"""Diagnostic-only Matsubara-index scan for the minimal Casimir q scan.

The scan fixes theta, separation, q values, phi values, nk, and shift policy,
then runs the q-scan diagnostic for each positive Matsubara index.  It reports
q-phi diagnostic integrals and their n-dependence.  This is not a Matsubara sum
and does not include an n=0 policy.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..io.writers import write_json
from .minimal_casimir_q_scan import run_minimal_casimir_q_scan
from .schur_effective_translation_rhs_audit import DEFAULT_CANDIDATE

SCHEMA_VERSION = "finite_q_tmte_minimal_casimir_n_scan_v1"
K_B_EV_PER_K = 8.617333262145e-5
CSV_COLUMNS = [
    "matsubara_index",
    "xi_eV",
    "q_trapezoid_integral_of_q_weighted_phi_average_logdet_real_diagnostic",
    "q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic",
    "q_trapezoid_integral_of_q_weighted_phi_integral_logdet_real_diagnostic",
    "q_trapezoid_integral_of_q_weighted_phi_integral_logdet_abs_diagnostic",
    "delta_abs_from_n_min",
    "ratio_abs_to_previous_n",
    "partial_sum_abs_diagnostic_no_prefactor",
    "partial_sum_real_diagnostic_no_prefactor",
    "max_q_weighted_phi_average_logdet_abs_diagnostic",
    "max_range_phi_logdet_abs",
    "max_Rdiff_over_q",
    "all_finite_R1",
    "all_finite_R2",
    "all_finite_logdet",
    "all_kappa_match",
]


def matsubara_xi_eV(index: int, temperature_K: float) -> float:
    if int(index) <= 0:
        raise ValueError("Matsubara index must be positive for this diagnostic; n=0 needs a separate policy")
    return float(2.0 * np.pi * int(index) * K_B_EV_PER_K * float(temperature_K))


def _normalise_matsubara_indices(indices: Sequence[int]) -> list[int]:
    if not indices:
        raise ValueError("matsubara_indices must not be empty")
    values = [int(n) for n in indices]
    if any(n <= 0 for n in values):
        raise ValueError("n scan currently supports only n>=1; n=0 is a separate static-limit problem")
    if len(set(values)) != len(values):
        raise ValueError("matsubara_indices must be unique")
    return sorted(values)


def _row_from_q_scan(*, index: int, temperature_K: float, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    avg_real = summary["q_trapezoid_integral_of_q_weighted_phi_average_logdet_real_diagnostic"]
    avg_abs = summary["q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic"]
    int_real = summary["q_trapezoid_integral_of_q_weighted_phi_integral_logdet_real_diagnostic"]
    int_abs = summary["q_trapezoid_integral_of_q_weighted_phi_integral_logdet_abs_diagnostic"]
    return {
        "matsubara_index": int(index),
        "xi_eV": matsubara_xi_eV(index, temperature_K),
        "q_trapezoid_integral_of_q_weighted_phi_average_logdet_real_diagnostic": None if avg_real is None else float(avg_real),
        "q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic": None if avg_abs is None else float(avg_abs),
        "q_trapezoid_integral_of_q_weighted_phi_integral_logdet_real_diagnostic": None if int_real is None else float(int_real),
        "q_trapezoid_integral_of_q_weighted_phi_integral_logdet_abs_diagnostic": None if int_abs is None else float(int_abs),
        "delta_abs_from_n_min": None,
        "ratio_abs_to_previous_n": None,
        "partial_sum_abs_diagnostic_no_prefactor": None,
        "partial_sum_real_diagnostic_no_prefactor": None,
        "max_q_weighted_phi_average_logdet_abs_diagnostic": float(summary["max_q_weighted_phi_average_logdet_abs_diagnostic"]),
        "max_range_phi_logdet_abs": float(summary["max_range_phi_logdet_abs"]),
        "max_Rdiff_over_q": float(summary["max_Rdiff_over_q"]),
        "all_finite_R1": bool(summary["all_finite_R1"]),
        "all_finite_R2": bool(summary["all_finite_R2"]),
        "all_finite_logdet": bool(summary["all_finite_logdet"]),
        "all_kappa_match": bool(summary["all_kappa_match"]),
    }


def _fill_n_diagnostics(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rows.sort(key=lambda row: int(row["matsubara_index"]))
    reference_abs = float(rows[0]["q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic"])
    prev_abs: float | None = None
    partial_abs = 0.0
    partial_real = 0.0
    for row in rows:
        abs_value = float(row["q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic"])
        real_value = float(row["q_trapezoid_integral_of_q_weighted_phi_average_logdet_real_diagnostic"])
        row["delta_abs_from_n_min"] = float(abs_value - reference_abs)
        row["ratio_abs_to_previous_n"] = None if prev_abs is None or prev_abs == 0.0 else float(abs_value / prev_abs)
        partial_abs += abs_value
        partial_real += real_value
        row["partial_sum_abs_diagnostic_no_prefactor"] = float(partial_abs)
        row["partial_sum_real_diagnostic_no_prefactor"] = float(partial_real)
        prev_abs = abs_value


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_rows": 0, "valid_for_casimir_input": False}
    abs_values = np.asarray([float(row["q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic"]) for row in rows], dtype=float)
    real_values = np.asarray([float(row["q_trapezoid_integral_of_q_weighted_phi_average_logdet_real_diagnostic"]) for row in rows], dtype=float)
    max_rdiff = np.asarray([float(row["max_Rdiff_over_q"]) for row in rows], dtype=float)
    max_phi_range = np.asarray([float(row["max_range_phi_logdet_abs"]) for row in rows], dtype=float)
    indices = [int(row["matsubara_index"]) for row in rows]
    return {
        "num_rows": len(rows),
        "min_matsubara_index": min(indices),
        "max_matsubara_index": max(indices),
        "min_xi_eV": float(min(float(row["xi_eV"]) for row in rows)),
        "max_xi_eV": float(max(float(row["xi_eV"]) for row in rows)),
        "all_finite_R1": all(bool(row["all_finite_R1"]) for row in rows),
        "all_finite_R2": all(bool(row["all_finite_R2"]) for row in rows),
        "all_finite_logdet": all(bool(row["all_finite_logdet"]) for row in rows),
        "all_kappa_match": all(bool(row["all_kappa_match"]) for row in rows),
        "min_q_phi_average_abs_diagnostic": float(np.min(abs_values)),
        "max_q_phi_average_abs_diagnostic": float(np.max(abs_values)),
        "last_q_phi_average_abs_diagnostic": float(abs_values[-1]),
        "last_to_first_abs_ratio_diagnostic": None if abs_values[0] == 0.0 else float(abs_values[-1] / abs_values[0]),
        "partial_sum_abs_diagnostic_no_prefactor": float(np.sum(abs_values)),
        "partial_sum_real_diagnostic_no_prefactor": float(np.sum(real_values)),
        "max_Rdiff_over_nq": float(np.max(max_rdiff)),
        "max_range_phi_logdet_abs_over_nq": float(np.max(max_phi_range)),
        "matsubara_tail_not_estimated": True,
        "n0_policy_included": False,
        "valid_for_casimir_input": False,
    }


def run_minimal_casimir_n_scan(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_indices: Sequence[int],
    temperature_K: float,
    q_values: Sequence[float],
    phi_values_deg: Sequence[float],
    plate2_theta_deg: float,
    nk: int,
    separation_nm: float,
    plate1_theta_deg: float = 0.0,
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    candidate_name: str = DEFAULT_CANDIDATE,
    include_rhs_aware_validation: bool = True,
    include_q_scan_payloads: bool = False,
) -> dict[str, Any]:
    if int(nk) <= 0:
        raise ValueError("nk must be positive")
    if float(separation_nm) <= 0.0:
        raise ValueError("separation_nm must be positive")
    n_list = _normalise_matsubara_indices(matsubara_indices)

    rows: list[dict[str, Any]] = []
    q_scan_payloads: list[dict[str, Any]] = []
    for n in n_list:
        q_scan = run_minimal_casimir_q_scan(
            model_name=model_name,
            pairing_name=pairing_name,
            matsubara_index=n,
            temperature_K=temperature_K,
            q_values=q_values,
            phi_values_deg=phi_values_deg,
            plate1_theta_deg=plate1_theta_deg,
            plate2_theta_deg=plate2_theta_deg,
            nk=nk,
            separation_nm=separation_nm,
            delta0_eV=delta0_eV,
            eta_eV=eta_eV,
            shift_fractions=shift_fractions,
            candidate_name=candidate_name,
            include_rhs_aware_validation=include_rhs_aware_validation,
            include_phi_scan_payloads=False,
        )
        rows.append(_row_from_q_scan(index=n, temperature_K=temperature_K, payload=q_scan))
        if include_q_scan_payloads:
            q_scan_payloads.append(q_scan)

    _fill_n_diagnostics(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "n_scan_only": True,
            "uses_q_scan_diagnostic": True,
            "uses_q_phi_average_diagnostic": True,
            "matsubara_indices_positive_only": True,
            "single_theta_only": True,
            "single_separation_only": True,
            "no_n0_policy": True,
            "no_matsubara_tail_extrapolation": True,
            "not_a_matsubara_sum": True,
            "not_a_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "input": {
            "model_name": model_name,
            "pairing_name": pairing_name,
            "matsubara_indices": n_list,
            "temperature_K": float(temperature_K),
            "q_values": [float(v) for v in q_values],
            "phi_values_deg": [float(v) for v in phi_values_deg],
            "plate1_theta_deg": float(plate1_theta_deg),
            "plate2_theta_deg": float(plate2_theta_deg),
            "nk": int(nk),
            "separation_nm": float(separation_nm),
            "delta0_eV": None if delta0_eV is None else float(delta0_eV),
            "eta_eV": float(eta_eV),
            "shift_fractions": [float(v) for v in shift_fractions],
            "candidate_name": candidate_name,
            "include_rhs_aware_validation": bool(include_rhs_aware_validation),
            "include_q_scan_payloads": bool(include_q_scan_payloads),
            "xi_eV_formula": "2*pi*n*k_B*T",
            "k_B_eV_per_K": K_B_EV_PER_K,
            "valid_for_casimir_input": False,
        },
        "summary": _summary_from_rows(rows),
        "rows": rows,
        "q_scan_payloads": q_scan_payloads if include_q_scan_payloads else None,
        "interpretation_guardrails": {
            "partial_sums_omit_matsubara_prefactor": True,
            "partial_sums_are_diagnostic_only": True,
            "n0_policy_not_included": True,
            "tail_not_estimated": True,
            "does_not_modify_main_validation": True,
            "does_not_modify_main_casimir_pipeline": True,
            "not_a_full_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def write_n_scan_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def run_and_write_minimal_casimir_n_scan(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_minimal_casimir_n_scan(**kwargs)
    output = Path(output_dir)
    write_json(output / "minimal_casimir_n_scan.json", payload)
    write_n_scan_csv(output / "minimal_casimir_n_scan.csv", payload["rows"])
    return payload
