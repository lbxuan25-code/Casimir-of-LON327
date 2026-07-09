"""Diagnostic-only q scan for the minimal Casimir phi scan path.

The scan evaluates fixed-n, fixed-theta phi scans over several positive q
magnitudes.  It reports phi-average/integral diagnostics and q-weighted radial
integrand diagnostics.  This is not a full q/phi/n Casimir integral.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..io.writers import write_json
from .minimal_casimir_phi_scan import run_minimal_casimir_phi_scan
from .schur_effective_translation_rhs_audit import DEFAULT_CANDIDATE

SCHEMA_VERSION = "finite_q_tmte_minimal_casimir_q_scan_v1"
CSV_COLUMNS = [
    "q_magnitude",
    "phi_average_logdet_real_diagnostic",
    "phi_average_logdet_abs_diagnostic",
    "phi_integral_logdet_real_diagnostic",
    "phi_integral_logdet_abs_diagnostic",
    "q_weighted_phi_average_logdet_real_diagnostic",
    "q_weighted_phi_average_logdet_abs_diagnostic",
    "q_weighted_phi_integral_logdet_real_diagnostic",
    "q_weighted_phi_integral_logdet_abs_diagnostic",
    "range_phi_logdet_abs",
    "max_Rdiff",
    "max_abs_d_logdet_abs_dphi_rad_diagnostic",
    "all_finite_R1",
    "all_finite_R2",
    "all_finite_logdet",
    "all_kappa_match",
    "d_q_weighted_phi_average_logdet_abs_dq_diagnostic",
    "d_q_weighted_phi_integral_logdet_abs_dq_diagnostic",
]


def _normalise_q_values(q_values: Sequence[float]) -> list[float]:
    if not q_values:
        raise ValueError("q_values must not be empty")
    values = [float(q) for q in q_values]
    if any(q <= 0.0 for q in values):
        raise ValueError("q_values must be positive; q=0 requires a separate policy")
    keys = [round(q, 14) for q in values]
    if len(set(keys)) != len(keys):
        raise ValueError("q_values must be unique")
    return sorted(values)


def _row_from_phi_scan(q: float, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    avg_real = summary["periodic_phi_average_logdet_real_diagnostic"]
    avg_abs = summary["periodic_phi_average_logdet_abs_diagnostic"]
    integral_real = summary["periodic_phi_integral_logdet_real_diagnostic"]
    integral_abs = summary["periodic_phi_integral_logdet_abs_diagnostic"]
    return {
        "q_magnitude": float(q),
        "phi_average_logdet_real_diagnostic": None if avg_real is None else float(avg_real),
        "phi_average_logdet_abs_diagnostic": None if avg_abs is None else float(avg_abs),
        "phi_integral_logdet_real_diagnostic": None if integral_real is None else float(integral_real),
        "phi_integral_logdet_abs_diagnostic": None if integral_abs is None else float(integral_abs),
        "q_weighted_phi_average_logdet_real_diagnostic": None if avg_real is None else float(q * float(avg_real)),
        "q_weighted_phi_average_logdet_abs_diagnostic": None if avg_abs is None else float(q * float(avg_abs)),
        "q_weighted_phi_integral_logdet_real_diagnostic": None if integral_real is None else float(q * float(integral_real)),
        "q_weighted_phi_integral_logdet_abs_diagnostic": None if integral_abs is None else float(q * float(integral_abs)),
        "range_phi_logdet_abs": float(summary["range_logdet_abs"]),
        "max_Rdiff": float(summary["max_Rdiff"]),
        "max_abs_d_logdet_abs_dphi_rad_diagnostic": summary["max_abs_d_logdet_abs_dphi_rad_diagnostic"],
        "all_finite_R1": bool(summary["all_finite_R1"]),
        "all_finite_R2": bool(summary["all_finite_R2"]),
        "all_finite_logdet": bool(summary["all_finite_logdet"]),
        "all_kappa_match": bool(summary["all_kappa_match"]),
        "d_q_weighted_phi_average_logdet_abs_dq_diagnostic": None,
        "d_q_weighted_phi_integral_logdet_abs_dq_diagnostic": None,
    }


def _fill_q_derivatives(rows: list[dict[str, Any]]) -> None:
    if len(rows) < 2:
        return
    rows.sort(key=lambda row: float(row["q_magnitude"]))
    q = np.asarray([float(row["q_magnitude"]) for row in rows], dtype=float)
    avg_abs = np.asarray([float(row["q_weighted_phi_average_logdet_abs_diagnostic"]) for row in rows], dtype=float)
    integral_abs = np.asarray([float(row["q_weighted_phi_integral_logdet_abs_diagnostic"]) for row in rows], dtype=float)
    d_avg = np.gradient(avg_abs, q)
    d_integral = np.gradient(integral_abs, q)
    for row, da, di in zip(rows, d_avg, d_integral):
        row["d_q_weighted_phi_average_logdet_abs_dq_diagnostic"] = float(da)
        row["d_q_weighted_phi_integral_logdet_abs_dq_diagnostic"] = float(di)


def _trapezoid_q_integral(rows: list[dict[str, Any]], field: str) -> float | None:
    if len(rows) < 2:
        return None
    rows_sorted = sorted(rows, key=lambda row: float(row["q_magnitude"]))
    q = np.asarray([float(row["q_magnitude"]) for row in rows_sorted], dtype=float)
    values = np.asarray([float(row[field]) for row in rows_sorted], dtype=float)
    return float(np.trapz(values, q))


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_rows": 0, "valid_for_casimir_input": False}
    q_values = np.asarray([float(row["q_magnitude"]) for row in rows], dtype=float)
    q_weighted_abs = np.asarray([float(row["q_weighted_phi_integral_logdet_abs_diagnostic"]) for row in rows], dtype=float)
    q_weighted_avg_abs = np.asarray([float(row["q_weighted_phi_average_logdet_abs_diagnostic"]) for row in rows], dtype=float)
    range_phi = np.asarray([float(row["range_phi_logdet_abs"]) for row in rows], dtype=float)
    max_rdiff = np.asarray([float(row["max_Rdiff"]) for row in rows], dtype=float)
    return {
        "num_rows": len(rows),
        "min_q": float(np.min(q_values)),
        "max_q": float(np.max(q_values)),
        "all_finite_R1": all(bool(row["all_finite_R1"]) for row in rows),
        "all_finite_R2": all(bool(row["all_finite_R2"]) for row in rows),
        "all_finite_logdet": all(bool(row["all_finite_logdet"]) for row in rows),
        "all_kappa_match": all(bool(row["all_kappa_match"]) for row in rows),
        "min_q_weighted_phi_integral_logdet_abs_diagnostic": float(np.min(q_weighted_abs)),
        "max_q_weighted_phi_integral_logdet_abs_diagnostic": float(np.max(q_weighted_abs)),
        "min_q_weighted_phi_average_logdet_abs_diagnostic": float(np.min(q_weighted_avg_abs)),
        "max_q_weighted_phi_average_logdet_abs_diagnostic": float(np.max(q_weighted_avg_abs)),
        "max_range_phi_logdet_abs": float(np.max(range_phi)),
        "max_Rdiff_over_q": float(np.max(max_rdiff)),
        "q_trapezoid_integral_of_q_weighted_phi_integral_logdet_real_diagnostic": _trapezoid_q_integral(rows, "q_weighted_phi_integral_logdet_real_diagnostic"),
        "q_trapezoid_integral_of_q_weighted_phi_integral_logdet_abs_diagnostic": _trapezoid_q_integral(rows, "q_weighted_phi_integral_logdet_abs_diagnostic"),
        "q_trapezoid_integral_of_q_weighted_phi_average_logdet_real_diagnostic": _trapezoid_q_integral(rows, "q_weighted_phi_average_logdet_real_diagnostic"),
        "q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic": _trapezoid_q_integral(rows, "q_weighted_phi_average_logdet_abs_diagnostic"),
        "valid_for_casimir_input": False,
    }


def run_minimal_casimir_q_scan(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
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
    include_phi_scan_payloads: bool = False,
) -> dict[str, Any]:
    if int(matsubara_index) <= 0:
        raise ValueError("q scan currently supports only n>=1; n=0 is a separate static-limit problem")
    if int(nk) <= 0:
        raise ValueError("nk must be positive")
    if float(separation_nm) <= 0.0:
        raise ValueError("separation_nm must be positive")
    q_list = _normalise_q_values(q_values)

    rows: list[dict[str, Any]] = []
    phi_scan_payloads: list[dict[str, Any]] = []
    for q in q_list:
        phi_scan = run_minimal_casimir_phi_scan(
            model_name=model_name,
            pairing_name=pairing_name,
            matsubara_index=matsubara_index,
            temperature_K=temperature_K,
            q_magnitude=q,
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
            include_point_payloads=False,
        )
        rows.append(_row_from_phi_scan(q, phi_scan))
        if include_phi_scan_payloads:
            phi_scan_payloads.append(phi_scan)

    _fill_q_derivatives(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "q_scan_only": True,
            "uses_phi_scan_diagnostic": True,
            "q_weighted_radial_integrand_diagnostic_only": True,
            "q_trapezoid_integral_diagnostic_only": True,
            "single_matsubara_index_only": True,
            "single_theta_only": True,
            "single_separation_only": True,
            "no_matsubara_sum": True,
            "no_n0_policy": True,
            "no_q0_policy": True,
            "not_a_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "input": {
            "model_name": model_name,
            "pairing_name": pairing_name,
            "matsubara_index": int(matsubara_index),
            "temperature_K": float(temperature_K),
            "q_values": q_list,
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
            "include_phi_scan_payloads": bool(include_phi_scan_payloads),
            "valid_for_casimir_input": False,
        },
        "summary": _summary_from_rows(rows),
        "rows": rows,
        "phi_scan_payloads": phi_scan_payloads if include_phi_scan_payloads else None,
        "interpretation_guardrails": {
            "q_weighted_quantities_include_2d_radial_measure_q_but_no_physical_prefactor": True,
            "q_trapezoid_integral_is_diagnostic_only": True,
            "does_not_modify_main_validation": True,
            "does_not_modify_main_casimir_pipeline": True,
            "not_a_full_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def write_q_scan_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def run_and_write_minimal_casimir_q_scan(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_minimal_casimir_q_scan(**kwargs)
    output = Path(output_dir)
    write_json(output / "minimal_casimir_q_scan.json", payload)
    write_q_scan_csv(output / "minimal_casimir_q_scan.csv", payload["rows"])
    return payload
