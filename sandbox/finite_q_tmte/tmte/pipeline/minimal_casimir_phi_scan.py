"""Diagnostic-only phi scan for the minimal Casimir theta path.

The scan evaluates the two-plate theta diagnostic over lab-frame q directions
at fixed q magnitude, plate angle, Matsubara index, separation, nk, and shift
fractions.  It emits compact rows plus periodic phi-average/integral
summaries.  These are diagnostics only, not a full q/phi/n Casimir integral.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..io.writers import write_json
from .minimal_casimir_qvec_path import q_geometry_payload, q_model_vector_from_polar
from .minimal_casimir_theta_path import run_minimal_casimir_theta_path
from .schur_effective_translation_rhs_audit import DEFAULT_CANDIDATE

SCHEMA_VERSION = "finite_q_tmte_minimal_casimir_phi_scan_v1"
PHI_PERIOD_DEG = 360.0
PHI_PERIOD_RAD = 2.0 * np.pi
CSV_COLUMNS = [
    "phi_deg",
    "phi_mod_deg",
    "phi_rad",
    "logdet_real",
    "logdet_imag",
    "logdet_abs",
    "delta_logdet_real_from_phi0",
    "delta_logdet_abs_from_phi0",
    "d_logdet_real_dphi_rad_diagnostic",
    "d_logdet_abs_dphi_rad_diagnostic",
    "Rdiff",
    "R1_norm",
    "R2_norm",
    "p1_Keff_norm",
    "p2_Keff_norm",
    "p1_q_crystal_phi_deg",
    "p2_q_crystal_phi_deg",
    "p1_ward_closed",
    "p2_ward_closed",
    "finite_R1",
    "finite_R2",
    "finite_logdet",
    "kappa_match",
]


def _complex_parts(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        return float(value.get("real", 0.0)), float(value.get("imag", 0.0))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    z = complex(value)
    return float(z.real), float(z.imag)


def _ward_closed_label(validation: object) -> object:
    if not isinstance(validation, dict):
        return None
    status = validation.get("status")
    if not isinstance(status, dict):
        return None
    return status.get("rhs_aware_ward_closed")


def _phi_mod_deg(phi_deg: float) -> float:
    value = float(phi_deg) % PHI_PERIOD_DEG
    if np.isclose(value, PHI_PERIOD_DEG, rtol=0.0, atol=1e-12):
        value = 0.0
    return float(value)


def _normalise_phi_values(phi_values_deg: Sequence[float]) -> list[tuple[float, float]]:
    if not phi_values_deg:
        raise ValueError("phi_values_deg must not be empty")
    pairs = [(float(phi), _phi_mod_deg(float(phi))) for phi in phi_values_deg]
    keys = [round(mod, 12) for _, mod in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("phi_values_deg must be unique modulo 360 degrees")
    pairs.sort(key=lambda item: item[1])
    return pairs


def _row_from_point(*, phi_deg: float, phi_mod_deg: float, point: dict[str, Any]) -> dict[str, Any]:
    mixed = point["mixed_trace_log"]
    real, imag = _complex_parts(mixed["logdet_integrand"])
    p1 = point["plate1"]
    p2 = point["plate2"]
    return {
        "phi_deg": float(phi_deg),
        "phi_mod_deg": float(phi_mod_deg),
        "phi_rad": float(np.deg2rad(float(phi_mod_deg))),
        "logdet_real": real,
        "logdet_imag": imag,
        "logdet_abs": float(mixed["logdet_abs"]),
        "delta_logdet_real_from_phi0": None,
        "delta_logdet_abs_from_phi0": None,
        "d_logdet_real_dphi_rad_diagnostic": None,
        "d_logdet_abs_dphi_rad_diagnostic": None,
        "Rdiff": float(mixed["R1_minus_R2_norm"]),
        "R1_norm": float(mixed["R1_TE_TM_norm"]),
        "R2_norm": float(mixed["R2_TE_TM_norm"]),
        "p1_Keff_norm": float(p1["sandbox_response_source"]["K_eff_norm"]),
        "p2_Keff_norm": float(p2["sandbox_response_source"]["K_eff_norm"]),
        "p1_q_crystal_phi_deg": float(point["geometry"]["plate1"]["q_crystal"]["phi_deg"]),
        "p2_q_crystal_phi_deg": float(point["geometry"]["plate2"]["q_crystal"]["phi_deg"]),
        "p1_ward_closed": _ward_closed_label(p1.get("rhs_aware_validation")),
        "p2_ward_closed": _ward_closed_label(p2.get("rhs_aware_validation")),
        "finite_R1": bool(point["sanity_checks"]["finite_R1"]),
        "finite_R2": bool(point["sanity_checks"]["finite_R2"]),
        "finite_logdet": bool(point["sanity_checks"]["finite_logdet"]),
        "kappa_match": bool(point["sanity_checks"]["kappa_match"]),
    }


def _fill_deltas_and_periodic_derivatives(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rows.sort(key=lambda row: float(row["phi_mod_deg"]))
    reference_real = float(rows[0]["logdet_real"])
    reference_abs = float(rows[0]["logdet_abs"])
    for row in rows:
        row["delta_logdet_real_from_phi0"] = float(row["logdet_real"]) - reference_real
        row["delta_logdet_abs_from_phi0"] = float(row["logdet_abs"]) - reference_abs

    if len(rows) < 3:
        return
    phi = np.asarray([float(row["phi_rad"]) for row in rows], dtype=float)
    real = np.asarray([float(row["logdet_real"]) for row in rows], dtype=float)
    absval = np.asarray([float(row["logdet_abs"]) for row in rows], dtype=float)
    n = len(rows)
    for i, row in enumerate(rows):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        x_prev = phi[prev_i]
        x_next = phi[next_i]
        if prev_i > i:
            x_prev -= PHI_PERIOD_RAD
        if next_i < i:
            x_next += PHI_PERIOD_RAD
        row["d_logdet_real_dphi_rad_diagnostic"] = float((real[next_i] - real[prev_i]) / (x_next - x_prev))
        row["d_logdet_abs_dphi_rad_diagnostic"] = float((absval[next_i] - absval[prev_i]) / (x_next - x_prev))


def _periodic_trapezoid(rows: list[dict[str, Any]], field: str) -> float | None:
    if len(rows) < 2:
        return None
    rows_sorted = sorted(rows, key=lambda row: float(row["phi_rad"]))
    phi = np.asarray([float(row["phi_rad"]) for row in rows_sorted], dtype=float)
    values = np.asarray([float(row[field]) for row in rows_sorted], dtype=float)
    total = 0.0
    for i in range(len(rows_sorted)):
        j = (i + 1) % len(rows_sorted)
        delta = phi[j] - phi[i]
        if j == 0:
            delta += PHI_PERIOD_RAD
        total += 0.5 * (values[i] + values[j]) * delta
    return float(total)


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_rows": 0, "valid_for_casimir_input": False}
    logdet_abs = np.asarray([float(row["logdet_abs"]) for row in rows], dtype=float)
    rdiff = np.asarray([float(row["Rdiff"]) for row in rows], dtype=float)
    integral_real = _periodic_trapezoid(rows, "logdet_real")
    integral_abs = _periodic_trapezoid(rows, "logdet_abs")
    d_abs_values = [row["d_logdet_abs_dphi_rad_diagnostic"] for row in rows if row["d_logdet_abs_dphi_rad_diagnostic"] is not None]
    d_real_values = [row["d_logdet_real_dphi_rad_diagnostic"] for row in rows if row["d_logdet_real_dphi_rad_diagnostic"] is not None]
    return {
        "num_rows": len(rows),
        "all_finite_R1": all(bool(row["finite_R1"]) for row in rows),
        "all_finite_R2": all(bool(row["finite_R2"]) for row in rows),
        "all_finite_logdet": all(bool(row["finite_logdet"]) for row in rows),
        "all_kappa_match": all(bool(row["kappa_match"]) for row in rows),
        "min_logdet_abs": float(np.min(logdet_abs)),
        "max_logdet_abs": float(np.max(logdet_abs)),
        "range_logdet_abs": float(np.max(logdet_abs) - np.min(logdet_abs)),
        "max_Rdiff": float(np.max(rdiff)),
        "periodic_phi_integral_logdet_real_diagnostic": integral_real,
        "periodic_phi_integral_logdet_abs_diagnostic": integral_abs,
        "periodic_phi_average_logdet_real_diagnostic": None if integral_real is None else float(integral_real / PHI_PERIOD_RAD),
        "periodic_phi_average_logdet_abs_diagnostic": None if integral_abs is None else float(integral_abs / PHI_PERIOD_RAD),
        "max_abs_d_logdet_real_dphi_rad_diagnostic": None if not d_real_values else float(np.max(np.abs(d_real_values))),
        "max_abs_d_logdet_abs_dphi_rad_diagnostic": None if not d_abs_values else float(np.max(np.abs(d_abs_values))),
        "valid_for_casimir_input": False,
    }


def run_minimal_casimir_phi_scan(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_magnitude: float,
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
    include_point_payloads: bool = False,
) -> dict[str, Any]:
    if int(matsubara_index) <= 0:
        raise ValueError("phi scan currently supports only n>=1; n=0 is a separate static-limit problem")
    if float(q_magnitude) <= 0.0:
        raise ValueError("q_magnitude must be positive")
    if int(nk) <= 0:
        raise ValueError("nk must be positive")
    if float(separation_nm) <= 0.0:
        raise ValueError("separation_nm must be positive")
    phi_pairs = _normalise_phi_values(phi_values_deg)

    point_payloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for phi_original, phi_mod in phi_pairs:
        q_lab = q_model_vector_from_polar(q_magnitude, phi_mod)
        point = run_minimal_casimir_theta_path(
            model_name=model_name,
            pairing_name=pairing_name,
            matsubara_index=matsubara_index,
            temperature_K=temperature_K,
            q_lab_vector=q_lab,
            plate1_theta_deg=plate1_theta_deg,
            plate2_theta_deg=plate2_theta_deg,
            nk=nk,
            separation_nm=separation_nm,
            delta0_eV=delta0_eV,
            eta_eV=eta_eV,
            shift_fractions=shift_fractions,
            candidate_name=candidate_name,
            include_rhs_aware_validation=include_rhs_aware_validation,
        )
        rows.append(_row_from_point(phi_deg=phi_original, phi_mod_deg=phi_mod, point=point))
        if include_point_payloads:
            point_payloads.append(point)

    _fill_deltas_and_periodic_derivatives(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "phi_scan_only": True,
            "periodic_phi_integral_diagnostic_only": True,
            "single_q_magnitude_only": True,
            "single_theta_only": True,
            "single_matsubara_index_only": True,
            "single_separation_only": True,
            "no_q_grid_integral": True,
            "no_matsubara_sum": True,
            "not_a_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "input": {
            "model_name": model_name,
            "pairing_name": pairing_name,
            "matsubara_index": int(matsubara_index),
            "temperature_K": float(temperature_K),
            "q_magnitude": float(q_magnitude),
            "phi_values_deg": [float(v) for v in phi_values_deg],
            "phi_values_mod_deg_sorted": [float(mod) for _, mod in phi_pairs],
            "plate1_theta_deg": float(plate1_theta_deg),
            "plate2_theta_deg": float(plate2_theta_deg),
            "nk": int(nk),
            "separation_nm": float(separation_nm),
            "delta0_eV": None if delta0_eV is None else float(delta0_eV),
            "eta_eV": float(eta_eV),
            "shift_fractions": [float(v) for v in shift_fractions],
            "candidate_name": candidate_name,
            "include_rhs_aware_validation": bool(include_rhs_aware_validation),
            "include_point_payloads": bool(include_point_payloads),
            "example_q_geometry_phi0": q_geometry_payload(q_model_vector_from_polar(q_magnitude, 0.0)),
            "valid_for_casimir_input": False,
        },
        "summary": _summary_from_rows(rows),
        "rows": rows,
        "point_payloads": point_payloads if include_point_payloads else None,
        "interpretation_guardrails": {
            "periodic_phi_integral_is_diagnostic_only": True,
            "does_not_modify_main_validation": True,
            "does_not_modify_main_casimir_pipeline": True,
            "not_a_full_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def write_phi_scan_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def run_and_write_minimal_casimir_phi_scan(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_minimal_casimir_phi_scan(**kwargs)
    output = Path(output_dir)
    write_json(output / "minimal_casimir_phi_scan.json", payload)
    write_phi_scan_csv(output / "minimal_casimir_phi_scan.csv", payload["rows"])
    return payload
