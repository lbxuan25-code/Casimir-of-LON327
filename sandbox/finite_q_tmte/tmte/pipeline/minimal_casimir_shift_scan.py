"""Diagnostic-only shift scan for the minimal Casimir phi path.

The scan fixes q, theta, Matsubara index, separation, nk, and a set of phi
values, then runs single-shift phi scans for each requested shift fraction.  It
is meant to expose shift-resolved reflection norm pathologies instead of hiding
them inside an averaged shifted mesh.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..io.writers import write_json
from .minimal_casimir_phi_scan import run_minimal_casimir_phi_scan
from .schur_effective_translation_rhs_audit import DEFAULT_CANDIDATE

SCHEMA_VERSION = "finite_q_tmte_minimal_casimir_shift_scan_v1"
DEFAULT_R_NORM_WARNING_THRESHOLD = 2.0
CSV_COLUMNS = [
    "shift_fraction",
    "phi_mod_deg",
    "logdet_abs",
    "delta_logdet_abs_from_shift_phi0",
    "Rdiff",
    "R1_norm",
    "R2_norm",
    "max_R_norm",
    "large_R_norm",
    "p1_Keff_norm",
    "p2_Keff_norm",
    "finite_R1",
    "finite_R2",
    "finite_logdet",
    "kappa_match",
]


def _normalise_shift_values(shift_values: Sequence[float]) -> list[float]:
    if not shift_values:
        raise ValueError("shift_values must not be empty")
    values = [float(v) for v in shift_values]
    keys = [round(v, 14) for v in values]
    if len(set(keys)) != len(keys):
        raise ValueError("shift_values must be unique")
    return values


def _shift_tag(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def _row_from_phi_row(*, shift: float, row: dict[str, Any], r_norm_warning_threshold: float) -> dict[str, Any]:
    r1 = float(row["R1_norm"])
    r2 = float(row["R2_norm"])
    max_r = max(r1, r2)
    return {
        "shift_fraction": float(shift),
        "phi_mod_deg": float(row["phi_mod_deg"]),
        "logdet_abs": float(row["logdet_abs"]),
        "delta_logdet_abs_from_shift_phi0": float(row["delta_logdet_abs_from_phi0"]),
        "Rdiff": float(row["Rdiff"]),
        "R1_norm": r1,
        "R2_norm": r2,
        "max_R_norm": max_r,
        "large_R_norm": bool(max_r > float(r_norm_warning_threshold)),
        "p1_Keff_norm": float(row["p1_Keff_norm"]),
        "p2_Keff_norm": float(row["p2_Keff_norm"]),
        "finite_R1": bool(row["finite_R1"]),
        "finite_R2": bool(row["finite_R2"]),
        "finite_logdet": bool(row["finite_logdet"]),
        "kappa_match": bool(row["kappa_match"]),
    }


def _summary_from_rows(rows: list[dict[str, Any]], *, r_norm_warning_threshold: float) -> dict[str, Any]:
    if not rows:
        return {"num_rows": 0, "valid_for_casimir_input": False}
    logdet_abs = np.asarray([float(row["logdet_abs"]) for row in rows], dtype=float)
    rdiff = np.asarray([float(row["Rdiff"]) for row in rows], dtype=float)
    max_r = np.asarray([float(row["max_R_norm"]) for row in rows], dtype=float)
    worst_row = max(rows, key=lambda row: float(row["max_R_norm"]))
    return {
        "num_rows": len(rows),
        "num_large_R_norm_rows": int(sum(bool(row["large_R_norm"]) for row in rows)),
        "has_large_R_norm": any(bool(row["large_R_norm"]) for row in rows),
        "r_norm_warning_threshold": float(r_norm_warning_threshold),
        "max_R_norm": float(np.max(max_r)),
        "max_Rdiff": float(np.max(rdiff)),
        "min_logdet_abs": float(np.min(logdet_abs)),
        "max_logdet_abs": float(np.max(logdet_abs)),
        "range_logdet_abs": float(np.max(logdet_abs) - np.min(logdet_abs)),
        "all_finite_R1": all(bool(row["finite_R1"]) for row in rows),
        "all_finite_R2": all(bool(row["finite_R2"]) for row in rows),
        "all_finite_logdet": all(bool(row["finite_logdet"]) for row in rows),
        "all_kappa_match": all(bool(row["kappa_match"]) for row in rows),
        "worst_R_norm_row": {
            "shift_fraction": float(worst_row["shift_fraction"]),
            "phi_mod_deg": float(worst_row["phi_mod_deg"]),
            "R1_norm": float(worst_row["R1_norm"]),
            "R2_norm": float(worst_row["R2_norm"]),
            "max_R_norm": float(worst_row["max_R_norm"]),
            "Rdiff": float(worst_row["Rdiff"]),
            "logdet_abs": float(worst_row["logdet_abs"]),
        },
        "valid_for_casimir_input": False,
    }


def _summary_by_shift(rows: list[dict[str, Any]], *, r_norm_warning_threshold: float) -> list[dict[str, Any]]:
    shifts = sorted({float(row["shift_fraction"]) for row in rows})
    out: list[dict[str, Any]] = []
    for shift in shifts:
        subset = [row for row in rows if float(row["shift_fraction"]) == shift]
        summary = _summary_from_rows(subset, r_norm_warning_threshold=r_norm_warning_threshold)
        out.append({"shift_fraction": float(shift), **summary})
    return out


def run_minimal_casimir_shift_scan(
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
    shift_values: Sequence[float],
    plate1_theta_deg: float = 0.0,
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    candidate_name: str = DEFAULT_CANDIDATE,
    include_rhs_aware_validation: bool = True,
    include_phi_scan_payloads: bool = False,
    r_norm_warning_threshold: float = DEFAULT_R_NORM_WARNING_THRESHOLD,
) -> dict[str, Any]:
    if int(matsubara_index) <= 0:
        raise ValueError("shift scan currently supports only n>=1; n=0 is a separate static-limit problem")
    if float(q_magnitude) <= 0.0:
        raise ValueError("q_magnitude must be positive")
    if int(nk) <= 0:
        raise ValueError("nk must be positive")
    if float(separation_nm) <= 0.0:
        raise ValueError("separation_nm must be positive")
    if float(r_norm_warning_threshold) <= 0.0:
        raise ValueError("r_norm_warning_threshold must be positive")

    shift_list = _normalise_shift_values(shift_values)
    rows: list[dict[str, Any]] = []
    phi_scan_payloads: list[dict[str, Any]] = []
    for shift in shift_list:
        phi_scan = run_minimal_casimir_phi_scan(
            model_name=model_name,
            pairing_name=pairing_name,
            matsubara_index=matsubara_index,
            temperature_K=temperature_K,
            q_magnitude=q_magnitude,
            phi_values_deg=phi_values_deg,
            plate1_theta_deg=plate1_theta_deg,
            plate2_theta_deg=plate2_theta_deg,
            nk=nk,
            separation_nm=separation_nm,
            delta0_eV=delta0_eV,
            eta_eV=eta_eV,
            shift_fractions=(shift,),
            candidate_name=candidate_name,
            include_rhs_aware_validation=include_rhs_aware_validation,
            include_point_payloads=False,
        )
        for row in phi_scan["rows"]:
            rows.append(_row_from_phi_row(shift=shift, row=row, r_norm_warning_threshold=r_norm_warning_threshold))
        if include_phi_scan_payloads:
            phi_scan_payloads.append(phi_scan)

    rows.sort(key=lambda row: (float(row["shift_fraction"]), float(row["phi_mod_deg"])))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "shift_scan_only": True,
            "single_shift_phi_scans_only": True,
            "r_norm_guard_included": True,
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
            "plate1_theta_deg": float(plate1_theta_deg),
            "plate2_theta_deg": float(plate2_theta_deg),
            "nk": int(nk),
            "separation_nm": float(separation_nm),
            "shift_values": [float(v) for v in shift_list],
            "delta0_eV": None if delta0_eV is None else float(delta0_eV),
            "eta_eV": float(eta_eV),
            "candidate_name": candidate_name,
            "include_rhs_aware_validation": bool(include_rhs_aware_validation),
            "include_phi_scan_payloads": bool(include_phi_scan_payloads),
            "r_norm_warning_threshold": float(r_norm_warning_threshold),
            "valid_for_casimir_input": False,
        },
        "summary": _summary_from_rows(rows, r_norm_warning_threshold=r_norm_warning_threshold),
        "summary_by_shift": _summary_by_shift(rows, r_norm_warning_threshold=r_norm_warning_threshold),
        "rows": rows,
        "phi_scan_payloads": phi_scan_payloads if include_phi_scan_payloads else None,
        "interpretation_guardrails": {
            "large_R_norm_is_a_diagnostic_warning_not_a_proof_of_physical_instability": True,
            "single_shift_results_should_not_be_averaged_without_checking_R_norm": True,
            "does_not_modify_main_validation": True,
            "does_not_modify_main_casimir_pipeline": True,
            "not_a_full_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def write_shift_scan_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def run_and_write_minimal_casimir_shift_scan(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_minimal_casimir_shift_scan(**kwargs)
    output = Path(output_dir)
    write_json(output / "minimal_casimir_shift_scan.json", payload)
    write_shift_scan_csv(output / "minimal_casimir_shift_scan.csv", payload["rows"])
    return payload
