"""Diagnostic-only theta/plate-rotation minimal Casimir path.

This module builds the first single-point two-plate diagnostic on top of the
q-vector minimal path.  Given a lab-frame q vector and a relative plate angle,
it evaluates plate 1 at q_lab and plate 2 at q_crystal=R(-theta)q_lab, converts
both local LT responses to TE/TM reflection matrices, and forms one mixed
trace-log point.  It performs no q/phi/n integration and is not Casimir-ready.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lno327.casimir.lifshitz_integrand import lifshitz_integrand_metadata, trace_log_point

from ..io.writers import write_json
from ..theory.frequency import frequency_payload
from .minimal_casimir_qvec_path import (
    DEFAULT_CANDIDATE,
    as_q_model_vector,
    q_geometry_payload,
    q_model_vector_from_polar,
    run_minimal_casimir_qvec_path,
)

SCHEMA_VERSION = "finite_q_tmte_minimal_casimir_theta_path_v1"
Q_TOL = 1e-14


def _norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(matrix, dtype=complex)))


def _finite_complex(value: complex) -> bool:
    z = complex(value)
    return bool(np.isfinite(z.real) and np.isfinite(z.imag))


def rotation_matrix_deg(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, -s], [s, c]], dtype=float)


def crystal_q_from_lab_q(q_lab: Sequence[float] | np.ndarray, plate_angle_deg: float) -> np.ndarray:
    """Return q in the crystal frame for a plate rotated by plate_angle_deg.

    Convention: plate_angle_deg is the crystal-axis angle relative to the lab
    axes.  Therefore q_crystal = R(-theta) q_lab.
    """

    q = as_q_model_vector(q_lab)
    return rotation_matrix_deg(-float(plate_angle_deg)) @ q


def _require_domain(*, matsubara_index: int, q_lab_vector: Sequence[float] | np.ndarray, separation_nm: float) -> None:
    if int(matsubara_index) <= 0:
        raise ValueError("theta minimal Casimir path currently supports only n>=1; n=0 is a separate static-limit problem")
    q = as_q_model_vector(q_lab_vector)
    if float(np.linalg.norm(q)) <= Q_TOL:
        raise ValueError("q_lab_vector must be nonzero")
    if float(separation_nm) <= 0.0:
        raise ValueError("separation_nm must be positive")


def _plate_point(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_crystal_vector: Sequence[float] | np.ndarray,
    nk: int,
    separation_nm: float,
    delta0_eV: float | None,
    eta_eV: float,
    shift_fractions: Sequence[float],
    candidate_name: str,
    include_rhs_aware_validation: bool,
) -> dict[str, Any]:
    return run_minimal_casimir_qvec_path(
        model_name=model_name,
        pairing_name=pairing_name,
        matsubara_index=matsubara_index,
        temperature_K=temperature_K,
        q_model_vector=q_crystal_vector,
        nk=nk,
        separation_nm=separation_nm,
        delta0_eV=delta0_eV,
        eta_eV=eta_eV,
        shift_fractions=shift_fractions,
        theta_deg=0.0,
        candidate_name=candidate_name,
        include_rhs_aware_validation=include_rhs_aware_validation,
    )


def run_minimal_casimir_theta_path(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_lab_vector: Sequence[float] | np.ndarray,
    plate2_theta_deg: float,
    nk: int,
    separation_nm: float,
    plate1_theta_deg: float = 0.0,
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    candidate_name: str = DEFAULT_CANDIDATE,
    include_rhs_aware_validation: bool = True,
) -> dict[str, Any]:
    _require_domain(matsubara_index=matsubara_index, q_lab_vector=q_lab_vector, separation_nm=separation_nm)
    if candidate_name != DEFAULT_CANDIDATE:
        raise ValueError("theta minimal Casimir path v1 only supports the default RHS-aware Ward candidate")

    q_lab = as_q_model_vector(q_lab_vector)
    q1_crystal = crystal_q_from_lab_q(q_lab, plate1_theta_deg)
    q2_crystal = crystal_q_from_lab_q(q_lab, plate2_theta_deg)

    plate1 = _plate_point(
        model_name=model_name,
        pairing_name=pairing_name,
        matsubara_index=matsubara_index,
        temperature_K=temperature_K,
        q_crystal_vector=q1_crystal,
        nk=nk,
        separation_nm=separation_nm,
        delta0_eV=delta0_eV,
        eta_eV=eta_eV,
        shift_fractions=shift_fractions,
        candidate_name=candidate_name,
        include_rhs_aware_validation=include_rhs_aware_validation,
    )
    plate2 = _plate_point(
        model_name=model_name,
        pairing_name=pairing_name,
        matsubara_index=matsubara_index,
        temperature_K=temperature_K,
        q_crystal_vector=q2_crystal,
        nk=nk,
        separation_nm=separation_nm,
        delta0_eV=delta0_eV,
        eta_eV=eta_eV,
        shift_fractions=shift_fractions,
        candidate_name=candidate_name,
        include_rhs_aware_validation=include_rhs_aware_validation,
    )

    point1 = plate1["minimal_casimir_point"]
    point2 = plate2["minimal_casimir_point"]
    r1 = np.asarray(point1["reflection"]["R1_TE_TM"], dtype=complex)
    r2 = np.asarray(point2["reflection"]["R1_TE_TM"], dtype=complex)
    kappa1 = float(point1["q_geometry"]["kappa_m_inv"])
    kappa2 = float(point2["q_geometry"]["kappa_m_inv"])
    if not np.isclose(kappa1, kappa2, rtol=1e-12, atol=1e-9):
        raise ValueError("plate kappa mismatch; theta diagnostic assumes common lab propagation kappa")
    separation_m = float(separation_nm) * 1.0e-9
    mixed_trace_log = trace_log_point(r1, r2, kappa1, separation_m)
    logdet = complex(mixed_trace_log["logdet_integrand"])

    theta_rel = float(plate2_theta_deg) - float(plate1_theta_deg)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "single_point_theta_trace_log_only": True,
            "n_ge_1_only": True,
            "q_positive_only": True,
            "theta_diagnostic_supported": True,
            "plate_rotation_geometry_v1": True,
            "no_q_grid_integral": True,
            "no_phi_integral": True,
            "no_matsubara_sum": True,
            "no_n0_policy": True,
            "no_q0_policy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
            "reason": "minimal_single_point_theta_diagnostic_path_only",
        },
        "input": {
            "model_name": model_name,
            "pairing_name": pairing_name,
            "matsubara_index": int(matsubara_index),
            "temperature_K": float(temperature_K),
            "q_lab_vector": q_lab,
            "q_lab_norm": float(np.linalg.norm(q_lab)),
            "plate1_theta_deg": float(plate1_theta_deg),
            "plate2_theta_deg": float(plate2_theta_deg),
            "relative_theta_deg": theta_rel,
            "nk": int(nk),
            "separation_nm": float(separation_nm),
            "delta0_eV": None if delta0_eV is None else float(delta0_eV),
            "eta_eV": float(eta_eV),
            "shift_fractions": [float(v) for v in shift_fractions],
            "candidate_name": candidate_name,
            "valid_for_casimir_input": False,
        },
        "frequency": frequency_payload(matsubara_index, temperature_K),
        "geometry": {
            "convention": "q_crystal = R(-plate_theta) q_lab; plate theta is crystal axes relative to lab axes",
            "q_lab": q_geometry_payload(q_lab),
            "plate1": {
                "theta_deg": float(plate1_theta_deg),
                "q_crystal": q_geometry_payload(q1_crystal),
            },
            "plate2": {
                "theta_deg": float(plate2_theta_deg),
                "q_crystal": q_geometry_payload(q2_crystal),
            },
            "common_lab_TE_TM_basis_assumed": True,
            "valid_for_casimir_input": False,
        },
        "plate1": plate1,
        "plate2": plate2,
        "mixed_trace_log": {
            "R1_TE_TM": r1,
            "R2_TE_TM": r2,
            "R1_TE_TM_norm": _norm(r1),
            "R2_TE_TM_norm": _norm(r2),
            "R1_minus_R2_norm": _norm(r1 - r2),
            "kappa_m_inv": kappa1,
            "kappa_relative_mismatch": float(abs(kappa1 - kappa2) / max(abs(kappa1), abs(kappa2), 1.0)),
            "separation_nm": float(separation_nm),
            "separation_m": separation_m,
            "round_trip_factor": float(mixed_trace_log["round_trip_factor"]),
            "trace_log_matrix": mixed_trace_log["trace_log_matrix"],
            "logdet_integrand": logdet,
            "logdet_abs": float(abs(logdet)),
            "logdet_is_finite": _finite_complex(logdet),
            "lifshitz_integrand_metadata": lifshitz_integrand_metadata(),
            "valid_for_casimir_input": False,
        },
        "sanity_checks": {
            "finite_R1": bool(np.isfinite(r1.real).all() and np.isfinite(r1.imag).all()),
            "finite_R2": bool(np.isfinite(r2.real).all() and np.isfinite(r2.imag).all()),
            "finite_logdet": _finite_complex(logdet),
            "kappa_match": bool(np.isclose(kappa1, kappa2, rtol=1e-12, atol=1e-9)),
            "single_point_theta_trace_log_only": True,
            "valid_for_casimir_input": False,
        },
        "interpretation_guardrails": {
            "tests_plate_rotation_chain_not_convergence": True,
            "uses_qvec_LT_native_reflection_path": True,
            "does_not_modify_main_validation": True,
            "does_not_modify_main_casimir_pipeline": True,
            "not_a_full_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_and_write_minimal_casimir_theta_path(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_minimal_casimir_theta_path(**kwargs)
    write_json(Path(output_dir) / "minimal_casimir_theta_path.json", payload)
    return payload
