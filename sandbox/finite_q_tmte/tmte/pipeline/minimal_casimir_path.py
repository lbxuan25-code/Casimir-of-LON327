"""Diagnostic-only minimal single-point Casimir path for sandbox finite-q TMTE.

This module stitches the sandbox Schur-effective K_eff response into the
existing electrodynamics and Lifshitz trace-log helpers.  It intentionally
supports only n>=1, q>0, q=(q,0), theta=0 single-point diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lno327.casimir.lifshitz_integrand import lifshitz_integrand_metadata, trace_log_point
from lno327.electrodynamics.conventions import (
    bilayer_sheet_conductivity_convention_metadata,
    model_response_to_sheet_conductivity,
    sheet_conductivity_to_reflection_dimensionless,
    spatial_response_to_bilayer_sheet_conductivity_model,
)
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.electrodynamics.reflection import sigma_tilde_xy_to_te_tm_reflection_matrix

from ..adapters.bubble_adapter import compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .rhs_aware_finite_q_validation import run_rhs_aware_finite_q_validation
from .schur_effective_translation_rhs_audit import DEFAULT_CANDIDATE, _solve_etaeta
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .ward_basis_convention_scan import primitive_blocks_from_baseline

SCHEMA_VERSION = "finite_q_tmte_minimal_casimir_path_v1"
SPATIAL_ORDER = ("L", "T")
TE_TM_ORDER = ("s", "p")


def _norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(matrix, dtype=complex)))


def _finite_complex(value: complex) -> bool:
    z = complex(value)
    return bool(np.isfinite(z.real) and np.isfinite(z.imag))


def _require_single_point_domain(*, matsubara_index: int, q_value: float, separation_nm: float, theta_deg: float) -> None:
    if int(matsubara_index) <= 0:
        raise ValueError("minimal Casimir path currently supports only n>=1; n=0 is a separate static-limit problem")
    if float(q_value) <= 0.0:
        raise ValueError("minimal Casimir path currently requires q>0; q=0 is a separate limit problem")
    if float(separation_nm) <= 0.0:
        raise ValueError("separation_nm must be positive")
    if abs(float(theta_deg)) > 1e-14:
        raise ValueError("minimal Casimir path v1 supports only theta_deg=0 until arbitrary q-vector sandbox response exists")


def _compute_sandbox_k_eff(
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
) -> dict[str, Any]:
    if int(nk) <= 0:
        raise ValueError("nk must be positive")
    xi = matsubara_xi_eV(matsubara_index, temperature_K)
    inputs = build_model_scan_inputs(
        model_name=model_name,
        pairing_name=pairing_name,
        xi_eV=xi,
        nk=nk,
        delta0_eV=delta0_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
    )
    q = np.asarray([float(q_value), 0.0], dtype=float)
    target_blocks = []
    for sx, sy in shift_pairs_from_fractions(shift_fractions):
        points = shifted_uniform_bz_mesh(int(nk), float(sx), float(sy))
        weights = weights_for_points(points)
        target_blocks.append(
            compute_target_bare_blocks(
                spec=inputs.spec,
                ansatz=inputs.ansatz,
                q_model=q,
                xi_eV=xi,
                k_points=points,
                weights=weights,
                config=inputs.config,
                pairing_params=inputs.pairing_params,
            )
        )
    bare = average_bare_blocks_then_schur(target_blocks).bare_blocks
    primitive = primitive_blocks_from_baseline(bare)
    k_ss = np.asarray(primitive["k_ss"], dtype=complex)
    k_seta = np.asarray(primitive["k_seta"], dtype=complex)
    k_etas = np.asarray(primitive["k_etas"], dtype=complex)
    k_etaeta = np.asarray(primitive["k_etaeta"], dtype=complex)
    action, solve_meta = _solve_etaeta(k_etaeta, k_etas)
    schur_correction = k_seta @ action
    k_eff = k_ss - schur_correction
    return {
        "K_eff": k_eff,
        "K_SS": k_ss,
        "K_Seta": k_seta,
        "K_etaS": k_etas,
        "K_etaeta": k_etaeta,
        "Schur_correction": schur_correction,
        "schur_solve_metadata": solve_meta,
        "metadata": {
            "q_model": q,
            "xi_eV": float(xi),
            "num_shifted_meshes": len(shift_pairs_from_fractions(shift_fractions)),
            "K_eff_norm": _norm(k_eff),
            "K_SS_norm": _norm(k_ss),
            "Schur_correction_norm": _norm(schur_correction),
            "K_etaeta_norm": _norm(k_etaeta),
            "valid_for_casimir_input": False,
        },
    }


def response_to_minimal_casimir_point(
    full_response: np.ndarray,
    *,
    omega_eV: float,
    q_value: float,
    separation_nm: float,
) -> dict[str, Any]:
    """Convert a 3x3 response into a same-plate TE/TM trace-log point."""

    matrix = np.asarray(full_response, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError("full_response must have shape (3, 3)")
    if float(omega_eV) <= 0.0:
        raise ValueError("omega_eV must be positive")
    if float(q_value) <= 0.0:
        raise ValueError("q_value must be positive")
    if float(separation_nm) <= 0.0:
        raise ValueError("separation_nm must be positive")
    separation_m = float(separation_nm) * 1.0e-9
    sigma_model = spatial_response_to_bilayer_sheet_conductivity_model(matrix, float(omega_eV))
    sigma_sheet = model_response_to_sheet_conductivity(sigma_model)
    sigma_tilde = sheet_conductivity_to_reflection_dimensionless(sigma_sheet)
    material = LNO327_THIN_FILM_SLAO_IN_PLANE
    reflection = sigma_tilde_xy_to_te_tm_reflection_matrix(
        sigma_tilde.tensor.matrix(),
        float(q_value),
        0.0,
        float(omega_eV),
        material.lattice_a_x_m,
        material.lattice_a_y_m,
        allow_q_zero=False,
    )
    r_te_tm = np.asarray(reflection["reflection_TE_TM"], dtype=complex)
    trace_log = trace_log_point(r_te_tm, r_te_tm, float(reflection["kappa_m_inv"]), separation_m)
    logdet = complex(trace_log["logdet_integrand"])
    return {
        "response": {
            "full_response_3x3": matrix,
            "spatial_response_2x2": matrix[1:3, 1:3],
            "full_response_norm": _norm(matrix),
            "spatial_response_norm": _norm(matrix[1:3, 1:3]),
            "spatial_order": list(SPATIAL_ORDER),
            "valid_for_casimir_input": False,
        },
        "conductivity": {
            "sigma_model_matrix": sigma_model,
            "sigma_model_norm": _norm(sigma_model),
            "sigma_sheet_matrix": sigma_sheet.tensor.matrix(),
            "sigma_sheet_norm": _norm(sigma_sheet.tensor.matrix()),
            "sigma_sheet_unit_stage": sigma_sheet.unit_stage,
            "sigma_sheet_unit_label": sigma_sheet.unit_label,
            "sigma_sheet_normalization_status": sigma_sheet.normalization_status,
            "sigma_tilde_matrix": sigma_tilde.tensor.matrix(),
            "sigma_tilde_norm": _norm(sigma_tilde.tensor.matrix()),
            "sigma_tilde_unit_stage": sigma_tilde.unit_stage,
            "sigma_tilde_unit_label": sigma_tilde.unit_label,
            "sigma_tilde_normalization_status": sigma_tilde.normalization_status,
            "conversion_metadata": bilayer_sheet_conductivity_convention_metadata(),
            "valid_for_casimir_input": False,
        },
        "reflection": {
            "R1_TE_TM": r_te_tm,
            "R2_TE_TM": r_te_tm,
            "R_TE_TM_norm": _norm(r_te_tm),
            "TE_TM_order": list(TE_TM_ORDER),
            "same_plate_theta0_diagnostic": True,
            "sigma_tilde_xy_matrix": reflection["sigma_tilde_xy_matrix"],
            "sigma_tilde_LT_matrix": reflection["sigma_tilde_LT_matrix"],
            "reflection_tangential_E_LT": reflection["reflection_tangential_E_LT"],
            "vacuum_admittance_Y0_LT": reflection["vacuum_admittance_Y0_LT"],
            "xy_to_lt_rotation_matrix": reflection["xy_to_lt_rotation_matrix"],
            "q_model_x": float(reflection["q_model_x"]),
            "q_model_y": float(reflection["q_model_y"]),
            "Q_x_m_inv": float(reflection["Q_x_m_inv"]),
            "Q_y_m_inv": float(reflection["Q_y_m_inv"]),
            "Q_m_inv": float(reflection["Q_m_inv"]),
            "xi_si_s_inv": float(reflection["xi_si_s_inv"]),
            "kappa_m_inv": float(reflection["kappa_m_inv"]),
            "basis_convention": reflection["basis_convention"],
            "valid_for_casimir_input": False,
        },
        "trace_log": {
            "separation_nm": float(separation_nm),
            "separation_m": separation_m,
            "round_trip_factor": float(trace_log["round_trip_factor"]),
            "trace_log_matrix": trace_log["trace_log_matrix"],
            "logdet_integrand": logdet,
            "logdet_abs": float(abs(logdet)),
            "logdet_is_finite": _finite_complex(logdet),
            "lifshitz_integrand_metadata": lifshitz_integrand_metadata(),
            "valid_for_casimir_input": False,
        },
        "sanity_checks": {
            "finite_reflection": bool(np.isfinite(r_te_tm.real).all() and np.isfinite(r_te_tm.imag).all()),
            "finite_logdet": _finite_complex(logdet),
            "single_point_trace_log_only": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_minimal_casimir_path(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_value: float,
    nk: int,
    separation_nm: float,
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    theta_deg: float = 0.0,
    candidate_name: str = DEFAULT_CANDIDATE,
    include_rhs_aware_validation: bool = True,
) -> dict[str, Any]:
    _require_single_point_domain(
        matsubara_index=matsubara_index,
        q_value=q_value,
        separation_nm=separation_nm,
        theta_deg=theta_deg,
    )
    if candidate_name != DEFAULT_CANDIDATE:
        raise ValueError("minimal Casimir path v1 only supports the default RHS-aware Ward candidate")
    response = _compute_sandbox_k_eff(
        model_name=model_name,
        pairing_name=pairing_name,
        matsubara_index=matsubara_index,
        temperature_K=temperature_K,
        q_value=q_value,
        nk=nk,
        delta0_eV=delta0_eV,
        eta_eV=eta_eV,
        shift_fractions=shift_fractions,
    )
    omega_eV = float(response["metadata"]["xi_eV"])
    point = response_to_minimal_casimir_point(
        response["K_eff"],
        omega_eV=omega_eV,
        q_value=q_value,
        separation_nm=separation_nm,
    )
    validation = None
    if include_rhs_aware_validation:
        validation = run_rhs_aware_finite_q_validation(
            model_name=model_name,
            pairing_name=pairing_name,
            matsubara_index=matsubara_index,
            temperature_K=temperature_K,
            q_value=q_value,
            nk=nk,
            delta0_eV=delta0_eV,
            eta_eV=eta_eV,
            shift_fractions=shift_fractions,
            candidate_name=candidate_name,
            include_raw_schur_audit=False,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "single_point_trace_log_only": True,
            "n_ge_1_only": True,
            "q_positive_only": True,
            "q_along_x_only": True,
            "theta0_only": True,
            "no_q_grid_integral": True,
            "no_phi_integral": True,
            "no_matsubara_sum": True,
            "no_n0_policy": True,
            "no_q0_policy": True,
            "valid_for_casimir_input": False,
            "reason": "minimal_single_point_diagnostic_path_only",
        },
        "input": {
            "model_name": model_name,
            "pairing_name": pairing_name,
            "matsubara_index": int(matsubara_index),
            "temperature_K": float(temperature_K),
            "q_value": float(q_value),
            "q_model_vector": [float(q_value), 0.0],
            "nk": int(nk),
            "separation_nm": float(separation_nm),
            "delta0_eV": None if delta0_eV is None else float(delta0_eV),
            "eta_eV": float(eta_eV),
            "shift_fractions": [float(v) for v in shift_fractions],
            "theta_deg": float(theta_deg),
            "candidate_name": candidate_name,
            "valid_for_casimir_input": False,
        },
        "frequency": frequency_payload(matsubara_index, temperature_K),
        "sandbox_response_source": {
            "source": "sandbox_finite_q_tmte_schur_effective_K_eff",
            "primitive_order": ["A0", "L", "T"],
            "K_eff": response["K_eff"],
            "K_SS_norm": response["metadata"]["K_SS_norm"],
            "K_eff_norm": response["metadata"]["K_eff_norm"],
            "Schur_correction_norm": response["metadata"]["Schur_correction_norm"],
            "K_etaeta_norm": response["metadata"]["K_etaeta_norm"],
            "schur_solve_metadata": response["schur_solve_metadata"],
            "valid_for_casimir_input": False,
        },
        "rhs_aware_validation": validation,
        "minimal_casimir_point": point,
        "interpretation_guardrails": {
            "tests_physical_chain_not_convergence": True,
            "uses_existing_conductivity_reflection_trace_log_helpers": True,
            "does_not_modify_main_validation": True,
            "does_not_modify_main_casimir_pipeline": True,
            "not_a_full_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_and_write_minimal_casimir_path(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_minimal_casimir_path(**kwargs)
    write_json(Path(output_dir) / "minimal_casimir_path.json", payload)
    return payload
