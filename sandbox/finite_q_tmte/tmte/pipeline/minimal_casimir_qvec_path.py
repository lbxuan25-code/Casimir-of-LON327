"""Diagnostic-only q-vector minimal single-point Casimir path.

This module extends the minimal sandbox response-to-Casimir chain to arbitrary
nonzero in-plane q_model=(qx,qy).  It keeps the response and reflection
conversion in the local LT basis defined by q, so it does not rely on the
q=(q,0) special case where L/T can be identified with lab x/y.
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
from lno327.electrodynamics.reflection import (
    model_q_to_si_wavevector,
    omega_eV_to_xi_si,
    tangential_electric_LT_to_TE_TM,
    tangential_electric_reflection_matrix_LT,
    te_tm_adapter_metadata,
    vacuum_admittance_LT,
    vacuum_kappa,
)

from ..adapters.bubble_adapter import compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.conventions import finite_q_conventions
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .rhs_aware_finite_q_validation import run_rhs_aware_finite_q_validation
from .schur_effective_translation_rhs_audit import DEFAULT_CANDIDATE, _solve_etaeta
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .ward_basis_convention_scan import primitive_blocks_from_baseline

SCHEMA_VERSION = "finite_q_tmte_minimal_casimir_qvec_path_v1"
PRIMITIVE_ORDER = ("A0", "L", "T")
SPATIAL_ORDER = ("L", "T")
TE_TM_ORDER = ("s", "p")
Q_TOL = 1e-14


def _norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(matrix, dtype=complex)))


def _finite_complex(value: complex) -> bool:
    z = complex(value)
    return bool(np.isfinite(z.real) and np.isfinite(z.imag))


def q_model_vector_from_polar(q_value: float, phi_deg: float = 0.0) -> np.ndarray:
    if float(q_value) <= 0.0:
        raise ValueError("q_value must be positive")
    phi = np.deg2rad(float(phi_deg))
    return np.asarray([float(q_value) * np.cos(phi), float(q_value) * np.sin(phi)], dtype=float)


def as_q_model_vector(q_model_vector: Sequence[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(q_model_vector, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model_vector must have shape (2,)")
    if float(np.linalg.norm(q)) <= Q_TOL:
        raise ValueError("q_model_vector must be nonzero")
    return q


def q_geometry_payload(q_model: Sequence[float] | np.ndarray, *, xi_eV: float | None = None) -> dict[str, Any]:
    q = as_q_model_vector(q_model)
    q_norm = float(np.linalg.norm(q))
    qhat = q / q_norm
    that = np.asarray([-qhat[1], qhat[0]], dtype=float)
    payload: dict[str, Any] = {
        "q_model_vector": q,
        "q_norm": q_norm,
        "qhat": qhat,
        "that": that,
        "phi_rad": float(np.arctan2(q[1], q[0])),
        "phi_deg": float(np.rad2deg(np.arctan2(q[1], q[0]))),
        "valid_for_casimir_input": False,
    }
    if xi_eV is not None:
        conventions = finite_q_conventions(q, float(xi_eV))
        payload.update(
            {
                "basis_normalization": conventions.basis_normalization,
                "g0": conventions.g0,
                "gL": conventions.gL,
            }
        )
    return payload


def _require_domain(*, matsubara_index: int, q_model_vector: Sequence[float] | np.ndarray, separation_nm: float, theta_deg: float) -> None:
    if int(matsubara_index) <= 0:
        raise ValueError("q-vector minimal Casimir path currently supports only n>=1; n=0 is a separate static-limit problem")
    _ = as_q_model_vector(q_model_vector)
    if float(separation_nm) <= 0.0:
        raise ValueError("separation_nm must be positive")
    if abs(float(theta_deg)) > 1e-14:
        raise ValueError("q-vector minimal Casimir path v1 supports only theta_deg=0; plate rotation/torque comes later")


def _compute_sandbox_k_eff_qvec(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_model_vector: Sequence[float] | np.ndarray,
    nk: int,
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
) -> dict[str, Any]:
    if int(nk) <= 0:
        raise ValueError("nk must be positive")
    xi = matsubara_xi_eV(matsubara_index, temperature_K)
    q = as_q_model_vector(q_model_vector)
    inputs = build_model_scan_inputs(
        model_name=model_name,
        pairing_name=pairing_name,
        xi_eV=xi,
        nk=nk,
        delta0_eV=delta0_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
    )
    target_blocks = []
    shifts = shift_pairs_from_fractions(shift_fractions)
    for sx, sy in shifts:
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
            **q_geometry_payload(q, xi_eV=xi),
            "xi_eV": float(xi),
            "num_shifted_meshes": len(shifts),
            "K_eff_norm": _norm(k_eff),
            "K_SS_norm": _norm(k_ss),
            "Schur_correction_norm": _norm(schur_correction),
            "K_etaeta_norm": _norm(k_etaeta),
            "valid_for_casimir_input": False,
        },
    }


def response_to_minimal_casimir_qvec_point(
    full_response: np.ndarray,
    *,
    omega_eV: float,
    q_model_vector: Sequence[float] | np.ndarray,
    separation_nm: float,
) -> dict[str, Any]:
    """Convert a 3x3 local LT response into a same-plate TE/TM trace-log point."""

    matrix = np.asarray(full_response, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError("full_response must have shape (3, 3)")
    if float(omega_eV) <= 0.0:
        raise ValueError("omega_eV must be positive")
    q = as_q_model_vector(q_model_vector)
    if float(separation_nm) <= 0.0:
        raise ValueError("separation_nm must be positive")

    separation_m = float(separation_nm) * 1.0e-9
    sigma_model_lt = spatial_response_to_bilayer_sheet_conductivity_model(matrix, float(omega_eV))
    sigma_sheet_lt = model_response_to_sheet_conductivity(sigma_model_lt)
    sigma_tilde_lt = sheet_conductivity_to_reflection_dimensionless(sigma_sheet_lt)
    sigma_tilde_lt_matrix = sigma_tilde_lt.tensor.matrix()

    material = LNO327_THIN_FILM_SLAO_IN_PLANE
    square_lattice = abs(material.lattice_a_x_m - material.lattice_a_y_m) / max(material.lattice_a_x_m, material.lattice_a_y_m) < 1e-12
    if not square_lattice:
        raise ValueError("LT-native q-vector path currently assumes square in-plane lattice so model-q and SI-Q directions agree")

    qx_si, qy_si, q_si = model_q_to_si_wavevector(float(q[0]), float(q[1]), material.lattice_a_x_m, material.lattice_a_y_m)
    xi_si = omega_eV_to_xi_si(float(omega_eV))
    kappa = vacuum_kappa(q_si, xi_si)
    y0_lt = vacuum_admittance_LT(xi_si, kappa)
    reflection_lt = tangential_electric_reflection_matrix_LT(sigma_tilde_lt_matrix, xi_si, kappa)
    reflection_te_tm = tangential_electric_LT_to_TE_TM(reflection_lt)
    trace_log = trace_log_point(reflection_te_tm, reflection_te_tm, kappa, separation_m)
    logdet = complex(trace_log["logdet_integrand"])

    return {
        "q_geometry": {
            **q_geometry_payload(q),
            "Q_x_m_inv": float(qx_si),
            "Q_y_m_inv": float(qy_si),
            "Q_m_inv": float(q_si),
            "xi_si_s_inv": float(xi_si),
            "kappa_m_inv": float(kappa),
            "square_lattice_model_LT_matches_SI_LT": bool(square_lattice),
            "valid_for_casimir_input": False,
        },
        "response": {
            "full_response_3x3": matrix,
            "spatial_response_LT_2x2": matrix[1:3, 1:3],
            "full_response_norm": _norm(matrix),
            "spatial_response_norm": _norm(matrix[1:3, 1:3]),
            "spatial_order": list(SPATIAL_ORDER),
            "spatial_basis_note": "local LT basis defined by q_model_vector; no xy reinterpretation is used",
            "valid_for_casimir_input": False,
        },
        "conductivity": {
            "sigma_model_LT_matrix": sigma_model_lt,
            "sigma_model_norm": _norm(sigma_model_lt),
            "sigma_sheet_LT_matrix": sigma_sheet_lt.tensor.matrix(),
            "sigma_sheet_norm": _norm(sigma_sheet_lt.tensor.matrix()),
            "sigma_sheet_unit_stage": sigma_sheet_lt.unit_stage,
            "sigma_sheet_unit_label": sigma_sheet_lt.unit_label,
            "sigma_sheet_normalization_status": sigma_sheet_lt.normalization_status,
            "sigma_tilde_LT_matrix": sigma_tilde_lt_matrix,
            "sigma_tilde_norm": _norm(sigma_tilde_lt_matrix),
            "sigma_tilde_unit_stage": sigma_tilde_lt.unit_stage,
            "sigma_tilde_unit_label": sigma_tilde_lt.unit_label,
            "sigma_tilde_normalization_status": sigma_tilde_lt.normalization_status,
            "conversion_metadata": bilayer_sheet_conductivity_convention_metadata(),
            "valid_for_casimir_input": False,
        },
        "reflection": {
            "R1_TE_TM": reflection_te_tm,
            "R2_TE_TM": reflection_te_tm,
            "R_TE_TM_norm": _norm(reflection_te_tm),
            "TE_TM_order": list(TE_TM_ORDER),
            "same_plate_theta0_diagnostic": True,
            "LT_native_reflection_path": True,
            "reflection_tangential_E_LT": reflection_lt,
            "vacuum_admittance_Y0_LT": y0_lt,
            "basis_convention": te_tm_adapter_metadata(),
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
            "finite_reflection": bool(np.isfinite(reflection_te_tm.real).all() and np.isfinite(reflection_te_tm.imag).all()),
            "finite_logdet": _finite_complex(logdet),
            "LT_native_reflection_path": True,
            "single_point_trace_log_only": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def _rhs_aware_validation_or_guard(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_model_vector: np.ndarray,
    nk: int,
    delta0_eV: float | None,
    eta_eV: float,
    shift_fractions: Sequence[float],
    candidate_name: str,
) -> dict[str, Any]:
    q = as_q_model_vector(q_model_vector)
    q_norm = float(np.linalg.norm(q))
    if abs(float(q[1])) <= Q_TOL:
        return run_rhs_aware_finite_q_validation(
            model_name=model_name,
            pairing_name=pairing_name,
            matsubara_index=matsubara_index,
            temperature_K=temperature_K,
            q_value=q_norm,
            nk=nk,
            delta0_eV=delta0_eV,
            eta_eV=eta_eV,
            shift_fractions=shift_fractions,
            candidate_name=candidate_name,
            include_raw_schur_audit=False,
        )
    return {
        "schema_version": "finite_q_tmte_rhs_aware_qvec_guard_v1",
        "status": {
            "diagnostic_run_completed": False,
            "rhs_aware_ward_closed": None,
            "valid_for_casimir_input": False,
            "reason": "existing_rhs_aware_validation_is_scalar_q_along_x_only; not run for qy!=0",
        },
        "q_geometry": q_geometry_payload(q),
        "valid_for_casimir_input": False,
    }


def run_minimal_casimir_qvec_path(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_model_vector: Sequence[float] | np.ndarray,
    nk: int,
    separation_nm: float,
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    theta_deg: float = 0.0,
    candidate_name: str = DEFAULT_CANDIDATE,
    include_rhs_aware_validation: bool = True,
) -> dict[str, Any]:
    q = as_q_model_vector(q_model_vector)
    _require_domain(matsubara_index=matsubara_index, q_model_vector=q, separation_nm=separation_nm, theta_deg=theta_deg)
    if candidate_name != DEFAULT_CANDIDATE:
        raise ValueError("q-vector minimal Casimir path v1 only supports the default RHS-aware Ward candidate")

    response = _compute_sandbox_k_eff_qvec(
        model_name=model_name,
        pairing_name=pairing_name,
        matsubara_index=matsubara_index,
        temperature_K=temperature_K,
        q_model_vector=q,
        nk=nk,
        delta0_eV=delta0_eV,
        eta_eV=eta_eV,
        shift_fractions=shift_fractions,
    )
    omega_eV = float(response["metadata"]["xi_eV"])
    point = response_to_minimal_casimir_qvec_point(response["K_eff"], omega_eV=omega_eV, q_model_vector=q, separation_nm=separation_nm)
    validation = None
    if include_rhs_aware_validation:
        validation = _rhs_aware_validation_or_guard(
            model_name=model_name,
            pairing_name=pairing_name,
            matsubara_index=matsubara_index,
            temperature_K=temperature_K,
            q_model_vector=q,
            nk=nk,
            delta0_eV=delta0_eV,
            eta_eV=eta_eV,
            shift_fractions=shift_fractions,
            candidate_name=candidate_name,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "single_point_trace_log_only": True,
            "n_ge_1_only": True,
            "q_positive_only": True,
            "arbitrary_q_vec_supported": True,
            "LT_native_reflection_path": True,
            "theta0_only": True,
            "no_q_grid_integral": True,
            "no_phi_integral": True,
            "no_matsubara_sum": True,
            "no_n0_policy": True,
            "no_q0_policy": True,
            "valid_for_casimir_input": False,
            "reason": "minimal_single_point_qvec_diagnostic_path_only",
        },
        "input": {
            "model_name": model_name,
            "pairing_name": pairing_name,
            "matsubara_index": int(matsubara_index),
            "temperature_K": float(temperature_K),
            "q_model_vector": q,
            "q_norm": float(np.linalg.norm(q)),
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
            "source": "sandbox_finite_q_tmte_schur_effective_K_eff_qvec",
            "primitive_order": list(PRIMITIVE_ORDER),
            "spatial_basis_note": "primitive L/T is local to q_model_vector and is passed directly to LT reflection helper",
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


def run_and_write_minimal_casimir_qvec_path(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_minimal_casimir_qvec_path(**kwargs)
    write_json(Path(output_dir) / "minimal_casimir_qvec_path.json", payload)
    return payload
