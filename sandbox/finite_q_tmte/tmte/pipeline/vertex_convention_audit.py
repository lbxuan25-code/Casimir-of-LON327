"""Diagnostic-only primitive vertex convention audit for finite-q Ward closure."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lno327.bdg.hamiltonian import bdg_hamiltonian_from_model_pairing

from ..adapters.collective_adapter import collective_vertices
from ..adapters.model_adapter import build_model_scan_inputs
from ..adapters.primitive_vertices_adapter import primitive_observable_vertices, primitive_source_vertices
from ..io.writers import write_json
from ..theory.conventions import finite_q_conventions
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from ..theory.vertices import longitudinal_transverse_vertices
from .collective_schur_factors import collective_order_from_ansatz
from .extended_ward_kernel import complex_vector_payload

SCHEMA_VERSION = "finite_q_tmte_vertex_convention_audit_v1"
PRIMITIVE_LABELS = ("A0", "L", "T")


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-30) -> float:
    return float(numerator) / max(float(denominator), float(eps))


def _complex_payload(value: complex) -> complex:
    return complex(value)


def matrix_report(name: str, matrix: np.ndarray) -> dict[str, Any]:
    """Return Hermiticity/anti-Hermiticity fingerprints for one matrix."""

    m = np.asarray(matrix, dtype=complex)
    norm = _norm(m)
    herm = _norm(m - m.conj().T)
    anti = _norm(m + m.conj().T)
    i_m = 1j * m
    i_herm = _norm(i_m - i_m.conj().T)
    return {
        "name": name,
        "shape": list(m.shape),
        "norm": norm,
        "hermitian_residual_norm": herm,
        "antihermitian_residual_norm": anti,
        "i_times_matrix_hermitian_residual_norm": i_herm,
        "hermitian_residual_over_norm": _safe_ratio(herm, norm),
        "antihermitian_residual_over_norm": _safe_ratio(anti, norm),
        "i_times_matrix_hermitian_residual_over_norm": _safe_ratio(i_herm, norm),
        "trace": _complex_payload(np.trace(m)),
        "valid_for_casimir_input": False,
    }


def relation_report(name: str, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Compare a row-side vertex to source-side dagger/sign conventions."""

    l = np.asarray(left, dtype=complex)
    r = np.asarray(right, dtype=complex)
    denom = max(_norm(l), _norm(r), 1e-30)
    minus_dagger = _norm(l - r.conj().T)
    plus_dagger = _norm(l + r.conj().T)
    minus_plain = _norm(l - r)
    plus_plain = _norm(l + r)
    return {
        "name": name,
        "left_minus_right_dagger_norm": minus_dagger,
        "left_plus_right_dagger_norm": plus_dagger,
        "left_minus_right_plain_norm": minus_plain,
        "left_plus_right_plain_norm": plus_plain,
        "left_minus_right_dagger_over_norm": _safe_ratio(minus_dagger, denom),
        "left_plus_right_dagger_over_norm": _safe_ratio(plus_dagger, denom),
        "left_minus_right_plain_over_norm": _safe_ratio(minus_plain, denom),
        "left_plus_right_plain_over_norm": _safe_ratio(plus_plain, denom),
        "valid_for_casimir_input": False,
    }


def linear_combination_report(name: str, matrix: np.ndarray, reference: np.ndarray | None = None) -> dict[str, Any]:
    m = np.asarray(matrix, dtype=complex)
    ref = np.zeros_like(m) if reference is None else np.asarray(reference, dtype=complex)
    residual = m - ref
    return {
        "name": name,
        "combination_norm": _norm(m),
        "reference_norm": _norm(ref),
        "residual_to_reference_norm": _norm(residual),
        "residual_to_reference_over_reference": _safe_ratio(_norm(residual), _norm(ref)),
        "trace": _complex_payload(np.trace(m)),
        "valid_for_casimir_input": False,
    }


def _candidate_coefficients(xi_eV: float, q_norm: float, delta0_eV: float) -> list[dict[str, Any]]:
    xi = float(xi_eV)
    q = float(q_norm)
    d = float(delta0_eV)
    return [
        {"name": "baseline_real_A0L_imag_phase", "a0": xi, "l": q, "phase": -2j * d},
        {"name": "temporal_i_A0_real_L_imag_phase", "a0": 1j * xi, "l": q, "phase": -2j * d},
        {"name": "temporal_minus_i_A0_real_L_imag_phase", "a0": -1j * xi, "l": q, "phase": -2j * d},
        {"name": "baseline_A0L_real_phase", "a0": xi, "l": q, "phase": -2.0 * d},
        {"name": "temporal_i_A0_real_L_real_phase", "a0": 1j * xi, "l": q, "phase": -2.0 * d},
        {"name": "spatial_i_L_imag_phase", "a0": xi, "l": 1j * q, "phase": -2j * d},
        {"name": "spatial_minus_i_L_imag_phase", "a0": xi, "l": -1j * q, "phase": -2j * d},
    ]


def _ward_candidate_reports(
    *,
    side: str,
    gamma0: np.ndarray,
    gamma_l: np.ndarray,
    gamma_phase: np.ndarray,
    delta_h: np.ndarray,
    xi_eV: float,
    q_norm: float,
    delta0_eV: float,
) -> list[dict[str, Any]]:
    reports = []
    for candidate in _candidate_coefficients(xi_eV, q_norm, delta0_eV):
        combo = candidate["a0"] * gamma0 + candidate["l"] * gamma_l + candidate["phase"] * gamma_phase
        reports.append(
            {
                "side": side,
                "candidate": candidate["name"],
                "coefficients": {"A0": complex(candidate["a0"]), "L": complex(candidate["l"]), "phase_eta2": complex(candidate["phase"])},
                "combination_to_zero": linear_combination_report("combination_to_zero", combo),
                "combination_minus_deltaH": linear_combination_report("combination_minus_deltaH", combo, delta_h),
                "combination_plus_deltaH": linear_combination_report("combination_plus_deltaH", combo, -delta_h),
                "valid_for_casimir_input": False,
            }
        )
    return reports


def run_vertex_convention_audit(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_value: float,
    kx: float,
    ky: float,
    nk_for_model: int = 5,
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    current_vertex: str = "peierls",
) -> dict[str, Any]:
    """Run a single-k diagnostic audit of primitive vertices and Ward-like matrix combinations."""

    xi_eV = matsubara_xi_eV(matsubara_index, temperature_K)
    inputs = build_model_scan_inputs(
        model_name=model_name,
        pairing_name=pairing_name,
        xi_eV=xi_eV,
        nk=nk_for_model,
        delta0_eV=delta0_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
    )
    q = np.asarray([float(q_value), 0.0], dtype=float)
    qx, qy = float(q[0]), float(q[1])
    conventions = finite_q_conventions(q, xi_eV)
    kx0 = float(kx)
    ky0 = float(ky)
    pairing_params = inputs.pairing_params
    delta_minus = inputs.ansatz.mean_pairing(kx0 - 0.5 * qx, ky0 - 0.5 * qy, pairing_params)
    delta_plus = inputs.ansatz.mean_pairing(kx0 + 0.5 * qx, ky0 + 0.5 * qy, pairing_params)
    h_minus = bdg_hamiltonian_from_model_pairing(inputs.spec, kx0 - 0.5 * qx, ky0 - 0.5 * qy, delta_minus)
    h_plus = bdg_hamiltonian_from_model_pairing(inputs.spec, kx0 + 0.5 * qx, ky0 + 0.5 * qy, delta_plus)
    delta_h = np.asarray(h_plus, dtype=complex) - np.asarray(h_minus, dtype=complex)

    src0, srcx, srcy = primitive_source_vertices(inputs.spec, kx0, ky0, qx, qy, current_vertex=current_vertex)
    obs0, obsx, obsy = primitive_observable_vertices(inputs.spec, kx0, ky0, qx, qy, current_vertex=current_vertex)
    src_l, src_t = longitudinal_transverse_vertices(srcx, srcy, conventions)
    obs_l, obs_t = longitudinal_transverse_vertices(obsx, obsy, conventions)
    coll = collective_vertices(inputs.ansatz, kx0, ky0, qx, qy, pairing_params)
    collective_order, raw_names = collective_order_from_ansatz(inputs.ansatz, len(coll))
    phase_index = collective_order.index("phase_eta2") if "phase_eta2" in collective_order else len(coll) - 1
    phase = np.asarray(coll[phase_index], dtype=complex)
    delta0 = float(getattr(pairing_params, "delta0_eV", 0.0))

    source_vertices = {"A0": src0, "L": src_l, "T": src_t}
    observable_vertices = {"A0": obs0, "L": obs_l, "T": obs_t}
    collective_vertices_by_label = {label: np.asarray(vertex, dtype=complex) for label, vertex in zip(collective_order, coll, strict=True)}

    finite_difference_reports = {
        "source": [
            linear_combination_report("deltaH_minus_qL_source", delta_h, conventions.gL * src_l),
            linear_combination_report("deltaH_plus_qL_source", delta_h, -conventions.gL * src_l),
            linear_combination_report("deltaH_minus_iqL_source", delta_h, 1j * conventions.gL * src_l),
            linear_combination_report("deltaH_plus_iqL_source", delta_h, -1j * conventions.gL * src_l),
        ],
        "observable": [
            linear_combination_report("deltaH_minus_qL_observable", delta_h, conventions.gL * obs_l),
            linear_combination_report("deltaH_plus_qL_observable", delta_h, -conventions.gL * obs_l),
            linear_combination_report("deltaH_minus_iqL_observable", delta_h, 1j * conventions.gL * obs_l),
            linear_combination_report("deltaH_plus_iqL_observable", delta_h, -1j * conventions.gL * obs_l),
        ],
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
            "reason": "primitive_vertex_convention_audit_not_production_convention",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "point": {
            "kx": kx0,
            "ky": ky0,
            "q_model": [qx, qy],
            "xi_eV": float(xi_eV),
            "matsubara_index": int(matsubara_index),
            "temperature_K": float(temperature_K),
            "delta0_eV": delta0,
            "current_vertex": current_vertex,
            "valid_for_casimir_input": False,
        },
        "frequency": frequency_payload(matsubara_index, temperature_K),
        "matrix_norms": {
            "H_minus_norm": _norm(h_minus),
            "H_plus_norm": _norm(h_plus),
            "deltaH_norm": _norm(delta_h),
            "valid_for_casimir_input": False,
        },
        "primitive_vertex_reports": {
            "source": [matrix_report(label, matrix) for label, matrix in source_vertices.items()],
            "observable": [matrix_report(label, matrix) for label, matrix in observable_vertices.items()],
            "source_observable_relations": [relation_report(label, observable_vertices[label], source_vertices[label]) for label in PRIMITIVE_LABELS],
            "valid_for_casimir_input": False,
        },
        "collective_vertex_reports": {
            "collective_order": list(collective_order),
            "raw_ansatz_channel_names": list(raw_names) if raw_names is not None else None,
            "reports": [matrix_report(label, matrix) for label, matrix in collective_vertices_by_label.items()],
            "phase_eta2_focus": {
                "phase_index": int(phase_index),
                "phase_label": collective_order[phase_index],
                "phase_report": matrix_report("phase_eta2", phase),
                "i_phase_report": matrix_report("i_times_phase_eta2", 1j * phase),
                "valid_for_casimir_input": False,
            },
            "valid_for_casimir_input": False,
        },
        "finite_difference_current_checks": finite_difference_reports,
        "ward_like_matrix_combinations": {
            "note": "Diagnostic combinations only. They are not accepted Ward identities without an analytic derivation.",
            "source": _ward_candidate_reports(side="source", gamma0=src0, gamma_l=src_l, gamma_phase=phase, delta_h=delta_h, xi_eV=xi_eV, q_norm=conventions.gL, delta0_eV=delta0),
            "observable": _ward_candidate_reports(side="observable", gamma0=obs0, gamma_l=obs_l, gamma_phase=phase, delta_h=delta_h, xi_eV=xi_eV, q_norm=conventions.gL, delta0_eV=delta0),
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_and_write_vertex_convention_audit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_vertex_convention_audit(**kwargs)
    write_json(Path(output_dir) / "vertex_convention_audit.json", payload)
    return payload
