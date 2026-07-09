"""Diagnostic-only BdG primitive EM translation-RHS audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lno327.bdg.hamiltonian import bdg_hamiltonian_from_model_pairing
from lno327.response.finite_q import kubo_factor, thermal_expectation_bdg_from_hamiltonian, vertex_band
from lno327.response.finite_q_bdg import bdg_eigensystem_from_model_pairing
from lno327.response.occupations import fermi_function

from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..adapters.primitive_vertices_adapter import primitive_observable_vertices, primitive_source_vertices, primitive_spatial_contact_vertices
from ..io.writers import write_json
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .contact_ablation import _shifted_payload
from .extended_ward_kernel import PRIMITIVE_ORDER, complex_vector_payload
from .primitive_response_ward_audit import primitive_ward_candidate_vectors
from .shifted_average import shift_pairs_from_fractions

SCHEMA_VERSION = "finite_q_tmte_primitive_em_translation_rhs_audit_v1"
DEFAULT_CANDIDATE = "matrix_inferred_matsubara_i_asymmetric"


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=complex)))


def _ratio(a: float, b: float, eps: float = 1e-30) -> float:
    return float(a) / max(float(b), eps)


def _vec(x: np.ndarray) -> list[dict[str, Any]]:
    return complex_vector_payload(np.asarray(x, dtype=complex).reshape(3), PRIMITIVE_ORDER)


def _fit(target: np.ndarray, cand: np.ndarray) -> dict[str, Any]:
    t = np.asarray(target, dtype=complex).reshape(3)
    c = np.asarray(cand, dtype=complex).reshape(3)
    den = np.vdot(c, c)
    alpha = 0j if abs(den) < 1e-30 else np.vdot(c, t) / den
    res = t - alpha * c
    return {
        "alpha": complex(alpha),
        "residual_norm": _norm(res),
        "residual_over_target_norm": _ratio(_norm(res), _norm(t)),
        "target_norm": _norm(t),
        "candidate_norm": _norm(c),
        "valid_for_casimir_input": False,
    }


def _match(name: str, vector: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    v = np.asarray(vector, dtype=complex).reshape(3)
    t = np.asarray(target, dtype=complex).reshape(3)
    diff = v - t
    return {
        "name": name,
        "values": _vec(v),
        "norm": _norm(v),
        "difference_norm": _norm(diff),
        "difference_over_target_norm": _ratio(_norm(diff), _norm(t)),
        "fit_to_target": _fit(t, v),
        "valid_for_casimir_input": False,
    }


def _accumulate(
    *,
    spec: object,
    ansatz: object,
    pairing_params: object,
    q: np.ndarray,
    xi_eV: float,
    points: np.ndarray,
    weights: np.ndarray,
    config: object,
) -> dict[str, Any]:
    qx, qy = float(q[0]), float(q[1])
    bubble = np.zeros((3, 3), dtype=complex)
    contact = np.zeros((3, 3), dtype=complex)
    equal_forward = np.zeros(3, dtype=complex)
    delta_v_mid = np.zeros(3, dtype=complex)
    qM_mid = np.zeros(3, dtype=complex)

    for weight, (kx0, ky0) in zip(weights, points, strict=True):
        kx = float(kx0)
        ky = float(ky0)
        delta_minus = ansatz.mean_pairing(kx - 0.5 * qx, ky - 0.5 * qy, pairing_params)
        delta_plus = ansatz.mean_pairing(kx + 0.5 * qx, ky + 0.5 * qy, pairing_params)
        delta_mid = ansatz.mean_pairing(kx, ky, pairing_params)
        bands_minus = bdg_eigensystem_from_model_pairing(spec, kx - 0.5 * qx, ky - 0.5 * qy, delta_minus)
        bands_plus = bdg_eigensystem_from_model_pairing(spec, kx + 0.5 * qx, ky + 0.5 * qy, delta_plus)
        occ_minus = fermi_function(bands_minus.energies, config.fermi_level_eV, config.temperature_eV)
        occ_plus = fermi_function(bands_plus.energies, config.fermi_level_eV, config.temperature_eV)
        obs = primitive_observable_vertices(spec, kx, ky, qx, qy)
        src = primitive_source_vertices(spec, kx, ky, qx, qy)
        left_band = tuple(vertex_band(bands_minus.states, v, bands_plus.states) for v in obs)
        right_band = tuple(vertex_band(bands_minus.states, v, bands_plus.states) for v in src)
        rho_band = right_band[0]
        for m, em in enumerate(bands_minus.energies):
            for n, ep in enumerate(bands_plus.energies):
                occ = float(occ_minus[m] - occ_plus[n])
                raw = kubo_factor(float(em), float(ep), float(occ_minus[m]), float(occ_plus[n]), xi_eV, fermi_level_eV=config.fermi_level_eV, temperature_eV=config.temperature_eV, eta_eV=config.eta_eV)
                if raw != 0.0:
                    factor = 0.5 * float(weight) * raw
                    for a, lv in enumerate(left_band):
                        for b, rv in enumerate(right_band):
                            bubble[a, b] += factor * lv[m, n] * np.conjugate(rv[m, n])
                if occ != 0.0:
                    equal_forward += 0.5 * float(weight) * np.asarray([occ * rho_band[m, n] * np.conjugate(right_band[j][m, n]) for j in range(3)], dtype=complex)
        h_mid = bdg_hamiltonian_from_model_pairing(spec, kx, ky, delta_mid)
        contacts = primitive_spatial_contact_vertices(spec, kx, ky, qx, qy)
        for i, di in enumerate(("x", "y")):
            qi = qx if di == "x" else qy
            for j, dj in enumerate(("x", "y")):
                direct = -float(weight) * thermal_expectation_bdg_from_hamiltonian(h_mid, contacts[(di, dj)], config)
                contact[1 + i, 1 + j] += direct
                qM_mid[1 + j] += qi * direct
        for j, direction in enumerate(("x", "y"), start=1):
            vp = primitive_source_vertices(spec, kx + 0.5 * qx, ky + 0.5 * qy, qx, qy)[j]
            vm = primitive_source_vertices(spec, kx - 0.5 * qx, ky - 0.5 * qy, qx, qy)[j]
            delta_v_mid[j] += float(weight) * thermal_expectation_bdg_from_hamiltonian(h_mid, vp - vm, config)
    return {"bubble": bubble, "contact": contact, "equal_forward": equal_forward, "delta_v_mid": delta_v_mid, "qM_mid": qM_mid}


def _average(rows: list[dict[str, Any]]) -> dict[str, Any]:
    inv = 1.0 / float(len(rows))
    return {key: sum(np.asarray(row[key], dtype=complex) for row in rows) * inv for key in ("bubble", "contact", "equal_forward", "delta_v_mid", "qM_mid")}


def _candidate_lookup(xi_eV: float, q_value: float, delta0_eV: float) -> dict[str, dict[str, Any]]:
    return {row["candidate"]: row for row in primitive_ward_candidate_vectors(xi_eV, abs(float(q_value)), delta0_eV)}


def run_primitive_em_translation_rhs_audit(
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
    candidate_name: str = DEFAULT_CANDIDATE,
) -> dict[str, Any]:
    if nk <= 0:
        raise ValueError("nk must be positive")
    if float(q_value) <= 0.0:
        raise ValueError("q must be positive")
    xi = matsubara_xi_eV(matsubara_index, temperature_K)
    inputs = build_model_scan_inputs(model_name=model_name, pairing_name=pairing_name, xi_eV=xi, nk=nk, delta0_eV=delta0_eV, temperature_K=temperature_K, eta_eV=eta_eV)
    q = np.asarray([float(q_value), 0.0], dtype=float)
    shifts = shift_pairs_from_fractions(shift_fractions)
    parts = []
    for sx, sy in shifts:
        pts = shifted_uniform_bz_mesh(nk, sx, sy)
        parts.append(_accumulate(spec=inputs.spec, ansatz=inputs.ansatz, pairing_params=inputs.pairing_params, q=q, xi_eV=xi, points=pts, weights=weights_for_points(pts), config=inputs.config))
    data = _average(parts)
    delta0 = float(getattr(inputs.pairing_params, "delta0_eV", 0.0))
    candidates = _candidate_lookup(xi, q_value, delta0)
    if candidate_name not in candidates:
        raise ValueError(f"unknown primitive Ward candidate {candidate_name!r}")
    candidate = candidates[candidate_name]
    u_left = np.asarray(candidate["left_u"], dtype=complex).reshape(3)
    u_right = np.asarray(candidate["right_u"], dtype=complex).reshape(3)
    bubble = data["bubble"]
    contact = data["contact"]
    total = bubble + contact
    left_total = u_left @ total
    right_total = total @ u_right
    left_missing = -left_total
    translation = data["equal_forward"] - data["delta_v_mid"]
    vectors = {
        "translation_forward": translation,
        "minus_translation_forward": -translation,
        "equal_forward": data["equal_forward"],
        "minus_equal_forward": -data["equal_forward"],
        "delta_v_mid": data["delta_v_mid"],
        "minus_delta_v_mid": -data["delta_v_mid"],
        "qM_mid": data["qM_mid"],
        "minus_qM_mid": -data["qM_mid"],
        "delta_v_plus_qM": data["delta_v_mid"] + data["qM_mid"],
        "minus_delta_v_plus_qM": -data["delta_v_mid"] - data["qM_mid"],
    }
    ranked = sorted((_match(name, vector, left_missing) for name, vector in vectors.items()), key=lambda row: float(row["difference_over_target_norm"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {"diagnostic_run_completed": True, "diagnostic_only_not_a_fix": True, "accepted_convention": False, "valid_for_casimir_input": False, "reason": "primitive_em_translation_rhs_audit_not_production_convention"},
        "model": {"name": model_name, "pairing": pairing_name, "delta0_eV": delta0, "valid_for_casimir_input": False},
        "frequency": frequency_payload(matsubara_index, temperature_K),
        "debug_parameters": {"q_value": float(q_value), "nk": int(nk), "eta_eV": float(eta_eV), "shift_fractions": [float(v) for v in shift_fractions], "shifted_mesh_average": _shifted_payload(shift_fractions, shifts), "candidate_name": candidate_name, "primitive_order": list(PRIMITIVE_ORDER), "valid_for_casimir_input": False},
        "block_norms": {"bubble_norm": _norm(bubble), "contact_norm": _norm(contact), "total_norm": _norm(total), "valid_for_casimir_input": False},
        "ward_decomposition": {
            "left": {"total": {"values": _vec(left_total), "norm": _norm(left_total)}, "missing_to_close": {"values": _vec(left_missing), "norm": _norm(left_missing)}, "u_vector": _vec(u_left), "valid_for_casimir_input": False},
            "right": {"total": {"values": _vec(right_total), "norm": _norm(right_total)}, "missing_to_close": {"values": _vec(-right_total), "norm": _norm(right_total)}, "u_vector": _vec(u_right), "valid_for_casimir_input": False},
            "valid_for_casimir_input": False,
        },
        "candidate_translation_vectors_ranked": ranked,
        "raw_translation_vectors": {name: {"values": _vec(vector), "norm": _norm(vector)} for name, vector in vectors.items()},
        "interpretation_guardrails": {"primitive_em_only_no_collective_schur": True, "not_a_fit_fix": True, "if_translation_matches": "finite-q translation RHS explains primitive EM residual before collective Schur", "if_no_match": "remaining residual is not the normal-like EM translation RHS alone", "valid_for_casimir_input": False},
        "valid_for_casimir_input": False,
    }


def run_and_write_primitive_em_translation_rhs_audit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_primitive_em_translation_rhs_audit(**kwargs)
    write_json(Path(output_dir) / "primitive_em_translation_rhs_audit.json", payload)
    return payload
