"""Diagnostic-only normal finite-q equal-time Ward audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lno327 import KuboConfig, fermi_function
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

from ..adapters.model_adapter import shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from .contact_ablation import _shifted_payload
from .extended_ward_kernel import complex_vector_payload
from .normal_contact_ward_control import NORMAL_ORDER, normal_peierls_vertex_ward_residual
from .normal_response_convention_audit import band_vertex, kubo_factor, normal_thermal_expectation, ward_vectors
from .shifted_average import shift_pairs_from_fractions

SCHEMA_VERSION = "finite_q_tmte_normal_equal_time_ward_audit_v1"


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=complex)))


def _ratio(a: float, b: float, eps: float = 1e-30) -> float:
    return float(a) / max(float(b), eps)


def _vec(x: np.ndarray) -> list[dict[str, Any]]:
    return complex_vector_payload(np.asarray(x, dtype=complex).reshape(3), NORMAL_ORDER)


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
    d = v - t
    return {
        "name": name,
        "values": _vec(v),
        "norm": _norm(v),
        "difference_norm": _norm(d),
        "difference_over_target_norm": _ratio(_norm(d), _norm(t)),
        "fit_to_target": _fit(t, v),
        "valid_for_casimir_input": False,
    }


def _accumulate(spec: object, q: np.ndarray, xi: float, points: np.ndarray, weights: np.ndarray, config: KuboConfig) -> dict[str, Any]:
    qx, qy = float(q[0]), float(q[1])
    dim = np.asarray(spec.normal_hamiltonian(float(points[0, 0]), float(points[0, 1]))).shape[0]
    rho = np.eye(dim, dtype=complex)
    bubble = np.zeros((3, 3), dtype=complex)
    contact = np.zeros((3, 3), dtype=complex)
    equal_forward = np.zeros(3, dtype=complex)
    equal_direct = np.zeros(3, dtype=complex)
    delta_v_mid = np.zeros(3, dtype=complex)
    qM_mid = np.zeros(3, dtype=complex)
    v_abs: list[float] = []
    v_rel: list[float] = []

    for weight, (kx0, ky0) in zip(weights, points, strict=True):
        kx = float(kx0)
        ky = float(ky0)
        hm = np.asarray(spec.normal_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy), dtype=complex)
        hp = np.asarray(spec.normal_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy), dtype=complex)
        h0 = np.asarray(spec.normal_hamiltonian(kx, ky), dtype=complex)
        em, um = np.linalg.eigh(hm)
        ep, up = np.linalg.eigh(hp)
        fm = fermi_function(em, config.fermi_level_eV, config.temperature_eV)
        fp = fermi_function(ep, config.fermi_level_eV, config.temperature_eV)
        vx = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x")
        vy = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "y")
        obs = (rho, -vx, -vy)
        src = (rho, vx, vy)
        left = tuple(band_vertex(um, v, up, "forward_minus_plus") for v in obs)
        right = tuple(band_vertex(um, v, up, "forward_minus_plus") for v in src)
        rho_f = band_vertex(um, rho, up, "forward_minus_plus")
        src_f = tuple(band_vertex(um, v, up, "forward_minus_plus") for v in src)
        rho_d = band_vertex(um, rho, up, "direct_minus_plus")
        src_d = tuple(band_vertex(um, v, up, "direct_minus_plus") for v in src)
        for m, e_m in enumerate(em):
            for n, e_p in enumerate(ep):
                occ = float(fm[m] - fp[n])
                kfac = kubo_factor(
                    energy_minus=float(e_m),
                    energy_plus=float(e_p),
                    occupation_minus=float(fm[m]),
                    occupation_plus=float(fp[n]),
                    xi_eV=xi,
                    convention="minus_plus",
                )
                if kfac != 0.0:
                    for a, lv in enumerate(left):
                        for b, rv in enumerate(right):
                            bubble[a, b] += float(weight) * kfac * lv[m, n] * np.conjugate(rv[m, n])
                if occ != 0.0:
                    equal_forward += float(weight) * np.array([occ * rho_f[m, n] * np.conjugate(src_f[j][m, n]) for j in range(3)])
                    equal_direct += float(weight) * np.array([occ * rho_d[m, n] * np.conjugate(src_d[j][m, n]) for j in range(3)])
        for i, di in enumerate(("x", "y")):
            qi = qx if di == "x" else qy
            for j, dj in enumerate(("x", "y")):
                mij = spec.peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, di, dj)
                direct = -normal_thermal_expectation(h0, mij, config)
                contact[1 + i, 1 + j] += float(weight) * direct
                qM_mid[1 + j] += float(weight) * qi * direct
        for j, direction in enumerate(("x", "y"), start=1):
            vp = spec.peierls_hamiltonian_vector_vertex(kx + 0.5 * qx, ky + 0.5 * qy, qx, qy, direction)
            vm = spec.peierls_hamiltonian_vector_vertex(kx - 0.5 * qx, ky - 0.5 * qy, qx, qy, direction)
            delta_v_mid[j] += float(weight) * normal_thermal_expectation(h0, vp - vm, config)
        check = normal_peierls_vertex_ward_residual(spec, kx, ky, qx, qy)
        v_abs.append(float(check["abs_error"]))
        v_rel.append(float(check["rel_error"]))
    return {
        "bubble": bubble,
        "contact": contact,
        "equal_forward": equal_forward,
        "equal_direct": equal_direct,
        "delta_v_mid": delta_v_mid,
        "qM_mid": qM_mid,
        "vertex_abs": max(v_abs) if v_abs else 0.0,
        "vertex_rel": max(v_rel) if v_rel else 0.0,
        "vertex_rel_mean": float(np.mean(v_rel)) if v_rel else 0.0,
    }


def _average(rows: list[dict[str, Any]]) -> dict[str, Any]:
    inv = 1.0 / len(rows)
    keys = ("bubble", "contact", "equal_forward", "equal_direct", "delta_v_mid", "qM_mid")
    out = {k: sum(np.asarray(r[k], dtype=complex) for r in rows) * inv for k in keys}
    out["vertex"] = {
        "max_abs_error_over_shifted_meshes": max(float(r["vertex_abs"]) for r in rows),
        "max_rel_error_over_shifted_meshes": max(float(r["vertex_rel"]) for r in rows),
        "mean_rel_error_over_shifted_meshes": float(np.mean([float(r["vertex_rel_mean"]) for r in rows])),
        "valid_for_casimir_input": False,
    }
    return out


def run_normal_equal_time_ward_audit(
    *,
    model_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_value: float,
    nk: int,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
) -> dict[str, Any]:
    if nk <= 0:
        raise ValueError("nk must be positive")
    xi = matsubara_xi_eV(matsubara_index, temperature_K)
    model = get_finite_q_validation_model(model_name)
    spec = model.spec
    config = KuboConfig.from_kelvin(omega_eV=xi, temperature_K=float(temperature_K), eta_eV=float(eta_eV), output_si=False)
    q = np.array([float(q_value), 0.0], dtype=float)
    shifts = shift_pairs_from_fractions(shift_fractions)
    parts = []
    for sx, sy in shifts:
        pts = shifted_uniform_bz_mesh(nk, sx, sy)
        parts.append(_accumulate(spec, q, xi, pts, weights_for_points(pts), config))
    data = _average(parts)
    bubble = data["bubble"]
    contact = data["contact"]
    total = bubble + contact
    left, right = ward_vectors(xi, q, "standard")
    lb = left @ bubble
    lc = left @ contact
    lt = left @ total
    rb = bubble @ right
    rc = contact @ right
    rt = total @ right
    missing = -lt
    cand = {
        "equal_forward": data["equal_forward"],
        "minus_equal_forward": -data["equal_forward"],
        "equal_direct": data["equal_direct"],
        "minus_equal_direct": -data["equal_direct"],
        "delta_v_mid": data["delta_v_mid"],
        "minus_delta_v_mid": -data["delta_v_mid"],
        "translation_forward": data["equal_forward"] - data["delta_v_mid"],
        "minus_translation_forward": -data["equal_forward"] + data["delta_v_mid"],
        "translation_direct": data["equal_direct"] - data["delta_v_mid"],
        "minus_translation_direct": -data["equal_direct"] + data["delta_v_mid"],
        "qM_mid": data["qM_mid"],
        "minus_qM_mid": -data["qM_mid"],
        "delta_v_plus_qM": data["delta_v_mid"] + data["qM_mid"],
        "minus_delta_v_plus_qM": -data["delta_v_mid"] - data["qM_mid"],
    }
    ranked = sorted((_match(k, v, missing) for k, v in cand.items()), key=lambda r: float(r["difference_over_target_norm"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {"diagnostic_run_completed": True, "diagnostic_only_not_a_fix": True, "accepted_convention": False, "valid_for_casimir_input": False},
        "model": {"name": model_name, "valid_for_casimir_input": False},
        "frequency": frequency_payload(matsubara_index, temperature_K),
        "debug_parameters": {"q_value": float(q_value), "nk": int(nk), "eta_eV": float(eta_eV), "shift_fractions": list(map(float, shift_fractions)), "shifted_mesh_average": _shifted_payload(shift_fractions, shifts), "valid_for_casimir_input": False},
        "vertex_identity": data["vertex"],
        "block_norms": {"bubble_norm": _norm(bubble), "contact_norm": _norm(contact), "total_norm": _norm(total), "valid_for_casimir_input": False},
        "ward_decomposition": {
            "left": {"bubble": {"values": _vec(lb), "norm": _norm(lb)}, "contact": {"values": _vec(lc), "norm": _norm(lc)}, "total": {"values": _vec(lt), "norm": _norm(lt)}, "missing_to_close": {"values": _vec(missing), "norm": _norm(missing)}, "contact_required_over_current": _fit(-lb, lc)},
            "right": {"bubble": {"values": _vec(rb), "norm": _norm(rb)}, "contact": {"values": _vec(rc), "norm": _norm(rc)}, "total": {"values": _vec(rt), "norm": _norm(rt)}, "missing_to_close": {"values": _vec(-rt), "norm": _norm(rt)}, "contact_required_over_current": _fit(-rb, rc)},
            "valid_for_casimir_input": False,
        },
        "candidate_equal_time_vectors_ranked": ranked,
        "valid_for_casimir_input": False,
    }


def run_and_write_normal_equal_time_ward_audit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_normal_equal_time_ward_audit(**kwargs)
    write_json(Path(output_dir) / "normal_equal_time_ward_audit.json", payload)
    return payload
