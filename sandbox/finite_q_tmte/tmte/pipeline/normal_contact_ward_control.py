"""Diagnostic-only normal-state Peierls contact Ward control.

This audit bypasses BdG pairing ansatz, collective phase vertices, and Schur
completion. It tests whether the normal-state Peierls density/current bubble plus
normal Peierls diamagnetic contact closes the finite-q Ward identity.
"""

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
from .shifted_average import shift_pairs_from_fractions

SCHEMA_VERSION = "finite_q_tmte_normal_contact_ward_control_v1"
NORMAL_ORDER = ("density", "current_x", "current_y")


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-30) -> float:
    return float(numerator) / max(float(denominator), float(eps))


def _vec(values: np.ndarray, labels: Sequence[str] = NORMAL_ORDER) -> list[dict[str, Any]]:
    return complex_vector_payload(np.asarray(values, dtype=complex).reshape(-1), labels)


def _matrix_payload(matrix: np.ndarray, rows: Sequence[str] = NORMAL_ORDER, cols: Sequence[str] = NORMAL_ORDER) -> list[dict[str, Any]]:
    a = np.asarray(matrix, dtype=complex)
    return [{"row": row, "values": complex_vector_payload(a[i, :], cols)} for i, row in enumerate(rows)]


def _vertex_band(states_minus: np.ndarray, vertex: np.ndarray, states_plus: np.ndarray) -> np.ndarray:
    """Return finite-q vertex elements stored as [minus, plus]."""

    return (states_plus.conjugate().T @ np.asarray(vertex, dtype=complex) @ states_minus).T


def _kubo_factor(em: float, en: float, fm: float, fn: float, omega_eV: float) -> complex:
    return (float(fm) - float(fn)) / (1j * float(omega_eV) + float(em) - float(en))


def _normal_thermal_expectation(hamiltonian: np.ndarray, vertex: np.ndarray, config: KuboConfig) -> complex:
    energies, states = np.linalg.eigh(np.asarray(hamiltonian, dtype=complex))
    occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
    vertex_band = states.conjugate().T @ np.asarray(vertex, dtype=complex) @ states
    return complex(np.sum(occupations * np.diag(vertex_band)))


def normal_peierls_vertex_ward_residual(spec: object, kx: float, ky: float, qx: float, qy: float) -> dict[str, Any]:
    vx = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x")
    vy = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "y")
    lhs = qx * vx + qy * vy
    h_plus = spec.normal_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
    h_minus = spec.normal_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
    rhs = h_plus - h_minus
    abs_error = _norm(lhs - rhs)
    rhs_norm = _norm(rhs)
    return {
        "abs_error": abs_error,
        "rhs_norm": rhs_norm,
        "lhs_norm": _norm(lhs),
        "rel_error": _safe_ratio(abs_error, rhs_norm),
        "valid_for_casimir_input": False,
    }


def accumulate_normal_blocks(
    *,
    spec: object,
    q_model: np.ndarray,
    xi_eV: float,
    k_points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
) -> dict[str, Any]:
    qx, qy = float(q_model[0]), float(q_model[1])
    if float(np.linalg.norm(q_model)) <= 0.0:
        raise ValueError("normal contact Ward control requires q > 0")
    first_h = np.asarray(spec.normal_hamiltonian(float(k_points[0, 0]), float(k_points[0, 1])), dtype=complex)
    dim = first_h.shape[0]
    rho = np.eye(dim, dtype=complex)
    bubble = np.zeros((3, 3), dtype=complex)
    contact = np.zeros((3, 3), dtype=complex)
    vertex_rel_errors: list[float] = []
    vertex_abs_errors: list[float] = []

    for weight, (kx_value, ky_value) in zip(weights, k_points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = np.asarray(spec.normal_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy), dtype=complex)
        h_plus = np.asarray(spec.normal_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy), dtype=complex)
        e_minus, u_minus = np.linalg.eigh(h_minus)
        e_plus, u_plus = np.linalg.eigh(h_plus)
        occ_minus = fermi_function(e_minus, config.fermi_level_eV, config.temperature_eV)
        occ_plus = fermi_function(e_plus, config.fermi_level_eV, config.temperature_eV)

        vx = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x")
        vy = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "y")
        observable_vertices = (rho, -vx, -vy)
        source_vertices = (rho, vx, vy)
        left_band = tuple(_vertex_band(u_minus, vertex, u_plus) for vertex in observable_vertices)
        right_band = tuple(_vertex_band(u_minus, vertex, u_plus) for vertex in source_vertices)
        for m, energy_minus in enumerate(e_minus):
            for n, energy_plus in enumerate(e_plus):
                raw_factor = _kubo_factor(
                    float(energy_minus),
                    float(energy_plus),
                    float(occ_minus[m]),
                    float(occ_plus[n]),
                    xi_eV,
                )
                if raw_factor == 0.0:
                    continue
                factor = float(weight) * raw_factor
                for mu, left in enumerate(left_band):
                    for nu, right in enumerate(right_band):
                        bubble[mu, nu] += factor * left[m, n] * np.conjugate(right[m, n])

        h_mid = np.asarray(spec.normal_hamiltonian(kx, ky), dtype=complex)
        for i, di in enumerate(("x", "y")):
            for j, dj in enumerate(("x", "y")):
                vertex = spec.peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, di, dj)
                contact[1 + i, 1 + j] += -float(weight) * _normal_thermal_expectation(h_mid, vertex, config)

        vertex_check = normal_peierls_vertex_ward_residual(spec, kx, ky, qx, qy)
        vertex_rel_errors.append(float(vertex_check["rel_error"]))
        vertex_abs_errors.append(float(vertex_check["abs_error"]))

    return {
        "bubble": bubble,
        "contact": contact,
        "total": bubble + contact,
        "vertex_identity": {
            "max_abs_error": max(vertex_abs_errors) if vertex_abs_errors else 0.0,
            "max_rel_error": max(vertex_rel_errors) if vertex_rel_errors else 0.0,
            "mean_rel_error": float(np.mean(vertex_rel_errors)) if vertex_rel_errors else 0.0,
            "identity": "qx Vx + qy Vy = H(k+q/2)-H(k-q/2)",
            "valid_for_casimir_input": False,
        },
    }


def ward_vectors(xi_eV: float, q_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qx, qy = float(q_model[0]), float(q_model[1])
    left = np.asarray([1j * float(xi_eV), qx, qy], dtype=complex)
    right = np.asarray([1j * float(xi_eV), -qx, -qy], dtype=complex)
    return left, right


def ward_residuals(matrix: np.ndarray, xi_eV: float, q_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = ward_vectors(xi_eV, q_model)
    response = np.asarray(matrix, dtype=complex)
    return left @ response, response @ right


def scalar_projection(required: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    r = np.asarray(required, dtype=complex).reshape(-1)
    c = np.asarray(current, dtype=complex).reshape(-1)
    denom = np.vdot(c, c)
    alpha = 0.0 + 0.0j if abs(denom) < 1e-30 else np.vdot(c, r) / denom
    residual = r - alpha * c
    return {
        "alpha_required_over_current": complex(alpha),
        "residual_norm": _norm(residual),
        "required_norm": _norm(r),
        "current_norm": _norm(c),
        "residual_over_required_norm": _safe_ratio(_norm(residual), _norm(r)),
        "valid_for_casimir_input": False,
    }


def contact_formula_payload(bubble: np.ndarray, contact: np.ndarray, xi_eV: float, q_model: np.ndarray) -> dict[str, Any]:
    left, right = ward_vectors(xi_eV, q_model)
    left_current = left @ contact
    left_required = -(left @ bubble)
    right_current = contact @ right
    right_required = -(bubble @ right)
    left_projection = scalar_projection(left_required, left_current)
    right_projection = scalar_projection(right_required, right_current)
    alpha_left = complex(left_projection["alpha_required_over_current"])
    alpha_right = complex(right_projection["alpha_required_over_current"])
    return {
        "formula": "normal_contact_required = -normal_bubble_contraction",
        "left_current_contact": {"values": _vec(left_current), "norm": _norm(left_current)},
        "left_required_contact": {"values": _vec(left_required), "norm": _norm(left_required)},
        "right_current_contact": {"values": _vec(right_current), "norm": _norm(right_current)},
        "right_required_contact": {"values": _vec(right_required), "norm": _norm(right_required)},
        "left_required_over_current_scalar_projection": left_projection,
        "right_required_over_current_scalar_projection": right_projection,
        "left_right_alpha_abs_diff": float(abs(alpha_left - alpha_right)),
        "valid_for_casimir_input": False,
    }


def response_payload(name: str, matrix: np.ndarray, xi_eV: float, q_model: np.ndarray, reference_norm: float) -> dict[str, Any]:
    left, right = ward_residuals(matrix, xi_eV, q_model)
    left_norm = _norm(left)
    right_norm = _norm(right)
    return {
        "name": name,
        "matrix": _matrix_payload(matrix),
        "matrix_norm": _norm(matrix),
        "left_residual": {"values": _vec(left), "norm": left_norm, "norm_over_reference": _safe_ratio(left_norm, reference_norm)},
        "right_residual": {"values": _vec(right), "norm": right_norm, "norm_over_reference": _safe_ratio(right_norm, reference_norm)},
        "max_residual_norm": max(left_norm, right_norm),
        "max_residual_over_reference": _safe_ratio(max(left_norm, right_norm), reference_norm),
        "valid_for_casimir_input": False,
    }


def run_normal_contact_ward_control(
    *,
    model_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_value: float,
    nk: int,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
) -> dict[str, Any]:
    xi_eV = matsubara_xi_eV(matsubara_index, temperature_K)
    if nk <= 0:
        raise ValueError("nk must be positive")
    model = get_finite_q_validation_model(model_name)
    spec = model.spec
    for required in ("normal_hamiltonian", "peierls_hamiltonian_vector_vertex", "peierls_hamiltonian_contact_vertex"):
        if not hasattr(spec, required):
            raise ValueError(f"model spec does not support required normal Peierls method {required!r}")
    config = KuboConfig.from_kelvin(omega_eV=xi_eV, temperature_K=float(temperature_K), eta_eV=float(eta_eV), output_si=False)
    q_model = np.asarray([float(q_value), 0.0], dtype=float)
    shifts = shift_pairs_from_fractions(shift_fractions)
    bubbles: list[np.ndarray] = []
    contacts: list[np.ndarray] = []
    vertex_reports: list[dict[str, Any]] = []
    for sx, sy in shifts:
        points = shifted_uniform_bz_mesh(nk, sx, sy)
        weights = weights_for_points(points)
        blocks = accumulate_normal_blocks(spec=spec, q_model=q_model, xi_eV=xi_eV, k_points=points, weights=weights, config=config)
        bubbles.append(np.asarray(blocks["bubble"], dtype=complex))
        contacts.append(np.asarray(blocks["contact"], dtype=complex))
        vertex_reports.append(blocks["vertex_identity"])
    bubble = sum(bubbles) / float(len(bubbles))
    contact = sum(contacts) / float(len(contacts))
    total = bubble + contact
    reference_norm = max(_norm(total), _norm(bubble), _norm(contact), 1e-30)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "valid_for_casimir_input": False,
            "reason": "normal_contact_ward_control_not_production_convention",
        },
        "model": {"name": model_name, "valid_for_casimir_input": False},
        "frequency": frequency_payload(matsubara_index, temperature_K),
        "debug_parameters": {
            "q_model_convention": "q_along_x_only",
            "q_value": float(q_value),
            "nk": int(nk),
            "eta_eV": float(eta_eV),
            "shift_fractions": [float(value) for value in shift_fractions],
            "num_shifted_meshes": len(shifts),
            "shifted_mesh_average": _shifted_payload(shift_fractions, shifts),
            "normal_state_only": True,
            "no_bdg_pairing_ansatz": True,
            "no_collective_phase_or_schur": True,
            "valid_for_casimir_input": False,
        },
        "normal_order": list(NORMAL_ORDER),
        "vertex_identity": {
            "max_abs_error_over_shifted_meshes": max(float(row["max_abs_error"]) for row in vertex_reports),
            "max_rel_error_over_shifted_meshes": max(float(row["max_rel_error"]) for row in vertex_reports),
            "mean_rel_error_over_shifted_meshes": float(np.mean([float(row["mean_rel_error"]) for row in vertex_reports])),
            "per_shift_summary": vertex_reports,
            "valid_for_casimir_input": False,
        },
        "block_norms": {
            "bubble_norm": _norm(bubble),
            "contact_norm": _norm(contact),
            "total_norm": _norm(total),
            "valid_for_casimir_input": False,
        },
        "responses": [
            response_payload("bubble", bubble, xi_eV, q_model, reference_norm),
            response_payload("contact", contact, xi_eV, q_model, reference_norm),
            response_payload("total", total, xi_eV, q_model, reference_norm),
        ],
        "contact_formula": contact_formula_payload(bubble, contact, xi_eV, q_model),
        "interpretation_guardrails": {
            "if_total_residual_small": "normal Peierls bubble+contact closes; superconducting pairing sector remains suspect",
            "if_total_residual_large_and_alpha_not_one": "normal contact formula or normal response convention is also suspect",
            "not_a_fit_fix": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_and_write_normal_contact_ward_control(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_normal_contact_ward_control(**kwargs)
    write_json(Path(output_dir) / "normal_contact_ward_control.json", payload)
    return payload
