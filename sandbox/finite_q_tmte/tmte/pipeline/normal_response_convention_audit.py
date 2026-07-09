"""Diagnostic-only normal response convention audit.

This audit keeps the normal Peierls vertex algebra fixed and scans response-level
assembly conventions: band-vertex orientation, Kubo factor, source/observable
current signs, contact sign/evaluation point, and Ward contraction vector. It is
diagnostic only and must not be used to accept a production convention by
residual minimization alone.
"""

from __future__ import annotations

from itertools import product
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
from .shifted_average import shift_pairs_from_fractions

SCHEMA_VERSION = "finite_q_tmte_normal_response_convention_audit_v1"

BAND_ORIENTATIONS = ("forward_minus_plus", "direct_minus_plus")
KUBO_CONVENTIONS = ("minus_plus", "denominator_flipped", "fully_reversed")
CURRENT_SIGN_CONVENTIONS = (
    "observable_minus_source_plus",
    "both_plus",
    "observable_plus_source_minus",
    "both_minus",
)
CONTACT_SIGNS = ("minus_expectation", "plus_expectation")
CONTACT_EVALUATIONS = ("mid", "plus", "minus", "sym_pm")
WARD_CONVENTIONS = (
    "standard",
    "right_spatial_plus",
    "left_spatial_minus_right_plus",
    "temporal_minus",
)

DEFAULT_CANDIDATE_NAMES = (
    "baseline_current",
    "contact_plus_expectation",
    "contact_eval_plus",
    "contact_eval_minus",
    "contact_eval_sym_pm",
    "kubo_denominator_flipped",
    "kubo_fully_reversed",
    "band_direct_minus_plus",
    "current_sign_both_plus",
    "current_sign_observable_plus_source_minus",
    "ward_right_spatial_plus",
    "ward_temporal_minus",
    "kubo_fully_reversed_contact_plus",
)


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-30) -> float:
    return float(numerator) / max(float(denominator), float(eps))


def _vec(values: np.ndarray, labels: Sequence[str] = NORMAL_ORDER) -> list[dict[str, Any]]:
    return complex_vector_payload(np.asarray(values, dtype=complex).reshape(-1), labels)


def _matrix_payload(matrix: np.ndarray, rows: Sequence[str] = NORMAL_ORDER, cols: Sequence[str] = NORMAL_ORDER) -> list[dict[str, Any]]:
    a = np.asarray(matrix, dtype=complex)
    return [{"row": row, "values": complex_vector_payload(a[i, :], cols)} for i, row in enumerate(rows)]


def default_candidate_specs() -> list[dict[str, str]]:
    base = {
        "band_orientation": "forward_minus_plus",
        "kubo_convention": "minus_plus",
        "current_sign_convention": "observable_minus_source_plus",
        "contact_sign": "minus_expectation",
        "contact_evaluation": "mid",
        "ward_convention": "standard",
    }

    def row(name: str, **updates: str) -> dict[str, str]:
        return {"name": name, **base, **updates}

    return [
        row("baseline_current"),
        row("contact_plus_expectation", contact_sign="plus_expectation"),
        row("contact_eval_plus", contact_evaluation="plus"),
        row("contact_eval_minus", contact_evaluation="minus"),
        row("contact_eval_sym_pm", contact_evaluation="sym_pm"),
        row("kubo_denominator_flipped", kubo_convention="denominator_flipped"),
        row("kubo_fully_reversed", kubo_convention="fully_reversed"),
        row("band_direct_minus_plus", band_orientation="direct_minus_plus"),
        row("current_sign_both_plus", current_sign_convention="both_plus"),
        row("current_sign_observable_plus_source_minus", current_sign_convention="observable_plus_source_minus"),
        row("ward_right_spatial_plus", ward_convention="right_spatial_plus"),
        row("ward_temporal_minus", ward_convention="temporal_minus"),
        row("kubo_fully_reversed_contact_plus", kubo_convention="fully_reversed", contact_sign="plus_expectation"),
    ]


def full_grid_candidate_specs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, (band, kubo, signs, csign, ceval, ward) in enumerate(
        product(BAND_ORIENTATIONS, KUBO_CONVENTIONS, CURRENT_SIGN_CONVENTIONS, CONTACT_SIGNS, CONTACT_EVALUATIONS, WARD_CONVENTIONS)
    ):
        rows.append(
            {
                "name": f"grid_{idx:04d}",
                "band_orientation": band,
                "kubo_convention": kubo,
                "current_sign_convention": signs,
                "contact_sign": csign,
                "contact_evaluation": ceval,
                "ward_convention": ward,
            }
        )
    return rows


def candidate_specs_from_names(names: Sequence[str] | None = None, *, full_grid: bool = False) -> list[dict[str, str]]:
    if full_grid:
        return full_grid_candidate_specs()
    candidates = default_candidate_specs()
    if names is None:
        return candidates
    by_name = {row["name"]: row for row in candidates}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown normal response convention candidate(s): {unknown}")
    return [by_name[name] for name in names]


def _current_signs(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if name == "observable_minus_source_plus":
        return (1.0, -1.0, -1.0), (1.0, 1.0, 1.0)
    if name == "both_plus":
        return (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
    if name == "observable_plus_source_minus":
        return (1.0, 1.0, 1.0), (1.0, -1.0, -1.0)
    if name == "both_minus":
        return (1.0, -1.0, -1.0), (1.0, -1.0, -1.0)
    raise ValueError(f"unknown current sign convention {name!r}")


def band_vertex(states_minus: np.ndarray, vertex: np.ndarray, states_plus: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "forward_minus_plus":
        return (states_plus.conjugate().T @ np.asarray(vertex, dtype=complex) @ states_minus).T
    if orientation == "direct_minus_plus":
        return states_minus.conjugate().T @ np.asarray(vertex, dtype=complex) @ states_plus
    raise ValueError(f"unknown band orientation {orientation!r}")


def kubo_factor(
    *,
    energy_minus: float,
    energy_plus: float,
    occupation_minus: float,
    occupation_plus: float,
    xi_eV: float,
    convention: str,
) -> complex:
    if convention == "minus_plus":
        numerator = float(occupation_minus) - float(occupation_plus)
        denominator = 1j * float(xi_eV) + float(energy_minus) - float(energy_plus)
    elif convention == "denominator_flipped":
        numerator = float(occupation_minus) - float(occupation_plus)
        denominator = 1j * float(xi_eV) + float(energy_plus) - float(energy_minus)
    elif convention == "fully_reversed":
        numerator = float(occupation_plus) - float(occupation_minus)
        denominator = 1j * float(xi_eV) + float(energy_plus) - float(energy_minus)
    else:
        raise ValueError(f"unknown Kubo convention {convention!r}")
    return numerator / denominator


def normal_thermal_expectation(hamiltonian: np.ndarray, vertex: np.ndarray, config: KuboConfig) -> complex:
    energies, states = np.linalg.eigh(np.asarray(hamiltonian, dtype=complex))
    occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
    vertex_band_matrix = states.conjugate().T @ np.asarray(vertex, dtype=complex) @ states
    return complex(np.sum(occupations * np.diag(vertex_band_matrix)))


def contact_expectation_for_point(
    *,
    spec: object,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction_i: str,
    direction_j: str,
    evaluation: str,
    config: KuboConfig,
) -> complex:
    vertex = spec.peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
    if evaluation == "mid":
        return normal_thermal_expectation(spec.normal_hamiltonian(kx, ky), vertex, config)
    if evaluation == "plus":
        return normal_thermal_expectation(spec.normal_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy), vertex, config)
    if evaluation == "minus":
        return normal_thermal_expectation(spec.normal_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy), vertex, config)
    if evaluation == "sym_pm":
        plus = normal_thermal_expectation(spec.normal_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy), vertex, config)
        minus = normal_thermal_expectation(spec.normal_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy), vertex, config)
        return 0.5 * (plus + minus)
    raise ValueError(f"unknown contact evaluation {evaluation!r}")


def ward_vectors(xi_eV: float, q_model: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray]:
    qx, qy = float(q_model[0]), float(q_model[1])
    if convention == "standard":
        return np.asarray([1j * xi_eV, qx, qy], dtype=complex), np.asarray([1j * xi_eV, -qx, -qy], dtype=complex)
    if convention == "right_spatial_plus":
        return np.asarray([1j * xi_eV, qx, qy], dtype=complex), np.asarray([1j * xi_eV, qx, qy], dtype=complex)
    if convention == "left_spatial_minus_right_plus":
        return np.asarray([1j * xi_eV, -qx, -qy], dtype=complex), np.asarray([1j * xi_eV, qx, qy], dtype=complex)
    if convention == "temporal_minus":
        return np.asarray([-1j * xi_eV, qx, qy], dtype=complex), np.asarray([-1j * xi_eV, -qx, -qy], dtype=complex)
    raise ValueError(f"unknown Ward convention {convention!r}")


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


def accumulate_candidate_blocks(
    *,
    spec: object,
    q_model: np.ndarray,
    xi_eV: float,
    k_points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    candidate: dict[str, str],
) -> dict[str, Any]:
    qx, qy = float(q_model[0]), float(q_model[1])
    if float(np.linalg.norm(q_model)) <= 0.0:
        raise ValueError("normal response convention audit requires q > 0")
    first_h = np.asarray(spec.normal_hamiltonian(float(k_points[0, 0]), float(k_points[0, 1])), dtype=complex)
    dim = first_h.shape[0]
    rho = np.eye(dim, dtype=complex)
    bubble = np.zeros((3, 3), dtype=complex)
    contact = np.zeros((3, 3), dtype=complex)
    left_signs, right_signs = _current_signs(candidate["current_sign_convention"])
    contact_prefactor = -1.0 if candidate["contact_sign"] == "minus_expectation" else 1.0
    vertex_abs_errors: list[float] = []
    vertex_rel_errors: list[float] = []

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
        primitive = (rho, vx, vy)
        left_vertices = tuple(sign * vertex for sign, vertex in zip(left_signs, primitive, strict=True))
        right_vertices = tuple(sign * vertex for sign, vertex in zip(right_signs, primitive, strict=True))
        left_band = tuple(band_vertex(u_minus, vertex, u_plus, candidate["band_orientation"]) for vertex in left_vertices)
        right_band = tuple(band_vertex(u_minus, vertex, u_plus, candidate["band_orientation"]) for vertex in right_vertices)
        for m, energy_minus in enumerate(e_minus):
            for n, energy_plus in enumerate(e_plus):
                raw_factor = kubo_factor(
                    energy_minus=float(energy_minus),
                    energy_plus=float(energy_plus),
                    occupation_minus=float(occ_minus[m]),
                    occupation_plus=float(occ_plus[n]),
                    xi_eV=xi_eV,
                    convention=candidate["kubo_convention"],
                )
                if raw_factor == 0.0:
                    continue
                factor = float(weight) * raw_factor
                for mu, left in enumerate(left_band):
                    for nu, right in enumerate(right_band):
                        bubble[mu, nu] += factor * left[m, n] * np.conjugate(right[m, n])
        for i, di in enumerate(("x", "y")):
            for j, dj in enumerate(("x", "y")):
                value = contact_expectation_for_point(
                    spec=spec,
                    kx=kx,
                    ky=ky,
                    qx=qx,
                    qy=qy,
                    direction_i=di,
                    direction_j=dj,
                    evaluation=candidate["contact_evaluation"],
                    config=config,
                )
                contact[1 + i, 1 + j] += float(weight) * contact_prefactor * value
        vertex_check = normal_peierls_vertex_ward_residual(spec, kx, ky, qx, qy)
        vertex_abs_errors.append(float(vertex_check["abs_error"]))
        vertex_rel_errors.append(float(vertex_check["rel_error"]))
    return {
        "bubble": bubble,
        "contact": contact,
        "total": bubble + contact,
        "vertex_identity": {
            "max_abs_error": max(vertex_abs_errors) if vertex_abs_errors else 0.0,
            "max_rel_error": max(vertex_rel_errors) if vertex_rel_errors else 0.0,
            "mean_rel_error": float(np.mean(vertex_rel_errors)) if vertex_rel_errors else 0.0,
            "valid_for_casimir_input": False,
        },
    }


def evaluate_candidate(candidate: dict[str, str], bubble: np.ndarray, contact: np.ndarray, xi_eV: float, q_model: np.ndarray) -> dict[str, Any]:
    total = bubble + contact
    left_vec, right_vec = ward_vectors(xi_eV, q_model, candidate["ward_convention"])
    reference_norm = max(_norm(bubble), _norm(contact), _norm(total), 1e-30)
    def residual_payload(name: str, matrix: np.ndarray) -> dict[str, Any]:
        left = left_vec @ matrix
        right = matrix @ right_vec
        max_norm = max(_norm(left), _norm(right))
        return {
            "name": name,
            "matrix_norm": _norm(matrix),
            "left_residual": {"values": _vec(left), "norm": _norm(left)},
            "right_residual": {"values": _vec(right), "norm": _norm(right)},
            "max_residual_norm": max_norm,
            "max_residual_over_reference": _safe_ratio(max_norm, reference_norm),
            "valid_for_casimir_input": False,
        }
    left_current = left_vec @ contact
    left_required = -(left_vec @ bubble)
    right_current = contact @ right_vec
    right_required = -(bubble @ right_vec)
    left_projection = scalar_projection(left_required, left_current)
    right_projection = scalar_projection(right_required, right_current)
    alpha_left = complex(left_projection["alpha_required_over_current"])
    alpha_right = complex(right_projection["alpha_required_over_current"])
    total_left = left_vec @ total
    total_right = total @ right_vec
    total_max = max(_norm(total_left), _norm(total_right))
    return {
        "candidate": candidate,
        "responses": [residual_payload("bubble", bubble), residual_payload("contact", contact), residual_payload("total", total)],
        "summary": {
            "total_left_norm": _norm(total_left),
            "total_right_norm": _norm(total_right),
            "total_max_residual_norm": total_max,
            "total_max_residual_over_reference": _safe_ratio(total_max, reference_norm),
            "alpha_left": alpha_left,
            "alpha_right": alpha_right,
            "alpha_real_mean": float(np.real(0.5 * (alpha_left + alpha_right))),
            "alpha_imag_mean": float(np.imag(0.5 * (alpha_left + alpha_right))),
            "left_right_alpha_abs_diff": float(abs(alpha_left - alpha_right)),
            "left_projection_residual_over_required": float(left_projection["residual_over_required_norm"]),
            "right_projection_residual_over_required": float(right_projection["residual_over_required_norm"]),
            "valid_for_casimir_input": False,
        },
        "contact_formula": {
            "left_current_contact": {"values": _vec(left_current), "norm": _norm(left_current)},
            "left_required_contact": {"values": _vec(left_required), "norm": _norm(left_required)},
            "right_current_contact": {"values": _vec(right_current), "norm": _norm(right_current)},
            "right_required_contact": {"values": _vec(right_required), "norm": _norm(right_required)},
            "left_required_over_current_scalar_projection": left_projection,
            "right_required_over_current_scalar_projection": right_projection,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_normal_response_convention_audit(
    *,
    model_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_value: float,
    nk: int,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    candidate_names: Sequence[str] | None = None,
    full_grid: bool = False,
) -> dict[str, Any]:
    xi_eV = matsubara_xi_eV(matsubara_index, temperature_K)
    if nk <= 0:
        raise ValueError("nk must be positive")
    candidates = candidate_specs_from_names(candidate_names, full_grid=full_grid)
    model = get_finite_q_validation_model(model_name)
    spec = model.spec
    for required in ("normal_hamiltonian", "peierls_hamiltonian_vector_vertex", "peierls_hamiltonian_contact_vertex"):
        if not hasattr(spec, required):
            raise ValueError(f"model spec does not support required normal Peierls method {required!r}")
    config = KuboConfig.from_kelvin(omega_eV=xi_eV, temperature_K=float(temperature_K), eta_eV=float(eta_eV), output_si=False)
    q_model = np.asarray([float(q_value), 0.0], dtype=float)
    shifts = shift_pairs_from_fractions(shift_fractions)
    candidate_payloads: list[dict[str, Any]] = []
    for candidate in candidates:
        bubbles: list[np.ndarray] = []
        contacts: list[np.ndarray] = []
        vertex_reports: list[dict[str, Any]] = []
        for sx, sy in shifts:
            points = shifted_uniform_bz_mesh(nk, sx, sy)
            weights = weights_for_points(points)
            blocks = accumulate_candidate_blocks(
                spec=spec,
                q_model=q_model,
                xi_eV=xi_eV,
                k_points=points,
                weights=weights,
                config=config,
                candidate=candidate,
            )
            bubbles.append(np.asarray(blocks["bubble"], dtype=complex))
            contacts.append(np.asarray(blocks["contact"], dtype=complex))
            vertex_reports.append(blocks["vertex_identity"])
        bubble = sum(bubbles) / float(len(bubbles))
        contact = sum(contacts) / float(len(contacts))
        candidate_payload = evaluate_candidate(candidate, bubble, contact, xi_eV, q_model)
        candidate_payload["vertex_identity"] = {
            "max_abs_error_over_shifted_meshes": max(float(row["max_abs_error"]) for row in vertex_reports),
            "max_rel_error_over_shifted_meshes": max(float(row["max_rel_error"]) for row in vertex_reports),
            "mean_rel_error_over_shifted_meshes": float(np.mean([float(row["mean_rel_error"]) for row in vertex_reports])),
            "valid_for_casimir_input": False,
        }
        candidate_payload["block_norms"] = {
            "bubble_norm": _norm(bubble),
            "contact_norm": _norm(contact),
            "total_norm": _norm(bubble + contact),
            "valid_for_casimir_input": False,
        }
        candidate_payloads.append(candidate_payload)
    sorted_by_residual = sorted(candidate_payloads, key=lambda row: float(row["summary"]["total_max_residual_over_reference"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "requires_analytic_derivation_before_convention_change": True,
            "valid_for_casimir_input": False,
            "reason": "normal_response_convention_audit_not_production_convention",
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
            "candidate_names": [row["name"] for row in candidates],
            "full_grid": bool(full_grid),
            "normal_state_only": True,
            "valid_for_casimir_input": False,
        },
        "available_conventions": {
            "band_orientations": list(BAND_ORIENTATIONS),
            "kubo_conventions": list(KUBO_CONVENTIONS),
            "current_sign_conventions": list(CURRENT_SIGN_CONVENTIONS),
            "contact_signs": list(CONTACT_SIGNS),
            "contact_evaluations": list(CONTACT_EVALUATIONS),
            "ward_conventions": list(WARD_CONVENTIONS),
            "valid_for_casimir_input": False,
        },
        "candidates": candidate_payloads,
        "ranked_candidates": [
            {
                "rank": idx + 1,
                "name": row["candidate"]["name"],
                "total_max_residual_over_reference": row["summary"]["total_max_residual_over_reference"],
                "total_max_residual_norm": row["summary"]["total_max_residual_norm"],
                "alpha_real_mean": row["summary"]["alpha_real_mean"],
                "left_right_alpha_abs_diff": row["summary"]["left_right_alpha_abs_diff"],
                "candidate": row["candidate"],
                "valid_for_casimir_input": False,
            }
            for idx, row in enumerate(sorted_by_residual)
        ],
        "interpretation_guardrails": {
            "not_a_residual_minimization_fix": True,
            "best_candidate_requires_analytic_derivation_before_change": True,
            "if_only_contact_sign_fixes": "inspect sign convention of diamagnetic term",
            "if_kubo_or_band_orientation_fixes": "inspect response assembly convention before any BdG pairing work",
            "if_no_candidate_closes": "normal response may need a missing finite-q equal-time/contact term beyond the scanned controls",
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_and_write_normal_response_convention_audit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_normal_response_convention_audit(**kwargs)
    write_json(Path(output_dir) / "normal_response_convention_audit.json", payload)
    return payload
