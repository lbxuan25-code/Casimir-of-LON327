#!/usr/bin/env python3
"""Normal-state finite-q Ward residual audit for diagnostic output only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.conductivity import (  # noqa: E402
    KuboConfig,
    fermi_function,
    k_weights,
    negative_fermi_derivative,
    uniform_bz_mesh,
)
from lno327.model import normal_state_hamiltonian  # noqa: E402
from lno327.tb_fourier import (  # noqa: E402
    normal_state_hopping_terms,
    peierls_hamiltonian_contact_vertex,
    peierls_hamiltonian_vector_vertex,
    peierls_vertex_ward_residual,
)
from lno327.ward_response import (  # noqa: E402
    normal_physical_density_current_response_components_imag_axis,
    physical_ward_residuals,
)

WARD_COMPONENT_LABELS = ("density", "current_x", "current_y")
RESPONSE_NAMES = ("bubble", "direct", "total")
DIRECTION_VECTORS = {
    "x": (1.0, 0.0),
    "y": (0.0, 1.0),
    "diagonal": (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
}


def _complex_vector_components(vector: np.ndarray) -> list[dict[str, float | str]]:
    array = np.asarray(vector, dtype=complex)
    if array.shape != (3,):
        raise ValueError("Ward residual vector must have shape (3,)")
    return [
        {
            "component": label,
            "real": float(np.real(value)),
            "imag": float(np.imag(value)),
        }
        for label, value in zip(WARD_COMPONENT_LABELS, array, strict=True)
    ]


def _complex_value(value: complex) -> dict[str, float]:
    return {
        "real": float(np.real(value)),
        "imag": float(np.imag(value)),
        "abs": float(abs(value)),
    }


def _component_vector(values: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "component": label,
            **_complex_value(value),
        }
        for label, value in zip(WARD_COMPONENT_LABELS, np.asarray(values, dtype=complex), strict=True)
    ]


def _vector_from_component_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    values = [complex(row["real"], row["imag"]) for row in rows]
    return np.asarray(values, dtype=complex)


def _complex_matrix_entries(matrix: np.ndarray) -> list[list[dict[str, float]]]:
    array = np.asarray(matrix, dtype=complex)
    return [[_complex_value(value) for value in row] for row in array]


def _ward_contraction_decomposition(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    response = np.asarray(matrix, dtype=complex)
    qx, qy = float(q[0]), float(q[1])
    left_rows = []
    right_rows = []
    for idx, label in enumerate(WARD_COMPONENT_LABELS):
        left_terms = {
            "iomega_Pi_0nu": 1j * omega_eV * response[0, idx],
            "qx_Pi_xnu": qx * response[1, idx],
            "qy_Pi_ynu": qy * response[2, idx],
        }
        right_terms = {
            "iomega_Pi_mu0": 1j * omega_eV * response[idx, 0],
            "minus_qx_Pi_mux": -qx * response[idx, 1],
            "minus_qy_Pi_muy": -qy * response[idx, 2],
        }
        left_rows.append(
            {
                "component": label,
                "terms": {name: _complex_value(value) for name, value in left_terms.items()},
                "residual": _complex_value(sum(left_terms.values())),
            }
        )
        right_rows.append(
            {
                "component": label,
                "terms": {name: _complex_value(value) for name, value in right_terms.items()},
                "residual": _complex_value(sum(right_terms.values())),
            }
        )
    return {
        "left_contraction": left_rows,
        "right_contraction": right_rows,
        "left_formula": "R_left[nu] = iomega*Pi[0,nu] + qx*Pi[x,nu] + qy*Pi[y,nu]",
        "right_formula": "R_right[mu] = iomega*Pi[mu,0] - qx*Pi[mu,x] - qy*Pi[mu,y]",
    }


def _response_residual_row(response_name: str, matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    left, right = physical_ward_residuals(matrix, omega_eV, q)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    return {
        "response_name": response_name,
        "residual_kind": "response_level",
        "residual_component_labels": list(WARD_COMPONENT_LABELS),
        "left_ward_residual_vector": _complex_vector_components(left),
        "right_ward_residual_vector": _complex_vector_components(right),
        "ward_contraction_decomposition": _ward_contraction_decomposition(matrix, omega_eV, q),
        "left_ward_residual_norm": left_norm,
        "right_ward_residual_norm": right_norm,
        "max_ward_residual_norm": float(max(left_norm, right_norm)),
        "valid_for_casimir_input": False,
    }


def _longitudinal_current_component(vector: np.ndarray, q: np.ndarray) -> complex:
    q_norm = float(np.linalg.norm(q))
    if q_norm <= 0.0:
        raise ValueError("q must be nonzero for longitudinal current projection")
    q_hat = np.asarray(q, dtype=float) / q_norm
    residual = np.asarray(vector, dtype=complex)
    return complex(q_hat[0] * residual[1] + q_hat[1] * residual[2])


def _longitudinal_current_scaling(
    response_rows: list[dict[str, Any]],
    q: np.ndarray,
) -> dict[str, Any]:
    by_name = {str(row["response_name"]): row for row in response_rows}
    q_norm = float(np.linalg.norm(q))
    output: dict[str, Any] = {
        "component": "longitudinal_current",
        "projection": {
            "qx_hat": float(q[0] / q_norm),
            "qy_hat": float(q[1] / q_norm),
            "definition": "qhat_x * current_x_residual + qhat_y * current_y_residual",
        },
    }
    for side, vector_key in (
        ("left", "left_ward_residual_vector"),
        ("right", "right_ward_residual_vector"),
    ):
        values = {}
        for response_name in RESPONSE_NAMES:
            vector = np.array(
                [
                    complex(component["real"], component["imag"])
                    for component in by_name[response_name][vector_key]
                ],
                dtype=complex,
            )
            values[response_name] = _longitudinal_current_component(vector, q)
        total = values["total"]
        output[f"{side}_contraction"] = {
            "bubble_residual": _complex_value(values["bubble"]),
            "direct_residual": _complex_value(values["direct"]),
            "total_residual": _complex_value(total),
            "total_residual_over_q": _complex_value(total / q_norm),
            "total_residual_over_q2": _complex_value(total / (q_norm * q_norm)),
        }
    return output


def _ward_residual_payload(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    left, right = physical_ward_residuals(matrix, omega_eV, q)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    q_norm = float(np.linalg.norm(q))
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "left_ward_residual_vector": _component_vector(left),
        "right_ward_residual_vector": _component_vector(right),
        "left_ward_residual_norm": left_norm,
        "right_ward_residual_norm": right_norm,
        "max_ward_residual_norm": float(max(left_norm, right_norm)),
        "left_ward_residual_over_q": _component_vector(left / q_norm),
        "left_ward_residual_over_q2": _component_vector(left / (q_norm * q_norm)),
        "left_ward_residual_over_q_norm": float(left_norm / q_norm),
        "left_ward_residual_over_q2_norm": float(left_norm / (q_norm * q_norm)),
    }


def _operator_level_rows(points: np.ndarray, q: np.ndarray) -> list[dict[str, Any]]:
    qx, qy = float(q[0]), float(q[1])
    rows: list[dict[str, Any]] = []
    for kx_value, ky_value in points:
        kx = float(kx_value)
        ky = float(ky_value)
        abs_error, rel_error, lhs_norm, rhs_norm = peierls_vertex_ward_residual(kx, ky, qx, qy)
        rows.append(
            {
                "residual_kind": "operator_level",
                "identity": "q_x V_x(k,q) + q_y V_y(k,q) = H(k+q/2)-H(k-q/2)",
                "k_model": [kx, ky],
                "absolute_error_norm": float(abs_error),
                "relative_error_norm": float(rel_error),
                "lhs_norm": float(lhs_norm),
                "rhs_norm": float(rhs_norm),
            }
        )
    return rows


def _operator_level_second_order_contact_ward(points: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    qx, qy = float(q[0]), float(q[1])
    peierls_terms = normal_state_hopping_terms()
    rows: list[dict[str, Any]] = []
    max_absolute_error = -1.0
    max_relative_error = -1.0
    max_error_k_model: list[float] | None = None
    max_error_component = ""

    for kx_value, ky_value in points:
        kx = float(kx_value)
        ky = float(ky_value)
        component_rows: list[dict[str, Any]] = []
        for component_label, source_direction in (("current_x", "x"), ("current_y", "y")):
            implemented_contact_contraction = np.zeros((4, 4), dtype=complex)
            hessian_q0_reference = np.zeros((4, 4), dtype=complex)
            for q_component, observable_direction in ((qx, "x"), (qy, "y")):
                implemented_contact_contraction += q_component * peierls_hamiltonian_contact_vertex(
                    kx,
                    ky,
                    qx,
                    qy,
                    observable_direction,
                    source_direction,
                    hopping_terms=peierls_terms,
                )
                hessian_q0_reference += q_component * peierls_hamiltonian_contact_vertex(
                    kx,
                    ky,
                    0.0,
                    0.0,
                    observable_direction,
                    source_direction,
                    hopping_terms=peierls_terms,
                )

            finite_difference_current_vertex_reference = peierls_hamiltonian_vector_vertex(
                kx + 0.5 * qx,
                ky + 0.5 * qy,
                qx,
                qy,
                source_direction,
                hopping_terms=peierls_terms,
            ) - peierls_hamiltonian_vector_vertex(
                kx - 0.5 * qx,
                ky - 0.5 * qy,
                qx,
                qy,
                source_direction,
                hopping_terms=peierls_terms,
            )
            residual_vs_finite_difference = (
                implemented_contact_contraction - finite_difference_current_vertex_reference
            )
            residual_vs_hessian_q0 = implemented_contact_contraction - hessian_q0_reference
            absolute_error = float(np.linalg.norm(residual_vs_finite_difference))
            reference_norm = float(np.linalg.norm(finite_difference_current_vertex_reference))
            relative_error = absolute_error / max(reference_norm, 1e-300)
            if absolute_error > max_absolute_error:
                max_absolute_error = absolute_error
                max_error_k_model = [kx, ky]
                max_error_component = component_label
            max_relative_error = max(max_relative_error, relative_error)
            component_rows.append(
                {
                    "component": component_label,
                    "implemented_contact_contraction": _complex_matrix_entries(
                        implemented_contact_contraction
                    ),
                    "finite_difference_current_vertex_reference": _complex_matrix_entries(
                        finite_difference_current_vertex_reference
                    ),
                    "hessian_q0_reference": _complex_matrix_entries(hessian_q0_reference),
                    "residual_vs_finite_difference_reference": {
                        "matrix": _complex_matrix_entries(residual_vs_finite_difference),
                        "norm": absolute_error,
                        "reference_norm": reference_norm,
                        "relative_error_norm": float(relative_error),
                    },
                    "residual_vs_hessian_q0_reference": {
                        "matrix": _complex_matrix_entries(residual_vs_hessian_q0),
                        "norm": float(np.linalg.norm(residual_vs_hessian_q0)),
                        "reference_norm": float(np.linalg.norm(hessian_q0_reference)),
                    },
                }
            )
        rows.append(
            {
                "k_model": [kx, ky],
                "components": component_rows,
            }
        )

    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "residual_kind": "operator_level",
        "identity": (
            "Hamiltonian Peierls convention check: "
            "q_i M_ij(k,q) = V_j(k+q/2,q) - V_j(k-q/2,q). "
            "The physical current is -V_j, so this block tracks the implemented "
            "Hamiltonian contact/source-vertex convention without changing response formulas."
        ),
        "implemented_contact_contraction": "qx * M[x,j](k,q) + qy * M[y,j](k,q)",
        "finite_difference_current_vertex_reference": "V_j(k+q/2,q) - V_j(k-q/2,q)",
        "hessian_q0_reference": "qx * M[x,j](k,0) + qy * M[y,j](k,0)",
        "residual_vs_finite_difference_reference": (
            "implemented_contact_contraction - finite_difference_current_vertex_reference"
        ),
        "residual_vs_hessian_q0_reference": "implemented_contact_contraction - hessian_q0_reference",
        "max_absolute_error_norm": float(max_absolute_error),
        "max_relative_error_norm": float(max_relative_error),
        "max_error_k_model": max_error_k_model,
        "max_error_component": max_error_component,
        "per_k_residuals": rows,
    }


def _shifted_pair_response_components(
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
) -> dict[str, np.ndarray]:
    qx, qy = float(q[0]), float(q[1])
    peierls_terms = normal_state_hopping_terms()
    rho = np.eye(4, dtype=complex)
    bubble = np.zeros((3, 3), dtype=complex)
    direct = np.zeros((3, 3), dtype=complex)

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = normal_state_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = normal_state_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
        energies_minus, states_minus = np.linalg.eigh(h_minus)
        energies_plus, states_plus = np.linalg.eigh(h_plus)
        occupations_minus = fermi_function(
            energies_minus,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        occupations_plus = fermi_function(
            energies_plus,
            config.fermi_level_eV,
            config.temperature_eV,
        )

        vector_x = peierls_hamiltonian_vector_vertex(
            kx,
            ky,
            qx,
            qy,
            "x",
            hopping_terms=peierls_terms,
        )
        vector_y = peierls_hamiltonian_vector_vertex(
            kx,
            ky,
            qx,
            qy,
            "y",
            hopping_terms=peierls_terms,
        )
        observable_vertices = (rho, -vector_x, -vector_y)
        source_vertices = (rho, vector_x, vector_y)
        observable_matrices = tuple(
            states_minus.conjugate().T @ vertex @ states_plus for vertex in observable_vertices
        )
        source_matrices = tuple(states_minus.conjugate().T @ vertex @ states_plus for vertex in source_vertices)
        for m, energy_minus in enumerate(energies_minus):
            for n, energy_plus in enumerate(energies_plus):
                occupation_diff = float(occupations_minus[m] - occupations_plus[n])
                if occupation_diff == 0.0:
                    continue
                denominator = 1j * config.omega_eV + float(energy_minus - energy_plus)
                factor = occupation_diff / denominator
                for mu, observable_matrix in enumerate(observable_matrices):
                    for nu, source_matrix in enumerate(source_matrices):
                        bubble[mu, nu] += (
                            weight
                            * factor
                            * observable_matrix[m, n]
                            * np.conjugate(source_matrix[m, n])
                        )

        h_midpoint = normal_state_hamiltonian(kx, ky)
        energies_midpoint, states_midpoint = np.linalg.eigh(h_midpoint)
        occupations_midpoint = fermi_function(
            energies_midpoint,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        for i, direction_i in enumerate(("x", "y")):
            for j, direction_j in enumerate(("x", "y")):
                contact_matrix = peierls_hamiltonian_contact_vertex(
                    kx,
                    ky,
                    qx,
                    qy,
                    direction_i,
                    direction_j,
                    hopping_terms=peierls_terms,
                )
                band_contact = states_midpoint.conjugate().T @ contact_matrix @ states_midpoint
                physical_direct_contact = -np.sum(occupations_midpoint * np.diag(band_contact))
                direct[1 + i, 1 + j] += weight * physical_direct_contact
    return {"bubble": bubble, "direct": direct, "total": bubble + direct}


def _ward_compatible_shifted_pair_quadrature_audit(
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    raw_components: dict[str, np.ndarray],
) -> dict[str, Any]:
    q_norm = float(np.linalg.norm(q))
    shifted_pair_components = _shifted_pair_response_components(points, weights, config, q)
    shifted_pair_total_left, _ = physical_ward_residuals(shifted_pair_components["total"], config.omega_eV, q)
    raw_total_left, _ = physical_ward_residuals(raw_components["total"], config.omega_eV, q)
    residual_difference = shifted_pair_total_left - raw_total_left
    current_current_block_difference = (
        raw_components["total"][1:3, 1:3] - shifted_pair_components["total"][1:3, 1:3]
    )
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "shifted_pair_response_is_raw_equivalent_diagnostic": True,
        "quadrature": "midpoint shifted pair raw-equivalent diagnostic",
        "mesh_definition": "k_mid = k, k_plus = k_mid + q/2, k_minus = k_mid - q/2",
        "weights": "bubble, equal-time diagnostics, and contact use the same midpoint mesh weights",
        "no_longitudinal_projection_completion": True,
        "shifted_pair_bubble_ward_residual": _ward_residual_payload(
            shifted_pair_components["bubble"],
            config.omega_eV,
            q,
        ),
        "shifted_pair_direct_ward_residual": _ward_residual_payload(
            shifted_pair_components["direct"],
            config.omega_eV,
            q,
        ),
        "shifted_pair_total_ward_residual": _ward_residual_payload(
            shifted_pair_components["total"],
            config.omega_eV,
            q,
        ),
        "shifted_pair_total_residual_over_q": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "components": _component_vector(shifted_pair_total_left / q_norm),
            "norm": float(np.linalg.norm(shifted_pair_total_left) / q_norm),
        },
        "shifted_pair_total_residual_over_q2": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "components": _component_vector(shifted_pair_total_left / (q_norm * q_norm)),
            "norm": float(np.linalg.norm(shifted_pair_total_left) / (q_norm * q_norm)),
        },
        "raw_total_ward_residual": _ward_residual_payload(raw_components["total"], config.omega_eV, q),
        "raw_total_residual_over_q": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "components": _component_vector(raw_total_left / q_norm),
            "norm": float(np.linalg.norm(raw_total_left) / q_norm),
        },
        "shifted_pair_minus_raw_response_norm": float(
            np.linalg.norm(shifted_pair_components["total"] - raw_components["total"])
        ),
        "shifted_pair_minus_raw_longitudinal_residual": _complex_value(
            _longitudinal_current_component(residual_difference, q)
        ),
        "raw_vs_shifted_pair_current_current_block_difference": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "raw total current-current block minus shifted-pair total current-current block",
            "matrix": _complex_matrix_entries(current_current_block_difference),
            "norm": float(np.linalg.norm(current_current_block_difference)),
        },
    }


def _finite_mesh_translation_error_audit(
    equal_time_audit: dict[str, Any],
    components: dict[str, np.ndarray],
    config: KuboConfig,
    q: np.ndarray,
) -> dict[str, Any]:
    shifted_grid = equal_time_audit["shifted_grid_equal_time_sum_rule"]
    actual_equal_time = _vector_from_component_rows(
        shifted_grid["actual_bubble_equal_time_term"]["components"]
    )
    shifted_equal_time_reference = _vector_from_component_rows(
        shifted_grid["shifted_equal_time_reference"]["components"]
    )
    contact_contraction = _vector_from_component_rows(shifted_grid["contact_contraction"]["components"])
    shifted_equal_time_plus_contact = shifted_equal_time_reference + contact_contraction
    translation_error = actual_equal_time - shifted_equal_time_reference
    total_left_residual, _ = physical_ward_residuals(components["total"], config.omega_eV, q)
    translation_error_minus_total_residual = translation_error - total_left_residual
    q_norm = float(np.linalg.norm(q))
    total_residual_norm = float(np.linalg.norm(total_left_residual))
    translation_error_norm = float(np.linalg.norm(translation_error))
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "scope": "finite_k_mesh_translation_invariance_failure_at_equal_time_contact_level",
        "density_component_note": (
            "density has no current-vertex finite-difference reference in this diagnostic; "
            "the shifted reference density component is stored as zero"
        ),
        "actual_equal_time": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "source": "normal_current_equal_time_sum_rule_audit.bubble_equal_time_term",
            "components": _component_vector(actual_equal_time),
        },
        "shifted_equal_time_reference": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "Tr[f(H(k)) * (V_j(k+q/2,q) - V_j(k-q/2,q))]",
            "components": _component_vector(shifted_equal_time_reference),
        },
        "contact_contraction": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "C_j(q) = q_i D_ij(q)",
            "components": _component_vector(contact_contraction),
        },
        "shifted_equal_time_plus_contact": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "shifted_equal_time_reference + contact_contraction",
            "components": _component_vector(shifted_equal_time_plus_contact),
        },
        "translation_error": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "actual_equal_time - shifted_equal_time_reference",
            "components": _component_vector(translation_error),
        },
        "translation_error_minus_total_residual": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "translation_error - left_ward_residual(total)",
            "components": _component_vector(translation_error_minus_total_residual),
        },
        "actual_equal_time_norm": float(np.linalg.norm(actual_equal_time)),
        "shifted_equal_time_reference_norm": float(np.linalg.norm(shifted_equal_time_reference)),
        "contact_contraction_norm": float(np.linalg.norm(contact_contraction)),
        "shifted_equal_time_plus_contact_norm": float(np.linalg.norm(shifted_equal_time_plus_contact)),
        "translation_error_norm": translation_error_norm,
        "translation_error_minus_total_residual_norm": float(
            np.linalg.norm(translation_error_minus_total_residual)
        ),
        "total_ward_residual_norm": total_residual_norm,
        "translation_error_over_q_norm": float(translation_error_norm / q_norm),
        "total_ward_residual_over_q_norm": float(total_residual_norm / q_norm),
    }


def _normal_equal_time_sum_rule_audit(
    points: np.ndarray,
    weights: np.ndarray,
    config: KuboConfig,
    q: np.ndarray,
    components: dict[str, np.ndarray],
) -> dict[str, Any]:
    qx, qy = float(q[0]), float(q[1])
    peierls_terms = normal_state_hopping_terms()
    rho = np.eye(4, dtype=complex)
    bubble_equal_time = np.zeros(3, dtype=complex)
    interband_contribution = np.zeros(3, dtype=complex)
    intraband_contribution = np.zeros(3, dtype=complex)
    intraband_finite_q_difference = np.zeros(3, dtype=complex)
    intraband_fprime_approximation = np.zeros(3, dtype=complex)
    direct_contact_contraction = np.zeros(3, dtype=complex)
    shifted_equal_time_reference = np.zeros(3, dtype=complex)

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        h_minus = normal_state_hamiltonian(kx - 0.5 * qx, ky - 0.5 * qy)
        h_plus = normal_state_hamiltonian(kx + 0.5 * qx, ky + 0.5 * qy)
        energies_minus, states_minus = np.linalg.eigh(h_minus)
        energies_plus, states_plus = np.linalg.eigh(h_plus)
        occupations_minus = fermi_function(
            energies_minus,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        occupations_plus = fermi_function(
            energies_plus,
            config.fermi_level_eV,
            config.temperature_eV,
        )

        vector_x = peierls_hamiltonian_vector_vertex(
            kx,
            ky,
            qx,
            qy,
            "x",
            hopping_terms=peierls_terms,
        )
        vector_y = peierls_hamiltonian_vector_vertex(
            kx,
            ky,
            qx,
            qy,
            "y",
            hopping_terms=peierls_terms,
        )
        source_vertices = (rho, vector_x, vector_y)
        rho_band = states_minus.conjugate().T @ rho @ states_plus
        source_matrices = tuple(states_minus.conjugate().T @ vertex @ states_plus for vertex in source_vertices)

        for m, energy_minus in enumerate(energies_minus):
            for n, energy_plus in enumerate(energies_plus):
                occupation_diff = float(occupations_minus[m] - occupations_plus[n])
                if occupation_diff == 0.0:
                    continue
                term = np.array(
                    [
                        occupation_diff * rho_band[m, n] * np.conjugate(source_matrix[m, n])
                        for source_matrix in source_matrices
                    ],
                    dtype=complex,
                )
                weighted_term = weight * term
                bubble_equal_time += weighted_term
                if m == n:
                    intraband_contribution += weighted_term
                    intraband_finite_q_difference += weighted_term
                else:
                    interband_contribution += weighted_term

        midpoint_energies = 0.5 * (energies_minus + energies_plus)
        finite_difference_delta = energies_plus - energies_minus
        fprime_occupation_diff = negative_fermi_derivative(
            midpoint_energies,
            config.fermi_level_eV,
            config.temperature_eV,
            config.eta_eV,
        ) * finite_difference_delta
        for band_index, occupation_diff_approx in enumerate(fprime_occupation_diff):
            intraband_fprime_approximation += weight * np.array(
                [
                    float(occupation_diff_approx)
                    * rho_band[band_index, band_index]
                    * np.conjugate(source_matrix[band_index, band_index])
                    for source_matrix in source_matrices
                ],
                dtype=complex,
            )

        h_midpoint = normal_state_hamiltonian(kx, ky)
        energies_midpoint, states_midpoint = np.linalg.eigh(h_midpoint)
        occupations_midpoint = fermi_function(
            energies_midpoint,
            config.fermi_level_eV,
            config.temperature_eV,
        )
        for source_index, source_direction in enumerate(("x", "y"), start=1):
            shifted_vertex_reference = peierls_hamiltonian_vector_vertex(
                kx + 0.5 * qx,
                ky + 0.5 * qy,
                qx,
                qy,
                source_direction,
                hopping_terms=peierls_terms,
            ) - peierls_hamiltonian_vector_vertex(
                kx - 0.5 * qx,
                ky - 0.5 * qy,
                qx,
                qy,
                source_direction,
                hopping_terms=peierls_terms,
            )
            band_shifted_reference = states_midpoint.conjugate().T @ shifted_vertex_reference @ states_midpoint
            shifted_equal_time_reference[source_index] += weight * np.sum(
                occupations_midpoint * np.diag(band_shifted_reference)
            )
        for source_index, source_direction in enumerate(("x", "y"), start=1):
            contraction_value = 0.0j
            for q_component, observable_direction in ((qx, "x"), (qy, "y")):
                contact_matrix = peierls_hamiltonian_contact_vertex(
                    kx,
                    ky,
                    qx,
                    qy,
                    observable_direction,
                    source_direction,
                    hopping_terms=peierls_terms,
                )
                band_contact = states_midpoint.conjugate().T @ contact_matrix @ states_midpoint
                physical_direct_contact = -np.sum(occupations_midpoint * np.diag(band_contact))
                contraction_value += q_component * physical_direct_contact
            direct_contact_contraction[source_index] += weight * contraction_value

    total_left_residual, _ = physical_ward_residuals(components["total"], config.omega_eV, q)
    equal_time_plus_contact = bubble_equal_time + direct_contact_contraction
    difference_from_total = equal_time_plus_contact - total_left_residual
    shifted_equal_time_plus_contact = shifted_equal_time_reference + direct_contact_contraction
    actual_minus_shifted_equal_time = bubble_equal_time - shifted_equal_time_reference
    actual_minus_shifted_vs_total = actual_minus_shifted_equal_time - total_left_residual
    return {
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "side": "left_contraction",
        "component_labels": list(WARD_COMPONENT_LABELS),
        "bubble_equal_time_term": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": (
                "Denominator-cancelled left bubble Ward term using the same finite-q band basis, "
                "source vertices, k weights, and Fermi occupations as the normal bubble."
            ),
            "components": _component_vector(bubble_equal_time),
            "interband_contribution": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "band_partition": "m != n in the finite-q sorted band labels",
                "components": _component_vector(interband_contribution),
            },
            "intraband_contribution": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "band_partition": "m == n in the finite-q sorted band labels",
                "components": _component_vector(intraband_contribution),
                "finite_q_difference_form": {
                    "diagnostic_only": True,
                    "valid_for_casimir_input": False,
                    "occupation_difference": "f(E_minus[m]) - f(E_plus[m])",
                    "components": _component_vector(intraband_finite_q_difference),
                },
                "fprime_approximation_form": {
                    "diagnostic_only": True,
                    "valid_for_casimir_input": False,
                    "occupation_difference": "(-df/dE at midpoint energy) * (E_plus[m] - E_minus[m])",
                    "components": _component_vector(intraband_fprime_approximation),
                },
            },
        },
        "direct_contact_contraction": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "qx * D[x,nu] + qy * D[y,nu] from the normal direct/contact response.",
            "components": _component_vector(direct_contact_contraction),
        },
        "equal_time_plus_contact": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "bubble_equal_time_term + direct_contact_contraction",
            "components": _component_vector(equal_time_plus_contact),
        },
        "difference_from_total_ward_residual": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "definition": "equal_time_plus_contact - left_ward_residual(total)",
            "components": _component_vector(difference_from_total),
            "norm": float(np.linalg.norm(difference_from_total)),
        },
        "shifted_grid_equal_time_sum_rule": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "purpose": (
                "Diagnose whether the remaining normal-state response residual is caused by finite-k-mesh "
                "failure of the k -> k +/- q/2 variable shift rather than by the implemented contact vertex."
            ),
            "actual_bubble_equal_time_term": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "source": "same vector as bubble_equal_time_term.components",
                "components": _component_vector(bubble_equal_time),
            },
            "shifted_equal_time_reference": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "definition": "Tr[f(H(k)) * (V_j(k+q/2,q) - V_j(k-q/2,q))]",
                "density_component_note": "not_applicable; stored as zero because there is no density current-vertex finite-difference reference",
                "components": _component_vector(shifted_equal_time_reference),
            },
            "contact_contraction": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "source": "same vector as direct_contact_contraction.components",
                "components": _component_vector(direct_contact_contraction),
            },
            "shifted_equal_time_plus_contact": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "definition": "shifted_equal_time_reference + direct_contact_contraction",
                "components": _component_vector(shifted_equal_time_plus_contact),
            },
            "actual_minus_shifted_equal_time": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "definition": "actual_bubble_equal_time_term - shifted_equal_time_reference",
                "components": _component_vector(actual_minus_shifted_equal_time),
            },
            "actual_minus_shifted_vs_total_residual_difference": {
                "diagnostic_only": True,
                "valid_for_casimir_input": False,
                "definition": "actual_minus_shifted_equal_time - left_ward_residual(total)",
                "components": _component_vector(actual_minus_shifted_vs_total),
            },
            "shifted_equal_time_plus_contact_norm": float(np.linalg.norm(shifted_equal_time_plus_contact)),
            "actual_minus_shifted_equal_time_norm": float(np.linalg.norm(actual_minus_shifted_equal_time)),
            "difference_from_total_ward_residual_norm": float(np.linalg.norm(actual_minus_shifted_vs_total)),
        },
    }


def run_normal_finite_q_ward_audit(
    *,
    omega_eV: float = 0.01,
    q_values: tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02),
    q_directions: tuple[str, ...] = ("x", "y", "diagonal"),
    nk_values: tuple[int, ...] = (3,),
    temperature_K: float = 10.0,
    eta_eV: float = 1e-8,
) -> dict[str, Any]:
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    unknown_directions = sorted(set(q_directions) - set(DIRECTION_VECTORS))
    if unknown_directions:
        raise ValueError(f"unknown q direction(s): {unknown_directions}")
    nk_reports: list[dict[str, Any]] = []
    quadrature_summary_rows: list[dict[str, Any]] = []
    translation_error_summary_rows: list[dict[str, Any]] = []
    for nk in nk_values:
        points = uniform_bz_mesh(int(nk))
        weights = k_weights(points)
        q_reports: list[dict[str, Any]] = []
        for direction_name in q_directions:
            direction = np.asarray(DIRECTION_VECTORS[direction_name], dtype=float)
            for q_value in q_values:
                q = float(q_value) * direction
                components = normal_physical_density_current_response_components_imag_axis(points, config, q, weights)
                response_rows = [
                    _response_residual_row(response_name, components[response_name], config.omega_eV, q)
                    for response_name in RESPONSE_NAMES
                ]
                operator_rows = _operator_level_rows(points, q)
                equal_time_audit = _normal_equal_time_sum_rule_audit(
                    points,
                    weights,
                    config,
                    q,
                    components,
                )
                second_order_audit = _operator_level_second_order_contact_ward(points, q)
                shifted_pair_audit = _ward_compatible_shifted_pair_quadrature_audit(
                    points,
                    weights,
                    config,
                    q,
                    components,
                )
                finite_mesh_translation_error_audit = _finite_mesh_translation_error_audit(
                    equal_time_audit,
                    components,
                    config,
                    q,
                )
                shifted_grid_summary = equal_time_audit["shifted_grid_equal_time_sum_rule"]
                raw_total_residual_norm = shifted_pair_audit["raw_total_ward_residual"]["left_ward_residual_norm"]
                shifted_pair_total_residual_norm = shifted_pair_audit["shifted_pair_total_ward_residual"][
                    "left_ward_residual_norm"
                ]
                q_norm = float(np.linalg.norm(q))
                compact_quadrature_summary = {
                    "diagnostic_only": True,
                    "valid_for_casimir_input": False,
                    "shifted_equal_time_plus_contact_norm": float(
                        shifted_grid_summary["shifted_equal_time_plus_contact_norm"]
                    ),
                    "actual_minus_shifted_equal_time_norm": float(
                        shifted_grid_summary["actual_minus_shifted_equal_time_norm"]
                    ),
                    "actual_minus_shifted_vs_total_residual_difference_norm": float(
                        shifted_grid_summary["difference_from_total_ward_residual_norm"]
                    ),
                }
                quadrature_summary_rows.append(
                    {
                        "diagnostic_only": True,
                        "valid_for_casimir_input": False,
                        "nk": int(nk),
                        "q_direction": direction_name,
                        "q_norm": q_norm,
                        "raw_total_residual_norm": float(raw_total_residual_norm),
                        "shifted_pair_total_residual_norm": float(shifted_pair_total_residual_norm),
                        "raw_total_residual_over_q_abs": float(raw_total_residual_norm / q_norm),
                        "shifted_pair_total_residual_over_q_abs": float(
                            shifted_pair_total_residual_norm / q_norm
                        ),
                        "second_order_contact_ward_max_absolute_error_norm": float(
                            second_order_audit["max_absolute_error_norm"]
                        ),
                        "shifted_equal_time_plus_contact_norm": compact_quadrature_summary[
                            "shifted_equal_time_plus_contact_norm"
                        ],
                        "actual_minus_shifted_vs_total_residual_difference_norm": compact_quadrature_summary[
                            "actual_minus_shifted_vs_total_residual_difference_norm"
                        ],
                    }
                )
                translation_error_summary_rows.append(
                    {
                        "diagnostic_only": True,
                        "valid_for_casimir_input": False,
                        "nk": int(nk),
                        "q_direction": direction_name,
                        "q_norm": q_norm,
                        "total_ward_residual_norm": finite_mesh_translation_error_audit[
                            "total_ward_residual_norm"
                        ],
                        "total_ward_residual_over_q_norm": finite_mesh_translation_error_audit[
                            "total_ward_residual_over_q_norm"
                        ],
                        "shifted_equal_time_plus_contact_norm": finite_mesh_translation_error_audit[
                            "shifted_equal_time_plus_contact_norm"
                        ],
                        "translation_error_norm": finite_mesh_translation_error_audit[
                            "translation_error_norm"
                        ],
                        "translation_error_over_q_norm": finite_mesh_translation_error_audit[
                            "translation_error_over_q_norm"
                        ],
                        "translation_error_minus_total_residual_norm": finite_mesh_translation_error_audit[
                            "translation_error_minus_total_residual_norm"
                        ],
                        "second_order_contact_ward_max_absolute_error_norm": float(
                            second_order_audit["max_absolute_error_norm"]
                        ),
                    }
                )
                q_reports.append(
                    {
                        "q_direction": direction_name,
                        "q_model": [float(q[0]), float(q[1])],
                        "q_norm": q_norm,
                        "response_level_residuals": response_rows,
                        "longitudinal_current_residual_scaling": _longitudinal_current_scaling(response_rows, q),
                        "normal_current_equal_time_sum_rule_audit": equal_time_audit,
                        "ward_compatible_shifted_pair_quadrature_audit": shifted_pair_audit,
                        "shifted_grid_equal_time_consistency_summary": compact_quadrature_summary,
                        "finite_mesh_translation_error_audit": finite_mesh_translation_error_audit,
                        "operator_level_peierls_ward": {
                            "residual_kind": "operator_level",
                            "identity": "q_x V_x(k,q) + q_y V_y(k,q) = H(k+q/2)-H(k-q/2)",
                            "max_absolute_error_norm": float(max(row["absolute_error_norm"] for row in operator_rows)),
                            "max_relative_error_norm": float(max(row["relative_error_norm"] for row in operator_rows)),
                            "per_k_residuals": operator_rows,
                        },
                        "operator_level_second_order_contact_ward": second_order_audit,
                    }
                )
        nk_reports.append(
            {
                "nk": int(nk),
                "mesh_size": int(points.shape[0]),
                "q_reports": q_reports,
            }
        )
    return {
        "audit_name": "normal_finite_q_ward_audit",
        "scope": "diagnostic_only_normal_state_finite_q_ward_residuals",
        "omega_eV": float(config.omega_eV),
        "temperature_K": float(temperature_K),
        "eta_eV": float(eta_eV),
        "nk_values": [int(value) for value in nk_values],
        "q_values": [float(value) for value in q_values],
        "q_directions": list(q_directions),
        "component_labels": list(WARD_COMPONENT_LABELS),
        "response_level_residuals_explain": (
            "bubble/direct/total are normal-state response-level residuals from physical_ward_residuals; "
            "ward_contraction_decomposition stores iomega, qx, qy terms before summing"
        ),
        "operator_level_residuals_explain": (
            "Peierls vertex identity is checked before response assembly and is distinct from response-level residuals"
        ),
        "ward_compatible_quadrature_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "rows": quadrature_summary_rows,
        },
        "finite_mesh_translation_error_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "rows": translation_error_summary_rows,
        },
        "nk_reports": nk_reports,
        "ward_identity_closed": False,
        "valid_for_casimir_input": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 normal-state finite-q Ward residual 诊断。")
    parser.add_argument("--omega", type=float, default=0.01)
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.001, 0.002, 0.005, 0.01, 0.02])
    parser.add_argument("--directions", nargs="+", choices=tuple(DIRECTION_VECTORS), default=["x", "y", "diagonal"])
    parser.add_argument("--nk", type=int, default=3, help="Backward-compatible single-nk shortcut.")
    parser.add_argument("--nk-values", nargs="+", type=int)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    nk_values = tuple(args.nk_values) if args.nk_values is not None else (int(args.nk),)
    payload = run_normal_finite_q_ward_audit(
        omega_eV=args.omega,
        q_values=tuple(args.q_values),
        q_directions=tuple(args.directions),
        nk_values=nk_values,
        temperature_K=args.temperature_K,
        eta_eV=args.eta,
    )
    if args.json_output is not None:
        _write_json(args.json_output, payload)
    print(
        "normal finite-q Ward audit prepared: "
        f"nk_values={payload['nk_values']}, q_values={payload['q_values']}, "
        f"directions={payload['q_directions']}, valid_for_casimir_input={payload['valid_for_casimir_input']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
