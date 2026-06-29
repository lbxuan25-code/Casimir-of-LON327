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
                q_reports.append(
                    {
                        "q_direction": direction_name,
                        "q_model": [float(q[0]), float(q[1])],
                        "q_norm": float(np.linalg.norm(q)),
                        "response_level_residuals": response_rows,
                        "longitudinal_current_residual_scaling": _longitudinal_current_scaling(response_rows, q),
                        "normal_current_equal_time_sum_rule_audit": _normal_equal_time_sum_rule_audit(
                            points,
                            weights,
                            config,
                            q,
                            components,
                        ),
                        "operator_level_peierls_ward": {
                            "residual_kind": "operator_level",
                            "identity": "q_x V_x(k,q) + q_y V_y(k,q) = H(k+q/2)-H(k-q/2)",
                            "max_absolute_error_norm": float(max(row["absolute_error_norm"] for row in operator_rows)),
                            "max_relative_error_norm": float(max(row["relative_error_norm"] for row in operator_rows)),
                            "per_k_residuals": operator_rows,
                        },
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
