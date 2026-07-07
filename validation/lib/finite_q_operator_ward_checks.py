"""Operator-level finite-q BdG Ward identity diagnostics.

The checks in this module are diagnostic-only.  They test matrix identities before
Kubo integration and do not define a new Casimir gate.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from lno327.bdg.finite_q import density_vertex
from lno327.bdg.hamiltonian import bdg_hamiltonian_from_model_pairing
from lno327.response.finite_q_bdg import bdg_contact_vertex_from_spec, bdg_vector_vertex_from_spec

DIRECTIONS = ("x", "y")
IDENTITY_VERSION = "finite_q_bdg_operator_ward_v1"


def _matrix_payload(matrix: np.ndarray, normal_dim: int) -> dict[str, Any]:
    array = np.asarray(matrix, dtype=complex)
    abs_values = np.abs(array)
    blocks = {
        "particle_particle": array[:normal_dim, :normal_dim],
        "particle_hole": array[:normal_dim, normal_dim:],
        "hole_particle": array[normal_dim:, :normal_dim],
        "hole_hole": array[normal_dim:, normal_dim:],
    }
    block_norms = {name: float(np.linalg.norm(block)) for name, block in blocks.items()}
    return {
        "norm": float(np.linalg.norm(array)),
        "max_abs": float(np.max(abs_values)) if abs_values.size else 0.0,
        "block_norms": block_norms,
        "dominant_block": max(block_norms.items(), key=lambda item: item[1])[0] if block_norms else None,
        "valid_for_casimir_input": False,
    }


def _bdg_hamiltonian(spec: Any, ansatz: Any, amp: Any, k: np.ndarray) -> np.ndarray:
    kx = float(k[0])
    ky = float(k[1])
    return bdg_hamiltonian_from_model_pairing(spec, kx, ky, ansatz.mean_pairing(kx, ky, amp))


def _first_order_error(
    *,
    spec: Any,
    ansatz: Any,
    amp: Any,
    k: np.ndarray,
    q: np.ndarray,
    delta0_eV: float,
    current_vertex: str,
    tau3: np.ndarray,
) -> np.ndarray:
    kx = float(k[0])
    ky = float(k[1])
    qx = float(q[0])
    qy = float(q[1])
    gamma_x = bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "x", current_vertex)
    gamma_y = bdg_vector_vertex_from_spec(spec, kx, ky, qx, qy, "y", current_vertex)
    eta2 = ansatz.collective_vertices(kx, ky, qx, qy, amp)[1]
    h_plus = _bdg_hamiltonian(spec, ansatz, amp, k + 0.5 * q)
    h_minus = _bdg_hamiltonian(spec, ansatz, amp, k - 0.5 * q)
    lhs = qx * gamma_x + qy * gamma_y + 2j * float(delta0_eV) * eta2
    rhs = h_plus @ tau3 - tau3 @ h_minus
    return np.asarray(lhs - rhs, dtype=complex)


def _contact_error(
    *,
    spec: Any,
    k: np.ndarray,
    q: np.ndarray,
    current_vertex: str,
    tau3: np.ndarray,
    direction_j: str,
) -> np.ndarray:
    kx = float(k[0])
    ky = float(k[1])
    qx = float(q[0])
    qy = float(q[1])
    lhs = np.zeros_like(tau3, dtype=complex)
    for component, direction_i in zip((qx, qy), DIRECTIONS, strict=True):
        lhs += float(component) * bdg_contact_vertex_from_spec(
            spec, kx, ky, qx, qy, direction_i, direction_j, current_vertex
        )
    gamma_plus = bdg_vector_vertex_from_spec(
        spec, kx + 0.5 * qx, ky + 0.5 * qy, -qx, -qy, direction_j, current_vertex
    )
    gamma_minus = bdg_vector_vertex_from_spec(
        spec, kx - 0.5 * qx, ky - 0.5 * qy, -qx, -qy, direction_j, current_vertex
    )
    rhs = gamma_plus @ tau3 - tau3 @ gamma_minus
    return np.asarray(lhs - rhs, dtype=complex)


def _ranked_errors(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: float(item["error"]["norm"]), reverse=True)


def _summary(records: list[dict[str, Any]], *, tolerance: float) -> dict[str, Any]:
    ranked = _ranked_errors(records)
    max_norm = float(ranked[0]["error"]["norm"]) if ranked else 0.0
    return {
        "max_error_norm": max_norm,
        "rms_error_norm": float(np.sqrt(np.mean([float(item["error"]["norm"]) ** 2 for item in records]))) if records else 0.0,
        "worst_record": ranked[0] if ranked else None,
        "passed_by_tolerance": bool(max_norm <= float(tolerance)),
        "valid_for_casimir_input": False,
    }


def evaluate_bdg_operator_ward_checks(
    *,
    pairing_name: str,
    q_model: Sequence[float] | np.ndarray,
    delta0_eV: float,
    spec: Any,
    ansatz: Any,
    amp: Any,
    k_points: np.ndarray,
    current_vertex: str = "peierls",
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Evaluate first-order and contact BdG operator Ward identities."""

    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model must have shape (2,)")
    points = np.asarray(k_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("k_points must have shape (N, 2)")
    normal_dim = int(np.asarray(spec.normal_hamiltonian(0.0, 0.0)).shape[0])
    tau3 = density_vertex(normal_dim)

    first_records: list[dict[str, Any]] = []
    contact_records: list[dict[str, Any]] = []
    for k in points:
        first_error = _first_order_error(
            spec=spec,
            ansatz=ansatz,
            amp=amp,
            k=k,
            q=q,
            delta0_eV=delta0_eV,
            current_vertex=current_vertex,
            tau3=tau3,
        )
        first_records.append(
            {
                "k_model": [float(k[0]), float(k[1])],
                "error": _matrix_payload(first_error, normal_dim),
                "valid_for_casimir_input": False,
            }
        )
        for direction_j in DIRECTIONS:
            contact_error = _contact_error(
                spec=spec,
                k=k,
                q=q,
                current_vertex=current_vertex,
                tau3=tau3,
                direction_j=direction_j,
            )
            contact_records.append(
                {
                    "k_model": [float(k[0]), float(k[1])],
                    "direction_j": direction_j,
                    "error": _matrix_payload(contact_error, normal_dim),
                    "valid_for_casimir_input": False,
                }
            )

    return {
        "identity_version": IDENTITY_VERSION,
        "diagnostic_role": "operator_identity_diagnostic_not_a_new_ward_criterion",
        "pairing_name": str(pairing_name),
        "q_model": [float(q[0]), float(q[1])],
        "delta0_eV": float(delta0_eV),
        "current_vertex": str(current_vertex),
        "identities": {
            "first_order_bdg": "q_i Gamma_i + 2i delta0 Gamma_eta2 = H_plus tau3 - tau3 H_minus",
            "contact_bdg": "q_i M_ij = Gamma_j(k+q/2,-q) tau3 - tau3 Gamma_j(k-q/2,-q)",
        },
        "first_order_bdg_identity": {
            **_summary(first_records, tolerance=tolerance),
            "ranked_errors": _ranked_errors(first_records),
            "valid_for_casimir_input": False,
        },
        "bdg_contact_identity": {
            **_summary(contact_records, tolerance=tolerance),
            "ranked_errors": _ranked_errors(contact_records),
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }
