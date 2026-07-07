"""Shifted-mesh averaged finite-q BdG response helpers.

This module is validation-only.  It averages response components over shifted
midpoint meshes and recomputes the nonlinear amplitude/eta Schur response from
averaged blocks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from lno327 import k_weights
from lno327.collective.schur import apply_amplitude_phase_schur
from lno327.response.finite_q_bdg import precompute_finite_q_bdg_workspace_from_model_ansatz
from lno327.workflows.finite_q_engine import bdg_finite_q_response_imag_axis_from_workspace
from validation.lib.finite_q_integrated_ward_chain import evaluate_integrated_ward_chain
from validation.lib.finite_q_integrated_ward_convergence import shifted_uniform_bz_mesh

MATRIX_FIELDS = (
    "bare_bubble",
    "direct",
    "bare_total",
    "minus_schur",
    "plus_schur",
    "collective_bubble",
    "collective_counterterm",
    "collective_total",
    "em_collective_left",
    "collective_em_right",
)
SCALAR_FIELDS = (
    "phase_phase_bubble",
    "phase_phase_direct",
    "phase_phase_total",
)
VECTOR_FIELDS = (
    "phase_coupling_left",
    "phase_coupling_right",
)


def shift_pairs_from_fractions(shift_fractions: Sequence[float]) -> tuple[tuple[float, float], ...]:
    shifts = tuple(float(value) for value in shift_fractions)
    if not shifts:
        raise ValueError("shift_fractions must not be empty")
    return tuple((sx, sy) for sx in shifts for sy in shifts)


def _mean_field(responses: Sequence[Any], field: str) -> Any:
    values = [np.asarray(getattr(response, field), dtype=complex) for response in responses]
    return sum(values, np.zeros_like(values[0], dtype=complex)) / len(values)


def _vector_from_payload(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray([entry["real"] + 1j * entry["imag"] for entry in payload["vector"]], dtype=complex)


def _vector_payload(vector: np.ndarray) -> dict[str, Any]:
    labels = ("density", "current_x", "current_y")
    array = np.asarray(vector, dtype=complex).reshape(3)
    abs_values = np.abs(array)
    dominant_index = int(np.argmax(abs_values))
    return {
        "labels": list(labels),
        "vector": [
            {"component": label, "real": float(np.real(value)), "imag": float(np.imag(value))}
            for label, value in zip(labels, array, strict=True)
        ],
        "norm": float(np.linalg.norm(array)),
        "max_abs": float(np.max(abs_values)),
        "dominant_component": labels[dominant_index],
        "valid_for_casimir_input": False,
    }


def _average_chain(chains: Sequence[dict[str, Any]], *, pairing_name: str, q_model: np.ndarray, omega_eV: float, delta0_eV: float) -> dict[str, Any]:
    left_vectors: dict[str, np.ndarray] = {}
    vector_keys = (
        "bubble_collective_contraction",
        "denominator_cancelled_equal_time",
        "contact_target_minus_direct_contraction",
    )
    for key in vector_keys:
        left_vectors[key] = sum(
            (_vector_from_payload(chain["left_chain"][key]) for chain in chains),
            np.zeros(3, dtype=complex),
        ) / len(chains)
    left_vectors["bubble_to_equal_time_difference"] = (
        left_vectors["bubble_collective_contraction"] - left_vectors["denominator_cancelled_equal_time"]
    )
    left_vectors["equal_time_to_contact_difference"] = (
        left_vectors["denominator_cancelled_equal_time"] - left_vectors["contact_target_minus_direct_contraction"]
    )
    left_vectors["full_chain_residual"] = (
        left_vectors["bubble_collective_contraction"] - left_vectors["contact_target_minus_direct_contraction"]
    )
    q = np.asarray(q_model, dtype=float)
    return {
        "identity_version": "finite_q_integrated_ward_chain_shift_average_v1",
        "diagnostic_role": "shifted_mesh_averaged_integrated_ward_proof_chain_not_a_new_ward_criterion",
        "pairing_name": str(pairing_name),
        "q_model": [float(q[0]), float(q[1])],
        "omega_eV": float(omega_eV),
        "delta0_eV": float(delta0_eV),
        "left_chain": {
            "formula": "shift average of W_L(K_AA_bubble)+R_L K_etaA -> equal-time trace -> -W_L(K_AA_direct)",
            **{key: _vector_payload(value) for key, value in left_vectors.items()},
            "valid_for_casimir_input": False,
        },
        "max_bubble_to_equal_time_difference_norm": float(np.linalg.norm(left_vectors["bubble_to_equal_time_difference"])),
        "max_equal_time_to_contact_difference_norm": float(np.linalg.norm(left_vectors["equal_time_to_contact_difference"])),
        "max_full_chain_residual_norm": float(np.linalg.norm(left_vectors["full_chain_residual"])),
        "valid_for_casimir_input": False,
    }


def average_finite_q_bdg_response_over_shifts(
    *,
    model: Any,
    ansatz: Any,
    q_model: np.ndarray,
    nk: int,
    config: Any,
    pairing_params: Any,
    options: Any,
    shift_fractions: Sequence[float],
) -> tuple[Any, dict[str, Any]]:
    """Average response components over shifted meshes and recompute Schur."""

    q = np.asarray(q_model, dtype=float)
    shift_pairs = shift_pairs_from_fractions(shift_fractions)
    responses: list[Any] = []
    chains: list[dict[str, Any]] = []
    per_shift: list[dict[str, Any]] = []
    for shift_x, shift_y in shift_pairs:
        points = shifted_uniform_bz_mesh(int(nk), shift_x, shift_y)
        weights = k_weights(points)
        workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
            model.spec,
            ansatz,
            q,
            points,
            weights,
            config,
            pairing_params,
            options,
        )
        response = bdg_finite_q_response_imag_axis_from_workspace(workspace, config=config)
        chain = evaluate_integrated_ward_chain(workspace=workspace, response=response, delta0_eV=float(pairing_params.delta0_eV))
        responses.append(response)
        chains.append(chain)
        per_shift.append(
            {
                "shift_fraction": [float(shift_x), float(shift_y)],
                "max_bubble_to_equal_time_difference_norm": float(chain["max_bubble_to_equal_time_difference_norm"]),
                "max_equal_time_to_contact_difference_norm": float(chain["max_equal_time_to_contact_difference_norm"]),
                "max_full_chain_residual_norm": float(chain["max_full_chain_residual_norm"]),
                "valid_for_casimir_input": False,
            }
        )
    averaged_fields = {field: _mean_field(responses, field) for field in MATRIX_FIELDS + VECTOR_FIELDS + SCALAR_FIELDS}
    schur_result = apply_amplitude_phase_schur(
        averaged_fields["bare_total"],
        averaged_fields["em_collective_left"],
        averaged_fields["collective_total"],
        averaged_fields["collective_em_right"],
    )
    averaged_fields["amplitude_phase_schur"] = schur_result.corrected_response
    averaged_fields["gauge_restored"] = schur_result.corrected_response
    metadata = dict(getattr(responses[0], "metadata", {}))
    metadata.update(
        {
            "shifted_mesh_average": True,
            "shift_fractions": [float(value) for value in shift_fractions],
            "shift_pairs": [[float(x), float(y)] for x, y in shift_pairs],
            "num_shifted_meshes": len(shift_pairs),
            "collective_total_condition_number": schur_result.condition_number,
            "collective_inverse_method": schur_result.inverse_method,
            "amplitude_phase_schur_status": schur_result.status,
            "valid_for_casimir_input": False,
        }
    )
    response = SimpleNamespace(**averaged_fields, metadata=metadata)
    averaged_chain = _average_chain(
        chains,
        pairing_name=ansatz.name,
        q_model=q,
        omega_eV=float(config.omega_eV),
        delta0_eV=float(pairing_params.delta0_eV),
    )
    averaged_chain["shifted_mesh_average"] = {
        "enabled": True,
        "shift_fractions": [float(value) for value in shift_fractions],
        "shift_pairs": [[float(x), float(y)] for x, y in shift_pairs],
        "per_shift_summary": per_shift,
        "valid_for_casimir_input": False,
    }
    return response, averaged_chain
