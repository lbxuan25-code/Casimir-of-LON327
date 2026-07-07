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
from lno327.collective.validation import validate_physical_ward_identity
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
WARD_LABELS = ("density", "current_x", "current_y")
CHAIN_VECTOR_KEYS = (
    "bubble_collective_contraction",
    "denominator_cancelled_equal_time",
    "contact_target_minus_direct_contraction",
    "bubble_to_equal_time_difference",
    "equal_time_to_contact_difference",
    "full_chain_residual",
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
    array = np.asarray(vector, dtype=complex).reshape(3)
    abs_values = np.abs(array)
    dominant_index = int(np.argmax(abs_values))
    return {
        "labels": list(WARD_LABELS),
        "vector": [
            {"component": label, "real": float(np.real(value)), "imag": float(np.imag(value))}
            for label, value in zip(WARD_LABELS, array, strict=True)
        ],
        "norm": float(np.linalg.norm(array)),
        "max_abs": float(np.max(abs_values)),
        "dominant_component": WARD_LABELS[dominant_index],
        "valid_for_casimir_input": False,
    }


def _matrix_payload(matrix: np.ndarray) -> dict[str, Any]:
    array = np.asarray(matrix, dtype=complex)
    abs_values = np.abs(array)
    return {
        "shape": list(array.shape),
        "norm": float(np.linalg.norm(array)),
        "max_abs": float(np.max(abs_values)) if abs_values.size else 0.0,
        "valid_for_casimir_input": False,
    }


def _ward_payload(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    ward = validate_physical_ward_identity(matrix, omega_eV, q)
    return {
        "left": _vector_payload(ward.left_residual),
        "right": _vector_payload(ward.right_residual),
        "left_norm": float(ward.left_norm),
        "right_norm": float(ward.right_norm),
        "max_norm": float(max(ward.left_norm, ward.right_norm)),
        "valid_for_casimir_input": False,
    }


def _distribution_payload(vectors: Sequence[np.ndarray]) -> dict[str, Any]:
    arrays = [np.asarray(vector, dtype=complex).reshape(3) for vector in vectors]
    if not arrays:
        zero = np.zeros(3, dtype=complex)
        return {
            "num_samples": 0,
            "mean_vector": _vector_payload(zero),
            "norm_of_mean": 0.0,
            "mean_norm": 0.0,
            "rms_norm": 0.0,
            "min_norm": 0.0,
            "max_norm": 0.0,
            "cancellation_ratio_vs_rms": None,
            "component_statistics": [],
            "valid_for_casimir_input": False,
        }
    stacked = np.vstack(arrays)
    mean_vector = np.mean(stacked, axis=0)
    norms = np.asarray([np.linalg.norm(vector) for vector in arrays], dtype=float)
    rms_norm = float(np.sqrt(np.mean(norms**2)))
    norm_of_mean = float(np.linalg.norm(mean_vector))
    component_statistics = []
    for index, label in enumerate(WARD_LABELS):
        values = stacked[:, index]
        real_values = np.real(values)
        imag_values = np.imag(values)
        component_statistics.append(
            {
                "component": label,
                "mean_real": float(np.mean(real_values)),
                "mean_imag": float(np.mean(imag_values)),
                "std_real": float(np.std(real_values)),
                "std_imag": float(np.std(imag_values)),
                "min_real": float(np.min(real_values)),
                "max_real": float(np.max(real_values)),
                "positive_real_count": int(np.sum(real_values > 0.0)),
                "negative_real_count": int(np.sum(real_values < 0.0)),
                "near_zero_real_count": int(np.sum(real_values == 0.0)),
            }
        )
    return {
        "num_samples": len(arrays),
        "mean_vector": _vector_payload(mean_vector),
        "norm_of_mean": norm_of_mean,
        "mean_norm": float(np.mean(norms)),
        "rms_norm": rms_norm,
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "cancellation_ratio_vs_rms": None if rms_norm == 0.0 else float(1.0 - norm_of_mean / rms_norm),
        "component_statistics": component_statistics,
        "valid_for_casimir_input": False,
    }


def _average_chain(chains: Sequence[dict[str, Any]], *, pairing_name: str, q_model: np.ndarray, omega_eV: float, delta0_eV: float) -> dict[str, Any]:
    left_vectors: dict[str, np.ndarray] = {}
    for key in CHAIN_VECTOR_KEYS[:3]:
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
    distributions = {
        key: _distribution_payload([_vector_from_payload(chain["left_chain"][key]) for chain in chains])
        for key in CHAIN_VECTOR_KEYS
    }
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
        "shift_distribution": distributions,
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
                "equal_time_to_contact_difference": chain["left_chain"]["equal_time_to_contact_difference"],
                "full_chain_residual": chain["left_chain"]["full_chain_residual"],
                "amplitude_phase_schur_ward": _ward_payload(response.amplitude_phase_schur, float(config.omega_eV), q),
                "collective_inverse_method": str(response.metadata.get("collective_inverse_method", "not_used")),
                "collective_total_condition_number": response.metadata.get("collective_total_condition_number"),
                "valid_for_casimir_input": False,
            }
        )
    averaged_fields = {field: _mean_field(responses, field) for field in MATRIX_FIELDS + VECTOR_FIELDS + SCALAR_FIELDS}
    mean_per_shift_amp_schur = _mean_field(responses, "amplitude_phase_schur")
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
    schur_difference = schur_result.corrected_response - mean_per_shift_amp_schur
    averaged_chain["shifted_mesh_average"] = {
        "enabled": True,
        "shift_fractions": [float(value) for value in shift_fractions],
        "shift_pairs": [[float(x), float(y)] for x, y in shift_pairs],
        "per_shift_summary": per_shift,
        "amplitude_phase_schur_ward_distribution": _distribution_payload(
            [
                _vector_from_payload(item["amplitude_phase_schur_ward"]["left"])
                for item in per_shift
            ]
        ),
        "schur_noncommutativity": {
            "definition": "Schur(averaged blocks) - average(Schur(per-shift blocks))",
            "matrix_difference": _matrix_payload(schur_difference),
            "ward_of_schur_averaged_blocks": _ward_payload(schur_result.corrected_response, float(config.omega_eV), q),
            "ward_of_average_per_shift_schur": _ward_payload(mean_per_shift_amp_schur, float(config.omega_eV), q),
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }
    return response, averaged_chain
