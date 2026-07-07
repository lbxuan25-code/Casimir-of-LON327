"""Convergence scans for the integrated finite-q Ward proof chain.

This diagnostic checks whether the equal-time-to-contact mismatch decreases with
quadrature refinement or shifted-mesh averaging.  It is diagnostic-only and does
not define a Casimir gate.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from lno327 import KuboConfig, k_weights
from lno327.workflows.finite_q_engine import FiniteQEngineOptions, bdg_finite_q_response_imag_axis_from_workspace
from lno327.response.finite_q_bdg import precompute_finite_q_bdg_workspace_from_model_ansatz
from validation.lib.finite_q_integrated_ward_chain import evaluate_integrated_ward_chain
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

IDENTITY_VERSION = "finite_q_integrated_ward_convergence_v1"
VECTOR_KEYS = (
    "bubble_to_equal_time_difference",
    "equal_time_to_contact_difference",
    "full_chain_residual",
)


def shifted_uniform_bz_mesh(nk: int, shift_fraction_x: float, shift_fraction_y: float) -> np.ndarray:
    """Return a shifted midpoint mesh over [-pi, pi) x [-pi, pi)."""

    if nk <= 0:
        raise ValueError("nk must be positive")
    spacing = 2.0 * np.pi / int(nk)
    kx_values = -np.pi + (np.arange(nk) + 0.5 + float(shift_fraction_x)) * spacing
    ky_values = -np.pi + (np.arange(nk) + 0.5 + float(shift_fraction_y)) * spacing
    kx_values = ((kx_values + np.pi) % (2.0 * np.pi)) - np.pi
    ky_values = ((ky_values + np.pi) % (2.0 * np.pi)) - np.pi
    return np.array([(kx, ky) for kx in kx_values for ky in ky_values], dtype=float)


def _vector_from_payload(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray([entry["real"] + 1j * entry["imag"] for entry in payload["vector"]], dtype=complex)


def _vector_summary(vectors: Sequence[np.ndarray]) -> dict[str, Any]:
    arrays = [np.asarray(vector, dtype=complex).reshape(3) for vector in vectors]
    if not arrays:
        zero = np.zeros(3, dtype=complex)
        return {
            "mean_vector": _vector_payload(zero),
            "mean_norm": 0.0,
            "rms_norm": 0.0,
            "max_norm": 0.0,
            "valid_for_casimir_input": False,
        }
    stacked = np.vstack(arrays)
    mean = np.mean(stacked, axis=0)
    norms = np.asarray([np.linalg.norm(vector) for vector in arrays], dtype=float)
    return {
        "mean_vector": _vector_payload(mean),
        "mean_norm": float(np.linalg.norm(mean)),
        "rms_norm": float(np.sqrt(np.mean(norms**2))),
        "max_norm": float(np.max(norms)),
        "valid_for_casimir_input": False,
    }


def _vector_payload(vector: np.ndarray) -> dict[str, Any]:
    array = np.asarray(vector, dtype=complex).reshape(3)
    labels = ("density", "current_x", "current_y")
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


def _compact_chain(chain: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity_version": chain["identity_version"],
        "diagnostic_role": chain["diagnostic_role"],
        "pairing_name": chain["pairing_name"],
        "q_model": list(chain["q_model"]),
        "max_bubble_to_equal_time_difference_norm": float(chain["max_bubble_to_equal_time_difference_norm"]),
        "max_equal_time_to_contact_difference_norm": float(chain["max_equal_time_to_contact_difference_norm"]),
        "max_full_chain_residual_norm": float(chain["max_full_chain_residual_norm"]),
        "left_chain": chain["left_chain"],
        "valid_for_casimir_input": False,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vectors_by_key = {
        key: [_vector_from_payload(row["chain"]["left_chain"][key]) for row in rows]
        for key in VECTOR_KEYS
    }
    return {
        "num_shifted_meshes": len(rows),
        "bubble_to_equal_time_difference": _vector_summary(vectors_by_key["bubble_to_equal_time_difference"]),
        "equal_time_to_contact_difference": _vector_summary(vectors_by_key["equal_time_to_contact_difference"]),
        "full_chain_residual": _vector_summary(vectors_by_key["full_chain_residual"]),
        "valid_for_casimir_input": False,
    }


def _run_single_chain(
    *,
    model: Any,
    pairing_name: str,
    q: np.ndarray,
    nk: int,
    shift_fraction: tuple[float, float],
    config: Any,
    amp: Any,
    options: Any,
) -> dict[str, Any]:
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    points = shifted_uniform_bz_mesh(nk, shift_fraction[0], shift_fraction[1])
    weights = k_weights(points)
    workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        q,
        points,
        weights,
        config,
        amp,
        options,
    )
    response = bdg_finite_q_response_imag_axis_from_workspace(workspace, config=config)
    chain = evaluate_integrated_ward_chain(workspace=workspace, response=response, delta0_eV=float(amp.delta0_eV))
    return {
        "pairing_name": pairing_name,
        "q_model": [float(q[0]), float(q[1])],
        "q_norm": float(np.linalg.norm(q)),
        "nk": int(nk),
        "mesh_size": int(points.shape[0]),
        "shift_fraction": [float(shift_fraction[0]), float(shift_fraction[1])],
        "chain": _compact_chain(chain),
        "valid_for_casimir_input": False,
    }


def run_integrated_ward_chain_convergence(
    *,
    model_name: str = "symmetry_bdg_2band",
    pairings: tuple[str, ...] = ("spm", "dwave"),
    q_values: tuple[float, ...] = (0.005, 0.01, 0.02),
    q_direction: tuple[float, float] = (1.0, 0.0),
    nk_values: tuple[int, ...] = (9, 13, 17),
    shift_fractions: tuple[float, ...] = (0.0,),
    omega_eV: float = 0.01,
    delta0_eV: float = 0.1,
) -> dict[str, Any]:
    """Run the integrated Ward chain convergence diagnostic."""

    model = get_finite_q_validation_model(model_name)
    for pairing in pairings:
        model.require_pairing(pairing)
    direction = np.asarray(q_direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("q_direction must be nonzero")
    direction = direction / norm
    config = KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    amp = model.build_pairing_params(delta0_eV)
    options = FiniteQEngineOptions(
        current_vertex="peierls",
        collective_mode="amplitude_phase",
        collective_counterterm="goldstone_gap_equation",
        include_phase_phase_direct=True,
    )
    shift_pairs = tuple((float(sx), float(sy)) for sx in shift_fractions for sy in shift_fractions)
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for pairing in pairings:
        for q_value in q_values:
            q = float(q_value) * direction
            for nk in nk_values:
                for shift_pair in shift_pairs:
                    row = _run_single_chain(
                        model=model,
                        pairing_name=pairing,
                        q=q,
                        nk=int(nk),
                        shift_fraction=shift_pair,
                        config=config,
                        amp=amp,
                        options=options,
                    )
                    rows.append(row)
                    grouped.setdefault((pairing, float(q_value), int(nk)), []).append(row)
    summaries = []
    for (pairing, q_value, nk), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        unshifted = next((row for row in group_rows if row["shift_fraction"] == [0.0, 0.0]), None)
        summaries.append(
            {
                "pairing_name": pairing,
                "q_model": [float(q_value * direction[0]), float(q_value * direction[1])],
                "q_norm": float(abs(q_value)),
                "nk": int(nk),
                "unshifted": None if unshifted is None else {
                    "bubble_to_equal_time": float(unshifted["chain"]["max_bubble_to_equal_time_difference_norm"]),
                    "equal_time_to_contact": float(unshifted["chain"]["max_equal_time_to_contact_difference_norm"]),
                    "full_chain": float(unshifted["chain"]["max_full_chain_residual_norm"]),
                    "valid_for_casimir_input": False,
                },
                "shifted_mesh_average": _aggregate_rows(group_rows),
                "valid_for_casimir_input": False,
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: float(row["chain"]["max_equal_time_to_contact_difference_norm"]),
        reverse=True,
    )
    return {
        "identity_version": IDENTITY_VERSION,
        "diagnostic_role": "integrated_ward_chain_convergence_not_a_new_ward_criterion",
        "model_name": model.name,
        "pairings": list(pairings),
        "omega_eV": float(omega_eV),
        "delta0_eV": float(delta0_eV),
        "q_values": [float(value) for value in q_values],
        "q_direction": [float(direction[0]), float(direction[1])],
        "nk_values": [int(value) for value in nk_values],
        "shift_fractions": [float(value) for value in shift_fractions],
        "shift_pairs": [[float(sx), float(sy)] for sx, sy in shift_pairs],
        "rows": rows,
        "summaries": summaries,
        "ranked_largest_equal_time_to_contact": ranked[: min(16, len(ranked))],
        "valid_for_casimir_input": False,
    }
