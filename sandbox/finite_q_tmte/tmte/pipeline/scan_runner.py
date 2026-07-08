"""High-level scan runner for the finite-q TM/TE sandbox."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..adapters.bubble_adapter import compute_component_reference_effective, compute_target_bare_blocks
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..io.writers import write_json
from .block_builder import EffectiveTargetResponse, build_effective_from_blocks, compute_effective_target_response
from .schema import basis_payload, scan_payload
from .shifted_average import average_bare_blocks_then_schur, per_shift_summaries, shift_pairs_from_fractions


def _q_vector(q_value: float, direction: tuple[float, float]) -> np.ndarray:
    direction_array = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction_array))
    if norm <= 0.0:
        raise ValueError("q direction must be nonzero")
    return float(q_value) * direction_array / norm


def _result_payload(
    *,
    response: EffectiveTargetResponse,
    q_model: np.ndarray,
    shifted: dict[str, Any],
) -> dict[str, Any]:
    return {
        "q_model": q_model,
        "q_norm": float(np.linalg.norm(q_model)),
        "basis": basis_payload(response.bare_blocks.conventions),
        "bare_blocks": {
            "K_SS_bubble": response.bare_blocks.k_ss_bubble,
            "K_SS_contact": response.bare_blocks.k_ss_contact,
            "K_SS": response.bare_blocks.k_ss,
            "K_Seta": response.bare_blocks.k_seta,
            "K_etaS": response.bare_blocks.k_etas,
            "K_etaeta": response.bare_blocks.k_etaeta,
        },
        "effective_response": {
            "K_GTMTE_eff": response.schur.effective,
            "K_TMTE_eff": response.k_tmte_eff,
        },
        "diagnostics": response.diagnostics,
        "schur": {
            "solve_method": response.schur.solve_method,
            "etaeta_condition_number": response.schur.etaeta_condition_number,
            "condition_threshold": response.schur.condition_threshold,
            "numerically_suspect": response.schur.numerically_suspect,
            "valid_for_casimir_input": False,
        },
        "shifted_mesh_average": shifted,
        "valid_for_casimir_input": False,
    }


def run_scan(
    *,
    model_name: str,
    pairing_name: str,
    xi: float,
    q_values: Sequence[float],
    nk: int,
    q_directions: Sequence[tuple[float, float]] = ((1.0, 0.0),),
    shift_fractions: Sequence[float] = (0.0,),
    omega_eV: float | None = None,
    delta0_eV: float | None = None,
    temperature_K: float = 10.0,
    eta_eV: float = 1e-8,
) -> dict[str, Any]:
    """Run a lightweight direct target-basis scan object without writing files."""

    inputs = build_model_scan_inputs(
        model_name=model_name,
        pairing_name=pairing_name,
        xi=xi,
        omega_eV=omega_eV,
        nk=nk,
        delta0_eV=delta0_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
    )
    shift_pairs = shift_pairs_from_fractions(shift_fractions)
    use_shifted_average = len(shift_pairs) > 1 or any(abs(sx) > 0.0 or abs(sy) > 0.0 for sx, sy in shift_pairs)
    results: list[dict[str, Any]] = []
    for q_value in q_values:
        for direction in q_directions:
            q = _q_vector(float(q_value), direction)
            if use_shifted_average:
                blocks = []
                per_shift_responses = []
                for sx, sy in shift_pairs:
                    points = shifted_uniform_bz_mesh(nk, sx, sy)
                    weights = weights_for_points(points)
                    block = compute_target_bare_blocks(
                        spec=inputs.spec,
                        ansatz=inputs.ansatz,
                        q_model=q,
                        xi=xi,
                        k_points=points,
                        weights=weights,
                        config=inputs.config,
                        pairing_params=inputs.pairing_params,
                    )
                    blocks.append(block)
                    per_shift_responses.append(build_effective_from_blocks(block))
                response = average_bare_blocks_then_schur(blocks)
                shifted_payload = {
                    "enabled": True,
                    "average_order": "average_blocks_then_schur",
                    "shift_fractions": [float(value) for value in shift_fractions],
                    "shift_pairs": [[float(sx), float(sy)] for sx, sy in shift_pairs],
                    "num_shifted_meshes": len(shift_pairs),
                    "per_shift_summaries": per_shift_summaries(per_shift_responses, shift_pairs),
                    "valid_for_casimir_input": False,
                }
            else:
                block = compute_target_bare_blocks(
                    spec=inputs.spec,
                    ansatz=inputs.ansatz,
                    q_model=q,
                    xi=xi,
                    k_points=inputs.k_points,
                    weights=inputs.weights,
                    config=inputs.config,
                    pairing_params=inputs.pairing_params,
                )
                response = build_effective_from_blocks(block)
                shifted_payload = {
                    "enabled": False,
                    "average_order": "average_blocks_then_schur",
                    "shift_fractions": [float(value) for value in shift_fractions],
                    "shift_pairs": [[float(sx), float(sy)] for sx, sy in shift_pairs],
                    "num_shifted_meshes": 1,
                    "valid_for_casimir_input": False,
                }
            results.append(_result_payload(response=response, q_model=q, shifted=shifted_payload))
    if not results:
        raise ValueError("q_values and q_directions produced no scan results")
    first = results[0]
    return scan_payload(
        model_name=model_name,
        pairing_name=pairing_name,
        xi=xi,
        nk=nk,
        first_result=first,
        results=results,
        shifted_mesh_average=first["shifted_mesh_average"],
    )


def run_and_write_scan(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_scan(**kwargs)
    write_json(Path(output_dir) / "tmte_scan.json", payload)
    return payload


def debug_compare_component_reference(
    *,
    model_name: str,
    pairing_name: str,
    xi: float,
    q_value: float,
    nk: int,
    omega_eV: float | None = None,
) -> dict[str, Any]:
    """Return debug-only direct target-basis versus component contraction norms."""

    inputs = build_model_scan_inputs(model_name=model_name, pairing_name=pairing_name, xi=xi, omega_eV=omega_eV, nk=nk)
    q = np.asarray([float(q_value), 0.0], dtype=float)
    direct = compute_effective_target_response(
        spec=inputs.spec,
        ansatz=inputs.ansatz,
        q_model=q,
        xi=xi,
        k_points=inputs.k_points,
        weights=inputs.weights,
        config=inputs.config,
        pairing_params=inputs.pairing_params,
    )
    reference = compute_component_reference_effective(
        spec=inputs.spec,
        ansatz=inputs.ansatz,
        q_model=q,
        xi=xi,
        k_points=inputs.k_points,
        weights=inputs.weights,
        config=inputs.config,
        pairing_params=inputs.pairing_params,
    )
    return {
        "debug_only_component_reference": True,
        "direct_norm": float(np.linalg.norm(direct.schur.effective)),
        "reference_norm": float(np.linalg.norm(reference)),
        "difference_norm": float(np.linalg.norm(direct.schur.effective - reference)),
        "valid_for_casimir_input": False,
    }
