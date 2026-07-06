"""Diagnostic-only finite-q BdG Schur Ward algebra localization helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.collective.validation import validate_physical_ward_identity
from lno327.response.finite_q_bdg import (
    finite_q_bdg_response_from_workspace,
    precompute_finite_q_bdg_workspace_from_model_ansatz,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def rectangular_ward_left(matrix: np.ndarray, omega_eV: float, q_model: tuple[float, float] | np.ndarray) -> np.ndarray:
    """Contract rows with the existing physical Ward left convention."""
    array = np.asarray(matrix, dtype=complex)
    if array.shape[0] != 3:
        raise ValueError("left Ward contraction requires the first axis to have length 3")
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model must have shape (2,)")
    return 1j * float(omega_eV) * array[0, :] + float(q[0]) * array[1, :] + float(q[1]) * array[2, :]


def rectangular_ward_right(matrix: np.ndarray, omega_eV: float, q_model: tuple[float, float] | np.ndarray) -> np.ndarray:
    """Contract columns with the existing physical Ward right convention."""
    array = np.asarray(matrix, dtype=complex)
    if array.shape[1] != 3:
        raise ValueError("right Ward contraction requires the second axis to have length 3")
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model must have shape (2,)")
    return 1j * float(omega_eV) * array[:, 0] - float(q[0]) * array[:, 1] - float(q[1]) * array[:, 2]


def _complex_payload(value: complex) -> dict[str, float]:
    scalar = complex(value)
    return {
        "real": float(np.real(scalar)),
        "imag": float(np.imag(scalar)),
        "abs": float(abs(scalar)),
    }


def _complex_vector_payload(vector: np.ndarray) -> list[dict[str, float]]:
    return [_complex_payload(value) for value in np.asarray(vector, dtype=complex).reshape(-1)]


def _r_candidates(delta0: float) -> tuple[tuple[str, np.ndarray], ...]:
    """Legacy diagnostic candidates retained for historical tests only.

    The primary localization result is analytic_identity below; these candidates
    must not be used to infer the gauge generator.
    """
    return (
        ("eta2_plus_1", np.asarray([0.0, 1.0], dtype=complex)),
        ("eta2_minus_1", np.asarray([0.0, -1.0], dtype=complex)),
        ("eta2_plus_delta0", np.asarray([0.0, delta0], dtype=complex)),
        ("eta2_minus_delta0", np.asarray([0.0, -delta0], dtype=complex)),
        ("eta2_plus_2delta0", np.asarray([0.0, 2.0 * delta0], dtype=complex)),
        ("eta2_minus_2delta0", np.asarray([0.0, -2.0 * delta0], dtype=complex)),
        ("eta2_plus_i", np.asarray([0.0, 1.0j], dtype=complex)),
        ("eta2_minus_i", np.asarray([0.0, -1.0j], dtype=complex)),
        ("eta2_plus_i_delta0", np.asarray([0.0, 1.0j * delta0], dtype=complex)),
        ("eta2_minus_i_delta0", np.asarray([0.0, -1.0j * delta0], dtype=complex)),
        ("eta2_plus_2i_delta0", np.asarray([0.0, 2.0j * delta0], dtype=complex)),
        ("eta2_minus_2i_delta0", np.asarray([0.0, -2.0j * delta0], dtype=complex)),
    )


def _classify_candidate(
    *,
    left_aa_norm: float,
    left_aeta_norm: float,
    right_aa_norm: float,
    right_etaa_norm: float,
    tolerance: float,
) -> str:
    aa_small = left_aa_norm <= tolerance and right_aa_norm <= tolerance
    mixed_small = left_aeta_norm <= tolerance and right_etaa_norm <= tolerance
    if aa_small and mixed_small:
        return "both_small"
    if aa_small and not mixed_small:
        return "first_small_second_large"
    if not aa_small and mixed_small:
        return "first_large_second_small"
    return "both_large"


def _candidate_payload(
    *,
    name: str,
    r_vector: np.ndarray,
    k_aa: np.ndarray,
    k_aeta: np.ndarray,
    k_etaa: np.ndarray,
    k_etaeta: np.ndarray,
    omega_eV: float,
    q_model: tuple[float, float],
    tolerance: float,
) -> dict[str, Any]:
    left_aa = rectangular_ward_left(k_aa, omega_eV, q_model) + r_vector @ k_etaa
    left_aeta = rectangular_ward_left(k_aeta, omega_eV, q_model) + r_vector @ k_etaeta
    right_aa = rectangular_ward_right(k_aa, omega_eV, q_model) + k_aeta @ r_vector
    right_etaa = rectangular_ward_right(k_etaa, omega_eV, q_model) + k_etaeta @ r_vector
    norms = {
        "left_aa_norm": float(np.linalg.norm(left_aa)),
        "left_aeta_norm": float(np.linalg.norm(left_aeta)),
        "right_aa_norm": float(np.linalg.norm(right_aa)),
        "right_etaa_norm": float(np.linalg.norm(right_etaa)),
    }
    return {
        "candidate_name": name,
        "R": _complex_vector_payload(r_vector),
        **norms,
        "max_norm": float(max(norms.values())),
        "classification": _classify_candidate(**norms, tolerance=tolerance),
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
    }


def _classify_analytic_identity(
    *,
    contact_aware_left_aa_norm: float,
    contact_aware_right_aa_norm: float,
    left_aeta_norm: float,
    right_etaa_norm: float,
    tolerance: float,
) -> str:
    aa_small = contact_aware_left_aa_norm <= tolerance and contact_aware_right_aa_norm <= tolerance
    mixed_small = left_aeta_norm <= tolerance and right_etaa_norm <= tolerance
    if aa_small and mixed_small:
        return "aa_small_mixed_small"
    if aa_small and not mixed_small:
        return "aa_small_mixed_large"
    if not aa_small and mixed_small:
        return "aa_large_mixed_small"
    return "aa_large_mixed_large"


def _analytic_identity_payload(
    *,
    k_aa: np.ndarray,
    k_direct: np.ndarray,
    k_aeta: np.ndarray,
    k_etaa: np.ndarray,
    k_etaeta: np.ndarray,
    omega_eV: float,
    q_model: tuple[float, float],
    delta0_eV: float,
    tolerance: float,
) -> dict[str, Any]:
    r_left = np.asarray([0.0, 2.0j * float(delta0_eV)], dtype=complex)
    r_right = np.asarray([0.0, -2.0j * float(delta0_eV)], dtype=complex)

    homogeneous_left_aa = rectangular_ward_left(k_aa, omega_eV, q_model) + r_left @ k_etaa
    homogeneous_right_aa = rectangular_ward_right(k_aa, omega_eV, q_model) + k_aeta @ r_right
    contact_aware_left_aa = rectangular_ward_left(k_aa, omega_eV, q_model) - rectangular_ward_left(
        k_direct, omega_eV, q_model
    ) + r_left @ k_etaa
    contact_aware_right_aa = rectangular_ward_right(k_aa, omega_eV, q_model) - rectangular_ward_right(
        k_direct, omega_eV, q_model
    ) + k_aeta @ r_right
    left_aeta = rectangular_ward_left(k_aeta, omega_eV, q_model) + r_left @ k_etaeta
    right_etaa = rectangular_ward_right(k_etaa, omega_eV, q_model) + k_etaeta @ r_right

    norms = {
        "homogeneous_left_aa_norm": float(np.linalg.norm(homogeneous_left_aa)),
        "homogeneous_right_aa_norm": float(np.linalg.norm(homogeneous_right_aa)),
        "contact_aware_left_aa_norm": float(np.linalg.norm(contact_aware_left_aa)),
        "contact_aware_right_aa_norm": float(np.linalg.norm(contact_aware_right_aa)),
        "left_aeta_norm": float(np.linalg.norm(left_aeta)),
        "right_etaa_norm": float(np.linalg.norm(right_etaa)),
    }
    return {
        "analytic_R_left": _complex_vector_payload(r_left),
        "analytic_R_right": _complex_vector_payload(r_right),
        "eta2_relation": "eta2 = delta0 * theta",
        "source_convention": "endpoint_average_delta_minus_plus_delta_plus_over_2delta0",
        "aa_identity": "contact_aware",
        **norms,
        "max_contact_aware_norm": float(
            max(
                norms["contact_aware_left_aa_norm"],
                norms["contact_aware_right_aa_norm"],
                norms["left_aeta_norm"],
                norms["right_etaa_norm"],
            )
        ),
        "classification": _classify_analytic_identity(
            contact_aware_left_aa_norm=norms["contact_aware_left_aa_norm"],
            contact_aware_right_aa_norm=norms["contact_aware_right_aa_norm"],
            left_aeta_norm=norms["left_aeta_norm"],
            right_etaa_norm=norms["right_etaa_norm"],
            tolerance=tolerance,
        ),
        "valid_for_casimir_input": False,
    }


@lru_cache(maxsize=4)
def run_bdg_schur_ward_algebra_localization(
    *,
    pairing_name: str,
    model_name: str = "symmetry_bdg_2band",
    nk: int = 9,
    q_model: tuple[float, float] = (0.02, 0.0),
    omega_eV: float = 0.01,
    delta0_eV: float = 0.1,
    phase_vertex: str = "bond_endpoint_gauge",
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Return compact Schur Ward algebra localization diagnostics without writing files."""
    model = get_finite_q_validation_model(model_name)
    model.require_pairing(pairing_name)
    pairing_params = model.build_pairing_params(delta0_eV)
    ansatz = model.build_ansatz(pairing_name, phase_vertex=phase_vertex)
    points = uniform_bz_mesh(int(nk))
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(omega_eV=float(omega_eV), temperature_K=10.0, eta_eV=1e-8, output_si=False)
    q = np.asarray(q_model, dtype=float)
    workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        q,
        points,
        weights,
        config,
        pairing_params,
        FiniteQEngineOptions(),
    )
    response = finite_q_bdg_response_from_workspace(workspace, config=config)
    k_aa = np.asarray(response.bare_total, dtype=complex)
    k_direct = np.asarray(response.direct, dtype=complex)
    k_aeta = np.asarray(response.em_collective_left, dtype=complex)
    k_etaa = np.asarray(response.collective_em_right, dtype=complex)
    k_etaeta = np.asarray(response.collective_total, dtype=complex)

    validator = validate_physical_ward_identity(k_aa, float(omega_eV), q, tolerance=tolerance)
    rectangular_left = rectangular_ward_left(k_aa, float(omega_eV), q)
    rectangular_right = rectangular_ward_right(k_aa, float(omega_eV), q)
    validator_reproduction = {
        "left_difference_norm": float(np.linalg.norm(rectangular_left - validator.left_residual)),
        "right_difference_norm": float(np.linalg.norm(rectangular_right - validator.right_residual)),
        "matches_existing_validator": bool(
            np.allclose(rectangular_left, validator.left_residual, rtol=0.0, atol=1e-14)
            and np.allclose(rectangular_right, validator.right_residual, rtol=0.0, atol=1e-14)
        ),
        "valid_for_casimir_input": False,
    }

    analytic_identity = _analytic_identity_payload(
        k_aa=k_aa,
        k_direct=k_direct,
        k_aeta=k_aeta,
        k_etaa=k_etaa,
        k_etaeta=k_etaeta,
        omega_eV=float(omega_eV),
        q_model=(float(q[0]), float(q[1])),
        delta0_eV=float(delta0_eV),
        tolerance=float(tolerance),
    )

    candidates = [
        _candidate_payload(
            name=name,
            r_vector=r_vector,
            k_aa=k_aa,
            k_aeta=k_aeta,
            k_etaa=k_etaa,
            k_etaeta=k_etaeta,
            omega_eV=float(omega_eV),
            q_model=(float(q[0]), float(q[1])),
            tolerance=float(tolerance),
        )
        for name, r_vector in _r_candidates(float(delta0_eV))
    ]
    best = min(candidates, key=lambda item: float(item["max_norm"]))

    return {
        "problem": "finite_q_bdg_schur_ward_algebra_localization",
        "model_name": model.name,
        "pairing_name": pairing_name,
        "run_config": {
            "nk": int(nk),
            "q_model": [float(q[0]), float(q[1])],
            "omega_eV": float(omega_eV),
            "delta0_eV": float(delta0_eV),
            "phase_vertex": phase_vertex,
            "tolerance": float(tolerance),
        },
        "matrix_shapes": {
            "K_AA": list(k_aa.shape),
            "K_Aeta": list(k_aeta.shape),
            "K_etaA": list(k_etaa.shape),
            "K_etaeta": list(k_etaeta.shape),
        },
        "direct_shape": list(k_direct.shape),
        "validator_reproduction": validator_reproduction,
        "analytic_identity": analytic_identity,
        "candidates": candidates,
        "best_candidate": best,
        "legacy_candidates_diagnostic_only": True,
        "valid_for_casimir_input": False,
    }
