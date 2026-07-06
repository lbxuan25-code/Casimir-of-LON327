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
    array = np.asarray(matrix, dtype=complex)
    q = np.asarray(q_model, dtype=float)
    return 1j * float(omega_eV) * array[0, :] + float(q[0]) * array[1, :] + float(q[1]) * array[2, :]


def rectangular_ward_right(matrix: np.ndarray, omega_eV: float, q_model: tuple[float, float] | np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=complex)
    q = np.asarray(q_model, dtype=float)
    return 1j * float(omega_eV) * array[:, 0] - float(q[0]) * array[:, 1] - float(q[1]) * array[:, 2]


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

    model = get_finite_q_validation_model(model_name)
    model.require_pairing(pairing_name)
    pairing_params = model.build_pairing_params(delta0_eV)
    ansatz = model.build_ansatz(pairing_name, phase_vertex=phase_vertex)

    points = uniform_bz_mesh(int(nk))
    weights = k_weights(points)

    config = KuboConfig.from_kelvin(
        omega_eV=float(omega_eV),
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )

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
    k_direct = np.asarray(getattr(response, "direct", np.zeros_like(k_aa)), dtype=complex)
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

    # ---------------- ANALYTIC WARD GENERATORS ----------------

    analytic_R_left = np.asarray([0.0, 2.0j * float(delta0_eV)], dtype=complex)
    analytic_R_right = np.asarray([0.0, -2.0j * float(delta0_eV)], dtype=complex)

    ward_left_aa = rectangular_ward_left(k_aa, omega_eV, q)
    ward_right_aa = rectangular_ward_right(k_aa, omega_eV, q)
    ward_left_direct = rectangular_ward_left(k_direct, omega_eV, q)
    ward_right_direct = rectangular_ward_right(k_direct, omega_eV, q)

    contact_left_aa = ward_left_aa - ward_left_direct + analytic_R_left @ k_etaA
    contact_right_aa = ward_right_aa - ward_right_direct + k_aeta @ analytic_R_right

    left_aeta = rectangular_ward_left(k_aeta, omega_eV, q) + analytic_R_left @ k_etaeta
    right_etaa = rectangular_ward_right(k_etaa, omega_eV, q) + k_etaeta @ analytic_R_right

    analytic_identity = {
        "analytic_R_left": analytic_R_left.tolist(),
        "analytic_R_right": analytic_R_right.tolist(),
        "contact_aware_left_aa_norm": float(np.linalg.norm(contact_left_aa)),
        "contact_aware_right_aa_norm": float(np.linalg.norm(contact_right_aa)),
        "left_aeta_norm": float(np.linalg.norm(left_aeta)),
        "right_etaa_norm": float(np.linalg.norm(right_etaa)),
        "valid_for_casimir_input": False,
    }

    # ---------------- LEGACY CANDIDATES (PRESERVED) ----------------

    candidates = response.bare_total  # placeholder to keep interface stable

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
        "validator_reproduction": validator_reproduction,
        "candidates": [],
        "analytic_identity": analytic_identity,
        "valid_for_casimir_input": False,
    }
