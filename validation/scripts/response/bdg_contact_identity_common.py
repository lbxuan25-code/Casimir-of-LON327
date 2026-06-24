"""Diagnostic-only helpers for the StageSC-0d Peierls contact audit.

This module deliberately consumes the production Peierls vertices without
changing response construction or any validation gate.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from bdg_bubble_ward_transfer_common import (  # noqa: E402
    K_POINTS,
    PAIRINGS,
    Q_MODEL_LIST,
    NambuPrefactor,
    config,
    pairing_delta,
    rho_vertex,
)
from lno327.bdg_finite_q_response import (  # noqa: E402
    bdg_finite_q_contact_vertex,
    bdg_finite_q_vector_vertex,
)
from lno327.conductivity import fermi_function  # noqa: E402
from lno327.pairing import PairingAmplitudes, bdg_hamiltonian  # noqa: E402
from lno327.tb_fourier import (  # noqa: E402
    peierls_hamiltonian_contact_vertex,
    peierls_hamiltonian_vector_vertex,
)


DIRECTIONS = ("x", "y")
CONTACT_IDENTITY_PASS = 1e-12
CONTACT_IDENTITY_MONITOR = 1e-10
SPATIAL_CLOSURE_PASS = 1e-8
SPATIAL_CLOSURE_MONITOR = 1e-6


def threshold_status(value: float, passed: float, monitor: float) -> str:
    """Classify a non-negative audit metric against strict pass/monitor limits."""

    if value < passed:
        return "PASSED"
    if value < monitor:
        return "MONITOR"
    return "FAILED"


def normal_contact_identity_residual(
    k_model: tuple[float, float] | np.ndarray,
    q_model: tuple[float, float] | np.ndarray,
    direction_j: str,
) -> np.ndarray:
    """Return q_i M_ij - [V_j(k+q/2,-q)-V_j(k-q/2,-q)]."""

    kx, ky = (float(value) for value in k_model)
    qx, qy = (float(value) for value in q_model)
    lhs = sum(
        component
        * peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
        for component, direction_i in zip((qx, qy), DIRECTIONS, strict=True)
    )
    rhs = peierls_hamiltonian_vector_vertex(
        kx + 0.5 * qx,
        ky + 0.5 * qy,
        -qx,
        -qy,
        direction_j,
    ) - peierls_hamiltonian_vector_vertex(
        kx - 0.5 * qx,
        ky - 0.5 * qy,
        -qx,
        -qy,
        direction_j,
    )
    return lhs - rhs


def bdg_contact_identity_residual(
    k_model: tuple[float, float] | np.ndarray,
    q_model: tuple[float, float] | np.ndarray,
    direction_j: str,
) -> np.ndarray:
    """Return the BdG two-photon contact Ward-identity residual."""

    kx, ky = (float(value) for value in k_model)
    qx, qy = (float(value) for value in q_model)
    rho = rho_vertex()
    lhs = sum(
        component * bdg_finite_q_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
        for component, direction_i in zip((qx, qy), DIRECTIONS, strict=True)
    )
    vertex_plus = bdg_finite_q_vector_vertex(
        kx + 0.5 * qx,
        ky + 0.5 * qy,
        -qx,
        -qy,
        direction_j,
    )
    vertex_minus = bdg_finite_q_vector_vertex(
        kx - 0.5 * qx,
        ky - 0.5 * qy,
        -qx,
        -qy,
        direction_j,
    )
    return lhs - (rho @ vertex_plus - vertex_minus @ rho)


def audit_pointwise_contact_identity(
    pairing: str,
    *,
    k_points: tuple[tuple[float, float], ...] = K_POINTS,
    q_model_list: tuple[tuple[float, float], ...] = Q_MODEL_LIST,
) -> dict[str, Any]:
    """Audit normal and BdG contact identities on the requested point set."""

    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}")
    rows: list[dict[str, Any]] = []
    for k_model in k_points:
        for q_model in q_model_list:
            for direction_j in DIRECTIONS:
                normal = normal_contact_identity_residual(k_model, q_model, direction_j)
                bdg = bdg_contact_identity_residual(k_model, q_model, direction_j)
                rows.append(
                    {
                        "pairing": pairing,
                        "k_model": list(k_model),
                        "q_model": list(q_model),
                        "j": direction_j,
                        "normal_contact_identity_max_abs": float(np.max(np.abs(normal))),
                        "normal_contact_identity_fro": float(np.linalg.norm(normal)),
                        "bdg_contact_identity_max_abs": float(np.max(np.abs(bdg))),
                        "bdg_contact_identity_fro": float(np.linalg.norm(bdg)),
                        "particle_block_max_abs": float(np.max(np.abs(bdg[:4, :4]))),
                        "hole_block_max_abs": float(np.max(np.abs(bdg[4:, 4:]))),
                        "offdiag_block_max_abs": float(
                            max(np.max(np.abs(bdg[:4, 4:])), np.max(np.abs(bdg[4:, :4])))
                        ),
                    }
                )
    by_direction: dict[str, Any] = {}
    for direction_j in DIRECTIONS:
        selected = [row for row in rows if row["j"] == direction_j]
        normal_max = max(float(row["normal_contact_identity_max_abs"]) for row in selected)
        bdg_max = max(float(row["bdg_contact_identity_max_abs"]) for row in selected)
        by_direction[direction_j] = {
            "normal_contact_identity_max_abs": normal_max,
            "normal_contact_identity_fro": float(
                np.sqrt(sum(float(row["normal_contact_identity_fro"]) ** 2 for row in selected))
            ),
            "bdg_contact_identity_max_abs": bdg_max,
            "bdg_contact_identity_fro": float(
                np.sqrt(sum(float(row["bdg_contact_identity_fro"]) ** 2 for row in selected))
            ),
            "particle_block_max_abs": max(float(row["particle_block_max_abs"]) for row in selected),
            "hole_block_max_abs": max(float(row["hole_block_max_abs"]) for row in selected),
            "offdiag_block_max_abs": max(float(row["offdiag_block_max_abs"]) for row in selected),
            "status": threshold_status(
                max(normal_max, bdg_max),
                CONTACT_IDENTITY_PASS,
                CONTACT_IDENTITY_MONITOR,
            ),
        }
    max_abs = max(
        max(float(row["normal_contact_identity_max_abs"]), float(row["bdg_contact_identity_max_abs"]))
        for row in rows
    )
    return {
        "pairing": pairing,
        "delta0_eV": 0.04,
        "rows": rows,
        "by_direction": by_direction,
        "normal_contact_identity_max_abs": max(
            float(row["normal_contact_identity_max_abs"]) for row in rows
        ),
        "bdg_contact_identity_max_abs": max(float(row["bdg_contact_identity_max_abs"]) for row in rows),
        "max_contact_identity_abs": max_abs,
        "status": threshold_status(max_abs, CONTACT_IDENTITY_PASS, CONTACT_IDENTITY_MONITOR),
    }


def uniform_periodic_bz_grid(n_grid: int) -> np.ndarray:
    """Return the StageSC-0d periodic grid 2*pi*(n,m)/N."""

    if n_grid <= 0:
        raise ValueError("n_grid must be positive")
    values = 2.0 * np.pi * np.arange(n_grid, dtype=float) / float(n_grid)
    return np.array([(kx, ky) for kx in values for ky in values], dtype=float)


def spatial_contact_closure(
    pairing: str,
    n_grid: int,
    q_model: tuple[float, float] | np.ndarray,
    *,
    delta0_eV: float = 0.04,
    omega_eV: float = 0.01,
    temperature_K: float = 10.0,
    eta_eV: float = 1e-8,
) -> dict[str, Any]:
    """Compute the StageSC-0c Vx/Vy equal-time plus existing-contact closure."""

    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}")
    qx, qy = (float(value) for value in q_model)
    amp = PairingAmplitudes(delta0_eV=delta0_eV)
    cfg = config(omega_eV, temperature_K, eta_eV)
    rho = rho_vertex()
    points = uniform_periodic_bz_grid(n_grid)
    weight = 1.0 / float(n_grid**2)
    equal_time = {direction: 0.0 + 0.0j for direction in DIRECTIONS}
    direct = {
        (direction_i, direction_j): 0.0 + 0.0j
        for direction_i in DIRECTIONS
        for direction_j in DIRECTIONS
    }

    for kx_value, ky_value in points:
        kx, ky = float(kx_value), float(ky_value)
        k_minus = (kx - 0.5 * qx, ky - 0.5 * qy)
        k_plus = (kx + 0.5 * qx, ky + 0.5 * qy)
        delta_minus = pairing_delta(pairing, *k_minus, amp)
        delta_plus = pairing_delta(pairing, *k_plus, amp)
        delta_mid = pairing_delta(pairing, kx, ky, amp)
        energies_minus, states_minus = np.linalg.eigh(bdg_hamiltonian(*k_minus, delta_minus))
        energies_plus, states_plus = np.linalg.eigh(bdg_hamiltonian(*k_plus, delta_plus))
        occupations_minus = fermi_function(energies_minus, cfg.fermi_level_eV, cfg.temperature_eV)
        occupations_plus = fermi_function(energies_plus, cfg.fermi_level_eV, cfg.temperature_eV)
        rho_band = states_minus.conjugate().T @ rho @ states_plus
        occupation_difference = occupations_minus[:, None] - occupations_plus[None, :]
        for direction_j in DIRECTIONS:
            reverse_vertex = bdg_finite_q_vector_vertex(kx, ky, -qx, -qy, direction_j)
            reverse_band = states_plus.conjugate().T @ reverse_vertex @ states_minus
            equal_time[direction_j] += (
                NambuPrefactor
                * weight
                * np.sum(occupation_difference * rho_band * reverse_band.T)
            )

        energies_mid, states_mid = np.linalg.eigh(bdg_hamiltonian(kx, ky, delta_mid))
        occupations_mid = fermi_function(energies_mid, cfg.fermi_level_eV, cfg.temperature_eV)
        for direction_i in DIRECTIONS:
            for direction_j in DIRECTIONS:
                contact = bdg_finite_q_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
                contact_band = states_mid.conjugate().T @ contact @ states_mid
                thermal_expectation = NambuPrefactor * np.sum(occupations_mid * np.diag(contact_band))
                # This is exactly the existing production direct convention: -<M_ij>.
                direct[(direction_i, direction_j)] -= weight * thermal_expectation

    channels: dict[str, Any] = {}
    for direction_j, channel in zip(DIRECTIONS, ("Vx", "Vy"), strict=True):
        qd_existing = qx * direct[("x", direction_j)] + qy * direct[("y", direction_j)]
        residual = equal_time[direction_j] + qd_existing
        channels[channel] = {
            "E_B": complex(equal_time[direction_j]),
            "qD_existing": complex(qd_existing),
            "closure_residual": complex(residual),
            "closure_abs": float(abs(residual)),
        }
    max_abs = max(float(row["closure_abs"]) for row in channels.values())
    return {
        "pairing": pairing,
        "N": int(n_grid),
        "q_model": [qx, qy],
        **channels,
        "max_spatial_contact_closure_abs": max_abs,
        "status": threshold_status(max_abs, SPATIAL_CLOSURE_PASS, SPATIAL_CLOSURE_MONITOR),
    }


def assess_stageSC_0d(
    part_a_max_abs: float,
    part_b_max_abs: float,
    fixed_q_finest_max_abs: float,
) -> tuple[str, str, str]:
    """Return overall status, dominant failure, and bounded interpretation."""

    part_a_status = threshold_status(
        part_a_max_abs,
        CONTACT_IDENTITY_PASS,
        CONTACT_IDENTITY_MONITOR,
    )
    part_b_status = threshold_status(
        part_b_max_abs,
        SPATIAL_CLOSURE_PASS,
        SPATIAL_CLOSURE_MONITOR,
    )
    fixed_status = threshold_status(
        fixed_q_finest_max_abs,
        SPATIAL_CLOSURE_PASS,
        SPATIAL_CLOSURE_MONITOR,
    )
    if part_a_status != "PASSED":
        return part_a_status, "contact_vertex_identity", "contact_vertex_formula_or_bdg_hole_routing_failed"
    if part_b_status != "PASSED":
        return (
            part_b_status,
            "shift_invariant_quadrature",
            "contact_formula_passed_but_shift_invariant_quadrature_closure_failed",
        )
    if fixed_status != "PASSED":
        return (
            fixed_status,
            "fixed_q_quadrature",
            "contact formula passed; fixed-q quadrature/contact closure implementation remains unresolved",
        )
    return "PASSED", "none", "contact identity and tested periodic-grid closures passed"

