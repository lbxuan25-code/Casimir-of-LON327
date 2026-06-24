"""Diagnostic-only helpers for the StageSC-0e shifted-grid assembly audit."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from bdg_bubble_ward_transfer_common import (  # noqa: E402
    PAIRINGS,
    NambuPrefactor,
    config,
    pairing_delta,
    rho_vertex,
)
from lno327.bdg_finite_q_response import (  # noqa: E402
    bdg_finite_q_contact_vertex,
    bdg_finite_q_vector_vertex,
)
from lno327.conductivity import KuboConfig, fermi_function  # noqa: E402
from lno327.pairing import PairingAmplitudes, bdg_hamiltonian  # noqa: E402


DIRECTIONS = ("x", "y")
SOURCE_CHANNELS = ("Vx", "Vy")
SHIFTED_DIRECT_PASS = 1e-10
SHIFTED_DIRECT_MONITOR = 1e-8
BAND_SHIFTED_PASS = 1e-8
BAND_SHIFTED_MONITOR = 1e-6


def matrix_fermi_function(hamiltonian: np.ndarray, cfg: KuboConfig) -> np.ndarray:
    """Return f(H) by Hermitian eigendecomposition."""

    matrix = np.asarray(hamiltonian, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be a square matrix")
    if not np.allclose(matrix, matrix.conjugate().T):
        raise ValueError("hamiltonian must be Hermitian")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    occupations = fermi_function(eigenvalues, cfg.fermi_level_eV, cfg.temperature_eV)
    return (eigenvectors @ np.diag(occupations) @ eigenvectors.conjugate().T).astype(complex)


def shifted_trace_direct_terms(
    hamiltonian: np.ndarray,
    rho: np.ndarray,
    vector_plus: np.ndarray,
    vector_minus: np.ndarray,
    q_contact: np.ndarray,
    cfg: KuboConfig,
) -> tuple[complex, complex]:
    """Return the shifted-trace remainder and production-sign direct term."""

    f_hamiltonian = matrix_fermi_function(hamiltonian, cfg)
    shifted = NambuPrefactor * np.trace(
        f_hamiltonian @ (rho @ vector_plus - vector_minus @ rho)
    )
    direct = -NambuPrefactor * np.trace(f_hamiltonian @ q_contact)
    return complex(shifted), complex(direct)


def commensurate_grid_spec(
    n_grid: int,
    grid_type: str,
    *,
    diagonal: bool = False,
) -> dict[str, Any]:
    """Return q and q/2 in grid-spacing units for a periodic uniform grid."""

    if n_grid <= 0:
        raise ValueError("n_grid must be positive")
    if grid_type not in {"half_step_commensurate", "grid_step_commensurate"}:
        raise ValueError("unknown commensurate grid_type")
    spacing = 2.0 * np.pi / float(n_grid)
    multiplier = 1.0 if grid_type == "half_step_commensurate" else 2.0
    q_component = multiplier * spacing
    q_model = [q_component, q_component if diagonal else 0.0]
    shift_steps = [value / (2.0 * spacing) for value in q_model]
    return {
        "grid_type": grid_type,
        "N": int(n_grid),
        "grid_spacing": spacing,
        "q_model": q_model,
        "q_half_in_grid_steps": shift_steps,
        "q_half_lands_on_grid": all(
            abs(value - round(value)) < 1e-12 for value in shift_steps
        ),
    }


def uniform_periodic_bz_grid(n_grid: int) -> np.ndarray:
    """Return points (2*pi*n/N, 2*pi*m/N), n,m=0,...,N-1."""

    values = 2.0 * np.pi * np.arange(n_grid, dtype=float) / float(n_grid)
    return np.array([(kx, ky) for kx in values for ky in values], dtype=float)


def case_dominant_failure(
    shifted_plus_direct_abs: float,
    band_minus_shifted_abs: float,
    band_plus_direct_abs: float,
) -> str:
    """Route a case failure without blaming the band remainder for direct mismatch."""

    if shifted_plus_direct_abs >= SHIFTED_DIRECT_MONITOR:
        return "direct_expectation_mismatch"
    if band_minus_shifted_abs >= BAND_SHIFTED_MONITOR:
        return "band_vs_shifted_remainder"
    if band_plus_direct_abs >= BAND_SHIFTED_MONITOR:
        return "quadrature_resolution"
    return "none"


def assembly_case_status(
    shifted_plus_direct_abs: float,
    band_minus_shifted_abs: float,
    band_plus_direct_abs: float,
) -> str:
    if (
        shifted_plus_direct_abs < SHIFTED_DIRECT_PASS
        and band_minus_shifted_abs < BAND_SHIFTED_PASS
        and band_plus_direct_abs < BAND_SHIFTED_PASS
    ):
        return "PASSED"
    if (
        shifted_plus_direct_abs < SHIFTED_DIRECT_MONITOR
        and band_minus_shifted_abs < BAND_SHIFTED_MONITOR
        and band_plus_direct_abs < BAND_SHIFTED_MONITOR
    ):
        return "MONITOR"
    return "FAILED"


def audit_shifted_grid_assembly(
    pairing: str,
    n_grid: int,
    q_model: tuple[float, float] | list[float] | np.ndarray,
    grid_type: str,
    *,
    delta0_eV: float = 0.04,
    omega_eV: float = 0.01,
    temperature_K: float = 10.0,
    eta_eV: float = 1e-8,
) -> list[dict[str, Any]]:
    """Compute E_band, E_shifted, and qD for both spatial source channels."""

    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}")
    if grid_type not in {"half_step_commensurate", "grid_step_commensurate", "fixed_q"}:
        raise ValueError("unknown grid_type")
    qx, qy = (float(value) for value in q_model)
    amp = PairingAmplitudes(delta0_eV=delta0_eV)
    cfg = config(omega_eV, temperature_K, eta_eV)
    rho = rho_vertex()
    weight = 1.0 / float(n_grid**2)
    e_band = {direction: 0.0 + 0.0j for direction in DIRECTIONS}
    e_band_shortcut = {direction: 0.0 + 0.0j for direction in DIRECTIONS}
    e_shifted = {direction: 0.0 + 0.0j for direction in DIRECTIONS}
    qd_existing = {direction: 0.0 + 0.0j for direction in DIRECTIONS}
    orientation_vertex_max = {direction: 0.0 for direction in DIRECTIONS}

    for kx_value, ky_value in uniform_periodic_bz_grid(n_grid):
        kx, ky = float(kx_value), float(ky_value)
        k_minus = (kx - 0.5 * qx, ky - 0.5 * qy)
        k_plus = (kx + 0.5 * qx, ky + 0.5 * qy)
        h_minus = bdg_hamiltonian(*k_minus, pairing_delta(pairing, *k_minus, amp))
        h_plus = bdg_hamiltonian(*k_plus, pairing_delta(pairing, *k_plus, amp))
        energies_minus, states_minus = np.linalg.eigh(h_minus)
        energies_plus, states_plus = np.linalg.eigh(h_plus)
        occupations_minus = fermi_function(energies_minus, cfg.fermi_level_eV, cfg.temperature_eV)
        occupations_plus = fermi_function(energies_plus, cfg.fermi_level_eV, cfg.temperature_eV)
        occupation_difference = occupations_minus[:, None] - occupations_plus[None, :]
        rho_band = states_minus.conjugate().T @ rho @ states_plus

        h_mid = bdg_hamiltonian(kx, ky, pairing_delta(pairing, kx, ky, amp))
        f_mid = matrix_fermi_function(h_mid, cfg)
        for direction_j in DIRECTIONS:
            forward_vertex = bdg_finite_q_vector_vertex(kx, ky, qx, qy, direction_j)
            reverse_vertex = bdg_finite_q_vector_vertex(kx, ky, -qx, -qy, direction_j)
            forward_band = states_minus.conjugate().T @ forward_vertex @ states_plus
            reverse_band = states_plus.conjugate().T @ reverse_vertex @ states_minus
            orientation_vertex_max[direction_j] = max(
                orientation_vertex_max[direction_j],
                float(np.max(np.abs(reverse_band.T - np.conjugate(forward_band)))),
            )
            e_band[direction_j] += (
                NambuPrefactor
                * weight
                * np.sum(occupation_difference * rho_band * reverse_band.T)
            )
            e_band_shortcut[direction_j] += (
                NambuPrefactor
                * weight
                * np.sum(occupation_difference * rho_band * np.conjugate(forward_band))
            )

            vector_plus = bdg_finite_q_vector_vertex(
                kx + 0.5 * qx,
                ky + 0.5 * qy,
                -qx,
                -qy,
                direction_j,
            )
            vector_minus = bdg_finite_q_vector_vertex(
                kx - 0.5 * qx,
                ky - 0.5 * qy,
                -qx,
                -qy,
                direction_j,
            )
            q_contact = (
                qx * bdg_finite_q_contact_vertex(kx, ky, qx, qy, "x", direction_j)
                + qy * bdg_finite_q_contact_vertex(kx, ky, qx, qy, "y", direction_j)
            )
            shifted_value = NambuPrefactor * np.trace(
                f_mid @ (rho @ vector_plus - vector_minus @ rho)
            )
            direct_value = -NambuPrefactor * np.trace(f_mid @ q_contact)
            e_shifted[direction_j] += weight * shifted_value
            qd_existing[direction_j] += weight * direct_value

    spacing = 2.0 * np.pi / float(n_grid)
    shift_steps = [qx / (2.0 * spacing), qy / (2.0 * spacing)]
    rows: list[dict[str, Any]] = []
    for direction_j, source_channel in zip(DIRECTIONS, SOURCE_CHANNELS, strict=True):
        band_minus_shifted = e_band[direction_j] - e_shifted[direction_j]
        shifted_plus_direct = e_shifted[direction_j] + qd_existing[direction_j]
        band_plus_direct = e_band[direction_j] + qd_existing[direction_j]
        metrics = (
            float(abs(shifted_plus_direct)),
            float(abs(band_minus_shifted)),
            float(abs(band_plus_direct)),
        )
        orientation_difference = e_band[direction_j] - e_band_shortcut[direction_j]
        rows.append(
            {
                "pairing": pairing,
                "grid_type": grid_type,
                "N": int(n_grid),
                "grid_spacing": spacing,
                "q_model": [qx, qy],
                "q_half_in_grid_steps": shift_steps,
                "q_half_lands_on_grid": all(
                    abs(value - round(value)) < 1e-12 for value in shift_steps
                ),
                "source_channel": source_channel,
                "E_band": complex(e_band[direction_j]),
                "E_band_shortcut": complex(e_band_shortcut[direction_j]),
                "E_band_orientation_diff": float(abs(orientation_difference)),
                "E_band_orientation_diff_complex": complex(orientation_difference),
                "right_vertex_orientation_max_abs": orientation_vertex_max[direction_j],
                "E_shifted": complex(e_shifted[direction_j]),
                "qD_existing": complex(qd_existing[direction_j]),
                "E_band_minus_E_shifted": complex(band_minus_shifted),
                "E_band_minus_E_shifted_abs": metrics[1],
                "E_shifted_plus_qD": complex(shifted_plus_direct),
                "E_shifted_plus_qD_abs": metrics[0],
                "E_band_plus_qD": complex(band_plus_direct),
                "E_band_plus_qD_abs": metrics[2],
                "dominant_failure": case_dominant_failure(*metrics),
                "status": assembly_case_status(*metrics),
            }
        )
    return rows
