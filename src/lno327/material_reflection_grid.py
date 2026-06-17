"""Small real-material reflection-grid prototype helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .casimir_grid import matsubara_xi_grid, xi_si_to_omega_eV


@dataclass(frozen=True)
class MaterialReflectionGridPoint:
    """One discrete real-material prototype point."""

    n: int
    Q_nm_inv: float
    phi_deg: float
    temperature_K: float = 10.0


def grid_point_to_si_and_model_q(
    point: MaterialReflectionGridPoint,
    lattice_a_x_m: float,
    lattice_a_y_m: float,
) -> dict[str, float | int]:
    """Convert (n,Q,phi) point to SI wave vector and model q."""

    if point.n <= 0:
        raise ValueError("Stage 5.11 excludes n=0; point.n must be positive")
    if point.Q_nm_inv <= 0.0:
        raise ValueError("Stage 5.11 excludes Q=0; Q_nm_inv must be positive")
    if lattice_a_x_m <= 0.0 or lattice_a_y_m <= 0.0:
        raise ValueError("lattice constants must be positive")
    q_si = float(point.Q_nm_inv) * 1.0e9
    phi_rad = float(np.deg2rad(point.phi_deg))
    qx_si = q_si * float(np.cos(phi_rad))
    qy_si = q_si * float(np.sin(phi_rad))
    xi = float(matsubara_xi_grid(point.temperature_K, point.n)[point.n])
    omega_eV = float(xi_si_to_omega_eV(xi))
    return {
        "n": int(point.n),
        "temperature_K": float(point.temperature_K),
        "Q_nm_inv": float(point.Q_nm_inv),
        "Q_m_inv": q_si,
        "phi_deg": float(point.phi_deg),
        "phi_rad": phi_rad,
        "Qx_m_inv": qx_si,
        "Qy_m_inv": qy_si,
        "q_model_x": qx_si * float(lattice_a_x_m),
        "q_model_y": qy_si * float(lattice_a_y_m),
        "omega_eV": omega_eV,
        "xi_si": xi,
    }


def default_stage5_11_points(*, smoke: bool = False, temperature_K: float = 10.0) -> list[MaterialReflectionGridPoint]:
    """Return default or smoke Stage 5.11 prototype points."""

    if smoke:
        n_values = [1, 2]
        q_values = [0.05, 0.10]
        phi_values = [0.0, 90.0]
    else:
        n_values = [1, 2, 4]
        q_values = [0.05, 0.10, 0.20]
        phi_values = [0.0, 45.0, 90.0, 135.0]
    return [
        MaterialReflectionGridPoint(n=n, Q_nm_inv=q, phi_deg=phi, temperature_K=temperature_K)
        for n in n_values
        for q in q_values
        for phi in phi_values
    ]


def complex_matrix_to_jsonable(matrix: np.ndarray) -> list[list[dict[str, float]]]:
    """Serialize a complex matrix as explicit real/imag fields."""

    array = np.asarray(matrix, dtype=complex)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return [[{"re": float(value.real), "im": float(value.imag)} for value in row] for row in array]


def material_reflection_grid_prototype_metadata() -> dict[str, Any]:
    """Return Stage 5.11 scope metadata."""

    return {
        "real_material_discrete_points": True,
        "full_integration_grid": False,
        "no_full_matsubara_sum": True,
        "no_full_Q_integral": True,
        "no_energy_output": True,
        "no_force_output": True,
        "no_torque_output": True,
        "not_production": True,
        "n0_excluded": True,
        "Q0_excluded": True,
        "q_model_conversion": "q_model_i = Q_i * a_i",
        "required_next_step": "material-grid convergence and response-cost audit",
    }
