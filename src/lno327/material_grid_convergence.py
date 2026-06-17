"""Planning helpers for zero-mode and material-grid convergence audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .casimir_grid import xi_si_to_omega_eV
from .constants import EV_TO_J, HBAR


@dataclass(frozen=True)
class SmallQAuditPoint:
    n: int
    Q_nm_inv: float
    phi_deg: float
    temperature_K: float = 10.0


@dataclass(frozen=True)
class ZeroModeAuditPoint:
    omega_eV: float
    Q_nm_inv: float
    phi_deg: float
    temperature_K: float = 10.0


def default_small_q_points(*, temperature_K: float = 10.0, smoke: bool = False) -> list[SmallQAuditPoint]:
    n_values = [1, 2] if smoke else [1, 2, 4]
    q_values = [0.005, 0.02] if smoke else [0.005, 0.01, 0.02, 0.05]
    phi_values = [0.0, 90.0] if smoke else [0.0, 45.0, 90.0, 135.0]
    return [SmallQAuditPoint(n=n, Q_nm_inv=q, phi_deg=phi, temperature_K=temperature_K) for n in n_values for q in q_values for phi in phi_values]


def default_zero_mode_points(*, temperature_K: float = 10.0, smoke: bool = False) -> list[ZeroModeAuditPoint]:
    omega_values = [1e-4, 1e-3] if smoke else [1e-4, 3e-4, 1e-3, 3e-3]
    q_values = [0.01, 0.05] if smoke else [0.01, 0.05, 0.10]
    phi_values = [0.0, 90.0] if smoke else [0.0, 45.0, 90.0]
    return [
        ZeroModeAuditPoint(omega_eV=omega, Q_nm_inv=q, phi_deg=phi, temperature_K=temperature_K)
        for omega in omega_values
        for q in q_values
        for phi in phi_values
    ]


def q_nm_phi_to_si_model(Q_nm_inv: float, phi_deg: float, lattice_a_x_m: float, lattice_a_y_m: float) -> dict[str, float]:
    if Q_nm_inv <= 0.0:
        raise ValueError("Q_nm_inv must be positive; Q=0 is excluded")
    phi_rad = float(np.deg2rad(phi_deg))
    q_m_inv = float(Q_nm_inv) * 1.0e9
    qx = q_m_inv * float(np.cos(phi_rad))
    qy = q_m_inv * float(np.sin(phi_rad))
    return {
        "Q_nm_inv": float(Q_nm_inv),
        "Q_m_inv": q_m_inv,
        "phi_deg": float(phi_deg),
        "phi_rad": phi_rad,
        "Qx_m_inv": qx,
        "Qy_m_inv": qy,
        "q_model_x": qx * float(lattice_a_x_m),
        "q_model_y": qy * float(lattice_a_y_m),
    }


def omega_eV_to_xi_si_scalar(omega_eV: float) -> float:
    if omega_eV <= 0.0:
        raise ValueError("omega_eV must be positive")
    return float(omega_eV) * EV_TO_J / HBAR


def xi_si_to_omega_eV_scalar(xi_si: float) -> float:
    return float(xi_si_to_omega_eV(float(xi_si)))


def grid_convergence_plan() -> dict[str, Any]:
    return {
        "coarse": {"n_max": 8, "n_Q": 16, "n_phi": 8},
        "medium": {"n_max": 16, "n_Q": 24, "n_phi": 12},
        "fine": {"n_max": 32, "n_Q": 32, "n_phi": 16},
        "Q0_policy": "exclude endpoint Q=0 and use interior quadrature nodes",
        "n0_policy": "use extrapolated xi->0+ reflection matrix; do not divide by omega=0",
        "Q_max_convergence": "scan Q_max separately",
        "n_max_convergence": "scan n_max separately",
        "angular_radial_convergence": "scan n_Q and n_phi separately",
        "response_grid_strategy": "start with direct response grid for audit, then evaluate interpolation grid only after error controls exist",
    }


def stage5_13_metadata() -> dict[str, Any]:
    return {
        "Q0_handling_recommendation": "exclude endpoint Q=0 and use interior quadrature nodes",
        "zero_mode_recommendation": "use extrapolated xi->0+ reflection matrix, not direct division by zero",
        "n0_weight": 0.5,
        "no_production_energy": True,
        "no_force": True,
        "no_torque": True,
        "not_casimir_ready_claim": True,
    }


def classify_threshold(value: float, *, pass_threshold: float, monitor_threshold: float) -> str:
    if not np.isfinite(value):
        return "FAIL"
    if value < pass_threshold:
        return "PASS"
    if value < monitor_threshold:
        return "MONITOR"
    return "FAIL"
