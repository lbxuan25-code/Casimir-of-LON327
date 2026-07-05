"""Toy-model Casimir integration convergence helpers.

This module performs full Matsubara/Q/phi integration only for analytic toy
reflection matrices.  It does not use a real material response grid and does not
produce material predictions, force, or torque.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import C0, EV_TO_J, HBAR, KB
from .grid import matsubara_prime_weights, matsubara_xi_grid, polar_measure_weights, q_polar_grid
from .integrand import casimir_integrand_single_point, rotate_2x2_te_tm_toy_matrix, toy_zero_reflection


def xi_c_from_omega_eV(omega_c_eV: float) -> float:
    """Return xi_c from hbar*xi_c in eV."""

    if omega_c_eV <= 0.0:
        raise ValueError("omega_c_eV must be positive")
    return float(omega_c_eV) * EV_TO_J / HBAR


def _toy_cutoff_factor(xi_si: float, Q_m_inv: float, *, Qc_m_inv: float, xi_c_si: float) -> float:
    if Qc_m_inv <= 0.0:
        raise ValueError("Qc_m_inv must be positive")
    if xi_c_si <= 0.0:
        raise ValueError("xi_c_si must be positive")
    if xi_si < 0.0 or Q_m_inv < 0.0:
        raise ValueError("xi_si and Q_m_inv must be non-negative")
    return float(np.exp(-(float(Q_m_inv) / float(Qc_m_inv)) ** 2) / (1.0 + float(xi_si) / float(xi_c_si)))


def toy_isotropic_reflection_te_tm(
    xi_si: float,
    Q_m_inv: float,
    *,
    r0: float = 0.3,
    Qc_m_inv: float,
    xi_c_si: float,
) -> np.ndarray:
    """Return diagonal toy TE/TM reflection matrix in (s,p) order."""

    if not 0.0 <= abs(r0) < 1.0:
        raise ValueError("abs(r0) must be less than 1")
    factor = _toy_cutoff_factor(xi_si, Q_m_inv, Qc_m_inv=Qc_m_inv, xi_c_si=xi_c_si)
    return np.diag([-float(r0) * factor, float(r0) * factor]).astype(complex)


def toy_anisotropic_reflection_te_tm(
    xi_si: float,
    Q_m_inv: float,
    theta_rad: float,
    *,
    rs0: float = -0.25,
    rp0: float = 0.35,
    mixing0: float = 0.05,
    Qc_m_inv: float,
    xi_c_si: float,
) -> np.ndarray:
    """Return symmetric anisotropic toy reflector with toy rotation only."""

    factor = _toy_cutoff_factor(xi_si, Q_m_inv, Qc_m_inv=Qc_m_inv, xi_c_si=xi_c_si)
    base = factor * np.array([[float(rs0), float(mixing0)], [float(mixing0), float(rp0)]], dtype=complex)
    return rotate_2x2_te_tm_toy_matrix(base, theta_rad)


def _kappa(Q_m_inv: float, xi_si: float) -> float:
    return float(np.sqrt(float(Q_m_inv) ** 2 + (float(xi_si) / C0) ** 2))


def toy_integrand_single_point(
    xi_si: float,
    Q_m_inv: float,
    separation_m: float,
    theta_rad: float,
    model: str,
    **params: Any,
) -> complex:
    """Return single-point toy logdet integrand."""

    if model == "zero":
        r1 = toy_zero_reflection()
        r2 = toy_zero_reflection()
    elif model == "isotropic_identical":
        r1 = toy_isotropic_reflection_te_tm(xi_si, Q_m_inv, **params)
        r2 = r1
    elif model == "anisotropic_relative_rotation":
        r1 = toy_anisotropic_reflection_te_tm(xi_si, Q_m_inv, 0.0, **params)
        r2 = toy_anisotropic_reflection_te_tm(xi_si, Q_m_inv, theta_rad, **params)
    else:
        raise ValueError(f"unknown toy model: {model}")
    return casimir_integrand_single_point(r1, r2, _kappa(Q_m_inv, xi_si), separation_m)["logdet_integrand"]


def integrate_toy_free_energy_density(
    *,
    temperature_K: float,
    n_max: int,
    Q_max_m_inv: float,
    n_Q: int,
    n_phi: int,
    separation_m: float,
    theta_rad: float,
    model: str,
    Q_min_m_inv: float = 0.0,
    exclude_Q0: bool = True,
    **params: Any,
) -> dict[str, Any]:
    """Integrate toy free energy density over a finite scaffold grid."""

    xi_grid = matsubara_xi_grid(temperature_K, n_max)
    n_weights = matsubara_prime_weights(n_max)
    grid = q_polar_grid(Q_max_m_inv, n_Q, n_phi, q_min_m_inv=Q_min_m_inv)
    q_values = grid["Q_m_inv"]
    phi_values = grid["phi_rad"]
    if exclude_Q0 and len(q_values) > 0 and np.isclose(q_values[0], 0.0):
        q_values = q_values[1:]
    measure = polar_measure_weights(q_values, phi_values)
    matsubara_terms: list[complex] = []
    for n_weight, xi_si in zip(n_weights, xi_grid, strict=True):
        q_phi_sum = 0.0 + 0.0j
        for i, q in enumerate(q_values):
            for j, phi in enumerate(phi_values):
                q_phi_sum += measure[i, j] * toy_integrand_single_point(
                    float(xi_si),
                    float(q),
                    separation_m,
                    float(theta_rad),
                    model,
                    **params,
                )
        matsubara_terms.append(complex(n_weight * q_phi_sum))
    total = KB * float(temperature_K) * sum(matsubara_terms)
    return {
        "free_energy_density_J_m2": float(np.real(total)),
        "imag_part_J_m2": float(np.imag(total)),
        "grid": {
            "temperature_K": float(temperature_K),
            "n_max": int(n_max),
            "Q_max_m_inv": float(Q_max_m_inv),
            "n_Q": int(n_Q),
            "n_phi": int(n_phi),
            "separation_m": float(separation_m),
            "theta_rad": float(theta_rad),
            "Q0_excluded": bool(exclude_Q0),
            "num_Q_used": int(len(q_values)),
        },
        "components": {
            "matsubara_sum_terms": matsubara_terms,
            "measure_sum_scaffold": float(np.sum(measure)),
        },
        "scope": "toy_model_not_material_prediction",
    }


def _relative_changes(values: list[float]) -> list[float | None]:
    changes: list[float | None] = [None]
    for previous, current in zip(values[:-1], values[1:], strict=False):
        changes.append(float(abs(current - previous) / max(abs(previous), 1e-300)))
    return changes


def _scan_status(values: list[float]) -> str:
    if not np.all(np.isfinite(values)):
        return "FAIL"
    changes = [change for change in _relative_changes(values)[1:] if change is not None]
    if len(changes) < 2:
        return "PASS"
    return "PASS" if changes[-1] <= 1.25 * changes[0] else "MONITOR"


def convergence_scan_toy(
    *,
    temperature_K: float,
    n_max_values: list[int],
    Q_max_values_m_inv: list[float],
    n_Q_values: list[int],
    n_phi_values: list[int],
    separation_m: float,
    theta_rad: float,
    model: str,
    **params: Any,
) -> dict[str, Any]:
    """Run light toy convergence scans over one grid direction at a time."""

    base_n_max = max(n_max_values)
    base_Q_max = Q_max_values_m_inv[min(1, len(Q_max_values_m_inv) - 1)]
    base_n_Q = n_Q_values[min(1, len(n_Q_values) - 1)]
    base_n_phi = n_phi_values[min(1, len(n_phi_values) - 1)]

    def run(**overrides: Any) -> dict[str, Any]:
        kwargs = {
            "temperature_K": temperature_K,
            "n_max": base_n_max,
            "Q_max_m_inv": base_Q_max,
            "n_Q": base_n_Q,
            "n_phi": base_n_phi,
            "separation_m": separation_m,
            "theta_rad": theta_rad,
            "model": model,
            **params,
        }
        kwargs.update(overrides)
        return integrate_toy_free_energy_density(**kwargs)

    scans: dict[str, Any] = {}
    for name, values, key in (
        ("n_max", n_max_values, "n_max"),
        ("Q_max", Q_max_values_m_inv, "Q_max_m_inv"),
        ("n_Q", n_Q_values, "n_Q"),
        ("n_phi", n_phi_values, "n_phi"),
    ):
        results = [run(**{key: value}) for value in values]
        energies = [float(result["free_energy_density_J_m2"]) for result in results]
        scans[name] = {
            "values": values,
            "free_energy_density_J_m2": energies,
            "imag_part_J_m2": [float(result["imag_part_J_m2"]) for result in results],
            "relative_changes": _relative_changes(energies),
            "status": _scan_status(energies),
        }
    return scans


def toy_integration_metadata() -> dict[str, Any]:
    """Return scope metadata for toy-only full integration audit."""

    return {
        "toy_model_only": True,
        "not_material_prediction": True,
        "no_real_LNO327_energy": True,
        "no_force": True,
        "no_torque": True,
        "full_sum_integral_only_for_toy": True,
        "toy_rotation_only_not_physical_material_rotation": True,
        "next_requires_material_response_grid": True,
    }
