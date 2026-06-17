"""Small real-material energy prototype from a stored reflection grid.

This module only reads Stage 5.11 reflection matrices and performs sparse
prototype quadrature.  It does not rerun response and does not produce
production energy, force, or torque.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .casimir_grid import kappa_si, polar_measure_weights
from .casimir_integrand import casimir_integrand_single_point
from .constants import KB

PASSED_STAGE5_11_STATUS = "STAGE5_11_REAL_MATERIAL_REFLECTION_GRID_PROTOTYPE_PASSED"

REQUIRED_WARNINGS = {
    "n0_excluded": "n0_excluded",
    "zero_mode_not_included": "zero_mode_not_included",
    "matsubara_grid_incomplete": "matsubara_grid_incomplete",
    "angular_grid_sparse": "angular_grid_sparse",
    "Q_grid_sparse": "Q_grid_sparse",
    "not_production_quadrature": "not_production_quadrature",
    "not_physical_prediction": "not_physical_prediction",
}


def load_stage5_11_reflection_grid(path: Path, *, allow_monitor: bool = True, require_no_fail: bool = True) -> dict[str, Any]:
    """Load a Stage 5.11 reflection grid and keep usable PASS/MONITOR points."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    status = data.get("diagnostic_status", {}).get("stage5_11_status")
    if status != PASSED_STAGE5_11_STATUS:
        raise ValueError("input must have STAGE5_11_REAL_MATERIAL_REFLECTION_GRID_PROTOTYPE_PASSED status")
    rows = list(data.get("point_results", []))
    if require_no_fail and any(row.get("status") == "FAIL" for row in rows):
        raise ValueError("input contains FAIL point_results")
    allowed = {"PASS", "MONITOR"} if allow_monitor else {"PASS"}
    usable = [row for row in rows if row.get("status") in allowed]
    if not usable:
        raise ValueError("input contains no usable PASS/MONITOR point_results")
    return {**data, "point_results": usable}


def _jsonable_complex(value: Any) -> complex:
    if isinstance(value, dict):
        return complex(float(value.get("re", value.get("real", 0.0))), float(value.get("im", value.get("imag", 0.0))))
    return complex(value)


def jsonable_complex_matrix_to_numpy(value: Any) -> np.ndarray:
    """Deserialize Stage 5.11 complex matrix JSON into a complex ndarray."""

    matrix = np.array([[_jsonable_complex(item) for item in row] for row in value], dtype=complex)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return matrix


def prototype_polar_weights(Q_values_m_inv: np.ndarray, phi_values_rad: np.ndarray) -> dict[str, Any]:
    """Return sparse scaffold polar weights for Q dQ dphi/(2*pi)^2."""

    q_values = np.asarray(Q_values_m_inv, dtype=float)
    phi_values = np.asarray(phi_values_rad, dtype=float)
    weights = polar_measure_weights(q_values, phi_values)
    return {
        "Q_m_inv": q_values,
        "phi_rad": phi_values,
        "weights": weights,
        "weight_by_pair": {
            (float(q), float(phi)): float(weights[i, j])
            for i, q in enumerate(q_values)
            for j, phi in enumerate(phi_values)
        },
    }


def _point_logdet(row: dict[str, Any], separation_m: float) -> complex:
    reflection = jsonable_complex_matrix_to_numpy(row["reflection_TE_TM"])
    if "kappa_m_inv" in row:
        kappa = float(row["kappa_m_inv"])
    else:
        kappa = float(kappa_si(float(row["Q_m_inv"]), float(row["xi_si"])))
    package = casimir_integrand_single_point(reflection, reflection, kappa, separation_m)
    return complex(package["logdet_integrand"])


def _group_add(group: dict[str, complex], key: Any, value: complex) -> None:
    group[str(key)] = group.get(str(key), 0.0 + 0.0j) + value


def integrate_small_real_material_energy_prototype(
    data: dict[str, Any],
    *,
    separation_m: float,
    allow_monitor: bool = True,
) -> dict[str, Any]:
    """Integrate the stored sparse real-material reflection grid."""

    if separation_m <= 0.0:
        raise ValueError("separation_m must be positive")
    rows = [
        row
        for row in data.get("point_results", [])
        if row.get("status") == "PASS" or (allow_monitor and row.get("status") == "MONITOR")
    ]
    if not rows:
        raise ValueError("no usable point_results")
    q_values = np.array(sorted({float(row["Q_m_inv"]) for row in rows}), dtype=float)
    phi_values = np.array(sorted({float(row["phi_rad"]) for row in rows}), dtype=float)
    weights = prototype_polar_weights(q_values, phi_values)["weight_by_pair"]
    partial_by_n: dict[str, complex] = {}
    partial_by_Q: dict[str, complex] = {}
    partial_by_phi: dict[str, complex] = {}
    total = 0.0 + 0.0j
    for row in rows:
        n_weight = 0.5 if int(row["n"]) == 0 else 1.0
        logdet = _point_logdet(row, separation_m)
        weight = weights[(float(row["Q_m_inv"]), float(row["phi_rad"]))]
        contribution = KB * float(row["temperature_K"]) * n_weight * weight * logdet
        total += contribution
        _group_add(partial_by_n, int(row["n"]), contribution)
        _group_add(partial_by_Q, float(row["Q_nm_inv"]), contribution)
        _group_add(partial_by_phi, float(row["phi_deg"]), contribution)
    warnings = list(REQUIRED_WARNINGS.values())
    return {
        "F_proto_over_area_J_m2": complex(total),
        "imag_part_J_m2": float(total.imag),
        "partial_by_n": partial_by_n,
        "partial_by_Q": partial_by_Q,
        "partial_by_phi": partial_by_phi,
        "num_points_used": len(rows),
        "warnings": warnings,
        "limitations": {
            "n0_excluded": True,
            "zero_mode_not_included": True,
            "matsubara_grid_incomplete": True,
            "angular_grid_sparse": True,
            "Q_grid_sparse": True,
            "not_production_quadrature": True,
            "not_physical_prediction": True,
        },
    }
