"""Helpers for Stage 5.14 real-material production-grid convergence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .casimir_grid import matsubara_prime_weights, polar_measure_weights
from .constants import KB
from .material_reflection_grid import MaterialReflectionGridPoint

PASSED_STAGE5_13_STATUS = "STAGE5_13_ZERO_MODE_GRID_CONVERGENCE_AUDIT_PASSED"

PRODUCTION_GRID_LEVELS: dict[str, dict[str, int]] = {
    "coarse": {"n_max": 8, "n_Q": 16, "n_phi": 8},
    "medium": {"n_max": 16, "n_Q": 24, "n_phi": 12},
    "fine": {"n_max": 32, "n_Q": 32, "n_phi": 16},
}

Q0_POLICY = "Q=0 is not a regular grid point; use interior quadrature nodes"
N0_POLICY = "n=0 uses xi->0+ extrapolated R_TE_TM; never divide by Omega=0"


@dataclass(frozen=True)
class ProductionGrid:
    level: str
    n_max: int
    n_Q: int
    n_phi: int
    Q_max_nm_inv: float
    temperature_K: float

    @property
    def Q_nm_inv(self) -> np.ndarray:
        return interior_q_nodes_nm_inv(self.Q_max_nm_inv, self.n_Q)

    @property
    def phi_deg(self) -> np.ndarray:
        return uniform_phi_nodes_deg(self.n_phi)

    @property
    def n_positive(self) -> list[int]:
        return list(range(1, self.n_max + 1))

    @property
    def num_response_points(self) -> int:
        return self.n_max * self.n_Q * self.n_phi

    @property
    def num_energy_points_including_n0(self) -> int:
        return (self.n_max + 1) * self.n_Q * self.n_phi


def validate_stage5_13_input(data: dict[str, Any]) -> str:
    status = data.get("diagnostic_status", {}).get("stage5_13_status")
    if status != PASSED_STAGE5_13_STATUS:
        raise ValueError("Stage 5.13 input must have STAGE5_13_ZERO_MODE_GRID_CONVERGENCE_AUDIT_PASSED status")
    return status


def production_grid_plan_from_stage5_13(data: dict[str, Any]) -> dict[str, Any]:
    plan = dict(data.get("grid_convergence_plan", {}))
    levels = {level: dict(plan.get(level, PRODUCTION_GRID_LEVELS[level])) for level in PRODUCTION_GRID_LEVELS}
    for level, expected in PRODUCTION_GRID_LEVELS.items():
        for key, value in expected.items():
            if int(levels[level].get(key, -1)) != value:
                raise ValueError(f"Stage 5.13 grid plan mismatch for {level}.{key}")
    return {
        **levels,
        "Q0_policy": plan.get("Q0_policy", Q0_POLICY),
        "n0_policy": plan.get("n0_policy", N0_POLICY),
    }


def interior_q_nodes_nm_inv(Q_max_nm_inv: float, n_Q: int) -> np.ndarray:
    if Q_max_nm_inv <= 0.0:
        raise ValueError("Q_max_nm_inv must be positive")
    if n_Q <= 0:
        raise ValueError("n_Q must be positive")
    width = float(Q_max_nm_inv) / int(n_Q)
    return (np.arange(int(n_Q), dtype=float) + 0.5) * width


def uniform_phi_nodes_deg(n_phi: int) -> np.ndarray:
    if n_phi <= 0:
        raise ValueError("n_phi must be positive")
    return np.linspace(0.0, 360.0, int(n_phi), endpoint=False)


def build_production_grid(level: str, *, Q_max_nm_inv: float, temperature_K: float) -> ProductionGrid:
    if level not in PRODUCTION_GRID_LEVELS:
        raise ValueError(f"unknown production grid level: {level}")
    spec = PRODUCTION_GRID_LEVELS[level]
    return ProductionGrid(
        level=level,
        n_max=int(spec["n_max"]),
        n_Q=int(spec["n_Q"]),
        n_phi=int(spec["n_phi"]),
        Q_max_nm_inv=float(Q_max_nm_inv),
        temperature_K=float(temperature_K),
    )


def material_reflection_points_for_grid(grid: ProductionGrid) -> list[MaterialReflectionGridPoint]:
    return [
        MaterialReflectionGridPoint(n=n, Q_nm_inv=float(q), phi_deg=float(phi), temperature_K=grid.temperature_K)
        for n in grid.n_positive
        for q in grid.Q_nm_inv
        for phi in grid.phi_deg
    ]


def _as_complex(value: Any) -> complex:
    if isinstance(value, dict):
        return complex(float(value.get("re", value.get("real", 0.0))), float(value.get("im", value.get("imag", 0.0))))
    return complex(value)


def row_logdet(row: dict[str, Any]) -> complex:
    if "integrand_identical_sheet" in row:
        return _as_complex(row["integrand_identical_sheet"]["logdet"])
    if "logdet_identical_sheet" in row:
        return _as_complex(row["logdet_identical_sheet"])
    raise KeyError("point row does not contain a logdet field")


def point_key(n: int, Q_nm_inv: float, phi_deg: float) -> tuple[int, float, float]:
    return int(n), round(float(Q_nm_inv), 12), round(float(phi_deg), 12)


def integrate_grid_energy_from_rows(grid: ProductionGrid, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Integrate cached/computed reflection rows for one grid level.

    The n=0 term is represented by the xi->0+ extrapolated reflection proxy:
    the first positive Matsubara row at the same Q,phi is reused with the
    Matsubara prime 1/2 weight.  This avoids any Omega=0 division in response.
    """

    usable = [row for row in rows if row.get("status") in {"PASS", "MONITOR"}]
    row_by_key = {point_key(row["n"], row["Q_nm_inv"], row["phi_deg"]): row for row in usable}
    q_m_inv = grid.Q_nm_inv * 1.0e9
    phi_rad = np.deg2rad(grid.phi_deg)
    weights = polar_measure_weights(q_m_inv, phi_rad)
    matsubara_weights = matsubara_prime_weights(grid.n_max)
    total = 0.0 + 0.0j
    missing: list[dict[str, Any]] = []
    for n in range(0, grid.n_max + 1):
        source_n = 1 if n == 0 else n
        for iq, q in enumerate(grid.Q_nm_inv):
            for iphi, phi in enumerate(grid.phi_deg):
                row = row_by_key.get(point_key(source_n, float(q), float(phi)))
                if row is None:
                    missing.append({"n": n, "source_n": source_n, "Q_nm_inv": float(q), "phi_deg": float(phi)})
                    continue
                total += KB * grid.temperature_K * float(matsubara_weights[n]) * float(weights[iq, iphi]) * row_logdet(row)
    finite = bool(np.isfinite(total.real) and np.isfinite(total.imag))
    return {
        "level": grid.level,
        "n_max": grid.n_max,
        "n_Q": grid.n_Q,
        "n_phi": grid.n_phi,
        "Q_max_nm_inv": grid.Q_max_nm_inv,
        "F_over_area_J_m2": total,
        "real_J_m2": float(total.real),
        "imag_J_m2": float(total.imag),
        "num_response_points_expected": grid.num_response_points,
        "num_energy_points_including_n0": grid.num_energy_points_including_n0,
        "num_rows_available": len(usable),
        "num_missing_points": len(missing),
        "missing_points_preview": missing[:10],
        "n0_policy": N0_POLICY,
        "Q0_policy": Q0_POLICY,
        "status": "PASS" if finite and not missing else "FAIL",
    }


def relative_change(previous: complex | float, current: complex | float) -> float:
    prev = complex(previous)
    curr = complex(current)
    denominator = max(abs(prev), abs(curr), 1e-300)
    return float(abs(curr - prev) / denominator)


def classify_energy_convergence(medium_to_fine: float, *, numerical_anomaly: bool = False) -> str:
    if numerical_anomaly or not np.isfinite(medium_to_fine):
        return "FAIL"
    if medium_to_fine < 0.05:
        return "PASS"
    if medium_to_fine < 0.15:
        return "MONITOR"
    return "FAIL"


def summarize_energy_convergence(grid_runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    coarse = complex(grid_runs["coarse"]["F_over_area_J_m2"])
    medium = complex(grid_runs["medium"]["F_over_area_J_m2"])
    fine = complex(grid_runs["fine"]["F_over_area_J_m2"])
    coarse_to_medium = relative_change(coarse, medium)
    medium_to_fine = relative_change(medium, fine)
    anomaly = any(run.get("status") == "FAIL" for run in grid_runs.values())
    status = classify_energy_convergence(medium_to_fine, numerical_anomaly=anomaly)
    return {
        "coarse_to_medium_relative_change": coarse_to_medium,
        "medium_to_fine_relative_change": medium_to_fine,
        "pass_threshold": 0.05,
        "monitor_threshold": 0.15,
        "status": status,
    }


PointResultProvider = Callable[[ProductionGrid], list[dict[str, Any]]]


def run_production_grid_sequence(*, Q_max_nm_inv: float, temperature_K: float, provider: PointResultProvider) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for level in ("coarse", "medium", "fine"):
        grid = build_production_grid(level, Q_max_nm_inv=Q_max_nm_inv, temperature_K=temperature_K)
        runs[level] = integrate_grid_energy_from_rows(grid, provider(grid))
    return runs
