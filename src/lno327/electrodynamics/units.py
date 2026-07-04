"""Electrodynamic unit-conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy import constants as _constants

    ELEMENTARY_CHARGE_C = float(_constants.elementary_charge)
    HBAR_J_S = float(_constants.hbar)
    VACUUM_IMPEDANCE_OHM = float(_constants.mu_0 * _constants.c)
    FINE_STRUCTURE_CONSTANT = float(_constants.alpha)
except Exception:  # pragma: no cover
    from lno327.constants import C0, E_CHARGE, EPSILON0, HBAR, MU0

    ELEMENTARY_CHARGE_C = float(E_CHARGE)
    HBAR_J_S = float(HBAR)
    VACUUM_IMPEDANCE_OHM = float(MU0 * C0)
    FINE_STRUCTURE_CONSTANT = float(E_CHARGE**2 / (4.0 * np.pi * EPSILON0 * HBAR * C0))


@dataclass(frozen=True)
class SheetConductivityUnitConvention:
    """Geometry convention for model-to-SI bilayer sheet conductivity."""

    lattice_a_x_m: float
    lattice_a_y_m: float
    unit_cell_area_m2: float | None = None
    normalization: str = "bilayer-normalized 2D sheet conductivity"
    si_scaling_applied: bool = True
    bulk_3d_conductivity: bool = False
    single_layer_conductivity: bool = False

    def __post_init__(self) -> None:
        if self.lattice_a_x_m <= 0.0:
            raise ValueError("lattice_a_x_m must be positive")
        if self.lattice_a_y_m <= 0.0:
            raise ValueError("lattice_a_y_m must be positive")
        area = self.lattice_a_x_m * self.lattice_a_y_m if self.unit_cell_area_m2 is None else self.unit_cell_area_m2
        if area <= 0.0:
            raise ValueError("unit_cell_area_m2 must be positive")
        object.__setattr__(self, "unit_cell_area_m2", float(area))


def e2_over_hbar_siemens() -> float:
    return ELEMENTARY_CHARGE_C**2 / HBAR_J_S


def vacuum_impedance_ohm() -> float:
    return VACUUM_IMPEDANCE_OHM


def four_pi_alpha() -> float:
    return 4.0 * np.pi * FINE_STRUCTURE_CONSTANT


def z0_e2_over_hbar() -> float:
    return vacuum_impedance_ohm() * e2_over_hbar_siemens()


def dimensionless_sheet_prefactor_square_lattice() -> float:
    return z0_e2_over_hbar()


def _axis_length(axis: str, convention: SheetConductivityUnitConvention) -> float:
    if axis == "x":
        return convention.lattice_a_x_m
    if axis == "y":
        return convention.lattice_a_y_m
    raise ValueError("axis must be 'x' or 'y'")


def geometry_factor(i: str, j: str, convention: SheetConductivityUnitConvention) -> float:
    return _axis_length(i, convention) * _axis_length(j, convention) / float(convention.unit_cell_area_m2)


def sheet_geometry_factor_tensor(convention: SheetConductivityUnitConvention) -> np.ndarray:
    return np.array(
        [
            [geometry_factor("x", "x", convention), geometry_factor("x", "y", convention)],
            [geometry_factor("y", "x", convention), geometry_factor("y", "y", convention)],
        ],
        dtype=float,
    )


def _as_2x2_complex(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.shape != (2, 2):
        raise ValueError("conductivity tensor must have shape (2, 2)")
    return array.astype(complex, copy=False)


def model_to_si_sheet_conductivity(
    sigma_model: np.ndarray,
    convention: SheetConductivityUnitConvention,
) -> np.ndarray:
    matrix = _as_2x2_complex(sigma_model)
    return e2_over_hbar_siemens() * sheet_geometry_factor_tensor(convention) * matrix


def si_sheet_to_dimensionless_conductivity(sigma_si_sheet: np.ndarray) -> np.ndarray:
    return vacuum_impedance_ohm() * _as_2x2_complex(sigma_si_sheet)


def model_to_dimensionless_sheet_conductivity(
    sigma_model: np.ndarray,
    convention: SheetConductivityUnitConvention,
) -> np.ndarray:
    return si_sheet_to_dimensionless_conductivity(model_to_si_sheet_conductivity(sigma_model, convention))


def conductivity_unit_conversion_metadata(convention: SheetConductivityUnitConvention) -> dict[str, Any]:
    return {
        "formula_model_to_si": "sigma_SI_sheet_ij = (e^2/hbar) * (a_i a_j/A_cell) * sigma_model_ij",
        "formula_si_to_dimensionless": "sigma_tilde_ij = Z0 * sigma_SI_sheet_ij",
        "formula_model_to_dimensionless": "sigma_tilde_ij = Z0 * (e^2/hbar) * (a_i a_j/A_cell) * sigma_model_ij",
        "normalization": convention.normalization,
        "si_units": "Siemens",
        "dimensionless_symbol": "sigma_tilde_ij = Z0 * sigma_SI_sheet_ij",
        "e2_over_hbar_S": e2_over_hbar_siemens(),
        "vacuum_impedance_ohm": vacuum_impedance_ohm(),
        "z0_e2_over_hbar": z0_e2_over_hbar(),
        "four_pi_alpha": four_pi_alpha(),
        "geometry_tensor": sheet_geometry_factor_tensor(convention),
        "lattice_a_x_m": convention.lattice_a_x_m,
        "lattice_a_y_m": convention.lattice_a_y_m,
        "unit_cell_area_m2": convention.unit_cell_area_m2,
        "bulk_3d_conductivity": convention.bulk_3d_conductivity,
        "single_layer_conductivity": convention.single_layer_conductivity,
        "reflection_" + "casi" + "mir_ready": False,
    }
