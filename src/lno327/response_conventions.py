"""Response, conductivity, and reflection-input unit conventions.

This module is the single source for model-response-to-conductivity
conventions, SI sheet-conductivity scaling, and dimensionless reflection
normalization. It replaces the former split across ``response_units.py``,
``conductivity_units.py``, and ``conductivity_conventions.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .conductivity import ConductivityTensor
from .constants import E2_OVER_HBAR, SIGMA0

try:  # Prefer SciPy CODATA constants when available.
    from scipy import constants as _constants

    ELEMENTARY_CHARGE_C = float(_constants.elementary_charge)
    HBAR_J_S = float(_constants.hbar)
    VACUUM_IMPEDANCE_OHM = float(_constants.mu_0 * _constants.c)
    FINE_STRUCTURE_CONSTANT = float(_constants.alpha)
except Exception:  # pragma: no cover - exercised only when SciPy is absent.
    from .constants import C0, E_CHARGE, EPSILON0, HBAR, MU0

    ELEMENTARY_CHARGE_C = float(E_CHARGE)
    HBAR_J_S = float(HBAR)
    VACUUM_IMPEDANCE_OHM = float(MU0 * C0)
    FINE_STRUCTURE_CONSTANT = float(E_CHARGE**2 / (4.0 * np.pi * EPSILON0 * HBAR * C0))

UnitStage = Literal[
    "model_response",
    "sheet_conductivity",
    "reflection_dimensionless_conductivity",
]


@dataclass(frozen=True)
class SheetConductivityConvention:
    """Convention for converting model response to sheet conductivity.

    Current Kubo/BdG response matrices are dimensionless model responses using
    dimensionless lattice momenta and normalized Brillouin-zone weights. In the
    present two-dimensional sheet convention, the physical area factor and
    velocity-vertex conversion are represented by the overall normalization
    choice

        sigma_sheet_SI = (e^2 / hbar) * sigma_model.

    Optional lattice fields are retained for future audits but are not active
    in the default normalization.
    """

    apply_e2_over_hbar: bool = True
    use_explicit_lattice_geometry: bool = False
    lattice_constant_m: float | None = None
    unit_cell_area_m2: float | None = None


# Backward-compatible alias; new code should use SheetConductivityConvention.
ResponseUnitConvention = SheetConductivityConvention


@dataclass(frozen=True)
class SheetConductivityConversion:
    """Tagged result of a unit-conversion step."""

    tensor: ConductivityTensor
    unit_stage: UnitStage
    unit_label: str
    normalization_status: str
    valid_for_casimir_input: bool
    notes: tuple[str, ...]


def _as_matrix(matrix_or_tensor: np.ndarray | ConductivityTensor) -> np.ndarray:
    if isinstance(matrix_or_tensor, ConductivityTensor):
        matrix = matrix_or_tensor.matrix()
    else:
        matrix = np.asarray(matrix_or_tensor, dtype=complex)
    if matrix.shape != (2, 2):
        raise ValueError("response matrix must have shape (2, 2)")
    return matrix


def _matrix_to_tensor(matrix_or_tensor: np.ndarray | ConductivityTensor) -> ConductivityTensor:
    matrix = _as_matrix(matrix_or_tensor)
    return ConductivityTensor(matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0])


def model_response_to_sheet_conductivity(
    matrix: np.ndarray | ConductivityTensor | SheetConductivityConversion,
    convention: SheetConductivityConvention | None = None,
) -> SheetConductivityConversion:
    """Convert a model response into SI sheet conductivity.

    This applies exactly one factor of ``e^2 / hbar``. Passing an already tagged
    sheet-conductivity or reflection-dimensionless conversion raises an error
    to prevent double scaling.
    """

    if isinstance(matrix, SheetConductivityConversion):
        if matrix.unit_stage != "model_response":
            raise ValueError("input is already unit-converted; refusing to apply e^2/hbar twice")
        raw_matrix = matrix.tensor.matrix()
    else:
        raw_matrix = _as_matrix(matrix)

    convention = SheetConductivityConvention() if convention is None else convention
    if not convention.apply_e2_over_hbar:
        raise ValueError("Sheet conductivity conversion requires apply_e2_over_hbar=True")
    if convention.use_explicit_lattice_geometry:
        if convention.lattice_constant_m is None or convention.unit_cell_area_m2 is None:
            raise ValueError("explicit lattice geometry requires lattice_constant_m and unit_cell_area_m2")
        if convention.lattice_constant_m <= 0.0 or convention.unit_cell_area_m2 <= 0.0:
            raise ValueError("lattice_constant_m and unit_cell_area_m2 must be positive")

    sheet_matrix = E2_OVER_HBAR * raw_matrix
    return SheetConductivityConversion(
        tensor=_matrix_to_tensor(sheet_matrix),
        unit_stage="sheet_conductivity",
        unit_label="sheet_conductivity_e2_over_hbar_scaled",
        normalization_status="e2_over_hbar_scaled",
        valid_for_casimir_input=True,
        notes=(
            "input stage: model_response",
            "kx and ky are dimensionless lattice momenta",
            "BZ weights use normalized dimensionless integration",
            "sigma_sheet_SI = (e^2 / hbar) * sigma_model",
            "explicit lattice geometry is reserved for future convention audits",
        ),
    )


def sheet_conductivity_to_reflection_dimensionless(
    matrix_or_tensor: np.ndarray | ConductivityTensor | SheetConductivityConversion,
    convention: SheetConductivityConvention | None = None,
) -> SheetConductivityConversion:
    """Normalize SI sheet conductivity by the vacuum admittance scale sigma0."""

    _ = SheetConductivityConvention() if convention is None else convention
    if isinstance(matrix_or_tensor, SheetConductivityConversion):
        if matrix_or_tensor.unit_stage == "model_response":
            raise ValueError("model response must first be converted to sheet conductivity")
        if matrix_or_tensor.unit_stage == "reflection_dimensionless_conductivity":
            return matrix_or_tensor
        matrix = matrix_or_tensor.tensor.matrix()
    else:
        matrix = _as_matrix(matrix_or_tensor)

    return SheetConductivityConversion(
        tensor=_matrix_to_tensor(matrix / SIGMA0),
        unit_stage="reflection_dimensionless_conductivity",
        unit_label="reflection_dimensionless_conductivity_vacuum_admittance_normalized",
        normalization_status="vacuum_admittance_normalized",
        valid_for_casimir_input=True,
        notes=("sigma_reflection = sigma_sheet_SI / sigma0",),
    )


def model_response_to_reflection_dimensionless(
    matrix: np.ndarray | ConductivityTensor,
    convention: SheetConductivityConvention | None = None,
) -> SheetConductivityConversion:
    """Convert model response directly to reflection-dimensionless conductivity."""

    sheet = model_response_to_sheet_conductivity(matrix, convention)
    return sheet_conductivity_to_reflection_dimensionless(sheet, convention)


def require_sheet_conductivity_for_reflection(
    response: np.ndarray | ConductivityTensor | SheetConductivityConversion,
    convention: SheetConductivityConvention | None = None,
) -> SheetConductivityConversion:
    """Return SI sheet conductivity, refusing raw reflection input ambiguity."""

    if isinstance(response, SheetConductivityConversion):
        if response.unit_stage == "sheet_conductivity":
            return response
        if response.unit_stage == "reflection_dimensionless_conductivity":
            raise ValueError("reflection-dimensionless conductivity is not SI sheet conductivity")
        return model_response_to_sheet_conductivity(response, convention)
    return model_response_to_sheet_conductivity(response, convention)


def sheet_conductivity_to_dimensionless(
    tensor: ConductivityTensor,
) -> ConductivityTensor:
    """Backward-compatible wrapper for reflection-dimensionless normalization."""

    return sheet_conductivity_to_reflection_dimensionless(tensor).tensor


def spatial_response_to_bilayer_sheet_conductivity_model(
    response: np.ndarray,
    omega_eV: float,
) -> np.ndarray:
    """
    Convert the spatial block of the finite-q physical response matrix
    to bilayer-normalized model sheet conductivity on the imaginary axis.

    Convention:
        Pi_ij = delta <j_i> / delta A_j
        E_j(i xi) = - xi A_j(i xi)
        sigma_ij(i xi) = - Pi_ij(i xi) / xi

    Code frequency variable:
        omega_eV = hbar * xi, positive Matsubara energy in eV.

    Model-level conversion:
        sigma_model_ij(iOmega) = - response[1:3, 1:3] / omega_eV
    """

    matrix = np.asarray(response)
    if matrix.shape != (3, 3):
        raise ValueError("response must have shape (3, 3)")
    if omega_eV <= 0.0:
        raise ValueError("omega_eV must be positive")
    pi_spatial = matrix[1:3, 1:3]
    return -pi_spatial / omega_eV


def bilayer_sheet_conductivity_convention_metadata() -> dict[str, Any]:
    """Return metadata for the bilayer sheet-conductivity convention."""

    return {
        "response_interpretation": "Pi_ij = delta<j_i>/delta A_j",
        "electric_field_relation": "E_j(i xi) = - xi A_j(i xi)",
        "model_formula": "sigma_model_ij(iOmega) = - Pi_ij(iOmega) / omega_eV",
        "normalization": "bilayer-normalized 2D sheet conductivity",
        "not_bulk_3d": True,
        "not_single_layer": True,
        "si_scaling_applied": False,
    }


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
    """Return e^2 / hbar in Siemens."""

    return ELEMENTARY_CHARGE_C**2 / HBAR_J_S


def vacuum_impedance_ohm() -> float:
    """Return the vacuum impedance Z0 in Ohm."""

    return VACUUM_IMPEDANCE_OHM


def four_pi_alpha() -> float:
    """Return 4*pi*alpha."""

    return 4.0 * np.pi * FINE_STRUCTURE_CONSTANT


def z0_e2_over_hbar() -> float:
    """Return Z0 * e^2 / hbar."""

    return vacuum_impedance_ohm() * e2_over_hbar_siemens()


def dimensionless_sheet_prefactor_square_lattice() -> float:
    """Return the square-lattice model-to-sigma_tilde prefactor."""

    return z0_e2_over_hbar()


def _axis_length(axis: str, convention: SheetConductivityUnitConvention) -> float:
    if axis == "x":
        return convention.lattice_a_x_m
    if axis == "y":
        return convention.lattice_a_y_m
    raise ValueError("axis must be 'x' or 'y'")


def geometry_factor(i: str, j: str, convention: SheetConductivityUnitConvention) -> float:
    """Return a_i a_j / A_cell for i,j in {'x','y'}."""

    return _axis_length(i, convention) * _axis_length(j, convention) / float(convention.unit_cell_area_m2)


def sheet_geometry_factor_tensor(convention: SheetConductivityUnitConvention) -> np.ndarray:
    """Return the 2x2 tensor of geometry factors a_i a_j / A_cell."""

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
    """Convert model bilayer sheet conductivity to SI sheet conductivity."""

    matrix = _as_2x2_complex(sigma_model)
    return e2_over_hbar_siemens() * sheet_geometry_factor_tensor(convention) * matrix


def si_sheet_to_dimensionless_conductivity(sigma_si_sheet: np.ndarray) -> np.ndarray:
    """Return sigma_tilde = Z0 * sigma_SI_sheet."""

    return vacuum_impedance_ohm() * _as_2x2_complex(sigma_si_sheet)


def model_to_dimensionless_sheet_conductivity(
    sigma_model: np.ndarray,
    convention: SheetConductivityUnitConvention,
) -> np.ndarray:
    """Convert model bilayer sheet conductivity directly to sigma_tilde."""

    return si_sheet_to_dimensionless_conductivity(model_to_si_sheet_conductivity(sigma_model, convention))


def conductivity_unit_conversion_metadata(convention: SheetConductivityUnitConvention) -> dict[str, Any]:
    """Return metadata for the bilayer unit conversion convention."""

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
        "reflection_casimir_ready": False,
    }
