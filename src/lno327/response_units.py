"""Neutral sheet-conductivity normalization for reflection inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .conductivity import ConductivityTensor
from .constants import E2_OVER_HBAR, SIGMA0

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
