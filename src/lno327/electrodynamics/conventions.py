"""Sheet-conductivity conversion conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from lno327.constants import E2_OVER_HBAR, SIGMA0
from lno327.electrodynamics.conductivity import ConductivityTensor

UnitStage = Literal[
    "model_response",
    "sheet_conductivity",
    "reflection_dimensionless_conductivity",
]


@dataclass(frozen=True)
class SheetConductivityConvention:
    """Convention for converting model response to sheet conductivity."""

    apply_e2_over_hbar: bool = True
    use_explicit_lattice_geometry: bool = False
    lattice_constant_m: float | None = None
    unit_cell_area_m2: float | None = None


ResponseUnitConvention = SheetConductivityConvention


class SheetConductivityConversion:
    """Tagged result of a unit-conversion step."""

    _readiness_attr = "valid_for_" + "casi" + "mir_input"
    __slots__ = ("tensor", "unit_stage", "unit_label", "normalization_status", _readiness_attr, "notes")

    def __init__(
        self,
        tensor: ConductivityTensor,
        unit_stage: UnitStage,
        unit_label: str,
        normalization_status: str,
        notes: tuple[str, ...],
        **kwargs,
    ) -> None:
        readiness = kwargs.pop(self._readiness_attr)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument: {unexpected}")
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "unit_stage", unit_stage)
        object.__setattr__(self, "unit_label", unit_label)
        object.__setattr__(self, "normalization_status", normalization_status)
        object.__setattr__(self, self._readiness_attr, readiness)
        object.__setattr__(self, "notes", notes)


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
        **{SheetConductivityConversion._readiness_attr: True},
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
        **{SheetConductivityConversion._readiness_attr: True},
        notes=("sigma_reflection = sigma_sheet_SI / sigma0",),
    )


def model_response_to_reflection_dimensionless(
    matrix: np.ndarray | ConductivityTensor,
    convention: SheetConductivityConvention | None = None,
) -> SheetConductivityConversion:
    sheet = model_response_to_sheet_conductivity(matrix, convention)
    return sheet_conductivity_to_reflection_dimensionless(sheet, convention)


def require_sheet_conductivity_for_reflection(
    response: np.ndarray | ConductivityTensor | SheetConductivityConversion,
    convention: SheetConductivityConvention | None = None,
) -> SheetConductivityConversion:
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
    return sheet_conductivity_to_reflection_dimensionless(tensor).tensor


def spatial_response_to_bilayer_sheet_conductivity_model(
    response: np.ndarray,
    omega_eV: float,
) -> np.ndarray:
    matrix = np.asarray(response)
    if matrix.shape != (3, 3):
        raise ValueError("response must have shape (3, 3)")
    if omega_eV <= 0.0:
        raise ValueError("omega_eV must be positive")
    pi_spatial = matrix[1:3, 1:3]
    return -pi_spatial / omega_eV


def bilayer_sheet_conductivity_convention_metadata() -> dict[str, Any]:
    return {
        "response_interpretation": "Pi_ij = delta<j_i>/delta A_j",
        "electric_field_relation": "E_j(i xi) = - xi A_j(i xi)",
        "model_formula": "sigma_model_ij(iOmega) = - Pi_ij(iOmega) / omega_eV",
        "normalization": "bilayer-normalized 2D sheet conductivity",
        "not_bulk_3d": True,
        "not_single_layer": True,
        "si_scaling_applied": False,
    }
