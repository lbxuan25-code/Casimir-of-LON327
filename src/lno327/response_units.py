"""Unit-convention helpers for pre-Casimir sheet responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .conductivity import ConductivityTensor
from .constants import E2_OVER_HBAR, SIGMA0

ResponseUnitMode = Literal["dimensionless_model", "si_sheet"]


@dataclass(frozen=True)
class ResponseUnitConvention:
    """Describe how a model response is interpreted before reflection input.

    Current response matrices are computed with dimensionless lattice momenta
    ``kx`` and ``ky``. They are model-unit diagnostics unless an explicit
    convention supplies the missing SI sheet-conductivity normalization.
    """

    mode: ResponseUnitMode = "dimensionless_model"
    lattice_constant_m: float | None = None
    unit_cell_area_m2: float | None = None
    include_e2_over_hbar: bool = False


@dataclass(frozen=True)
class SheetConductivityConversion:
    """Result of converting a model response matrix toward sheet conductivity."""

    tensor: ConductivityTensor
    unit_label: str
    normalization_status: str
    valid_for_casimir_input: bool
    notes: tuple[str, ...]


def _matrix_to_tensor(matrix: np.ndarray) -> ConductivityTensor:
    response_matrix = np.asarray(matrix, dtype=complex)
    if response_matrix.shape != (2, 2):
        raise ValueError("response_matrix must have shape (2, 2)")
    return ConductivityTensor(
        xx=response_matrix[0, 0],
        yy=response_matrix[1, 1],
        xy=response_matrix[0, 1],
        yx=response_matrix[1, 0],
    )


def model_response_to_sheet_conductivity(
    response_matrix: np.ndarray,
    convention: ResponseUnitConvention | None = None,
) -> SheetConductivityConversion:
    """Convert a 2x2 model response matrix through an explicit unit convention.

    The default ``dimensionless_model`` convention preserves the matrix and
    explicitly marks it as not valid for direct Casimir input. For ``si_sheet``,
    the conversion only becomes unit-valid when the lattice scale, unit-cell
    area, and ``e^2/hbar`` normalization choice are all specified.
    """

    convention = ResponseUnitConvention() if convention is None else convention
    matrix = np.asarray(response_matrix, dtype=complex)
    if matrix.shape != (2, 2):
        raise ValueError("response_matrix must have shape (2, 2)")

    base_notes = (
        "kx and ky are dimensionless lattice momenta",
        "model response is not automatically a final SI sheet conductivity",
        "reflection input must pass through this conversion interface",
    )

    if convention.mode == "dimensionless_model":
        return SheetConductivityConversion(
            tensor=_matrix_to_tensor(matrix),
            unit_label="dimensionless_model_response",
            normalization_status="dimensionless_model_not_si_sheet",
            valid_for_casimir_input=False,
            notes=base_notes + ("dimensionless_model fallback",),
        )

    if convention.mode != "si_sheet":
        raise ValueError("convention.mode must be 'dimensionless_model' or 'si_sheet'")

    missing: list[str] = []
    if convention.lattice_constant_m is None:
        missing.append("lattice_constant_m")
    if convention.unit_cell_area_m2 is None:
        missing.append("unit_cell_area_m2")
    if not convention.include_e2_over_hbar:
        missing.append("include_e2_over_hbar")
    if missing:
        return SheetConductivityConversion(
            tensor=_matrix_to_tensor(matrix),
            unit_label="model_response_missing_si_normalization",
            normalization_status="missing_" + "_".join(missing),
            valid_for_casimir_input=False,
            notes=base_notes + (f"missing SI normalization inputs: {', '.join(missing)}",),
        )

    if convention.lattice_constant_m <= 0.0 or convention.unit_cell_area_m2 <= 0.0:
        raise ValueError("lattice_constant_m and unit_cell_area_m2 must be positive")

    sheet_matrix = matrix * E2_OVER_HBAR
    return SheetConductivityConversion(
        tensor=_matrix_to_tensor(sheet_matrix),
        unit_label="si_sheet_conductivity_e2_over_hbar_scaled",
        normalization_status="si_sheet_normalization_specified",
        valid_for_casimir_input=True,
        notes=base_notes
        + (
            "e^2/hbar factor applied",
            "lattice scale recorded for audit; no finite-q normalization is implied",
        ),
    )


def sheet_conductivity_to_dimensionless(tensor: ConductivityTensor) -> ConductivityTensor:
    """Return conductivity components divided by the Dai-Jiang scale sigma0."""

    return ConductivityTensor(
        xx=tensor.xx / SIGMA0,
        yy=tensor.yy / SIGMA0,
        xy=tensor.xy / SIGMA0,
        yx=tensor.yx / SIGMA0,
    )
