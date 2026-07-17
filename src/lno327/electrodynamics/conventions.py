"""Sheet-conductivity conversion conventions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from lno327.constants import E2_OVER_HBAR, SIGMA0
from lno327.electrodynamics.conductivity import ConductivityTensor
from lno327.response.effective_kernel import EffectiveEMKernel

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


@dataclass(frozen=True)
class SheetConductivityConversion:
    """Tagged result of a unit-conversion step."""

    tensor: ConductivityTensor
    unit_stage: UnitStage
    unit_label: str
    normalization_status: str
    valid_for_casimir_input: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PositiveMatsubaraSheetResponse:
    """Typed positive-frequency sheet response in the crystal ``(x, y)`` basis."""

    sigma_model_xy: ConductivityTensor
    sigma_sheet_si_xy: SheetConductivityConversion
    sigma_tilde_xy: SheetConductivityConversion
    q_model: np.ndarray
    xi_eV: float
    degeneracy: float
    basis: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.sigma_sheet_si_xy.unit_stage != "sheet_conductivity":
            raise ValueError("sigma_sheet_si_xy must be at the sheet_conductivity stage")
        if self.sigma_tilde_xy.unit_stage != "reflection_dimensionless_conductivity":
            raise ValueError("sigma_tilde_xy must be reflection-dimensionless")

        q = np.array(self.q_model, dtype=float, copy=True)
        if q.shape != (2,):
            raise ValueError(f"q_model must have shape (2,), got {q.shape}")
        if not np.isfinite(q).all():
            raise ValueError("q_model must contain only finite values")
        q.setflags(write=False)
        object.__setattr__(self, "q_model", q)

        xi = float(self.xi_eV)
        if not np.isfinite(xi) or xi <= 0.0:
            raise ValueError("positive-Matsubara sheet response requires finite xi_eV > 0")
        object.__setattr__(self, "xi_eV", xi)

        degeneracy = float(self.degeneracy)
        if not np.isfinite(degeneracy) or degeneracy <= 0.0:
            raise ValueError("degeneracy must be finite and positive")
        object.__setattr__(self, "degeneracy", degeneracy)
        object.__setattr__(self, "basis", str(self.basis))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def matrix_model(self) -> np.ndarray:
        return self.sigma_model_xy.matrix()

    @property
    def matrix_sheet_si(self) -> np.ndarray:
        return self.sigma_sheet_si_xy.tensor.matrix()

    @property
    def matrix_tilde(self) -> np.ndarray:
        return self.sigma_tilde_xy.tensor.matrix()


@dataclass(frozen=True)
class SheetResponseValidation:
    """Single-point physical diagnostics for an imaginary-axis sheet tensor."""

    finite: bool
    relative_imaginary_norm: float
    relative_symmetry_residual: float
    minimum_symmetric_eigenvalue: float
    reality_tolerance: float
    symmetry_tolerance: float
    passivity_tolerance: float

    @property
    def passed(self) -> bool:
        return bool(
            self.finite
            and self.relative_imaginary_norm <= self.reality_tolerance
            and self.relative_symmetry_residual <= self.symmetry_tolerance
            and self.minimum_symmetric_eigenvalue >= -self.passivity_tolerance
        )

    def require_passed(self) -> None:
        if not self.passed:
            raise ValueError(
                "sheet response failed physical validation: "
                f"finite={self.finite}, "
                f"relative_imaginary_norm={self.relative_imaginary_norm:.3e}, "
                f"relative_symmetry_residual={self.relative_symmetry_residual:.3e}, "
                f"minimum_symmetric_eigenvalue={self.minimum_symmetric_eigenvalue:.3e}"
            )


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


def positive_matsubara_kernel_to_sheet_response(
    kernel: EffectiveEMKernel,
    *,
    degeneracy: float = 1.0,
    convention: SheetConductivityConvention | None = None,
) -> PositiveMatsubaraSheetResponse:
    """Convert a primitive effective kernel to the positive-Matsubara sheet tensor.

    The fixed sign convention is ``sigma_model(i xi) = -g K_eff,xy(i xi) / xi``.
    Square-lattice geometry is already implicit in the normalized model response;
    this function therefore applies only the explicit degeneracy and the existing
    ``e^2/hbar`` and vacuum-admittance conversions.
    """

    if kernel.xi_eV <= 0.0:
        raise ValueError("positive Matsubara conversion requires kernel.xi_eV > 0")
    factor = float(degeneracy)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("degeneracy must be finite and positive")

    sigma_model_matrix = -factor * np.asarray(kernel.spatial_xy, dtype=complex) / kernel.xi_eV
    sigma_model = _matrix_to_tensor(sigma_model_matrix)
    sheet = model_response_to_sheet_conductivity(sigma_model, convention)
    sigma_tilde = sheet_conductivity_to_reflection_dimensionless(sheet, convention)
    return PositiveMatsubaraSheetResponse(
        sigma_model_xy=sigma_model,
        sigma_sheet_si_xy=sheet,
        sigma_tilde_xy=sigma_tilde,
        q_model=kernel.q_model,
        xi_eV=kernel.xi_eV,
        degeneracy=factor,
        basis="crystal_xy",
        metadata={
            "source": "EffectiveEMKernel.spatial_xy",
            "formula": "sigma_model_xy(i xi) = - degeneracy * K_eff_xy(i xi) / xi_eV",
            "square_lattice_geometry_factor": 1.0,
            "frequency_sector": "positive_matsubara",
            "kernel_basis": kernel.metadata.get("basis", "crystal_A0_xy"),
        },
    )


def validate_positive_matsubara_sheet_response(
    response: PositiveMatsubaraSheetResponse,
    *,
    reality_tolerance: float = 1e-9,
    symmetry_tolerance: float = 1e-9,
    passivity_tolerance: float = 1e-10,
) -> SheetResponseValidation:
    """Validate finite, real-symmetric and passive single-point sheet response."""

    matrix = np.asarray(response.matrix_tilde, dtype=complex)
    finite = bool(np.isfinite(matrix.real).all() and np.isfinite(matrix.imag).all())
    scale = max(float(np.linalg.norm(matrix)), 1.0)
    relative_imaginary = float(np.linalg.norm(matrix.imag) / scale)
    relative_symmetry = float(np.linalg.norm(matrix - matrix.T) / scale)
    symmetric_real = 0.5 * (matrix.real + matrix.real.T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric_real))) if finite else float("-inf")
    return SheetResponseValidation(
        finite=finite,
        relative_imaginary_norm=relative_imaginary,
        relative_symmetry_residual=relative_symmetry,
        minimum_symmetric_eigenvalue=minimum_eigenvalue,
        reality_tolerance=float(reality_tolerance),
        symmetry_tolerance=float(symmetry_tolerance),
        passivity_tolerance=float(passivity_tolerance),
    )


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
