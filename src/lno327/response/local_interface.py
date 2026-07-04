"""Local sheet-response interface before the formal Casimir stage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from lno327.electrodynamics.conductivity import ConductivityTensor
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.electrodynamics.conventions import SheetConductivityConvention, model_response_to_sheet_conductivity
from lno327.response.config import KuboConfig
from lno327.response.local_bdg import bdg_local_superconducting_response_imag_axis
from lno327.response.local_normal import kubo_conductivity_imag_axis_from_model

ResponseKind = Literal["normal", "spm", "dwave"]


@dataclass(frozen=True)
class LocalSheetResponse:
    """Local q=0 response matrix prepared for future reflection-matrix plumbing."""

    kind: ResponseKind
    omega_eV: float
    matrix: np.ndarray
    unit_label: str
    source: str
    valid_for_casimir_input: bool
    notes: tuple[str, ...]
    normalization_status: str = "model_response_unconverted"
    recommended_unit_conversion: str = "model_response_to_sheet_conductivity"
    static_policy: str = "finite_matsubara_only"
    momentum_status: str = "local_q0_only"


def conductivity_tensor_from_matrix(matrix: np.ndarray) -> ConductivityTensor:
    """Convert a 2x2 complex matrix into a ``ConductivityTensor``."""

    response_matrix = np.asarray(matrix, dtype=complex)
    if response_matrix.shape != (2, 2):
        raise ValueError("matrix must have shape (2, 2)")
    return ConductivityTensor(
        xx=response_matrix[0, 0],
        yy=response_matrix[1, 1],
        xy=response_matrix[0, 1],
        yx=response_matrix[1, 0],
    )


def local_response_imag_axis(
    kind: ResponseKind,
    omega_eV: float,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    temperature_K: float,
    eta_eV: float = 1e-4,
    pairing_params: PairingAmplitudes | None = None,
    k_weights: Sequence[float] | np.ndarray | None = None,
    fermi_level_eV: float = 0.0,
    unit_convention: SheetConductivityConvention | None = None,
) -> LocalSheetResponse:
    """Return a local q=0 sheet response at one imaginary-axis energy."""

    if kind not in {"normal", "spm", "dwave"}:
        raise ValueError("kind must be one of: normal, spm, dwave")
    if omega_eV < 0.0:
        raise ValueError("omega_eV must be non-negative")
    if eta_eV <= 0.0:
        raise ValueError("eta_eV must be positive")

    base_notes = (
        "local q=0 response only",
        "n=0 Matsubara treatment unresolved",
        "SI sheet conductivity conversion is provided by response_conventions",
        "finite momentum response is not part of the current code path",
    )
    amp = pairing_params or PairingAmplitudes()
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=amp)

    if kind == "normal":
        config = KuboConfig.from_kelvin(
            omega_eV=omega_eV,
            temperature_K=temperature_K,
            fermi_level_eV=fermi_level_eV,
            eta_eV=eta_eV,
            output_si=False,
        )
        conductivity = kubo_conductivity_imag_axis_from_model(spec, k_points, config, k_weights)
        notes = base_notes
        if omega_eV == 0.0:
            notes += ("normal-state n=0 response retained as unresolved diagnostic",)
        conversion = model_response_to_sheet_conductivity(
            conductivity.matrix(),
            unit_convention,
        )
        return LocalSheetResponse(
            kind=kind,
            omega_eV=omega_eV,
            matrix=conductivity.matrix(),
            unit_label="model_units_normal_state_sigma_iomega",
            source="kubo_conductivity_imag_axis_from_model",
            valid_for_casimir_input=False,
            notes=notes,
            normalization_status=conversion.normalization_status,
            static_policy="n0_unresolved" if omega_eV == 0.0 else "finite_matsubara",
            momentum_status="local_q0_only",
        )

    if omega_eV <= 0.0:
        raise ValueError("omega_eV must be positive for BdG Sigma_SC; n=0 is unresolved")

    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=temperature_K,
        fermi_level_eV=fermi_level_eV,
        eta_eV=eta_eV,
        output_si=False,
    )
    response = bdg_local_superconducting_response_imag_axis(
        spec,
        kind,
        k_points,
        config,
        k_weights,
    )
    conversion = model_response_to_sheet_conductivity(
        response.sigma_like_response,
        unit_convention,
    )
    return LocalSheetResponse(
        kind=kind,
        omega_eV=omega_eV,
        matrix=response.sigma_like_response,
        unit_label="model_units_BdG_Sigma_SC_iomega",
        source="bdg_local_superconducting_response_imag_axis",
        valid_for_casimir_input=False,
        notes=base_notes + ("Sigma_SC = K_total / omega_eV for n >= 1",),
        normalization_status=conversion.normalization_status,
        recommended_unit_conversion="model_response_to_sheet_conductivity_then_reflection_dimensionless",
        static_policy="finite_matsubara",
        momentum_status="local_q0_only",
    )


def matrix_symmetry_diagnostics(
    matrix: np.ndarray,
    tolerance: float = 1e-8,
) -> dict[str, complex | float | bool]:
    """Return compact C4/local-response diagnostics for a 2x2 matrix."""

    response_matrix = np.asarray(matrix, dtype=complex)
    if response_matrix.shape != (2, 2):
        raise ValueError("matrix must have shape (2, 2)")

    diagonal_sum = response_matrix[0, 0] + response_matrix[1, 1]
    delta = (
        complex(0.0)
        if np.isclose(diagonal_sum, 0.0)
        else (response_matrix[0, 0] - response_matrix[1, 1]) / diagonal_sum
    )
    diagonal_scale = 0.5 * (abs(response_matrix[0, 0]) + abs(response_matrix[1, 1]))
    offdiag_norm = float(np.linalg.norm([response_matrix[0, 1], response_matrix[1, 0]]))
    relative_offdiag = 0.0 if np.isclose(diagonal_scale, 0.0) else float(offdiag_norm / diagonal_scale)

    eigenvalues = np.linalg.eigvals(response_matrix)
    eigen_scale = 0.5 * (abs(eigenvalues[0]) + abs(eigenvalues[1]))
    relative_eigen_split = (
        0.0 if np.isclose(eigen_scale, 0.0) else float(abs(eigenvalues[0] - eigenvalues[1]) / eigen_scale)
    )
    isotropic = (
        abs(delta) <= tolerance
        and relative_offdiag <= tolerance
        and relative_eigen_split <= tolerance
    )
    return {
        "delta": delta,
        "relative_offdiag": relative_offdiag,
        "relative_eigen_split": relative_eigen_split,
        "isotropic_within_tolerance": isotropic,
    }


def validate_local_response_symmetry(
    response: LocalSheetResponse,
    tolerance: float = 1e-8,
) -> dict[str, complex | float | bool]:
    """Return compact C4/local-response diagnostics."""

    return matrix_symmetry_diagnostics(response.matrix, tolerance=tolerance)


def compare_local_responses_imag_axis(
    kinds: Sequence[ResponseKind],
    omega_eV: Sequence[float] | np.ndarray,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    temperature_K: float,
    eta_eV: float = 1e-4,
    pairing_params: PairingAmplitudes | None = None,
    k_weights: Sequence[float] | np.ndarray | None = None,
    fermi_level_eV: float = 0.0,
    tolerance: float = 1e-8,
) -> list[dict[str, object]]:
    """Evaluate normal/spm/dwave local responses on the same omega grid."""

    energies = np.asarray(omega_eV, dtype=float)
    if energies.ndim != 1 or energies.size == 0:
        raise ValueError("omega_eV must be a non-empty 1D array")

    rows: list[dict[str, object]] = []
    for kind in kinds:
        for omega in energies:
            response = local_response_imag_axis(
                kind,
                float(omega),
                k_points,
                temperature_K,
                eta_eV,
                pairing_params,
                k_weights,
                fermi_level_eV,
            )
            diagnostics = validate_local_response_symmetry(response, tolerance=tolerance)
            rows.append(
                {
                    "kind": response.kind,
                    "omega_eV": response.omega_eV,
                    "matrix": response.matrix,
                    "unit_label": response.unit_label,
                    "source": response.source,
                    "valid_for_casimir_input": response.valid_for_casimir_input,
                    "notes": response.notes,
                    **diagnostics,
                }
            )
    return rows
