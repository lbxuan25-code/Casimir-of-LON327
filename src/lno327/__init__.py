"""Compatibility facade for common LNO327 analysis utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .bdg.finite_q import bdg_finite_q_vertex_from_normal_blocks
from .bdg.hamiltonian import bdg_hamiltonian_from_model_pairing
from .bdg.nambu import charge_current_vertex_from_model, diamagnetic_vertex_from_model
from .bdg.spectrum import diagonalize_hermitian, transform_operator_to_band_basis
from .workflows.finite_q_engine import (
    BdGPhaseCorrectionError,
    bdg_finite_q_response_imag_axis,
    collective_form_factor,
    collective_goldstone_counterterm,
    pairing_form_factor_matrix,
)
from .casimir import CasimirSetup, casimir_energy_integrand, casimir_torque_integrand
from .electrodynamics.conductivity import (
    ConductivityTensor,
    anisotropy_delta,
    anisotropy_summary,
    conductivity_matrix_diagnostics,
    rotate_conductivity,
)
from .electrodynamics.conventions import (
    ResponseUnitConvention,
    SheetConductivityConversion,
    SheetConductivityConvention,
    model_response_to_reflection_dimensionless,
    model_response_to_sheet_conductivity,
    require_sheet_conductivity_for_reflection,
    sheet_conductivity_to_dimensionless,
    sheet_conductivity_to_reflection_dimensionless,
)
from .analysis.gap import (
    FermiSurfacePoints,
    GapStatistics,
    band_gap_projection,
    fermi_surface_points,
    gap_statistics_by_band,
    gap_statistics_on_fermi_surface,
)
from .models.lno327_four_orbital.normal import normal_state_hamiltonian
from .models.lno327_four_orbital.parameters import (
    NormalStateParameters,
    ORBITAL_BASIS,
    PairingAmplitudes,
    PairingKind,
)
from .models.lno327_four_orbital.vertices import (
    normal_state_mass_operator,
    normal_state_mass_operators,
    normal_state_velocity_operator,
    normal_state_velocity_operators,
)
from .models.lno327_four_orbital.bdg import bdg_hamiltonian
from .models.lno327_four_orbital.pairing import (
    dwave_pairing_matrix,
    pairing_matrix,
    spm_pairing_matrix,
)
from .models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from .numerics.grids import uniform_bz_mesh
from .numerics.matsubara import bosonic_matsubara_energy_eV
from .numerics.weights import k_weights
from .response.config import KuboConfig
from .response.containers import KernelComponents as BdGKernelComponents
from .response.finite_q import BdGFiniteQResponseComponents
from .response.local_bdg import (
    BdGLocalEigensystem as BdGEigensystem,
    BdGLocalSuperconductingResponse as BdGSuperconductingResponse,
    bdg_local_diamagnetic_kernel,
    bdg_local_eigensystem_from_model,
    bdg_local_paramagnetic_kernel_imag_axis,
    bdg_local_superconducting_response_imag_axis,
    bdg_local_total_kernel_imag_axis,
)
from .response.local_interface import (
    LocalSheetResponse,
    ResponseKind,
    compare_local_responses_imag_axis,
    conductivity_tensor_from_matrix,
    local_response_imag_axis,
    matrix_symmetry_diagnostics,
    validate_local_response_symmetry,
)
from .response.local_normal import (
    NormalConductivityEigensystem as ConductivityEigensystem,
    kubo_conductivity_imag_axis_from_model,
    kubo_conductivity_real_axis_from_model,
    normal_conductivity_eigensystem_from_model,
)
from .response.occupations import fermi_function, negative_fermi_derivative
from .response.static_policy import (
    StaticResponsePolicy,
    StaticResponseResult,
    local_response_matsubara_index,
)


def conductivity_eigensystem(kx: float, ky: float, config: KuboConfig) -> ConductivityEigensystem:
    return normal_conductivity_eigensystem_from_model(LNO327FourOrbitalSpec(), kx, ky, config)


class _NormalModelAdapter:
    def __init__(self, base_spec, hamiltonian=None, velocity=None):
        self._base_spec = base_spec
        self._hamiltonian = hamiltonian
        self._velocity = velocity

    def normal_hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        if self._hamiltonian is None:
            return self._base_spec.normal_hamiltonian(kx, ky)
        return self._hamiltonian(kx, ky)

    def velocity_operator(self, kx: float, ky: float, direction: str) -> np.ndarray:
        if self._velocity is None:
            return self._base_spec.velocity_operator(kx, ky, direction)
        return self._velocity(kx, ky, direction)


def _normal_facade_spec(hamiltonian=None, velocity=None):
    base_spec = LNO327FourOrbitalSpec()
    if hamiltonian is None and velocity is None:
        return base_spec
    return _NormalModelAdapter(base_spec, hamiltonian=hamiltonian, velocity=velocity)


def kubo_conductivity_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
    hamiltonian=None,
    velocity=None,
) -> ConductivityTensor:
    return kubo_conductivity_imag_axis_from_model(
        _normal_facade_spec(hamiltonian=hamiltonian, velocity=velocity),
        k_points,
        config,
        k_weights,
    )


def kubo_conductivity_real_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
    hamiltonian=None,
    velocity=None,
) -> ConductivityTensor:
    return kubo_conductivity_real_axis_from_model(
        _normal_facade_spec(hamiltonian=hamiltonian, velocity=velocity),
        k_points,
        config,
        k_weights,
    )


def bdg_current_vertex(kx: float, ky: float, direction: str) -> np.ndarray:
    return charge_current_vertex_from_model(LNO327FourOrbitalSpec(), kx, ky, direction)


def bdg_current_vertices(kx: float, ky: float) -> tuple[np.ndarray, np.ndarray]:
    return bdg_current_vertex(kx, ky, "x"), bdg_current_vertex(kx, ky, "y")


def bdg_diamagnetic_vertex(kx: float, ky: float, direction_a: str, direction_b: str) -> np.ndarray:
    return diamagnetic_vertex_from_model(LNO327FourOrbitalSpec(), kx, ky, direction_a, direction_b)


def bdg_eigensystem(
    kx: float,
    ky: float,
    pairing: np.ndarray,
    config: KuboConfig | None = None,
) -> BdGEigensystem:
    spec = LNO327FourOrbitalSpec()
    bands = diagonalize_hermitian(bdg_hamiltonian_from_model_pairing(spec, kx, ky, pairing))
    if config is None:
        occupations = np.zeros_like(bands.energies, dtype=float)
        minus_df = np.zeros_like(bands.energies, dtype=float)
    else:
        occupations = fermi_function(bands.energies, config.fermi_level_eV, config.temperature_eV)
        minus_df = negative_fermi_derivative(
            bands.energies,
            config.fermi_level_eV,
            config.temperature_eV,
            config.eta_eV,
        )
    jx = transform_operator_to_band_basis(bands.states, bdg_current_vertex(kx, ky, "x"))
    jy = transform_operator_to_band_basis(bands.states, bdg_current_vertex(kx, ky, "y"))
    return BdGEigensystem(bands.energies, bands.states, occupations, minus_df, jx, jy)


def bdg_paramagnetic_kernel_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None = None,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=pairing_params or PairingAmplitudes())
    return bdg_local_paramagnetic_kernel_imag_axis(spec, pairing_kind, k_points, config, k_weights)


def bdg_diamagnetic_kernel(
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None,
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=pairing_params or PairingAmplitudes())
    return bdg_local_diamagnetic_kernel(spec, pairing_kind, k_points, config, k_weights)


def bdg_total_kernel_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None = None,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> BdGKernelComponents:
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=pairing_params or PairingAmplitudes())
    return bdg_local_total_kernel_imag_axis(spec, pairing_kind, k_points, config, k_weights)


def bdg_superconducting_response_imag_axis(
    k_points: Sequence[tuple[float, float]] | np.ndarray,
    config: KuboConfig,
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None = None,
    k_weights: Sequence[float] | np.ndarray | None = None,
) -> BdGSuperconductingResponse:
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=pairing_params or PairingAmplitudes())
    return bdg_local_superconducting_response_imag_axis(spec, pairing_kind, k_points, config, k_weights)


def bdg_finite_q_vector_vertex(kx: float, ky: float, qx: float, qy: float, direction: str) -> np.ndarray:
    spec = LNO327FourOrbitalSpec()
    particle = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, direction)
    hole = spec.peierls_hamiltonian_vector_vertex(-kx, -ky, -qx, -qy, direction)
    return bdg_finite_q_vertex_from_normal_blocks(particle, hole)


def bdg_finite_q_contact_vertex(
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    direction_i: str,
    direction_j: str,
) -> np.ndarray:
    spec = LNO327FourOrbitalSpec()
    particle = spec.peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
    hole = spec.peierls_hamiltonian_contact_vertex(-kx, -ky, -qx, -qy, direction_i, direction_j)
    return bdg_finite_q_vertex_from_normal_blocks(particle, hole)

__all__ = [
    "ORBITAL_BASIS",
    "BdGEigensystem",
    "BdGKernelComponents",
    "BdGFiniteQResponseComponents",
    "BdGPhaseCorrectionError",
    "BdGSuperconductingResponse",
    "CasimirSetup",
    "ConductivityEigensystem",
    "ConductivityTensor",
    "FermiSurfacePoints",
    "GapStatistics",
    "KuboConfig",
    "LocalSheetResponse",
    "NormalStateParameters",
    "PairingAmplitudes",
    "PairingKind",
    "ResponseKind",
    "ResponseUnitConvention",
    "SheetConductivityConvention",
    "SheetConductivityConversion",
    "StaticResponsePolicy",
    "StaticResponseResult",
    "anisotropy_delta",
    "anisotropy_summary",
    "bdg_current_vertex",
    "bdg_current_vertices",
    "bdg_diamagnetic_kernel",
    "bdg_diamagnetic_vertex",
    "bdg_eigensystem",
    "bdg_finite_q_contact_vertex",
    "bdg_finite_q_response_imag_axis",
    "bdg_finite_q_vector_vertex",
    "bdg_hamiltonian",
    "bdg_paramagnetic_kernel_imag_axis",
    "bdg_superconducting_response_imag_axis",
    "bdg_total_kernel_imag_axis",
    "bosonic_matsubara_energy_eV",
    "band_gap_projection",
    "casimir_energy_integrand",
    "casimir_torque_integrand",
    "conductivity_eigensystem",
    "collective_form_factor",
    "collective_goldstone_counterterm",
    "conductivity_matrix_diagnostics",
    "conductivity_tensor_from_matrix",
    "compare_local_responses_imag_axis",
    "dwave_pairing_matrix",
    "fermi_function",
    "fermi_surface_points",
    "gap_statistics_by_band",
    "gap_statistics_on_fermi_surface",
    "k_weights",
    "kubo_conductivity_imag_axis",
    "kubo_conductivity_real_axis",
    "local_response_imag_axis",
    "local_response_matsubara_index",
    "matrix_symmetry_diagnostics",
    "model_response_to_sheet_conductivity",
    "model_response_to_reflection_dimensionless",
    "normal_state_hamiltonian",
    "normal_state_mass_operator",
    "normal_state_mass_operators",
    "normal_state_velocity_operator",
    "normal_state_velocity_operators",
    "pairing_matrix",
    "pairing_form_factor_matrix",
    "rotate_conductivity",
    "spm_pairing_matrix",
    "require_sheet_conductivity_for_reflection",
    "sheet_conductivity_to_reflection_dimensionless",
    "sheet_conductivity_to_dimensionless",
    "uniform_bz_mesh",
    "validate_local_response_symmetry",
]
