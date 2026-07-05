"""Stable public API for downstream analysis notebooks and scripts.

The package root remains a compatibility facade for existing diagnostics.
New code should prefer this narrower module unless it needs an internal
diagnostic helper from a specific submodule.
"""

from __future__ import annotations

from .casimir import CasimirSetup, casimir_energy_integrand, casimir_torque_integrand
from .electrodynamics.conductivity import ConductivityTensor
from .analysis.gap import gap_statistics_by_band, gap_statistics_on_fermi_surface
from .models.lno327_four_orbital.bdg import bdg_hamiltonian
from .models.lno327_four_orbital.collective import PairingAnsatz, build_pairing_ansatz
from .models.lno327_four_orbital.normal import normal_state_hamiltonian
from .models.lno327_four_orbital.pairing import pairing_matrix
from .models.lno327_four_orbital.parameters import NormalStateParameters, PairingAmplitudes, PairingKind
from .models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from .numerics.grids import uniform_bz_mesh
from .numerics.matsubara import bosonic_matsubara_energy_eV
from .response.config import KuboConfig
from .response.containers import KernelComponents as BdGKernelComponents
from .response.local_bdg import (
    BdGLocalSuperconductingResponse as BdGSuperconductingResponse,
)
from .response.local_bdg import (
    bdg_local_superconducting_response_imag_axis,
    bdg_local_total_kernel_imag_axis,
)
from .response.local_interface import (
    LocalSheetResponse,
    ResponseKind,
    compare_local_responses_imag_axis,
    local_response_imag_axis,
    validate_local_response_symmetry,
)
from .response.local_normal import (
    kubo_conductivity_imag_axis_from_model,
    kubo_conductivity_real_axis_from_model,
)
from .electrodynamics.conventions import (
    ResponseUnitConvention,
    SheetConductivityConvention,
    model_response_to_sheet_conductivity,
    require_sheet_conductivity_for_reflection,
    sheet_conductivity_to_reflection_dimensionless,
)
from .response.static_policy import (
    StaticResponsePolicy,
    StaticResponseResult,
    local_response_matsubara_index,
    matsubara_response_series,
)
from .collective.validation import WardValidationReport, validate_physical_ward_identity


def kubo_conductivity_imag_axis(k_points, config, k_weights=None):
    spec = LNO327FourOrbitalSpec()
    return kubo_conductivity_imag_axis_from_model(spec, k_points, config, k_weights)


def kubo_conductivity_real_axis(k_points, config, k_weights=None):
    spec = LNO327FourOrbitalSpec()
    return kubo_conductivity_real_axis_from_model(spec, k_points, config, k_weights)


def bdg_superconducting_response_imag_axis(
    k_points,
    config,
    pairing,
    pairing_params=None,
    k_weights=None,
):
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=pairing_params or PairingAmplitudes())
    return bdg_local_superconducting_response_imag_axis(spec, pairing, k_points, config, k_weights)


def bdg_total_kernel_imag_axis(
    k_points,
    config,
    pairing,
    pairing_params=None,
    k_weights=None,
):
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=pairing_params or PairingAmplitudes())
    return bdg_local_total_kernel_imag_axis(spec, pairing, k_points, config, k_weights)


__all__ = [
    "BdGKernelComponents",
    "BdGSuperconductingResponse",
    "CasimirSetup",
    "ConductivityTensor",
    "KuboConfig",
    "LocalSheetResponse",
    "NormalStateParameters",
    "PairingAmplitudes",
    "PairingAnsatz",
    "PairingKind",
    "ResponseKind",
    "ResponseUnitConvention",
    "SheetConductivityConvention",
    "StaticResponsePolicy",
    "StaticResponseResult",
    "WardValidationReport",
    "bdg_hamiltonian",
    "bdg_superconducting_response_imag_axis",
    "bdg_total_kernel_imag_axis",
    "bosonic_matsubara_energy_eV",
    "build_pairing_ansatz",
    "casimir_energy_integrand",
    "casimir_torque_integrand",
    "compare_local_responses_imag_axis",
    "gap_statistics_by_band",
    "gap_statistics_on_fermi_surface",
    "kubo_conductivity_imag_axis",
    "kubo_conductivity_real_axis",
    "local_response_imag_axis",
    "local_response_matsubara_index",
    "matsubara_response_series",
    "model_response_to_sheet_conductivity",
    "normal_state_hamiltonian",
    "pairing_matrix",
    "require_sheet_conductivity_for_reflection",
    "sheet_conductivity_to_reflection_dimensionless",
    "uniform_bz_mesh",
    "validate_local_response_symmetry",
    "validate_physical_ward_identity",
]
