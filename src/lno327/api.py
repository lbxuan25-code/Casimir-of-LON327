"""Stable public API for downstream analysis notebooks and scripts.

The package root remains a compatibility facade for existing diagnostics.
New code should prefer this narrower module unless it needs an internal
diagnostic helper from a specific submodule.
"""

from __future__ import annotations

from .bdg_response import (
    BdGKernelComponents,
    BdGSuperconductingResponse,
    bdg_superconducting_response_imag_axis,
    bdg_total_kernel_imag_axis,
)
from .casimir import CasimirSetup, casimir_energy_integrand, casimir_torque_integrand
from .conductivity import (
    ConductivityTensor,
    KuboConfig,
    bosonic_matsubara_energy_eV,
    kubo_conductivity_imag_axis,
    kubo_conductivity_real_axis,
    uniform_bz_mesh,
)
from .gap_analysis import gap_statistics_by_band, gap_statistics_on_fermi_surface
from .model import NormalStateParameters, normal_state_hamiltonian
from .pairing import PairingAmplitudes, PairingKind, bdg_hamiltonian, pairing_matrix
from .response_interface import (
    LocalSheetResponse,
    ResponseKind,
    compare_local_responses_imag_axis,
    local_response_imag_axis,
    validate_local_response_symmetry,
)
from .response_units import (
    ResponseUnitConvention,
    SheetConductivityConvention,
    model_response_to_sheet_conductivity,
    require_sheet_conductivity_for_reflection,
    sheet_conductivity_to_reflection_dimensionless,
)

__all__ = [
    "BdGKernelComponents",
    "BdGSuperconductingResponse",
    "CasimirSetup",
    "ConductivityTensor",
    "KuboConfig",
    "LocalSheetResponse",
    "NormalStateParameters",
    "PairingAmplitudes",
    "PairingKind",
    "ResponseKind",
    "ResponseUnitConvention",
    "SheetConductivityConvention",
    "bdg_hamiltonian",
    "bdg_superconducting_response_imag_axis",
    "bdg_total_kernel_imag_axis",
    "bosonic_matsubara_energy_eV",
    "casimir_energy_integrand",
    "casimir_torque_integrand",
    "compare_local_responses_imag_axis",
    "gap_statistics_by_band",
    "gap_statistics_on_fermi_surface",
    "kubo_conductivity_imag_axis",
    "kubo_conductivity_real_axis",
    "local_response_imag_axis",
    "model_response_to_sheet_conductivity",
    "normal_state_hamiltonian",
    "pairing_matrix",
    "require_sheet_conductivity_for_reflection",
    "sheet_conductivity_to_reflection_dimensionless",
    "uniform_bz_mesh",
    "validate_local_response_symmetry",
]
