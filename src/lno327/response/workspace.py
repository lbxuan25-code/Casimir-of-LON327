"""Explicit precompute workspaces for repeated response evaluations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.finite_q_bdg import (
    FiniteQBdGWorkspace,
    finite_q_bdg_response_from_workspace,
    precompute_finite_q_bdg_workspace_from_model_ansatz,
)
from lno327.response.finite_q_material import (
    precompute_finite_q_material_workspace_from_model_ansatz,
)
from lno327.response.finite_q_optimized import (
    FiniteQMaterialWorkspace,
    FiniteQQWorkspace,
    finite_q_bdg_response_from_q_workspace as _finite_q_bdg_response_from_q_workspace,
    finite_q_bdg_responses_from_q_workspace as _finite_q_bdg_responses_from_q_workspace,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327.response.phase_hessian import (
    apply_phase_hessian_policy_to_components,
    phase_hessian_policy_from_options,
)
from lno327.response.local_bdg import (
    BdGLocalWorkspace,
    bdg_local_diamagnetic_kernel_from_workspace,
    bdg_local_paramagnetic_kernel_imag_axis_from_workspace,
    bdg_local_superconducting_response_imag_axis_from_workspace,
    bdg_local_total_kernel_imag_axis_from_workspace,
    precompute_bdg_local_workspace_from_model,
)
from lno327.response.local_normal import (
    NormalLocalWorkspace,
    kubo_conductivity_imag_axis_from_workspace,
    kubo_conductivity_real_axis_from_workspace,
    precompute_normal_local_workspace_from_model,
)
from lno327.response.nonlocal_bdg import (
    BdGNonlocalWorkspace,
    bdg_current_current_kernel_imag_axis_from_workspace,
    precompute_bdg_nonlocal_workspace_from_model,
)
from lno327.response.nonlocal_normal import (
    NormalNonlocalWorkspace,
    normal_current_current_kernel_imag_axis_from_workspace,
    precompute_normal_nonlocal_workspace_from_model,
)
from lno327.response.normal_density_current import (
    NormalDensityCurrentWorkspace,
    normal_density_current_response_imag_axis_from_workspace,
    normal_physical_density_current_response_components_imag_axis_from_workspace,
    normal_physical_density_current_response_imag_axis_from_workspace,
    precompute_normal_density_current_workspace_from_model,
    precompute_normal_physical_density_current_workspace_from_model,
)


def _apply_q_workspace_phase_hessian_policy(
    workspace: FiniteQQWorkspace,
    components: BdGFiniteQResponseComponents,
) -> BdGFiniteQResponseComponents:
    corrected, _ = apply_phase_hessian_policy_to_components(
        components,
        workspace.material.ansatz,
        workspace.q_model,
        phase_hessian_policy_from_options(workspace.material.options),
    )
    return corrected


def finite_q_bdg_responses_from_q_workspace(
    workspace: FiniteQQWorkspace,
    omega_eV_values: Sequence[float] | np.ndarray,
    *,
    config=None,
) -> tuple[BdGFiniteQResponseComponents, ...]:
    """Evaluate a q workspace and apply one shared phase-Hessian policy per result."""

    base = _finite_q_bdg_responses_from_q_workspace(
        workspace,
        omega_eV_values,
        config=config,
    )
    return tuple(
        _apply_q_workspace_phase_hessian_policy(workspace, components)
        for components in base
    )


def finite_q_bdg_response_from_q_workspace(
    workspace: FiniteQQWorkspace,
    omega_eV: float | None = None,
    *,
    config=None,
) -> BdGFiniteQResponseComponents:
    """Evaluate one q-workspace frequency with its explicit phase-Hessian policy."""

    base = _finite_q_bdg_response_from_q_workspace(
        workspace,
        omega_eV,
        config=config,
    )
    return _apply_q_workspace_phase_hessian_policy(workspace, base)


__all__ = [
    "BdGLocalWorkspace",
    "BdGNonlocalWorkspace",
    "FiniteQBdGWorkspace",
    "FiniteQMaterialWorkspace",
    "FiniteQQWorkspace",
    "NormalLocalWorkspace",
    "NormalDensityCurrentWorkspace",
    "NormalNonlocalWorkspace",
    "bdg_current_current_kernel_imag_axis_from_workspace",
    "bdg_local_diamagnetic_kernel_from_workspace",
    "bdg_local_paramagnetic_kernel_imag_axis_from_workspace",
    "bdg_local_superconducting_response_imag_axis_from_workspace",
    "bdg_local_total_kernel_imag_axis_from_workspace",
    "finite_q_bdg_response_from_q_workspace",
    "finite_q_bdg_responses_from_q_workspace",
    "finite_q_bdg_response_from_workspace",
    "kubo_conductivity_imag_axis_from_workspace",
    "kubo_conductivity_real_axis_from_workspace",
    "normal_current_current_kernel_imag_axis_from_workspace",
    "normal_density_current_response_imag_axis_from_workspace",
    "normal_physical_density_current_response_components_imag_axis_from_workspace",
    "normal_physical_density_current_response_imag_axis_from_workspace",
    "precompute_bdg_local_workspace_from_model",
    "precompute_bdg_nonlocal_workspace_from_model",
    "precompute_finite_q_bdg_workspace_from_model_ansatz",
    "precompute_finite_q_material_workspace_from_model_ansatz",
    "precompute_finite_q_q_workspace",
    "precompute_normal_local_workspace_from_model",
    "precompute_normal_density_current_workspace_from_model",
    "precompute_normal_nonlocal_workspace_from_model",
    "precompute_normal_physical_density_current_workspace_from_model",
    "primitive_ward_rhs_from_q_workspace",
]
