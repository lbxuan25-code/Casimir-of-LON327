"""Workflow-level orchestration helpers.

Only retained workflow surfaces are exported here. Historical q-specific local
refinement remains importable from its explicit module for archaeology/tests but is
not part of the public workflow API.
"""

from __future__ import annotations

from lno327.workflows.dwave_periodic_multishift_quadrature import (
    DWavePeriodicMultishiftOptions,
    build_dwave_periodic_multishift_quadrature,
)
from lno327.workflows.dwave_periodic_shift_ensemble import (
    DWavePeriodicShiftEnsembleOptions,
    build_dwave_periodic_shift_ensemble,
    merge_shift_components_before_schur,
    nested_c4_antithetic_shifts,
    periodic_shift_mesh,
)
from lno327.workflows.finite_q_engine import (
    BdGPhaseCorrectionError,
    FiniteQEngineOptions,
    bdg_finite_q_response_imag_axis_from_workspace,
    bdg_finite_q_response_imag_axis,
    collective_form_factor,
    collective_goldstone_counterterm,
    finite_q_bdg_response_from_ansatz,
    pairing_form_factor_matrix,
    precompute_finite_q_engine_workspace,
)

__all__ = [
    "BdGPhaseCorrectionError",
    "DWavePeriodicMultishiftOptions",
    "DWavePeriodicShiftEnsembleOptions",
    "FiniteQEngineOptions",
    "bdg_finite_q_response_imag_axis_from_workspace",
    "bdg_finite_q_response_imag_axis",
    "build_dwave_periodic_multishift_quadrature",
    "build_dwave_periodic_shift_ensemble",
    "collective_form_factor",
    "collective_goldstone_counterterm",
    "finite_q_bdg_response_from_ansatz",
    "merge_shift_components_before_schur",
    "nested_c4_antithetic_shifts",
    "pairing_form_factor_matrix",
    "periodic_shift_mesh",
    "precompute_finite_q_engine_workspace",
]
