"""Workflow-level orchestration helpers."""

from __future__ import annotations

from lno327.workflows.arbitrary_q_vector_adaptive import (
    AdaptiveConvergenceError,
    ArbitraryQVectorAdaptiveOptions,
    ArbitraryQVectorAdaptiveProfile,
    ArbitraryQVectorAdaptiveResponseCache,
    HierarchicalMaterialNodeCache,
    build_hierarchical_material_node_cache,
    integrate_arbitrary_q_vector_adaptive,
    integrate_two_plate_angle_batch_vector_adaptive,
)
from lno327.workflows.arbitrary_q_vector_adaptive_parallel import (
    ArbitraryQVectorAdaptiveParallelEvaluator,
)
from lno327.workflows.dwave_nodal_quadrature import (
    DWaveNodalQuadratureOptions,
    build_dwave_nodal_quadrature,
)
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
from lno327.workflows.dwave_vector_adaptive_cubature import (
    DWaveCubatureCell,
    DWaveVectorAdaptiveOptions,
    cubature_cell_gauss_rule,
    initial_cubature_cells,
    merge_cell_components_before_schur,
    primitive_component_vector,
    primitive_ward_residual_vector,
    subdivide_cubature_cell,
    validate_vector_adaptive_options,
    vector_error_metrics,
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
from lno327.workflows.finite_q_quadrature import (
    FiniteQQuadratureOptions,
    finite_q_quadrature_points,
)

__all__ = [
    "AdaptiveConvergenceError",
    "ArbitraryQVectorAdaptiveOptions",
    "ArbitraryQVectorAdaptiveParallelEvaluator",
    "ArbitraryQVectorAdaptiveProfile",
    "ArbitraryQVectorAdaptiveResponseCache",
    "BdGPhaseCorrectionError",
    "DWaveCubatureCell",
    "DWaveNodalQuadratureOptions",
    "DWavePeriodicMultishiftOptions",
    "DWavePeriodicShiftEnsembleOptions",
    "DWaveVectorAdaptiveOptions",
    "FiniteQEngineOptions",
    "FiniteQQuadratureOptions",
    "HierarchicalMaterialNodeCache",
    "bdg_finite_q_response_imag_axis_from_workspace",
    "bdg_finite_q_response_imag_axis",
    "build_dwave_nodal_quadrature",
    "build_dwave_periodic_multishift_quadrature",
    "build_dwave_periodic_shift_ensemble",
    "build_hierarchical_material_node_cache",
    "collective_form_factor",
    "collective_goldstone_counterterm",
    "cubature_cell_gauss_rule",
    "finite_q_bdg_response_from_ansatz",
    "finite_q_quadrature_points",
    "initial_cubature_cells",
    "integrate_arbitrary_q_vector_adaptive",
    "integrate_two_plate_angle_batch_vector_adaptive",
    "merge_cell_components_before_schur",
    "merge_shift_components_before_schur",
    "nested_c4_antithetic_shifts",
    "pairing_form_factor_matrix",
    "periodic_shift_mesh",
    "precompute_finite_q_engine_workspace",
    "primitive_component_vector",
    "primitive_ward_residual_vector",
    "subdivide_cubature_cell",
    "validate_vector_adaptive_options",
    "vector_error_metrics",
]
