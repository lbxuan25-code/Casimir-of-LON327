from __future__ import annotations

import numpy as np

import lno327.workflows as workflows
from lno327.numerics.fixed_outer_adaptive_inner import FixedOuterAdaptiveInnerOptions
from lno327.workflows.arbitrary_q_fixed_outer_adaptive_inner import (
    EXECUTOR_ID,
    METHOD_ID,
    integrate_arbitrary_q_fixed_outer_adaptive_inner,
)
from validation.__main__ import resolve_command
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def test_candidate_route_is_diagnostic_only_and_local_refinement_is_not_public() -> None:
    assert resolve_command(
        "diagnostic", "arbitrary-q-fixed-outer-adaptive-inner"
    ) == (
        "validation.commands.matsubara."
        "arbitrary_q_fixed_outer_adaptive_inner_diagnostic"
    )
    assert not hasattr(workflows, "FiniteQQuadratureOptions")
    assert not hasattr(workflows, "finite_q_quadrature_points")


def test_generalized_workflow_runs_all_matsubara_on_one_orientation() -> None:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    result = integrate_arbitrary_q_fixed_outer_adaptive_inner(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=np.asarray([0.0, 0.025]),
        temperature_K=10.0,
        eta_eV=1e-8,
        q_model=np.asarray([0.03, 0.02]),
        outer_order=2,
        inner_options=FixedOuterAdaptiveInnerOptions(
            epsabs=1.0,
            epsrel=1.0,
            inner_limit=10,
            max_point_evaluations=5000,
            cache_size_bytes=1_000_000,
            quadrature="gk15",
            norm="max",
            split_points=(0.0,),
        ),
        orders=("xy",),
        primary_order="xy",
    )
    assert result.metadata["method_id"] == METHOD_ID
    assert result.metadata["executor_id"] == EXECUTOR_ID
    assert result.metadata["diagnostic_only"] is True
    assert result.metadata["production_reference_established"] is False
    assert result.metadata["valid_for_casimir_input"] is False
    assert len(result.orientations) == 1
    orientation = result.primary
    assert orientation.quadrature.success
    assert orientation.operator_ward.passed
    assert len(orientation.components) == 2
    assert len(orientation.rhs) == 2
    assert orientation.metadata["matsubara_batch_shared_nodes"] is True
    assert orientation.pointwise_profile.unique_point_count > 0
    assert orientation.pointwise_profile.q_workspace_build_count == (
        orientation.pointwise_profile.unique_point_count
    )
