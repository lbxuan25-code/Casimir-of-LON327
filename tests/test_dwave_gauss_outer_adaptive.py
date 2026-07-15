from __future__ import annotations

import numpy as np
import pytest

from lno327 import KuboConfig
from lno327.response.workspace import (
    finite_q_bdg_response_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327.workflows.dwave_periodic_shift_ensemble import periodic_shift_mesh
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.dwave_adaptive_bond_metric import (
    AdaptiveStaticValidationConfig,
    postprocess_adaptive_bond_metric_static,
)
from validation.lib.dwave_static_primitives import (
    assemble_dwave_static_primitives,
    build_dwave_static_integrand_context,
    pack_complex_vector,
    unpack_complex_vector,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.gauss_outer_adaptive import (
    EvaluationBudgetExceeded,
    GaussAdaptiveOptions,
    gauss_outer_adaptive_integral,
)


def test_gauss_outer_adaptive_matches_analytic_vector_in_both_orders():
    def integrand(kx: float, ky: float) -> np.ndarray:
        return np.asarray([1.0, kx, ky, kx * ky, kx * kx + ky * ky])

    options = GaussAdaptiveOptions(
        epsabs=1e-11,
        epsrel=1e-11,
        inner_limit=40,
        max_point_evaluations=50_000,
        quadrature="gk15",
        split_points=(0.0,),
    )
    expected = np.asarray(
        [4.0 * np.pi**2, 0.0, 0.0, 0.0, 8.0 * np.pi**4 / 3.0]
    )
    xy = gauss_outer_adaptive_integral(
        integrand, order="xy", outer_order=8, options=options
    )
    yx = gauss_outer_adaptive_integral(
        integrand, order="yx", outer_order=8, options=options
    )
    assert xy.success
    assert yx.success
    assert xy.outer_evaluations == 8
    assert yx.outer_evaluations == 8
    assert np.allclose(xy.value, expected, rtol=1e-11, atol=1e-11)
    assert np.allclose(yx.value, expected, rtol=1e-11, atol=1e-11)
    assert np.allclose(xy.value, yx.value, rtol=1e-12, atol=1e-12)
    assert "excludes outer discretization" in xy.message


def test_gauss_outer_adaptive_point_budget_is_fail_closed():
    options = GaussAdaptiveOptions(
        max_point_evaluations=1,
        inner_limit=10,
    )
    with pytest.raises(EvaluationBudgetExceeded):
        gauss_outer_adaptive_integral(
            lambda kx, ky: np.asarray([kx + ky]),
            order="xy",
            outer_order=4,
            options=options,
        )


def test_complex_primitive_vector_pack_round_trip():
    value = np.arange(48, dtype=float) + 1j * np.arange(48, dtype=float)[::-1]
    assert np.array_equal(unpack_complex_vector(pack_complex_vector(value)), value)


def _uniform_fixture():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    q = np.asarray([0.11, 0.07])
    points, weights = periodic_shift_mesh(4, (0.5, 0.5))
    base_options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    context = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        q,
        config,
        pairing,
        base_options,
    )
    primitive = np.tensordot(weights, context.evaluate_complex(points), axes=(0, 0))
    components, rhs, _ = assemble_dwave_static_primitives(
        context,
        primitive,
        metadata={"test": "uniform_primitive_average"},
    )
    return model, ansatz, pairing, config, q, points, weights, components, rhs


def test_static_primitive_average_matches_periodic_workspace_before_bond_metric():
    model, ansatz, pairing, config, q, points, weights, components, rhs = _uniform_fixture()
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        config,
        pairing,
        options,
    )
    workspace = precompute_finite_q_q_workspace(material, q)
    expected_components = finite_q_bdg_response_from_q_workspace(workspace, 0.0)
    expected_rhs = primitive_ward_rhs_from_q_workspace(workspace, 0.0)

    for field in (
        "bare_bubble",
        "direct",
        "collective_bubble",
        "collective_counterterm",
        "em_collective_left",
        "collective_em_right",
        "amplitude_phase_schur",
    ):
        assert np.allclose(
            getattr(components, field),
            getattr(expected_components, field),
            rtol=2e-11,
            atol=2e-12,
        ), field
    assert np.allclose(rhs.left, expected_rhs.left, rtol=2e-11, atol=2e-12)
    assert np.allclose(rhs.right, expected_rhs.right, rtol=2e-11, atol=2e-12)


def test_postintegral_bond_metric_matches_policy_aware_workspace():
    model, ansatz, pairing, config, q, points, weights, components, rhs = _uniform_fixture()
    expected_options = FiniteQEngineOptions(
        phase_hessian_policy="nearest_neighbor_bond_metric"
    )
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        config,
        pairing,
        expected_options,
    )
    workspace = precompute_finite_q_q_workspace(material, q)
    expected = finite_q_bdg_response_from_q_workspace(workspace, 0.0)

    result = postprocess_adaptive_bond_metric_static(
        components,
        rhs,
        ansatz=ansatz,
        q_model=q,
        config=AdaptiveStaticValidationConfig(
            mixed_ward_tolerance=100.0,
            mixed_ward_absolute_tolerance=100.0,
            primitive_tolerance=100.0,
            amplitude_tolerance=100.0,
            phase_tolerance=100.0,
            effective_direct_tolerance=100.0,
            effective_residual_tolerance=100.0,
            longitudinal_tolerance=100.0,
            reality_tolerance=100.0,
            mixing_tolerance=100.0,
            passivity_tolerance=100.0,
        ),
    )

    for field in (
        "collective_counterterm",
        "collective_total",
        "amplitude_phase_schur",
        "gauge_restored",
    ):
        assert np.allclose(
            getattr(result.components, field),
            getattr(expected, field),
            rtol=2e-11,
            atol=2e-12,
        ), field

    base = np.asarray(components.collective_counterterm, dtype=complex)
    applied = np.asarray(result.components.collective_counterterm, dtype=complex)
    assert np.array_equal(applied[0, :], base[0, :])
    assert applied[1, 0] == base[1, 0]
    assert applied[1, 1] != base[1, 1]
    assert result.application.policy == "nearest_neighbor_bond_metric"
    assert result.components.metadata["phase_hessian_changed_only_22"] is True
    assert result.to_row_fields()["projection_applied"] is False
    assert result.to_row_fields()["valid_for_casimir_input"] is False

    with pytest.raises(ValueError, match="already have the bond metric applied"):
        postprocess_adaptive_bond_metric_static(
            result.components,
            rhs,
            ansatz=ansatz,
            q_model=q,
            config=AdaptiveStaticValidationConfig(),
        )
