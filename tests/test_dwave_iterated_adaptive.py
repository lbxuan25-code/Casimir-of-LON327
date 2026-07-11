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
from validation.lib.dwave_iterated_adaptive import (
    assemble_dwave_static_primitives,
    build_dwave_static_integrand_context,
    pack_complex_vector,
    unpack_complex_vector,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.iterated_adaptive import (
    EvaluationBudgetExceeded,
    IteratedAdaptiveOptions,
    iterated_adaptive_integral,
)


def test_iterated_adaptive_matches_analytic_vector_in_both_orders():
    def integrand(kx: float, ky: float) -> np.ndarray:
        return np.asarray([1.0, kx, ky, kx * ky, kx * kx + ky * ky])

    options = IteratedAdaptiveOptions(
        epsabs=1e-10,
        epsrel=1e-10,
        inner_limit=40,
        outer_limit=40,
        max_point_evaluations=50_000,
        quadrature="gk15",
        split_points=(0.0,),
    )
    expected = np.asarray(
        [4.0 * np.pi**2, 0.0, 0.0, 0.0, 8.0 * np.pi**4 / 3.0]
    )
    xy = iterated_adaptive_integral(integrand, order="xy", options=options)
    yx = iterated_adaptive_integral(integrand, order="yx", options=options)
    assert xy.success
    assert yx.success
    assert np.allclose(xy.value, expected, rtol=1e-10, atol=1e-10)
    assert np.allclose(yx.value, expected, rtol=1e-10, atol=1e-10)
    assert np.allclose(xy.value, yx.value, rtol=1e-12, atol=1e-12)


def test_iterated_adaptive_point_budget_is_fail_closed():
    options = IteratedAdaptiveOptions(max_point_evaluations=1)
    with pytest.raises(EvaluationBudgetExceeded):
        iterated_adaptive_integral(
            lambda kx, ky: np.asarray([kx + ky]),
            order="xy",
            options=options,
        )


def test_complex_vector_pack_round_trip():
    value = np.arange(48, dtype=float) + 1j * np.arange(48, dtype=float)[::-1]
    assert np.array_equal(unpack_complex_vector(pack_complex_vector(value)), value)


def test_pointwise_primitive_average_matches_periodic_workspace():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    q = np.asarray([0.03, 0.02])
    options = FiniteQEngineOptions()
    points, weights = periodic_shift_mesh(4, (0.5, 0.5))

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

    context = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        q,
        config,
        pairing,
        options,
    )
    primitive = np.tensordot(weights, context.evaluate_complex(points), axes=(0, 0))
    actual_components, actual_rhs, _ = assemble_dwave_static_primitives(
        context,
        primitive,
        metadata={"test": "uniform_average"},
    )

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
            getattr(actual_components, field),
            getattr(expected_components, field),
            rtol=2e-11,
            atol=2e-12,
        ), field
    assert np.allclose(actual_rhs.left, expected_rhs.left, rtol=2e-11, atol=2e-12)
    assert np.allclose(actual_rhs.right, expected_rhs.right, rtol=2e-11, atol=2e-12)
    assert np.isclose(
        complex(actual_components.metadata["phase_phase_direct_plus_convention"]),
        complex(expected_components.metadata["phase_phase_direct_plus_convention"]),
        rtol=2e-11,
        atol=2e-12,
    )
