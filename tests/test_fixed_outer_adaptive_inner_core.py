from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lno327 import KuboConfig
from lno327.numerics.fixed_outer_adaptive_inner import (
    FixedOuterAdaptiveInnerOptions,
    integrate_fixed_outer_adaptive_inner_orientation,
)
from lno327.response.arbitrary_q_pointwise_primitives import (
    evaluate_arbitrary_q_pointwise_primitives,
    pack_complex_density_to_real,
    unpack_real_integral_to_complex,
)
from lno327.response.finite_q_material_workspace_batched import (
    precompute_finite_q_material_workspace_batched,
)
from lno327.response.periodic_bz_grid import build_periodic_bz_grid
from lno327.response.primitive_kernel_v2 import evaluate_primitive_batch_from_material
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _inputs():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    return model, ansatz, pairing, config, options


def test_pointwise_density_weighted_sum_matches_existing_integrated_kernel() -> None:
    model, ansatz, pairing, config, options = _inputs()
    grid = build_periodic_bz_grid(4, (0.5, 0.5))
    q = np.asarray([0.03, 0.02])
    xi = np.asarray([0.0, 0.025])
    material = precompute_finite_q_material_workspace_batched(
        model.spec,
        ansatz,
        grid.points,
        grid.weights,
        config,
        pairing,
        options,
    )
    reference = evaluate_primitive_batch_from_material(
        material,
        q,
        xi,
        include_counterterm=True,
    )
    pointwise = evaluate_arbitrary_q_pointwise_primitives(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        config=config,
        q_model=q,
        xi_eV_values=xi,
        k_points=grid.points,
        options=options,
    )
    integrated = np.einsum("k,kw->w", grid.weights, pointwise.densities, optimize=True)
    np.testing.assert_allclose(integrated, reference.packed, rtol=2e-11, atol=2e-12)
    assert pointwise.operator_ward.passed
    assert pointwise.metrics.q_workspace_build_count == 1
    assert pointwise.metrics.counterterm_q0_workspace_build_count == 1


def test_pointwise_density_is_independent_of_probe_batch_partition() -> None:
    model, ansatz, pairing, config, options = _inputs()
    points = build_periodic_bz_grid(4, (0.5, 0.5)).points[:6]
    q = np.asarray([0.03, 0.02])
    xi = np.asarray([0.0, 0.025])
    all_at_once = evaluate_arbitrary_q_pointwise_primitives(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        config=config,
        q_model=q,
        xi_eV_values=xi,
        k_points=points,
        options=options,
    ).densities
    split = np.concatenate(
        [
            evaluate_arbitrary_q_pointwise_primitives(
                spec=model.spec,
                ansatz=ansatz,
                pairing=pairing,
                config=config,
                q_model=q,
                xi_eV_values=xi,
                k_points=chunk,
                options=options,
            ).densities
            for chunk in (points[:2], points[2:])
        ],
        axis=0,
    )
    np.testing.assert_allclose(split, all_at_once, rtol=2e-11, atol=2e-12)


def test_complex_real_pointwise_contract_roundtrips() -> None:
    value = np.asarray([1 + 2j, -3 + 0.25j, -4j])
    packed = pack_complex_density_to_real(value)
    np.testing.assert_allclose(unpack_real_integral_to_complex(packed), value)


def test_retained_integrator_matches_analytic_vector_integral_in_both_orders() -> None:
    options = FixedOuterAdaptiveInnerOptions(
        epsabs=1e-10,
        epsrel=1e-10,
        inner_limit=30,
        max_point_evaluations=20_000,
        split_points=(0.0,),
    )

    def integrand(x: float, y: float) -> np.ndarray:
        return np.asarray([1.0, x * x, y * y, x * y], dtype=float)

    expected = np.asarray(
        [
            4.0 * np.pi**2,
            4.0 * np.pi**4 / 3.0,
            4.0 * np.pi**4 / 3.0,
            0.0,
        ]
    )
    for order in ("xy", "yx"):
        result = integrate_fixed_outer_adaptive_inner_orientation(
            integrand,
            order=order,
            outer_order=8,
            options=options,
        )
        assert result.success
        np.testing.assert_allclose(result.value, expected, rtol=1e-10, atol=1e-10)


def test_method_registry_prevents_rejected_routes_from_becoming_primary() -> None:
    registry = json.loads(
        Path("docs/numerical_method_registry.json").read_text(encoding="utf-8")
    )
    methods = registry["methods"]
    assert methods["DWaveFixedOuterAdaptiveInner-v1"]["status"] == "canonical_target"
    assert methods["DWaveFixedOuterAdaptiveInner-v1"]["main_route_allowed"] is False
    assert methods["FullBZGL2GL3CellAdaptive"]["status"] == "rejected"
    assert methods["FermiWindowLocalRefinement"]["main_route_allowed"] is False
