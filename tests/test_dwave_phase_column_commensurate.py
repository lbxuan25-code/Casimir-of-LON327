from __future__ import annotations

import numpy as np

from lno327 import KuboConfig
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.commensurate_periodic import (
    CommensuratePeriodicGrid,
    integrate_commensurate_periodic_vector,
)
from validation.lib.dwave_iterated_adaptive import (
    assemble_dwave_static_primitives,
    build_dwave_static_integrand_context,
)
from validation.lib.dwave_phase_column_commensurate import (
    DWavePhaseColumnContext,
    assemble_phase_column_result,
    phase_column_result_as_audit_payload,
)
from validation.lib.dwave_phase_hessian_analysis import (
    analyze_dwave_phase_hessian_payload,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def test_reduced_phase_column_matches_full_primitive_integral():
    grid = CommensuratePeriodicGrid(nk=4, mx=1, my=1, max_points=100)
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    full = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        grid.q_model,
        config,
        pairing,
        FiniteQEngineOptions(),
    )

    full_integral = integrate_commensurate_periodic_vector(
        grid, full.evaluate_complex, chunk_size=5
    )
    components, _, _ = assemble_dwave_static_primitives(
        full,
        full_integral.value,
        metadata={"test": True},
    )

    reduced_context = DWavePhaseColumnContext(full)
    reduced_integral = integrate_commensurate_periodic_vector(
        grid, reduced_context.evaluate_complex, chunk_size=5
    )
    reduced = assemble_phase_column_result(
        reduced_context,
        reduced_integral.value,
    )

    qx, qy = grid.q_model
    u = np.asarray([0.0, qx, qy], dtype=complex)
    w_phase = -2j * pairing.delta0_eV
    expected_left = complex((u @ components.em_collective_left)[1])
    expected_right = complex((components.collective_em_right @ u)[1])
    expected_bubble = complex(w_phase * components.collective_bubble[1, 1])
    expected_counterterm = complex(w_phase * components.collective_counterterm[1, 1])

    assert np.allclose(reduced.left_em_collective_phase, expected_left, rtol=1e-12, atol=1e-12)
    assert np.allclose(reduced.right_collective_em_phase, expected_right, rtol=1e-12, atol=1e-12)
    assert np.allclose(reduced.phase_rotation_bubble, expected_bubble, rtol=1e-12, atol=1e-12)
    assert np.allclose(reduced.phase_rotation_counterterm, expected_counterterm, rtol=1e-12, atol=1e-12)


def test_reduced_phase_column_payload_is_consumed_by_postprocessor():
    grid = CommensuratePeriodicGrid(nk=4, mx=1, my=1, max_points=100)
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    full = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        grid.q_model,
        config,
        pairing,
        FiniteQEngineOptions(),
    )
    context = DWavePhaseColumnContext(full)
    integral = integrate_commensurate_periodic_vector(
        grid, context.evaluate_complex, chunk_size=8
    )
    result = assemble_phase_column_result(context, integral.value)
    payload = phase_column_result_as_audit_payload(result)

    analysis = analyze_dwave_phase_hessian_payload(payload)

    assert np.allclose(
        analysis.left.required_counterterm_multiplier,
        result.left_required_counterterm_multiplier,
    )
    assert np.allclose(
        analysis.right.required_counterterm_multiplier,
        result.right_required_counterterm_multiplier,
    )
    assert analysis.diagnostic_only
    assert not analysis.valid_for_casimir_input
