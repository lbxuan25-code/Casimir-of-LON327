from __future__ import annotations

import warnings

import numpy as np
import pytest

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.response.finite_q_bdg import finite_q_bdg_response_from_model_ansatz
from lno327.response.phase_hessian import (
    apply_phase_hessian_policy_to_components,
    finite_q_bdg_response_from_model_ansatz_with_phase_hessian,
    nearest_neighbor_dwave_bond_metric,
    validate_phase_hessian_policy,
)
from lno327.response.workspace import (
    finite_q_bdg_response_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


_RESPONSE_FIELDS = (
    "bare_bubble",
    "direct",
    "bare_total",
    "collective_bubble",
    "collective_counterterm",
    "collective_total",
    "em_collective_left",
    "collective_em_right",
    "amplitude_phase_schur",
    "gauge_restored",
)


def _problem(*, pairing_name: str = "dwave", phase_vertex: str = "bond_endpoint_gauge"):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex=phase_vertex)
    pairing = model.build_pairing_params(0.1)
    points = uniform_bz_mesh(4)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    q = np.asarray([0.03, 0.02], dtype=float)
    return model, ansatz, pairing, points, weights, config, q


def _direct(problem, options: FiniteQEngineOptions):
    model, ansatz, pairing, points, weights, config, q = problem
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return finite_q_bdg_response_from_model_ansatz_with_phase_hessian(
            model.spec,
            ansatz,
            0.0,
            q,
            points,
            weights,
            config,
            pairing,
            options,
        )


def test_phase_hessian_policy_validation_and_metric():
    assert validate_phase_hessian_policy("q_independent") == "q_independent"
    assert (
        validate_phase_hessian_policy("nearest_neighbor_bond_metric")
        == "nearest_neighbor_bond_metric"
    )
    with pytest.raises(ValueError, match="phase_hessian_policy"):
        validate_phase_hessian_policy("unknown")

    q = np.asarray([0.4, -0.2])
    expected = 0.5 * (np.cos(0.2) ** 2 + np.cos(-0.1) ** 2)
    assert nearest_neighbor_dwave_bond_metric(q) == pytest.approx(expected)
    assert nearest_neighbor_dwave_bond_metric(np.zeros(2)) == pytest.approx(1.0)


def test_q_independent_policy_preserves_default_numerics():
    problem = _problem()
    model, ansatz, pairing, points, weights, config, q = problem
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = finite_q_bdg_response_from_model_ansatz(
            model.spec,
            ansatz,
            0.0,
            q,
            points,
            weights,
            config,
            pairing,
            options,
        )
    wrapped = _direct(problem, options)
    for field in _RESPONSE_FIELDS:
        np.testing.assert_allclose(
            getattr(wrapped, field),
            getattr(raw, field),
            rtol=1e-13,
            atol=1e-13,
            err_msg=field,
        )
    assert wrapped.metadata["phase_hessian_policy"] == "q_independent"
    assert wrapped.metadata["phase_hessian_policy_opt_in"] is False
    assert wrapped.metadata["phase_hessian_multiplier"] == pytest.approx(1.0)


def test_bond_metric_changes_only_phase_counterterm_diagonal():
    problem = _problem()
    model, ansatz, pairing, points, weights, config, q = problem
    base_options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    corrected_options = FiniteQEngineOptions(
        phase_hessian_policy="nearest_neighbor_bond_metric"
    )
    base = _direct(problem, base_options)
    corrected = _direct(problem, corrected_options)

    metric = nearest_neighbor_dwave_bond_metric(q)
    np.testing.assert_array_equal(
        corrected.collective_counterterm[0:1, :],
        base.collective_counterterm[0:1, :],
    )
    assert corrected.collective_counterterm[1, 0] == base.collective_counterterm[1, 0]
    assert corrected.collective_counterterm[1, 1] == pytest.approx(
        metric * base.collective_counterterm[1, 1]
    )
    assert corrected.metadata["phase_hessian_policy"] == "nearest_neighbor_bond_metric"
    assert corrected.metadata["phase_hessian_policy_opt_in"] is True
    assert corrected.metadata["phase_hessian_changed_only_22"] is True
    assert corrected.metadata["valid_for_casimir_input"] is False


def test_bond_metric_rejects_unsupported_ansatz_and_vertex():
    for pairing_name, phase_vertex, message in (
        ("spm", "bond_endpoint_gauge", "d-wave ansatz"),
        ("dwave", "midpoint", "bond_endpoint_gauge"),
    ):
        problem = _problem(pairing_name=pairing_name, phase_vertex=phase_vertex)
        model, ansatz, pairing, points, weights, config, q = problem
        options = FiniteQEngineOptions()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            components = finite_q_bdg_response_from_model_ansatz(
                model.spec,
                ansatz,
                0.0,
                q,
                points,
                weights,
                config,
                pairing,
                options,
            )
        with pytest.raises(ValueError, match=message):
            apply_phase_hessian_policy_to_components(
                components,
                ansatz,
                q,
                "nearest_neighbor_bond_metric",
            )


def test_direct_and_optimized_q_workspace_share_bond_metric_policy():
    problem = _problem()
    model, ansatz, pairing, points, weights, config, q = problem
    options = FiniteQEngineOptions(
        phase_hessian_policy="nearest_neighbor_bond_metric"
    )
    direct = _direct(problem, options)

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        optimized = finite_q_bdg_response_from_q_workspace(workspace, 0.0)

    for field in _RESPONSE_FIELDS:
        np.testing.assert_allclose(
            optimized.__getattribute__(field),
            direct.__getattribute__(field),
            rtol=2e-11,
            atol=2e-12,
            err_msg=field,
        )
    assert optimized.metadata["phase_hessian_policy"] == "nearest_neighbor_bond_metric"
    assert optimized.metadata["phase_hessian_multiplier"] == pytest.approx(
        nearest_neighbor_dwave_bond_metric(q)
    )
