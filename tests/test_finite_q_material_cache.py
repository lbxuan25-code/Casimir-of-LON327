from __future__ import annotations

import numpy as np
import pytest

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.response.workspace import (
    finite_q_bdg_response_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


@pytest.mark.parametrize("pairing_name", ["spm", "dwave"])
def test_cached_material_counterterm_matches_ansatz_reference(pairing_name: str):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    points = uniform_bz_mesh(3)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.01,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    options = FiniteQEngineOptions()

    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        config,
        pairing,
        options,
    )
    reference = ansatz.hs_counterterm(config, points, weights, pairing)
    np.testing.assert_allclose(
        material.collective_counterterm_matrix,
        reference,
        rtol=2e-11,
        atol=2e-12,
    )
    assert material.metadata["duplicate_midpoint_eigensystem_passes"] == 0
    assert material.metadata["goldstone_counterterm_from_cached_midpoint_bands"] is True

    q_workspace = precompute_finite_q_q_workspace(material, np.asarray([0.03, 0.02]))
    response = finite_q_bdg_response_from_q_workspace(q_workspace, 0.01)
    np.testing.assert_allclose(
        response.collective_counterterm,
        reference,
        rtol=2e-11,
        atol=2e-12,
    )


def test_optimized_material_workspace_rejects_non_peierls_current_vertex():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("spm", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    points = uniform_bz_mesh(2)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.01,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    options = FiniteQEngineOptions(current_vertex="q0_velocity")

    with pytest.raises(ValueError, match="requires current_vertex='peierls'"):
        precompute_finite_q_material_workspace_from_model_ansatz(
            model.spec,
            ansatz,
            points,
            weights,
            config,
            pairing,
            options,
        )
