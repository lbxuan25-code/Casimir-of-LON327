from __future__ import annotations

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.response.finite_q_material_workspace_batched import (
    precompute_finite_q_material_workspace_batched,
    supports_batched_finite_q_material_workspace,
)
from lno327.response.finite_q_optimized import (
    precompute_finite_q_material_workspace_from_model_ansatz,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import (
    get_finite_q_validation_model,
)


def test_batched_material_workspace_matches_scalar_reference() -> None:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(
        "dwave",
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(0.1)
    points = uniform_bz_mesh(4)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.01,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")

    assert supports_batched_finite_q_material_workspace(model.spec, ansatz)

    scalar = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        config,
        pairing,
        options,
    )
    batched = precompute_finite_q_material_workspace_batched(
        model.spec,
        ansatz,
        points,
        weights,
        config,
        pairing,
        options,
    )

    np.testing.assert_allclose(
        batched.midpoint_energies,
        scalar.midpoint_energies,
        rtol=2e-12,
        atol=2e-13,
    )
    # Eigenvector phases are arbitrary, so compare reconstructed Hamiltonians.
    scalar_h = np.einsum(
        "kpi,ki,kqi->kpq",
        scalar.midpoint_states,
        scalar.midpoint_energies,
        np.conjugate(scalar.midpoint_states),
        optimize=True,
    )
    batched_h = np.einsum(
        "kpi,ki,kqi->kpq",
        batched.midpoint_states,
        batched.midpoint_energies,
        np.conjugate(batched.midpoint_states),
        optimize=True,
    )
    np.testing.assert_allclose(batched_h, scalar_h, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        batched.midpoint_occupations,
        scalar.midpoint_occupations,
        rtol=2e-12,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        batched.collective_counterterm_matrix,
        scalar.collective_counterterm_matrix,
        rtol=2e-12,
        atol=2e-13,
    )
    assert batched.metadata["material_workspace_implementation"] == (
        "batched_model_capability"
    )
