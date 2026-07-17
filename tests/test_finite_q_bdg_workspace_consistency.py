from __future__ import annotations

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.casimir.microscopic_model import get_finite_q_microscopic_model
from lno327.response.finite_q_bdg import (
    finite_q_bdg_response_from_model_ansatz,
    finite_q_bdg_response_from_workspace,
    precompute_finite_q_bdg_workspace_from_model_ansatz,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions


_RESPONSE_MATRIX_NAMES = (
    "bare_bubble",
    "direct",
    "bare_total",
    "collective_bubble",
    "collective_counterterm",
    "collective_total",
    "em_collective_left",
    "collective_em_right",
    "amplitude_phase_schur",
)


def _assert_response_matrices_close(left, right):
    for name in _RESPONSE_MATRIX_NAMES:
        np.testing.assert_allclose(
            getattr(left, name),
            getattr(right, name),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"response matrix mismatch for {name}",
        )


# Keep this as a small algebra/assembly regression, not a full validation run.
def test_precomputed_workspace_matches_direct_finite_q_bdg_response_for_dwave():
    model = get_finite_q_microscopic_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing_params = model.build_pairing_params(0.1)
    points = uniform_bz_mesh(3)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.01,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    q_model = np.asarray([0.02, 0.0], dtype=float)
    options = FiniteQEngineOptions()

    direct = finite_q_bdg_response_from_model_ansatz(
        model.spec,
        ansatz,
        config.omega_eV,
        q_model,
        points,
        weights,
        config,
        pairing_params,
        options,
    )
    workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        q_model,
        points,
        weights,
        config,
        pairing_params,
        options,
    )
    cached = finite_q_bdg_response_from_workspace(workspace, config=config)

    _assert_response_matrices_close(direct, cached)
