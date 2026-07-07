from __future__ import annotations

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.workflows.finite_q_engine import FiniteQEngineOptions, bdg_finite_q_response_imag_axis_from_workspace
from lno327.response.finite_q_bdg import precompute_finite_q_bdg_workspace_from_model_ansatz
from validation.lib.finite_q_integrated_ward_chain import evaluate_integrated_ward_chain
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _payload_vector(payload):
    return np.asarray([entry["real"] + 1j * entry["imag"] for entry in payload["vector"]])


def test_integrated_ward_chain_payload_reconstructs_left_full_chain_residual():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    amp = model.build_pairing_params(delta0_eV=0.1)
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    points = uniform_bz_mesh(3)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    options = FiniteQEngineOptions(
        current_vertex="peierls",
        collective_mode="amplitude_phase",
        collective_counterterm="goldstone_gap_equation",
        include_phase_phase_direct=True,
    )
    workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        np.asarray([0.02, 0.0]),
        points,
        weights,
        config,
        amp,
        options,
    )
    response = bdg_finite_q_response_imag_axis_from_workspace(workspace, config=config)

    payload = evaluate_integrated_ward_chain(workspace=workspace, response=response, delta0_eV=0.1)
    chain = payload["left_chain"]
    bubble_collective = _payload_vector(chain["bubble_collective_contraction"])
    contact_target = _payload_vector(chain["contact_target_minus_direct_contraction"])
    full_residual = _payload_vector(chain["full_chain_residual"])

    np.testing.assert_allclose(full_residual, bubble_collective - contact_target)
    assert payload["identity_version"] == "finite_q_integrated_ward_chain_v1"
    assert payload["diagnostic_role"] == "integrated_ward_proof_chain_not_a_new_ward_criterion"
    assert payload["valid_for_casimir_input"] is False


def test_integrated_ward_chain_requires_vector_q_model():
    class DummyWorkspace:
        q_model = np.asarray([0.01, 0.0, 0.0])
        config = type("Config", (), {"omega_eV": 0.01})()
        ansatz = type("Ansatz", (), {"name": "dummy"})()
        entries = ()

    class DummyResponse:
        bare_bubble = np.zeros((3, 3), dtype=complex)
        direct = np.zeros((3, 3), dtype=complex)
        collective_em_right = np.zeros((2, 3), dtype=complex)

    try:
        evaluate_integrated_ward_chain(workspace=DummyWorkspace(), response=DummyResponse(), delta0_eV=0.1)
    except ValueError as exc:
        assert "q_model" in str(exc)
    else:
        raise AssertionError("bad q_model shape should raise ValueError")
