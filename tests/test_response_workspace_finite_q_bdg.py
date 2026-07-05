import numpy as np
import pytest
from dataclasses import replace

from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from lno327.models.lno327_four_orbital.collective import build_pairing_ansatz
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.config import KuboConfig
from lno327.response.finite_q_bdg import (
    finite_q_bdg_response_from_model_ansatz,
    finite_q_bdg_response_from_workspace,
    precompute_finite_q_bdg_workspace_from_model_ansatz,
)


def _inputs(pairing: str, q: np.ndarray, collective_mode: str):
    points = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float)
    weights = np.array([0.4, 0.6])
    config = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    amp = PairingAmplitudes(delta0_eV=0.04)
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=amp)
    ansatz = build_pairing_ansatz(pairing)
    options = FiniteQEngineOptions(include_phase_correction=False, collective_mode=collective_mode)
    return spec, ansatz, q, points, weights, config, amp, options


@pytest.mark.parametrize(
    ("pairing", "q", "collective_mode"),
    [
        ("spm", np.array([0.0, 0.0]), "phase_only"),
        ("spm", np.array([0.01, 0.0]), "none"),
        ("spm", np.array([0.01, 0.0]), "amplitude_phase"),
        ("dwave", np.array([0.02, -0.01]), "amplitude_phase"),
    ],
)
def test_finite_q_bdg_workspace_matches_direct_and_keeps_metadata_not_casimir_ready(pairing, q, collective_mode):
    spec, ansatz, q, points, weights, config, amp, options = _inputs(pairing, q, collective_mode)
    workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
        spec, ansatz, q, points, weights, config, amp, options
    )
    direct = finite_q_bdg_response_from_model_ansatz(spec, ansatz, config.omega_eV, q, points, weights, config, amp, options)
    from_ws = finite_q_bdg_response_from_workspace(workspace, config=config)

    np.testing.assert_allclose(from_ws.bare_bubble, direct.bare_bubble)
    np.testing.assert_allclose(from_ws.direct, direct.direct)
    np.testing.assert_allclose(from_ws.bare_total, direct.bare_total)
    np.testing.assert_allclose(from_ws.gauge_restored, direct.gauge_restored)
    assert from_ws.metadata["valid_for_casimir_input"] is False
    assert workspace.metadata["valid_for_casimir_input"] is False
    assert from_ws.metadata["shared_eigenbasis_q0"] is bool(np.linalg.norm(q) <= 1e-14)


def test_finite_q_bdg_workspace_multi_omega_reuses_precomputed_expensive_objects(monkeypatch):
    spec, ansatz, q, points, weights, config, amp, options = _inputs(
        "spm",
        np.array([0.01, 0.0]),
        "amplitude_phase",
    )
    workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
        spec, ansatz, q, points, weights, config, amp, options
    )

    import lno327.response.finite_q_bdg as module

    def fail(*_args, **_kwargs):
        raise AssertionError("workspace evaluation should not rebuild omega-independent objects")

    monkeypatch.setattr(module, "bdg_eigensystem_from_model_pairing", fail)
    monkeypatch.setattr(module, "bdg_vector_vertex_from_spec", fail)
    monkeypatch.setattr(module, "bdg_contact_vertex_from_spec", fail)
    monkeypatch.setattr(module, "thermal_expectation_bdg_from_hamiltonian", fail)

    for omega in (0.01, 0.03, 0.07):
        eval_config = replace(config, omega_eV=omega)
        response = finite_q_bdg_response_from_workspace(workspace, config=eval_config)
        assert response.bare_total.shape == (3, 3)
        assert response.metadata["valid_for_casimir_input"] is False
