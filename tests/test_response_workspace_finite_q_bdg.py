import numpy as np

from lno327.finite_q_engine import FiniteQEngineOptions
from lno327.models.lno327_four_orbital.collective import build_pairing_ansatz
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.config import KuboConfig
from lno327.response.finite_q_bdg import (
    finite_q_bdg_response_from_model_ansatz,
    finite_q_bdg_response_from_workspace,
    precompute_finite_q_bdg_workspace_from_model_ansatz,
)


def test_finite_q_bdg_workspace_matches_direct_and_keeps_metadata_not_casimir_ready():
    points = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float)
    weights = np.array([0.4, 0.6])
    q = np.array([0.01, 0.0])
    config = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    amp = PairingAmplitudes(delta0_eV=0.04)
    spec = LNO327FourOrbitalSpec(pairing_amplitudes=amp)
    ansatz = build_pairing_ansatz("spm")
    options = FiniteQEngineOptions(include_phase_correction=False)

    workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
        spec, ansatz, q, points, weights, config, amp, options
    )
    direct = finite_q_bdg_response_from_model_ansatz(spec, ansatz, config.omega_eV, q, points, weights, config, amp, options)
    from_ws = finite_q_bdg_response_from_workspace(workspace, config=config)

    np.testing.assert_allclose(from_ws.bare_total, direct.bare_total)
    assert from_ws.metadata["valid_for_casimir_input"] is False
    assert workspace.metadata["valid_for_casimir_input"] is False
