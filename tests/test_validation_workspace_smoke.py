import numpy as np
from dataclasses import replace

from lno327.models.lno327_four_orbital.collective import build_pairing_ansatz
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.config import KuboConfig
from lno327.response.workspace import (
    finite_q_bdg_response_from_workspace,
    kubo_conductivity_imag_axis_from_workspace,
    precompute_finite_q_bdg_workspace_from_model_ansatz,
    precompute_normal_local_workspace_from_model,
)


def test_validation_can_build_explicit_workspace_without_outputs():
    points = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float)
    weights = np.array([0.4, 0.6])
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    workspace = precompute_normal_local_workspace_from_model(LNO327FourOrbitalSpec(), points, config, weights)

    sigma = kubo_conductivity_imag_axis_from_workspace(workspace, config)

    assert sigma.matrix().shape == (2, 2)
    assert np.all(np.isfinite(sigma.matrix()))


def test_validation_can_reuse_finite_q_bdg_workspace_without_outputs():
    points = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float)
    weights = np.array([0.4, 0.6])
    config = KuboConfig(omega_eV=0.02, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    amp = PairingAmplitudes(delta0_eV=0.04)
    workspace = precompute_finite_q_bdg_workspace_from_model_ansatz(
        LNO327FourOrbitalSpec(pairing_amplitudes=amp),
        build_pairing_ansatz("spm"),
        np.array([0.02, 0.01]),
        points,
        weights,
        config,
        amp,
    )

    for omega in (0.02, 0.05):
        response = finite_q_bdg_response_from_workspace(workspace, config=replace(config, omega_eV=omega))
        assert response.bare_total.shape == (3, 3)
        assert response.metadata["valid_for_casimir_input"] is False
