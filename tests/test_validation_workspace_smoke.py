import numpy as np

from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.config import KuboConfig
from lno327.response.workspace import (
    kubo_conductivity_imag_axis_from_workspace,
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
