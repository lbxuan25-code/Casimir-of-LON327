import numpy as np

from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.config import KuboConfig
from lno327.response.local_normal import (
    kubo_conductivity_imag_axis_from_model,
    kubo_conductivity_imag_axis_from_workspace,
    kubo_conductivity_real_axis_from_model,
    kubo_conductivity_real_axis_from_workspace,
    precompute_normal_local_workspace_from_model,
)


def _mesh():
    return np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float), np.array([0.4, 0.6])


def test_normal_local_workspace_matches_direct_imag_and_real():
    points, weights = _mesh()
    spec = LNO327FourOrbitalSpec()
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    workspace = precompute_normal_local_workspace_from_model(spec, points, config, weights)

    np.testing.assert_allclose(
        kubo_conductivity_imag_axis_from_workspace(workspace, config).matrix(),
        kubo_conductivity_imag_axis_from_model(spec, points, config, weights).matrix(),
    )
    np.testing.assert_allclose(
        kubo_conductivity_real_axis_from_workspace(workspace, config).matrix(),
        kubo_conductivity_real_axis_from_model(spec, points, config, weights).matrix(),
    )


def test_normal_local_workspace_multi_omega_reuses_precomputed_eigensystems():
    points, weights = _mesh()
    spec = LNO327FourOrbitalSpec()
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    workspace = precompute_normal_local_workspace_from_model(spec, points, config, weights)
    first_states = tuple(item.states for item in workspace.eigensystems)

    other = KuboConfig(omega_eV=0.11, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    kubo_conductivity_imag_axis_from_workspace(workspace, other)
    kubo_conductivity_real_axis_from_workspace(workspace, other)

    assert first_states == tuple(item.states for item in workspace.eigensystems)
