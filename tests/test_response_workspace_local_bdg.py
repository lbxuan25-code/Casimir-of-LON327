import numpy as np

from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.config import KuboConfig
from lno327.response.local_bdg import (
    bdg_local_diamagnetic_kernel_from_workspace,
    bdg_local_paramagnetic_kernel_imag_axis_from_workspace,
    bdg_local_superconducting_response_imag_axis_from_workspace,
    bdg_local_total_kernel_imag_axis,
    bdg_local_total_kernel_imag_axis_from_workspace,
    precompute_bdg_local_workspace_from_model,
)


def test_bdg_local_workspace_matches_direct_and_reuses_entries():
    points = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float)
    weights = np.array([0.4, 0.6])
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    spec = LNO327FourOrbitalSpec()

    for channel in ("spm", "dwave"):
        workspace = precompute_bdg_local_workspace_from_model(spec, channel, points, config, weights)
        direct = bdg_local_total_kernel_imag_axis(spec, channel, points, config, weights)
        total = bdg_local_total_kernel_imag_axis_from_workspace(workspace, config)
        para = bdg_local_paramagnetic_kernel_imag_axis_from_workspace(workspace, config)
        dia = bdg_local_diamagnetic_kernel_from_workspace(workspace, config)
        response = bdg_local_superconducting_response_imag_axis_from_workspace(workspace, config)

        np.testing.assert_allclose(total.paramagnetic, direct.paramagnetic)
        np.testing.assert_allclose(total.diamagnetic, direct.diamagnetic)
        np.testing.assert_allclose(total.total, dia - para)
        np.testing.assert_allclose(response.sigma_like_response, total.total / config.omega_eV)
        assert len(workspace.entries) == len(points)
