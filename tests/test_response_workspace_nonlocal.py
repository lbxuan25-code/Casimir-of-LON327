import numpy as np

from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.config import KuboConfig
from lno327.response.nonlocal_bdg import (
    bdg_current_current_kernel_imag_axis_from_model,
    bdg_current_current_kernel_imag_axis_from_workspace,
    precompute_bdg_nonlocal_workspace_from_model,
)
from lno327.response.nonlocal_normal import (
    normal_current_current_kernel_imag_axis_from_model,
    normal_current_current_kernel_imag_axis_from_workspace,
    precompute_normal_nonlocal_workspace_from_model,
)


def test_nonlocal_workspaces_match_direct_for_q_zero_and_finite_q():
    points = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float)
    weights = np.array([0.4, 0.6])
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    spec = LNO327FourOrbitalSpec()

    for q in (np.array([0.0, 0.0]), np.array([0.17, -0.09])):
        normal_ws = precompute_normal_nonlocal_workspace_from_model(spec, points, config, q, weights)
        np.testing.assert_allclose(
            normal_current_current_kernel_imag_axis_from_workspace(normal_ws, config),
            normal_current_current_kernel_imag_axis_from_model(spec, points, config, q, weights),
        )
        assert normal_ws.shared_eigenbasis_q0 is bool(np.all(q == 0.0))

        bdg_ws = precompute_bdg_nonlocal_workspace_from_model(spec, points, config, q, "spm", weights)
        np.testing.assert_allclose(
            bdg_current_current_kernel_imag_axis_from_workspace(bdg_ws, config),
            bdg_current_current_kernel_imag_axis_from_model(spec, points, config, q, "spm", weights),
        )
        assert bdg_ws.shared_eigenbasis_q0 is bool(np.all(q == 0.0))
