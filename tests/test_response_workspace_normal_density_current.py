import numpy as np
import pytest
from dataclasses import replace

from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.config import KuboConfig
from lno327.response.normal_density_current import (
    normal_density_current_response_imag_axis,
    normal_density_current_response_imag_axis_from_workspace,
    normal_physical_density_current_response_components_imag_axis,
    normal_physical_density_current_response_components_imag_axis_from_workspace,
    normal_physical_density_current_response_imag_axis_from_workspace,
    precompute_normal_physical_density_current_workspace_from_model,
    precompute_normal_density_current_workspace_from_model,
)


def test_normal_density_current_q0_and_finite_q_direct_paths_are_finite_and_fallback_stays_disabled():
    points = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float)
    weights = np.array([0.4, 0.6])
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)

    for q in (np.array([0.0, 0.0]), np.array([0.17, -0.09])):
        components = normal_physical_density_current_response_components_imag_axis(points, config, q, weights)
        assert components["total"].shape == (3, 3)
        assert np.all(np.isfinite(components["total"]))
        workspace = precompute_normal_density_current_workspace_from_model(
            LNO327FourOrbitalSpec(),
            points,
            config,
            q,
            weights,
        )
        np.testing.assert_allclose(
            normal_density_current_response_imag_axis_from_workspace(workspace, config),
            normal_density_current_response_imag_axis(points, config, q, weights),
        )
        physical_workspace = precompute_normal_physical_density_current_workspace_from_model(
            LNO327FourOrbitalSpec(),
            points,
            config,
            q,
            weights,
        )
        direct_physical = normal_physical_density_current_response_components_imag_axis(points, config, q, weights)
        from_workspace = normal_physical_density_current_response_components_imag_axis_from_workspace(
            physical_workspace,
            config,
        )
        for key in ("bubble", "direct", "total"):
            np.testing.assert_allclose(from_workspace[key], direct_physical[key])
        np.testing.assert_allclose(
            normal_physical_density_current_response_imag_axis_from_workspace(physical_workspace, config),
            direct_physical["total"],
        )

    with pytest.raises(ValueError, match="explicit hamiltonian fallback is not supported"):
        normal_physical_density_current_response_components_imag_axis(
            points,
            config,
            np.array([0.0, 0.0]),
            weights,
            hamiltonian=lambda kx, ky: np.eye(4),
        )


def test_normal_density_current_workspace_multi_omega_reuses_precomputed_objects(monkeypatch):
    points = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float)
    weights = np.array([0.4, 0.6])
    q = np.array([0.17, -0.09])
    config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4, output_si=False)
    spec = LNO327FourOrbitalSpec()
    workspace = precompute_normal_physical_density_current_workspace_from_model(
        spec,
        points,
        config,
        q,
        weights,
    )

    import lno327.response.normal_density_current as module

    def fail(*_args, **_kwargs):
        raise AssertionError("workspace evaluation should not rebuild omega-independent objects")

    monkeypatch.setattr(module, "diagonalize_hermitian", fail)
    monkeypatch.setattr(spec, "peierls_hamiltonian_vector_vertex", fail)
    monkeypatch.setattr(spec, "peierls_hamiltonian_contact_vertex", fail)

    for omega in (0.02, 0.08, 0.13):
        eval_config = replace(config, omega_eV=omega)
        components = normal_physical_density_current_response_components_imag_axis_from_workspace(
            workspace,
            eval_config,
        )
        assert components["total"].shape == (3, 3)
