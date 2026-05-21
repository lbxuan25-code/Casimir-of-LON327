import numpy as np

from lno327 import (
    CasimirSetup,
    ConductivityTensor,
    KuboConfig,
    anisotropy_delta,
    anisotropy_summary,
    bosonic_matsubara_energy_eV,
    casimir_torque_integrand,
    conductivity_eigensystem,
    conductivity_matrix_diagnostics,
    k_weights,
    kubo_conductivity_imag_axis,
    kubo_conductivity_real_axis,
    rotate_conductivity,
    uniform_bz_mesh,
)
from lno327.casimir import matsubara_frequency, reflection_matrix_weak_2d


def test_conductivity_rotation_preserves_trace():
    tensor = ConductivityTensor(xx=3.0, yy=1.0)
    rotated = rotate_conductivity(tensor, np.pi / 7.0)

    assert np.isclose(rotated.xx + rotated.yy, 4.0)
    assert np.isclose(anisotropy_delta(tensor), 0.5)


def test_isotropic_reflection_has_no_cross_polarization():
    refl = reflection_matrix_weak_2d(1e12, 2e6, 0.4, ConductivityTensor(xx=1e-4, yy=1e-4))

    assert np.isclose(refl[0, 1], 0.0)
    assert np.isclose(refl[1, 0], 0.0)


def test_isotropic_torque_integrand_vanishes():
    setup = CasimirSetup(temperature=30.0, distance=30e-9)
    xi = matsubara_frequency(1, setup.temperature)
    tensor = ConductivityTensor(xx=1e-4, yy=1e-4)

    torque = casimir_torque_integrand(setup, xi, 1e6, 0.2, 0.7, tensor, tensor)

    assert abs(torque) < 1e-20


def test_kubo_conductivity_returns_finite_tensor():
    k_points = np.array([[0.0, 0.0], [0.4, 0.2], [-0.3, 0.5]])
    config = KuboConfig.from_kelvin(omega_eV=0.1, temperature_K=30.0, output_si=False)

    sigma = kubo_conductivity_imag_axis(k_points, config)

    assert np.isfinite(sigma.matrix()).all()
    np.testing.assert_allclose(sigma.matrix(), sigma.matrix().conjugate().T, atol=1e-12)


def test_kubo_conductivity_vanishes_for_zero_velocity_vertices():
    def zero_velocity(_kx, _ky, _direction):
        return np.zeros((4, 4))

    config = KuboConfig(omega_eV=0.2, temperature_eV=0.01, output_si=False)

    sigma = kubo_conductivity_imag_axis([(0.1, 0.2)], config, velocity=zero_velocity)

    np.testing.assert_allclose(sigma.matrix(), np.zeros((2, 2)))


def test_bosonic_matsubara_energy_uses_eV_units():
    assert np.isclose(bosonic_matsubara_energy_eV(0, 30.0), 0.0)
    assert np.isclose(bosonic_matsubara_energy_eV(1, 30.0), 2.0 * np.pi * 30.0 * 8.617333262e-5)


def test_uniform_bz_mesh_and_weights_are_normalized():
    mesh = uniform_bz_mesh(4, 3)
    weights = k_weights(mesh)

    assert mesh.shape == (12, 2)
    assert np.isclose(weights.sum(), 1.0)


def test_conductivity_eigensystem_shapes():
    config = KuboConfig(omega_eV=0.1, temperature_eV=0.01, output_si=False)
    eigensystem = conductivity_eigensystem(0.2, -0.3, config)

    assert eigensystem.energies_eV.shape == (4,)
    assert eigensystem.states.shape == (4, 4)
    assert eigensystem.velocity_x_band.shape == (4, 4)
    assert eigensystem.velocity_y_band.shape == (4, 4)


def test_conductivity_matrix_diagnostics_fields():
    tensor = ConductivityTensor(xx=3.0, yy=1.0, xy=0.5, yx=-0.25)

    diagnostics = conductivity_matrix_diagnostics(tensor)

    assert set(diagnostics) == {
        "sigma_matrix",
        "eigenvalues",
        "eigenvectors",
        "anisotropy_delta",
        "offdiag_norm",
        "relative_xx_yy_error",
    }
    np.testing.assert_allclose(diagnostics["sigma_matrix"], tensor.matrix())
    assert diagnostics["eigenvalues"].shape == (2,)
    assert diagnostics["eigenvectors"].shape == (2, 2)
    assert np.isclose(diagnostics["anisotropy_delta"], 0.5)
    assert np.isclose(diagnostics["offdiag_norm"], np.sqrt(0.5**2 + 0.25**2))
    assert np.isclose(diagnostics["relative_xx_yy_error"], 1.0)


def test_c4_symmetric_bz_conductivity_has_isotropic_diagonal_and_zero_xy():
    mesh = uniform_bz_mesh(10)
    config = KuboConfig.from_kelvin(omega_eV=0.2, temperature_K=30.0, eta_eV=0.02, output_si=False)

    sigma = kubo_conductivity_imag_axis(mesh, config, k_weights(mesh))
    summary = anisotropy_summary(sigma)

    assert np.isclose(sigma.xx, sigma.yy, atol=1e-10)
    assert abs(sigma.xy) < 1e-10
    assert abs(sigma.yx) < 1e-10
    assert abs(summary["delta"]) < 1e-10


def test_real_axis_kubo_returns_finite_tensor():
    mesh = uniform_bz_mesh(4)
    config = KuboConfig(omega_eV=0.2, temperature_eV=0.01, eta_eV=0.03, output_si=False)

    sigma = kubo_conductivity_real_axis(mesh, config, k_weights(mesh))

    assert np.isfinite(sigma.matrix()).all()
