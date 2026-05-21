import numpy as np

from lno327 import (
    CasimirSetup,
    ConductivityTensor,
    KuboConfig,
    anisotropy_delta,
    bosonic_matsubara_energy_eV,
    casimir_torque_integrand,
    kubo_conductivity,
    rotate_conductivity,
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

    sigma = kubo_conductivity(k_points, config)

    assert np.isfinite(sigma.matrix()).all()
    np.testing.assert_allclose(sigma.matrix(), sigma.matrix().conjugate().T, atol=1e-12)


def test_kubo_conductivity_vanishes_for_zero_velocity_vertices():
    def zero_velocity(_kx, _ky, _direction):
        return np.zeros((4, 4))

    config = KuboConfig(omega_eV=0.2, temperature_eV=0.01, output_si=False)

    sigma = kubo_conductivity([(0.1, 0.2)], config, velocity=zero_velocity)

    np.testing.assert_allclose(sigma.matrix(), np.zeros((2, 2)))


def test_bosonic_matsubara_energy_uses_eV_units():
    assert np.isclose(bosonic_matsubara_energy_eV(0, 30.0), 0.0)
    assert np.isclose(bosonic_matsubara_energy_eV(1, 30.0), 2.0 * np.pi * 30.0 * 8.617333262e-5)
