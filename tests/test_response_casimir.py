import numpy as np

from lno327 import (
    CasimirSetup,
    ConductivityTensor,
    anisotropy_delta,
    casimir_torque_integrand,
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
