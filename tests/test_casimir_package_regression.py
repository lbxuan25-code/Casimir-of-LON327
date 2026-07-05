from pathlib import Path

import numpy as np
import pytest

from lno327.casimir import (
    CasimirSetup,
    casimir_energy_integrand,
    casimir_layer_metadata,
    casimir_torque_integrand,
    matsubara_frequency,
    reflection_matrix_weak_2d,
)
from lno327.electrodynamics.conductivity import ConductivityTensor


def test_active_casimir_import_uses_new_package():
    import lno327.casimir as active_casimir

    assert Path(active_casimir.__file__).name == "__init__.py"
    assert Path(active_casimir.__file__).parent.name == "casimir"


def test_casimir_setup_fields_are_stable():
    new_setup = CasimirSetup(temperature=30.0, distance=30e-9, area=2.0)

    assert new_setup.temperature == 30.0
    assert new_setup.distance == 30e-9
    assert new_setup.area == 2.0


def test_matsubara_frequency_formula():
    for n in (0, 1, 5):
        np.testing.assert_allclose(
            matsubara_frequency(n, 30.0),
            2.0 * np.pi * n * 1.380649e-23 * 30.0 / 1.054571817e-34,
            rtol=1e-12,
            atol=1e-12,
        )


def test_reflection_matrix_is_well_formed_for_isotropic_and_anisotropic_tensors():
    cases = (
        ConductivityTensor(xx=1e-4, yy=1e-4),
        ConductivityTensor(xx=1.4e-4, yy=0.7e-4, xy=0.2e-5, yx=-0.1e-5),
    )

    for tensor in cases:
        new_refl = reflection_matrix_weak_2d(1.3e12, 2.4e6, 0.37, tensor)
        assert new_refl.shape == (2, 2)
        assert np.all(np.isfinite(new_refl))


def test_lifshitz_energy_integrand_is_finite():
    setup = CasimirSetup(temperature=30.0, distance=35e-9, area=1.7)
    left = ConductivityTensor(xx=1.4e-4, yy=0.7e-4)
    right = ConductivityTensor(xx=0.8e-4, yy=1.3e-4)

    new_value = casimir_energy_integrand(setup, 1.1e12, 1.8e6, 0.22, 0.61, left, right)

    assert np.isfinite(new_value)


def test_torque_integrand_is_finite():
    setup = CasimirSetup(temperature=30.0, distance=35e-9)
    left = ConductivityTensor(xx=1.4e-4, yy=0.7e-4)
    right = ConductivityTensor(xx=0.8e-4, yy=1.3e-4)

    new_value = casimir_torque_integrand(setup, 1.1e12, 1.8e6, 0.22, 0.61, left, right, dtheta=2e-5)

    assert np.isfinite(new_value)


def test_zero_trace_conductivity_error():
    with pytest.raises(ValueError, match="sigma_xx \\+ sigma_yy must be nonzero"):
        reflection_matrix_weak_2d(1.3e12, 2.4e6, 0.37, ConductivityTensor(xx=1e-4, yy=-1e-4))


def test_casimir_layer_metadata_keeps_readiness_flags_false():
    metadata = casimir_layer_metadata()

    assert metadata["valid_for_casimir_input"] is False
    assert metadata["requires_gauge_closed_response"] is True
    assert metadata["ward_identity_closed_by_this_module"] is False
