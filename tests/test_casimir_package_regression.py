import importlib.util
import sys
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


def load_old_casimir_reference():
    old_path = Path(__file__).resolve().parents[1] / "src" / "lno327" / "casimir.py"
    spec = importlib.util.spec_from_file_location("lno327._old_casimir_reference", old_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_active_casimir_import_uses_new_package():
    import lno327.casimir as active_casimir

    assert Path(active_casimir.__file__).name == "__init__.py"
    assert Path(active_casimir.__file__).parent.name == "casimir"


def test_casimir_setup_fields_match_old_reference():
    old = load_old_casimir_reference()

    new_setup = CasimirSetup(temperature=30.0, distance=30e-9, area=2.0)
    old_setup = old.CasimirSetup(temperature=30.0, distance=30e-9, area=2.0)

    assert new_setup.temperature == old_setup.temperature
    assert new_setup.distance == old_setup.distance
    assert new_setup.area == old_setup.area


def test_matsubara_frequency_matches_old_reference():
    old = load_old_casimir_reference()

    for n in (0, 1, 5):
        np.testing.assert_allclose(
            matsubara_frequency(n, 30.0),
            old.matsubara_frequency(n, 30.0),
            rtol=1e-12,
            atol=1e-12,
        )


def test_reflection_matrix_matches_old_reference_for_isotropic_and_anisotropic_tensors():
    old = load_old_casimir_reference()

    cases = (
        ConductivityTensor(xx=1e-4, yy=1e-4),
        ConductivityTensor(xx=1.4e-4, yy=0.7e-4, xy=0.2e-5, yx=-0.1e-5),
    )

    for tensor in cases:
        new_refl = reflection_matrix_weak_2d(1.3e12, 2.4e6, 0.37, tensor)
        old_tensor = old.ConductivityTensor(tensor.xx, tensor.yy, tensor.xy, tensor.yx)
        old_refl = old.reflection_matrix_weak_2d(1.3e12, 2.4e6, 0.37, old_tensor)
        np.testing.assert_allclose(new_refl, old_refl, rtol=1e-12, atol=1e-12)


def test_lifshitz_energy_integrand_matches_old_reference():
    old = load_old_casimir_reference()

    setup = CasimirSetup(temperature=30.0, distance=35e-9, area=1.7)
    old_setup = old.CasimirSetup(setup.temperature, setup.distance, setup.area)
    left = ConductivityTensor(xx=1.4e-4, yy=0.7e-4)
    right = ConductivityTensor(xx=0.8e-4, yy=1.3e-4)
    old_left = old.ConductivityTensor(left.xx, left.yy, left.xy, left.yx)
    old_right = old.ConductivityTensor(right.xx, right.yy, right.xy, right.yx)

    new_value = casimir_energy_integrand(setup, 1.1e12, 1.8e6, 0.22, 0.61, left, right)
    old_value = old.casimir_energy_integrand(old_setup, 1.1e12, 1.8e6, 0.22, 0.61, old_left, old_right)

    np.testing.assert_allclose(new_value, old_value, rtol=1e-12, atol=1e-12)


def test_torque_integrand_matches_old_reference():
    old = load_old_casimir_reference()

    setup = CasimirSetup(temperature=30.0, distance=35e-9)
    old_setup = old.CasimirSetup(setup.temperature, setup.distance, setup.area)
    left = ConductivityTensor(xx=1.4e-4, yy=0.7e-4)
    right = ConductivityTensor(xx=0.8e-4, yy=1.3e-4)
    old_left = old.ConductivityTensor(left.xx, left.yy, left.xy, left.yx)
    old_right = old.ConductivityTensor(right.xx, right.yy, right.xy, right.yx)

    new_value = casimir_torque_integrand(setup, 1.1e12, 1.8e6, 0.22, 0.61, left, right, dtheta=2e-5)
    old_value = old.casimir_torque_integrand(
        old_setup,
        1.1e12,
        1.8e6,
        0.22,
        0.61,
        old_left,
        old_right,
        dtheta=2e-5,
    )

    np.testing.assert_allclose(new_value, old_value, rtol=1e-12, atol=1e-12)


def test_zero_trace_conductivity_error_matches_old_reference():
    old = load_old_casimir_reference()

    with pytest.raises(ValueError, match="sigma_xx \\+ sigma_yy must be nonzero"):
        reflection_matrix_weak_2d(1.3e12, 2.4e6, 0.37, ConductivityTensor(xx=1e-4, yy=-1e-4))

    with pytest.raises(ValueError, match="sigma_xx \\+ sigma_yy must be nonzero"):
        old.reflection_matrix_weak_2d(1.3e12, 2.4e6, 0.37, old.ConductivityTensor(xx=1e-4, yy=-1e-4))


def test_casimir_layer_metadata_keeps_readiness_flags_false():
    metadata = casimir_layer_metadata()

    assert metadata["valid_for_casimir_input"] is False
    assert metadata["requires_gauge_closed_response"] is True
    assert metadata["ward_identity_closed_by_this_module"] is False
