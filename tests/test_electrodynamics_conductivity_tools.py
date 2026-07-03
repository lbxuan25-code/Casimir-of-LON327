import numpy as np
import pytest

from lno327.conductivity import (
    ConductivityTensor as OldConductivityTensor,
    anisotropy_delta as old_delta,
    anisotropy_summary as old_summary,
    conductivity_matrix_diagnostics as old_diagnostics,
    rotate_conductivity as old_rotate,
)
from lno327.electrodynamics.conductivity import (
    ConductivityTensor,
    anisotropy_delta,
    anisotropy_summary,
    conductivity_matrix_diagnostics,
    rotate_conductivity,
)


def _old_tensor() -> OldConductivityTensor:
    return OldConductivityTensor(1.2 + 0.1j, 0.7 - 0.2j, 0.05j, -0.03j)


def _new_tensor() -> ConductivityTensor:
    return ConductivityTensor(1.2 + 0.1j, 0.7 - 0.2j, 0.05j, -0.03j)


def test_conductivity_tensor_matrix_matches_legacy():
    np.testing.assert_allclose(_new_tensor().matrix(), _old_tensor().matrix())


def test_rotate_conductivity_matches_legacy():
    old = old_rotate(_old_tensor(), 0.37).matrix()
    new = rotate_conductivity(_new_tensor(), 0.37).matrix()

    np.testing.assert_allclose(new, old)


def test_anisotropy_tools_match_legacy():
    assert anisotropy_delta(_new_tensor()) == old_delta(_old_tensor())
    assert anisotropy_summary(_new_tensor()) == old_summary(_old_tensor())


def test_anisotropy_delta_exception_matches_legacy():
    old = OldConductivityTensor(1.0, -1.0)
    new = ConductivityTensor(1.0, -1.0)

    with pytest.raises(ValueError):
        old_delta(old)
    with pytest.raises(ValueError):
        anisotropy_delta(new)


def test_conductivity_matrix_diagnostics_matches_legacy():
    old = old_diagnostics(_old_tensor())
    new = conductivity_matrix_diagnostics(_new_tensor())

    for key in old:
        np.testing.assert_allclose(new[key], old[key])
