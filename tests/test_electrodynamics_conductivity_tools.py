import numpy as np
import pytest

from lno327.electrodynamics.conductivity import (
    ConductivityTensor,
    anisotropy_delta,
    anisotropy_summary,
    conductivity_matrix_diagnostics,
    rotate_conductivity,
)


def _new_tensor() -> ConductivityTensor:
    return ConductivityTensor(1.2 + 0.1j, 0.7 - 0.2j, 0.05j, -0.03j)


def test_conductivity_tensor_matrix_layout():
    np.testing.assert_allclose(
        _new_tensor().matrix(),
        np.array([[1.2 + 0.1j, 0.05j], [-0.03j, 0.7 - 0.2j]], dtype=complex),
    )


def test_rotate_conductivity_preserves_trace():
    new = rotate_conductivity(_new_tensor(), 0.37).matrix()

    np.testing.assert_allclose(np.trace(new), np.trace(_new_tensor().matrix()))


def test_anisotropy_tools_are_well_formed():
    assert np.isfinite(anisotropy_delta(_new_tensor()))
    summary = anisotropy_summary(_new_tensor())
    assert {"sigma_trace", "delta", "sigma_xy", "sigma_yx"}.issubset(summary)


def test_anisotropy_delta_exception():
    new = ConductivityTensor(1.0, -1.0)

    with pytest.raises(ValueError):
        anisotropy_delta(new)


def test_conductivity_matrix_diagnostics_fields():
    new = conductivity_matrix_diagnostics(_new_tensor())

    assert set(new) == {
        "sigma_matrix",
        "eigenvalues",
        "eigenvectors",
        "anisotropy_delta",
        "offdiag_norm",
        "relative_xx_yy_error",
    }
