import numpy as np
import pytest

from lno327.collective import ward as new_ward


def _response():
    return np.array(
        [[1.0, 0.2j, 0.1], [-0.2j, 0.7, 0.05j], [0.1, -0.05j, 0.4]],
        dtype=complex,
    )


def test_collective_ward_residual_helpers_are_consistent():
    response = _response()
    q = np.array([0.1, -0.03])
    for name in (
        "physical_ward_residuals",
        "physical_ward_residuals_corrected",
        "physical_ward_residuals_legacy",
        "hamiltonian_vector_ward_residuals",
        "ward_residuals",
    ):
        new_left, new_right = getattr(new_ward, name)(response, 0.01, q)
        assert new_left.shape == (3,)
        assert new_right.shape == (3,)
        assert np.all(np.isfinite(new_left))
        assert np.all(np.isfinite(new_right))

    left_error, right_error, max_error = new_ward.ward_errors(response, 0.01, q)
    assert max_error == max(left_error, right_error)


def test_collective_ward_metadata_is_pure_diagnostic():
    response = _response()
    before = response.copy()
    q = np.array([0.1, -0.03])

    actual = new_ward.ward_metadata(response, 0.01, q)
    left, right = new_ward.physical_ward_residuals(response, 0.01, q)
    expected = {
        "left_norm": float(np.linalg.norm(left)),
        "right_norm": float(np.linalg.norm(right)),
        "max_norm": float(max(np.linalg.norm(left), np.linalg.norm(right))),
    }

    assert actual == expected
    np.testing.assert_allclose(response, before)


def test_collective_ward_rejects_bad_shapes():
    with pytest.raises(ValueError, match="response must have shape"):
        new_ward.physical_ward_residuals(np.eye(2), 0.01, np.array([0.1, 0.0]))
    with pytest.raises(ValueError, match="q must have shape"):
        new_ward.physical_ward_residuals(np.eye(3), 0.01, np.array([0.1]))
