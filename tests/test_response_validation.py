import numpy as np
import pytest

from lno327.response.config import KuboConfig
from lno327.response.validation import validate_k_points_and_weights


def test_validate_k_points_and_weights_returns_points_and_weights():
    points = np.array([[0.1, 0.2], [0.3, -0.4]])
    weights = np.array([0.25, 0.75])
    config = KuboConfig(omega_eV=0.1, temperature_eV=0.02)

    actual_points, actual_weights = validate_k_points_and_weights(points, config, weights)

    np.testing.assert_allclose(actual_points, points)
    np.testing.assert_allclose(actual_weights, weights)


def test_validate_k_points_and_weights_builds_uniform_default_weights():
    points = np.array([[0.1, 0.2], [0.3, -0.4]])
    config = KuboConfig(omega_eV=0.1, temperature_eV=0.02)

    _, weights = validate_k_points_and_weights(points, config)

    np.testing.assert_allclose(weights, np.array([0.5, 0.5]))


@pytest.mark.parametrize(
    "points, match",
    [
        (np.array([0.1, 0.2]), "k_points must have shape"),
        (np.empty((0, 2)), "k_points must not be empty"),
    ],
)
def test_validate_k_points_and_weights_rejects_bad_points(points, match):
    config = KuboConfig(omega_eV=0.1, temperature_eV=0.02)

    with pytest.raises(ValueError, match=match):
        validate_k_points_and_weights(points, config)


def test_validate_k_points_and_weights_rejects_negative_omega():
    config = KuboConfig(omega_eV=-0.1, temperature_eV=0.02)

    with pytest.raises(ValueError, match="omega_eV must be non-negative"):
        validate_k_points_and_weights(np.array([[0.1, 0.2]]), config)


def test_validate_k_points_and_weights_rejects_nonpositive_eta():
    config = KuboConfig(omega_eV=0.1, temperature_eV=0.02, eta_eV=0.0)

    with pytest.raises(ValueError, match="eta_eV must be positive"):
        validate_k_points_and_weights(np.array([[0.1, 0.2]]), config)


def test_validate_k_points_and_weights_rejects_bad_weight_shape():
    config = KuboConfig(omega_eV=0.1, temperature_eV=0.02)

    with pytest.raises(ValueError, match="k_weights must have shape"):
        validate_k_points_and_weights(np.array([[0.1, 0.2]]), config, np.array([0.5, 0.5]))
