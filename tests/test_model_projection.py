import numpy as np
import pytest

from lno327.models.projection import anchor_eigenvector_phases, band_project_pairing


def test_anchor_eigenvector_phases_preserves_shape_and_makes_anchor_positive():
    states = np.array(
        [
            [1.0j, 1.0],
            [2.0 + 2.0j, -3.0j],
            [0.5, 0.2],
        ],
        dtype=complex,
    )

    anchored = anchor_eigenvector_phases(states)

    assert anchored.shape == states.shape
    for band in range(anchored.shape[1]):
        index = int(np.argmax(np.abs(anchored[:, band])))
        assert np.isclose(anchored[index, band].imag, 0.0)
        assert anchored[index, band].real > 0.0


def test_anchor_eigenvector_phases_removes_global_phase():
    vector = np.array([[1.0 + 1.0j], [2.0 - 0.5j]], dtype=complex)
    phase = np.exp(0.37j)

    anchored = anchor_eigenvector_phases(vector)
    anchored_phased = anchor_eigenvector_phases(phase * vector)

    np.testing.assert_allclose(anchored, anchored_phased)


def test_band_project_pairing_raw_matches_old_formula():
    delta = np.array([[0.2, 0.1j], [-0.1j, -0.3]], dtype=complex)
    states_k = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
    states_minus_k = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    projected = band_project_pairing(delta, states_k, states_minus_k, gauge="raw")
    expected = np.asarray(
        [
            states_k[:, band].conjugate().T @ delta @ states_minus_k[:, band].conjugate()
            for band in range(2)
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(projected, expected)


def test_band_project_pairing_anchor_is_stable_under_global_phases():
    delta = np.array([[0.4, 0.05], [0.05, -0.2]], dtype=complex)
    states_k = np.array([[1.0, 0.2], [0.3, 1.0]], dtype=complex)
    states_minus_k = np.array([[0.8, -0.1], [0.4, 1.0]], dtype=complex)
    states_k_phased = states_k * np.array([np.exp(0.4j), np.exp(-0.2j)])
    states_minus_k_phased = states_minus_k * np.array([np.exp(-0.7j), np.exp(0.9j)])

    anchored = band_project_pairing(delta, states_k, states_minus_k, gauge="anchor")
    anchored_phased = band_project_pairing(delta, states_k_phased, states_minus_k_phased, gauge="anchor")
    raw = band_project_pairing(delta, states_k, states_minus_k, gauge="raw")
    raw_phased = band_project_pairing(delta, states_k_phased, states_minus_k_phased, gauge="raw")

    np.testing.assert_allclose(anchored, anchored_phased)
    assert not np.allclose(raw, raw_phased)


def test_band_project_pairing_rejects_unknown_gauge():
    with pytest.raises(ValueError, match="gauge must be 'anchor' or 'raw'"):
        band_project_pairing(np.eye(2), np.eye(2), np.eye(2), gauge="other")
