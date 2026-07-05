import numpy as np
import pytest

from lno327.bdg.nambu import (
    charge_current_vertex_from_blocks,
    charge_current_vertex_from_model,
    diamagnetic_vertex_from_blocks,
    diamagnetic_vertex_from_model,
    nambu_block,
)
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec


def test_two_band_normal_vertex_builds_four_dimensional_bdg_vertex():
    particle = np.array([[1.0, 0.2], [0.2, -0.4]], dtype=complex)
    hole_at_minus_k = np.array([[0.5, 0.1j], [-0.1j, -0.2]], dtype=complex)

    vertex = charge_current_vertex_from_blocks(particle, hole_at_minus_k)

    assert vertex.shape == (4, 4)
    np.testing.assert_allclose(vertex[:2, :2], particle)
    np.testing.assert_allclose(vertex[2:, 2:], -hole_at_minus_k.T)


def test_four_orbital_normal_vertex_builds_eight_dimensional_bdg_vertex():
    particle = np.eye(4, dtype=complex)
    hole_at_minus_k = 2.0 * np.eye(4, dtype=complex)

    vertex = diamagnetic_vertex_from_blocks(particle, hole_at_minus_k)

    assert vertex.shape == (8, 8)
    np.testing.assert_allclose(vertex[:4, :4], particle)
    np.testing.assert_allclose(vertex[4:, 4:], -hole_at_minus_k.T)


def test_nambu_block_shape_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="same square shape"):
        nambu_block(np.eye(2), np.eye(3))


def test_invalid_direction_raises_value_error():
    spec = LNO327FourOrbitalSpec()

    with pytest.raises(ValueError, match="direction must be"):
        charge_current_vertex_from_model(spec, 0.1, 0.2, "z")


def test_four_orbital_current_vertex_matches_model_velocity_blocks():
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.31, -0.27

    new = charge_current_vertex_from_model(spec, kx, ky, "x")
    expected = charge_current_vertex_from_blocks(
        spec.velocity_operator(kx, ky, "x"),
        spec.velocity_operator(-kx, -ky, "x"),
    )

    np.testing.assert_allclose(new, expected)


def test_four_orbital_diamagnetic_vertex_matches_model_mass_blocks():
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.31, -0.27

    new = diamagnetic_vertex_from_model(spec, kx, ky, "x", "y")
    expected = diamagnetic_vertex_from_blocks(
        spec.mass_operator(kx, ky, "x", "y"),
        spec.mass_operator(-kx, -ky, "x", "y"),
    )

    np.testing.assert_allclose(new, expected)
