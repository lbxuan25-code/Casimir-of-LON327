import numpy as np
import pytest

from lno327.bdg.finite_q import bdg_finite_q_vertex_from_normal_blocks
from lno327.bdg.nambu import charge_current_vertex_from_model, diamagnetic_vertex_from_model
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec
from lno327.response.finite_q_bdg import bdg_contact_vertex_from_spec, bdg_vector_vertex_from_spec


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
@pytest.mark.parametrize("direction", ("x", "y"))
def test_four_orbital_peierls_vector_vertex_matches_model_peierls_blocks(q, direction):
    qx, qy = q
    spec = LNO327FourOrbitalSpec()
    actual = bdg_vector_vertex_from_spec(spec, 0.21, -0.34, qx, qy, direction, "peierls")
    expected = bdg_finite_q_vertex_from_normal_blocks(
        spec.peierls_hamiltonian_vector_vertex(0.21, -0.34, qx, qy, direction),
        spec.peierls_hamiltonian_vector_vertex(-0.21, 0.34, -qx, -qy, direction),
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
@pytest.mark.parametrize("directions", (("x", "x"), ("y", "y"), ("x", "y")))
def test_four_orbital_peierls_contact_vertex_matches_model_peierls_blocks(q, directions):
    qx, qy = q
    direction_i, direction_j = directions
    spec = LNO327FourOrbitalSpec()
    actual = bdg_contact_vertex_from_spec(
        spec,
        0.21,
        -0.34,
        qx,
        qy,
        direction_i,
        direction_j,
        "peierls",
    )
    expected = bdg_finite_q_vertex_from_normal_blocks(
        spec.peierls_hamiltonian_contact_vertex(0.21, -0.34, qx, qy, direction_i, direction_j),
        spec.peierls_hamiltonian_contact_vertex(-0.21, 0.34, -qx, -qy, direction_i, direction_j),
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("direction", ("x", "y"))
def test_four_orbital_q0_velocity_vector_vertex_matches_model_current(direction):
    spec = LNO327FourOrbitalSpec()
    actual = bdg_vector_vertex_from_spec(spec, 0.21, -0.34, 0.17, -0.09, direction, "q0_velocity")
    np.testing.assert_allclose(actual, charge_current_vertex_from_model(spec, 0.21, -0.34, direction))


@pytest.mark.parametrize("directions", (("x", "x"), ("y", "y"), ("x", "y")))
def test_four_orbital_q0_velocity_contact_vertex_matches_model_diamagnetic_vertex(directions):
    direction_i, direction_j = directions
    spec = LNO327FourOrbitalSpec()
    actual = bdg_contact_vertex_from_spec(
        spec,
        0.21,
        -0.34,
        0.17,
        -0.09,
        direction_i,
        direction_j,
        "q0_velocity",
    )
    np.testing.assert_allclose(actual, diamagnetic_vertex_from_model(spec, 0.21, -0.34, direction_i, direction_j))


def test_two_band_peierls_vertices_have_bdg_shape_and_are_finite():
    spec = SymmetryBdG2BandSpec()
    vector = bdg_vector_vertex_from_spec(spec, 0.21, -0.34, 0.17, -0.09, "x", "peierls")
    contact = bdg_contact_vertex_from_spec(spec, 0.21, -0.34, 0.17, -0.09, "x", "y", "peierls")
    assert vector.shape == (4, 4)
    assert contact.shape == (4, 4)
    assert np.all(np.isfinite(vector))
    assert np.all(np.isfinite(contact))


def test_peierls_current_requires_spec_support():
    class DummySpec:
        def normal_hamiltonian(self, _kx, _ky):
            return np.eye(2, dtype=complex)

    with pytest.raises(ValueError, match="spec must support Peierls finite-q vertices"):
        bdg_vector_vertex_from_spec(DummySpec(), 0.0, 0.0, 0.1, 0.0, "x", "peierls")
