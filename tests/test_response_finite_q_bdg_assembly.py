import numpy as np
import pytest

from lno327.bdg_response import bdg_current_vertex, bdg_diamagnetic_vertex
from lno327.finite_q_primitives import (
    bdg_finite_q_contact_vertex as old_contact_vertex,
    bdg_finite_q_vector_vertex as old_vector_vertex,
)
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec
from lno327.response.finite_q_bdg import bdg_contact_vertex_from_spec, bdg_vector_vertex_from_spec


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
@pytest.mark.parametrize("direction", ("x", "y"))
def test_four_orbital_peierls_vector_vertex_matches_legacy(q, direction):
    qx, qy = q
    actual = bdg_vector_vertex_from_spec(LNO327FourOrbitalSpec(), 0.21, -0.34, qx, qy, direction, "peierls")
    np.testing.assert_allclose(actual, old_vector_vertex(0.21, -0.34, qx, qy, direction))


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
@pytest.mark.parametrize("directions", (("x", "x"), ("y", "y"), ("x", "y")))
def test_four_orbital_peierls_contact_vertex_matches_legacy(q, directions):
    qx, qy = q
    direction_i, direction_j = directions
    actual = bdg_contact_vertex_from_spec(
        LNO327FourOrbitalSpec(),
        0.21,
        -0.34,
        qx,
        qy,
        direction_i,
        direction_j,
        "peierls",
    )
    np.testing.assert_allclose(actual, old_contact_vertex(0.21, -0.34, qx, qy, direction_i, direction_j))


@pytest.mark.parametrize("direction", ("x", "y"))
def test_four_orbital_q0_velocity_vector_vertex_matches_legacy(direction):
    actual = bdg_vector_vertex_from_spec(LNO327FourOrbitalSpec(), 0.21, -0.34, 0.17, -0.09, direction, "q0_velocity")
    np.testing.assert_allclose(actual, bdg_current_vertex(0.21, -0.34, direction))


@pytest.mark.parametrize("directions", (("x", "x"), ("y", "y"), ("x", "y")))
def test_four_orbital_q0_velocity_contact_vertex_matches_legacy(directions):
    direction_i, direction_j = directions
    actual = bdg_contact_vertex_from_spec(
        LNO327FourOrbitalSpec(),
        0.21,
        -0.34,
        0.17,
        -0.09,
        direction_i,
        direction_j,
        "q0_velocity",
    )
    np.testing.assert_allclose(actual, bdg_diamagnetic_vertex(0.21, -0.34, direction_i, direction_j))


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
