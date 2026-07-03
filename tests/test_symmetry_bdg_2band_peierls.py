import numpy as np
import pytest

from lno327.bdg.finite_q import bdg_finite_q_vertex_from_normal_blocks
from lno327.models.symmetry_bdg_2band import peierls
from lno327.models.symmetry_bdg_2band.normal import normal_hamiltonian
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec


def test_two_band_hopping_terms_have_stable_sorted_order():
    terms = peierls.normal_state_hopping_terms()

    assert tuple(r for r, _matrix in terms) == tuple(sorted(r for r, _matrix in terms))


def test_two_band_hopping_hamiltonian_matches_normal_hamiltonian():
    kx, ky = 0.21, -0.34

    np.testing.assert_allclose(
        peierls.normal_state_hamiltonian_from_hoppings(kx, ky),
        normal_hamiltonian(kx, ky),
    )


def test_two_band_spec_hopping_hamiltonian_matches_normal_hamiltonian():
    spec = SymmetryBdG2BandSpec()
    kx, ky = 0.21, -0.34

    np.testing.assert_allclose(spec.normal_hamiltonian_from_hoppings(kx, ky), spec.normal_hamiltonian(kx, ky))


@pytest.mark.parametrize("direction", ("x", "y"))
def test_two_band_peierls_vector_q_zero_matches_velocity(direction):
    spec = SymmetryBdG2BandSpec()
    kx, ky = 0.21, -0.34

    np.testing.assert_allclose(
        spec.peierls_hamiltonian_vector_vertex(kx, ky, 0.0, 0.0, direction),
        spec.velocity_operator(kx, ky, direction),
    )


@pytest.mark.parametrize("directions", (("x", "x"), ("y", "y"), ("x", "y")))
def test_two_band_peierls_contact_q_zero_matches_mass(directions):
    spec = SymmetryBdG2BandSpec()
    kx, ky = 0.21, -0.34
    i, j = directions

    np.testing.assert_allclose(
        spec.peierls_hamiltonian_contact_vertex(kx, ky, 0.0, 0.0, i, j),
        spec.mass_operator(kx, ky, i, j),
    )


def test_two_band_peierls_vertex_ward_residual_is_small_and_hermiticity_validates():
    residual = peierls.peierls_vertex_ward_residual(0.21, -0.34, 0.17, -0.09)

    assert residual[0] < 1e-14
    assert residual[1] < 1e-12
    assert peierls.validate_hopping_hermiticity() == 0.0


def test_two_band_peierls_rejects_invalid_direction():
    with pytest.raises(ValueError, match="direction must be"):
        peierls.peierls_hamiltonian_vector_vertex(0.21, -0.34, 0.17, -0.09, "z")


def test_two_band_peierls_blocks_work_with_generic_bdg_lifting():
    spec = SymmetryBdG2BandSpec()
    kx, ky = 0.21, -0.34
    qx, qy = 0.17, -0.09
    particle = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x")
    hole_normal = spec.peierls_hamiltonian_vector_vertex(-kx, -ky, -qx, -qy, "x")
    bdg_vertex = bdg_finite_q_vertex_from_normal_blocks(particle, hole_normal)

    assert bdg_vertex.shape == (4, 4)

    particle_contact = spec.peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "x", "y")
    hole_contact = spec.peierls_hamiltonian_contact_vertex(-kx, -ky, -qx, -qy, "x", "y")
    bdg_contact = bdg_finite_q_vertex_from_normal_blocks(particle_contact, hole_contact)

    assert bdg_contact.shape == (4, 4)
