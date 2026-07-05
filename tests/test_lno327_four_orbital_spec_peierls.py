import numpy as np

from lno327.bdg.finite_q import bdg_finite_q_vertex_from_normal_blocks
from lno327.models.lno327_four_orbital import peierls
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec


def _assert_terms_allclose(new_terms, expected_terms):
    assert [term[0] for term in new_terms] == [term[0] for term in expected_terms]
    for (_, new_matrix), (_, expected_matrix) in zip(new_terms, expected_terms, strict=True):
        np.testing.assert_allclose(new_matrix, expected_matrix)


def test_spec_hopping_terms_match_model_peierls_module():
    spec = LNO327FourOrbitalSpec()

    _assert_terms_allclose(spec.hopping_terms(), peierls.normal_state_hopping_terms(spec.normal_params))


def test_spec_normal_hamiltonian_from_hoppings_matches_peierls_and_normal_hamiltonian():
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.21, -0.34

    from_hoppings = spec.normal_hamiltonian_from_hoppings(kx, ky)

    np.testing.assert_allclose(
        from_hoppings,
        peierls.normal_state_hamiltonian_from_hoppings(kx, ky, spec.normal_params),
    )
    np.testing.assert_allclose(from_hoppings, spec.normal_hamiltonian(kx, ky))


def test_spec_peierls_vertices_match_model_peierls_module():
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.21, -0.34
    qx, qy = 0.17, -0.09

    np.testing.assert_allclose(
        spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x"),
        peierls.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x", spec.normal_params),
    )
    np.testing.assert_allclose(
        spec.peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "x", "y"),
        peierls.peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "x", "y", spec.normal_params),
    )
    np.testing.assert_allclose(
        spec.peierls_vertex_ward_residual(kx, ky, qx, qy),
        peierls.peierls_vertex_ward_residual(kx, ky, qx, qy, spec.normal_params),
    )


def test_spec_peierls_vector_vertex_with_generic_bdg_lifting_has_expected_blocks():
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.21, -0.34
    qx, qy = 0.17, -0.09
    particle = spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, "x")
    hole_normal = spec.peierls_hamiltonian_vector_vertex(-kx, -ky, -qx, -qy, "x")

    lifted = bdg_finite_q_vertex_from_normal_blocks(particle, hole_normal)

    assert lifted.shape == (8, 8)
    np.testing.assert_allclose(lifted[:4, :4], particle)
    np.testing.assert_allclose(lifted[4:, 4:], -hole_normal.T)


def test_spec_peierls_contact_vertex_with_generic_bdg_lifting_has_expected_blocks():
    spec = LNO327FourOrbitalSpec()
    kx, ky = 0.21, -0.34
    qx, qy = 0.17, -0.09
    particle = spec.peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, "x", "y")
    hole_normal = spec.peierls_hamiltonian_contact_vertex(-kx, -ky, -qx, -qy, "x", "y")

    lifted = bdg_finite_q_vertex_from_normal_blocks(particle, hole_normal)

    assert lifted.shape == (8, 8)
    np.testing.assert_allclose(lifted[:4, :4], particle)
    np.testing.assert_allclose(lifted[4:, 4:], -hole_normal.T)
