import numpy as np
import pytest

from lno327.models.lno327_four_orbital import peierls as new


def test_normal_state_hopping_terms_are_hermitian_and_nonempty():
    terms = new.normal_state_hopping_terms()
    assert terms
    assert new.validate_hopping_hermiticity(terms) == 0.0


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
def test_normal_state_hamiltonian_from_hoppings_is_hermitian(q):
    qx, qy = q
    kx, ky = 0.21 + qx, -0.34 + qy

    hamiltonian = new.normal_state_hamiltonian_from_hoppings(kx, ky)
    np.testing.assert_allclose(hamiltonian, hamiltonian.conjugate().T)


def test_sinc_stable_scalar_and_array_limits():
    assert new.sinc_stable(0.0) == 1.0
    assert np.isfinite(new.sinc_stable(0.2))
    values = np.array([0.0, 1e-13, 0.2, -0.4])
    assert np.all(np.isfinite(new.sinc_stable(values)))


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
@pytest.mark.parametrize("direction", ("x", "y"))
def test_peierls_hamiltonian_vector_vertex_is_well_formed(q, direction):
    qx, qy = q

    vertex = new.peierls_hamiltonian_vector_vertex(0.21, -0.34, qx, qy, direction)
    assert vertex.shape == (4, 4)
    assert np.all(np.isfinite(vertex))


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
@pytest.mark.parametrize("directions", (("x", "x"), ("y", "y"), ("x", "y")))
def test_peierls_hamiltonian_contact_vertex_is_well_formed(q, directions):
    qx, qy = q
    direction_i, direction_j = directions

    vertex = new.peierls_hamiltonian_contact_vertex(0.21, -0.34, qx, qy, direction_i, direction_j)
    assert vertex.shape == (4, 4)
    assert np.all(np.isfinite(vertex))


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
def test_peierls_vertex_ward_residual_is_finite(q):
    qx, qy = q

    residual = new.peierls_vertex_ward_residual(0.21, -0.34, qx, qy)
    assert len(residual) == 4
    assert np.all(np.isfinite(residual))


@pytest.mark.parametrize("sign_convention", ("plus", "minus"))
def test_peierls_vector_vertex_sign_audit_residual_matches_legacy(sign_convention):
    residual = new.peierls_vector_vertex_sign_audit_residual(
        0.21,
        -0.34,
        0.17,
        -0.09,
        sign_convention=sign_convention,
    )
    assert len(residual) == 4
    assert np.all(np.isfinite(residual))


def test_validate_hopping_hermiticity_accepts_default_terms():
    assert new.validate_hopping_hermiticity() == 0.0


def test_peierls_error_paths_match_legacy():
    with pytest.raises(ValueError, match="direction must be"):
        new.peierls_hamiltonian_vector_vertex(0.21, -0.34, 0.17, -0.09, "z")
    with pytest.raises(ValueError, match="directions must be"):
        new.peierls_hamiltonian_contact_vertex(0.21, -0.34, 0.17, -0.09, "x", "z")
    with pytest.raises(ValueError, match="sign_convention must be"):
        new.peierls_vector_vertex_sign_audit_residual(
            0.21,
            -0.34,
            0.17,
            -0.09,
            sign_convention="bad",
        )
    with pytest.raises(ValueError, match="missing Hermitian partner"):
        new.validate_hopping_hermiticity([((1, 0), np.eye(4, dtype=complex))])
