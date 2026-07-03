import numpy as np
import pytest

from lno327 import tb_fourier as old
from lno327.models.lno327_four_orbital import peierls as new


def _assert_terms_allclose(new_terms, old_terms):
    assert [term[0] for term in new_terms] == [term[0] for term in old_terms]
    for (_, new_matrix), (_, old_matrix) in zip(new_terms, old_terms, strict=True):
        np.testing.assert_allclose(new_matrix, old_matrix)


def test_normal_state_hopping_terms_match_legacy():
    _assert_terms_allclose(new.normal_state_hopping_terms(), old.normal_state_hopping_terms())


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
def test_normal_state_hamiltonian_from_hoppings_matches_legacy(q):
    qx, qy = q
    kx, ky = 0.21 + qx, -0.34 + qy

    np.testing.assert_allclose(
        new.normal_state_hamiltonian_from_hoppings(kx, ky),
        old.normal_state_hamiltonian_from_hoppings(kx, ky),
    )


def test_sinc_stable_matches_legacy_for_scalar_and_array():
    assert new.sinc_stable(0.0) == old.sinc_stable(0.0)
    assert new.sinc_stable(0.2) == old.sinc_stable(0.2)
    values = np.array([0.0, 1e-13, 0.2, -0.4])
    np.testing.assert_allclose(new.sinc_stable(values), old.sinc_stable(values))


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
@pytest.mark.parametrize("direction", ("x", "y"))
def test_peierls_hamiltonian_vector_vertex_matches_legacy(q, direction):
    qx, qy = q

    np.testing.assert_allclose(
        new.peierls_hamiltonian_vector_vertex(0.21, -0.34, qx, qy, direction),
        old.peierls_hamiltonian_vector_vertex(0.21, -0.34, qx, qy, direction),
    )


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
@pytest.mark.parametrize("directions", (("x", "x"), ("y", "y"), ("x", "y")))
def test_peierls_hamiltonian_contact_vertex_matches_legacy(q, directions):
    qx, qy = q
    direction_i, direction_j = directions

    np.testing.assert_allclose(
        new.peierls_hamiltonian_contact_vertex(0.21, -0.34, qx, qy, direction_i, direction_j),
        old.peierls_hamiltonian_contact_vertex(0.21, -0.34, qx, qy, direction_i, direction_j),
    )


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
def test_peierls_vertex_ward_residual_matches_legacy(q):
    qx, qy = q

    np.testing.assert_allclose(
        new.peierls_vertex_ward_residual(0.21, -0.34, qx, qy),
        old.peierls_vertex_ward_residual(0.21, -0.34, qx, qy),
    )


@pytest.mark.parametrize("sign_convention", ("plus", "minus"))
def test_peierls_vector_vertex_sign_audit_residual_matches_legacy(sign_convention):
    np.testing.assert_allclose(
        new.peierls_vector_vertex_sign_audit_residual(
            0.21,
            -0.34,
            0.17,
            -0.09,
            sign_convention=sign_convention,
        ),
        old.peierls_vector_vertex_sign_audit_residual(
            0.21,
            -0.34,
            0.17,
            -0.09,
            sign_convention=sign_convention,
        ),
    )


def test_validate_hopping_hermiticity_matches_legacy():
    assert new.validate_hopping_hermiticity() == old.validate_hopping_hermiticity()


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
