import numpy as np
import pytest

from lno327.models.hopping import (
    normal_state_hamiltonian_from_hoppings,
    peierls_hamiltonian_contact_vertex_from_hoppings,
    peierls_hamiltonian_vector_vertex_from_hoppings,
    peierls_vector_vertex_sign_audit_residual_from_hoppings,
    peierls_vertex_ward_residual_from_hoppings,
    sinc_stable,
    validate_hopping_hermiticity,
)


def _hopping_terms(a=0.3, b=-0.7, c=0.2):
    eye = np.eye(2, dtype=complex)
    return (
        ((-1, 0), b * eye),
        ((0, -1), c * eye),
        ((0, 0), a * eye),
        ((0, 1), c * eye),
        ((1, 0), b * eye),
    )


def test_normal_state_hamiltonian_from_hoppings_matches_cos_formula():
    kx, ky = 0.21, -0.34
    a, b, c = 0.3, -0.7, 0.2

    hamiltonian = normal_state_hamiltonian_from_hoppings(kx, ky, _hopping_terms(a, b, c))

    expected = (a + 2.0 * b * np.cos(kx) + 2.0 * c * np.cos(ky)) * np.eye(2)
    np.testing.assert_allclose(hamiltonian, expected)


def test_sinc_stable_scalar_and_array():
    assert sinc_stable(0.0) == 1.0
    assert sinc_stable(1e-13) == 1.0
    assert sinc_stable(0.2) == np.sin(0.2) / 0.2
    values = np.array([0.0, 1e-13, 0.2])
    np.testing.assert_allclose(sinc_stable(values), np.array([1.0, 1.0, np.sin(0.2) / 0.2]))


@pytest.mark.parametrize("direction", ("x", "y"))
def test_peierls_vector_q_zero_matches_derivative(direction):
    kx, ky = 0.21, -0.34
    a, b, c = 0.3, -0.7, 0.2

    vertex = peierls_hamiltonian_vector_vertex_from_hoppings(
        kx,
        ky,
        0.0,
        0.0,
        direction,
        _hopping_terms(a, b, c),
    )

    derivative = -2.0 * b * np.sin(kx) if direction == "x" else -2.0 * c * np.sin(ky)
    np.testing.assert_allclose(vertex, derivative * np.eye(2))


@pytest.mark.parametrize("directions", (("x", "x"), ("y", "y"), ("x", "y")))
def test_peierls_contact_q_zero_matches_second_derivative(directions):
    kx, ky = 0.21, -0.34
    a, b, c = 0.3, -0.7, 0.2
    direction_i, direction_j = directions

    vertex = peierls_hamiltonian_contact_vertex_from_hoppings(
        kx,
        ky,
        0.0,
        0.0,
        direction_i,
        direction_j,
        _hopping_terms(a, b, c),
    )

    if directions == ("x", "x"):
        expected = -2.0 * b * np.cos(kx)
    elif directions == ("y", "y"):
        expected = -2.0 * c * np.cos(ky)
    else:
        expected = 0.0
    np.testing.assert_allclose(vertex, expected * np.eye(2), atol=1e-14)


def test_peierls_vertex_ward_residual_is_small_at_finite_q():
    residual = peierls_vertex_ward_residual_from_hoppings(0.21, -0.34, 0.17, -0.09, _hopping_terms())

    assert residual[0] < 1e-14
    assert residual[1] < 1e-12


def test_validate_hopping_hermiticity_and_error_paths():
    assert validate_hopping_hermiticity(_hopping_terms()) == 0.0

    with pytest.raises(ValueError, match="missing Hermitian partner"):
        validate_hopping_hermiticity((((1, 0), np.eye(2, dtype=complex)),))
    with pytest.raises(ValueError, match="direction must be"):
        peierls_hamiltonian_vector_vertex_from_hoppings(0.0, 0.0, 0.0, 0.0, "z", _hopping_terms())
    with pytest.raises(ValueError, match="directions must be"):
        peierls_hamiltonian_contact_vertex_from_hoppings(0.0, 0.0, 0.0, 0.0, "x", "z", _hopping_terms())
    with pytest.raises(ValueError, match="sign_convention must be"):
        peierls_vector_vertex_sign_audit_residual_from_hoppings(
            0.0,
            0.0,
            0.0,
            0.0,
            _hopping_terms(),
            "bad",
        )
