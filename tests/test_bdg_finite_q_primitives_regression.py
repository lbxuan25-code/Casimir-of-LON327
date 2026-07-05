import numpy as np
import pytest

from lno327.bdg.finite_q import (
    bdg_block_diagonal_vertex,
    bdg_finite_q_vertex_from_normal_blocks,
    density_vertex,
    phase_phase_direct_vertex,
    phase_vertex,
)
from lno327.models.lno327_four_orbital.peierls import (
    peierls_hamiltonian_contact_vertex,
    peierls_hamiltonian_vector_vertex,
)


def test_phase_vertices_have_expected_nambu_blocks():
    pairing = np.array(
        [[0.1, 0.02j, 0.0], [-0.02j, -0.03, 0.04], [0.0, 0.04, 0.2]],
        dtype=complex,
    )
    delta_theta = 0.5 * pairing

    zero = np.zeros_like(pairing)
    np.testing.assert_allclose(
        phase_vertex(pairing),
        np.block([[zero, 1j * pairing], [-1j * pairing.conjugate().T, zero]]),
    )
    np.testing.assert_allclose(
        phase_phase_direct_vertex(delta_theta),
        np.block([[zero, -delta_theta], [-delta_theta.conjugate().T, zero]]),
    )


def test_density_vertex_has_expected_nambu_charge_structure_for_four_orbitals():
    eye = np.eye(4, dtype=complex)
    zero = np.zeros((4, 4), dtype=complex)
    np.testing.assert_allclose(density_vertex(4), np.block([[eye, zero], [zero, -eye]]))


@pytest.mark.parametrize("orbital_dim", [0, -1, 1.5])
def test_density_vertex_rejects_nonpositive_or_noninteger_dimension(orbital_dim):
    with pytest.raises(ValueError, match="orbital_dim must be a positive integer"):
        density_vertex(orbital_dim)  # type: ignore[arg-type]


@pytest.mark.parametrize("direction", ("x", "y"))
def test_bdg_finite_q_vector_vertex_block_lifting_uses_transposed_hole_block(direction):
    kx, ky = 0.21, -0.34
    qx, qy = 0.17, -0.09
    particle = peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, direction)
    hole_normal = peierls_hamiltonian_vector_vertex(-kx, -ky, -qx, -qy, direction)

    new = bdg_finite_q_vertex_from_normal_blocks(particle, hole_normal)
    old = bdg_block_diagonal_vertex(particle, -hole_normal.T)

    np.testing.assert_allclose(new, old)


@pytest.mark.parametrize("directions", (("x", "x"), ("y", "y"), ("x", "y")))
def test_bdg_finite_q_contact_vertex_block_lifting_uses_transposed_hole_block(directions):
    kx, ky = 0.21, -0.34
    qx, qy = 0.17, -0.09
    direction_i, direction_j = directions
    particle = peierls_hamiltonian_contact_vertex(kx, ky, qx, qy, direction_i, direction_j)
    hole_normal = peierls_hamiltonian_contact_vertex(
        -kx,
        -ky,
        -qx,
        -qy,
        direction_i,
        direction_j,
    )

    new = bdg_finite_q_vertex_from_normal_blocks(particle, hole_normal)
    old = bdg_block_diagonal_vertex(particle, -hole_normal.T)

    np.testing.assert_allclose(new, old)
