import numpy as np
import pytest

from lno327.models.symmetry_bdg_2band import (
    TwoBandParameters,
    bdg_hamiltonian,
    mass_operator,
    normal_hamiltonian,
    pairing_matrix,
    velocity_operator,
)
from lno327.models.symmetry_bdg_2band.normal import TAU0, TAUX
from lno327.models.symmetry_bdg_2band.pairing import d_wave_form_factor


def test_normal_pairing_and_bdg_shapes():
    kx, ky = 0.37, -0.22

    assert normal_hamiltonian(kx, ky).shape == (2, 2)
    for channel in ("normal", "spm", "dwave"):
        assert pairing_matrix(channel, kx, ky).shape == (2, 2)
        assert bdg_hamiltonian(kx, ky, channel).shape == (4, 4)


def test_normal_and_bdg_hamiltonians_are_hermitian():
    kx, ky = 0.37, -0.22

    h = normal_hamiltonian(kx, ky)
    np.testing.assert_allclose(h, h.conjugate().T)
    for channel in ("normal", "spm", "dwave"):
        bdg = bdg_hamiltonian(kx, ky, channel)
        np.testing.assert_allclose(bdg, bdg.conjugate().T)


def test_bdg_spectrum_has_particle_hole_pairing():
    kx, ky = 0.37, -0.22

    for channel in ("normal", "spm", "dwave"):
        energies = np.linalg.eigvalsh(bdg_hamiltonian(kx, ky, channel))
        np.testing.assert_allclose(energies, -energies[::-1], atol=1e-12)


def test_pairing_channels_match_definitions():
    params = TwoBandParameters(delta_s=0.13, delta_d=0.17)
    kx, ky = 0.41, -0.29

    np.testing.assert_allclose(pairing_matrix("normal", kx, ky, params), np.zeros((2, 2), dtype=complex))
    np.testing.assert_allclose(pairing_matrix("spm", kx, ky, params), params.delta_s * TAUX)
    np.testing.assert_allclose(
        pairing_matrix("dwave", kx, ky, params),
        params.delta_d * d_wave_form_factor(kx, ky) * TAU0,
    )


def test_removed_control_channel_is_rejected():
    removed_channel = "s" + "pp"
    with pytest.raises(ValueError, match="channel must be 'normal', 'spm', or 'dwave'"):
        pairing_matrix(removed_channel, 0.41, -0.29)  # type: ignore[arg-type]


def test_dwave_form_factor_symmetry():
    k = 0.43
    kx, ky = 0.37, -0.21

    assert d_wave_form_factor(k, k) == 0.0
    np.testing.assert_allclose(d_wave_form_factor(kx, ky), -d_wave_form_factor(ky, kx))


def test_velocity_and_mass_shapes():
    kx, ky = 0.37, -0.22

    assert velocity_operator(kx, ky, "x").shape == (2, 2)
    assert velocity_operator(kx, ky, "y").shape == (2, 2)
    assert mass_operator(kx, ky, "x", "x").shape == (2, 2)
    assert mass_operator(kx, ky, "x", "y").shape == (2, 2)


def test_velocity_operator_matches_finite_difference():
    kx, ky = 0.37, -0.22
    step = 1e-6

    finite_difference_x = (
        normal_hamiltonian(kx + step, ky) - normal_hamiltonian(kx - step, ky)
    ) / (2.0 * step)
    finite_difference_y = (
        normal_hamiltonian(kx, ky + step) - normal_hamiltonian(kx, ky - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(velocity_operator(kx, ky, "x"), finite_difference_x, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(velocity_operator(kx, ky, "y"), finite_difference_y, rtol=1e-8, atol=1e-8)


def test_mass_operator_matches_finite_difference_of_velocity():
    kx, ky = 0.37, -0.22
    step = 1e-6

    finite_difference_xx = (
        velocity_operator(kx + step, ky, "x") - velocity_operator(kx - step, ky, "x")
    ) / (2.0 * step)
    finite_difference_xy = (
        velocity_operator(kx, ky + step, "x") - velocity_operator(kx, ky - step, "x")
    ) / (2.0 * step)

    np.testing.assert_allclose(mass_operator(kx, ky, "x", "x"), finite_difference_xx, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(mass_operator(kx, ky, "x", "y"), finite_difference_xy, rtol=1e-8, atol=1e-8)
