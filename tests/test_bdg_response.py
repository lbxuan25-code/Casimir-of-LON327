import numpy as np

from lno327 import (
    KuboConfig,
    PairingAmplitudes,
    bdg_current_vertex,
    bdg_paramagnetic_kernel_imag_axis,
    k_weights,
    normal_state_velocity_operator,
    uniform_bz_mesh,
)


def test_bdg_current_vertex_shape():
    vertex = bdg_current_vertex(0.2, -0.3, "x")

    assert vertex.shape == (8, 8)


def test_bdg_current_vertex_is_hermitian():
    vertex = bdg_current_vertex(0.2, -0.3, "y")

    np.testing.assert_allclose(vertex, vertex.conjugate().T)


def test_bdg_current_vertex_blocks_match_charge_current_definition():
    kx, ky = 0.2, -0.3
    vertex = bdg_current_vertex(kx, ky, "x")

    np.testing.assert_allclose(vertex[:4, :4], normal_state_velocity_operator(kx, ky, "x"))
    np.testing.assert_allclose(vertex[4:, 4:], -normal_state_velocity_operator(-kx, -ky, "x").T)
    np.testing.assert_allclose(vertex[:4, 4:], np.zeros((4, 4)))
    np.testing.assert_allclose(vertex[4:, :4], np.zeros((4, 4)))


def test_bdg_paramagnetic_kernel_imag_axis_returns_complex_2x2_matrix():
    mesh = uniform_bz_mesh(3)
    config = KuboConfig.from_kelvin(omega_eV=0.2, temperature_K=30.0, eta_eV=0.02, output_si=False)

    kernel = bdg_paramagnetic_kernel_imag_axis(
        mesh,
        config,
        "spm",
        PairingAmplitudes(delta0_eV=0.04),
        k_weights(mesh),
    )

    assert kernel.shape == (2, 2)
    assert kernel.dtype == complex
    assert np.isfinite(kernel).all()


def test_bdg_paramagnetic_kernel_is_c4_symmetric_without_magnetic_field():
    mesh = uniform_bz_mesh(8)
    config = KuboConfig.from_kelvin(omega_eV=0.2, temperature_K=30.0, eta_eV=0.02, output_si=False)

    kernel = bdg_paramagnetic_kernel_imag_axis(
        mesh,
        config,
        "dwave",
        PairingAmplitudes(delta0_eV=0.04),
        k_weights(mesh),
    )

    assert np.isclose(kernel[0, 0], kernel[1, 1], atol=1e-10)
    assert abs(kernel[0, 1]) < 1e-10
    assert abs(kernel[1, 0]) < 1e-10
