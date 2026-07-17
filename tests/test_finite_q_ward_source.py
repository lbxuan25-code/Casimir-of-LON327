import numpy as np
import pytest

from lno327.casimir.microscopic_model import get_finite_q_microscopic_model


@pytest.mark.parametrize("pairing_name", ["spm", "dwave"])
@pytest.mark.parametrize(
    "vertex_name",
    ["midpoint", "symmetric_kpm", "bond_endpoint_gauge"],
)
def test_two_band_collective_vertices_follow_form_factor(pairing_name, vertex_name):
    model = get_finite_q_microscopic_model("symmetry_bdg_2band")
    amp = model.build_pairing_params()
    ansatz = model.build_ansatz(pairing_name, vertex_name)
    kx, ky, qx, qy = 0.37, -0.22, 0.13, -0.07

    phi = ansatz.collective_form_factor(kx, ky, qx, qy, amp)
    zero = np.zeros_like(phi)
    expected_amplitude = np.block(
        [[zero, phi], [phi.conjugate().T, zero]]
    ).astype(complex)
    expected_eta2 = np.block(
        [[zero, 1j * phi], [-1j * phi.conjugate().T, zero]]
    ).astype(complex)
    amplitude, eta2 = ansatz.collective_vertices(kx, ky, qx, qy, amp)

    np.testing.assert_allclose(amplitude, expected_amplitude)
    np.testing.assert_allclose(eta2, expected_eta2)
    np.testing.assert_allclose(
        ansatz.phase_pairing_matrix(kx, ky, qx, qy, amp) / amp.delta0_eV,
        phi,
    )


def test_two_band_dwave_bond_endpoint_form_factor_is_endpoint_average():
    model = get_finite_q_microscopic_model("symmetry_bdg_2band")
    amp = model.build_pairing_params()
    ansatz = model.build_ansatz("dwave", "bond_endpoint_gauge")
    kx, ky, qx, qy = 0.37, -0.22, 0.13, -0.07
    delta0 = float(amp.delta0_eV)

    phi_minus = (
        ansatz.mean_pairing(kx - 0.5 * qx, ky - 0.5 * qy, amp) / delta0
    )
    phi_plus = (
        ansatz.mean_pairing(kx + 0.5 * qx, ky + 0.5 * qy, amp) / delta0
    )

    np.testing.assert_allclose(
        ansatz.collective_form_factor(kx, ky, qx, qy, amp),
        0.5 * (phi_minus + phi_plus),
    )
