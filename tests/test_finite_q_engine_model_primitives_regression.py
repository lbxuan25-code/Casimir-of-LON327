import numpy as np
import pytest

from lno327.bdg.finite_q import bdg_finite_q_vertex_from_normal_blocks
from lno327.bdg.nambu import charge_current_vertex_from_model, diamagnetic_vertex_from_model
from lno327.response.config import KuboConfig
from lno327.numerics.weights import k_weights
from lno327.numerics.grids import uniform_bz_mesh
from lno327.finite_q_engine import (
    FiniteQEngineOptions,
    finite_q_bdg_response_from_ansatz,
)
from lno327.models.lno327_four_orbital.collective import build_pairing_ansatz
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.finite_q_bdg import (
    bdg_contact_vertex_from_spec,
    bdg_vector_vertex_from_spec,
    finite_q_bdg_response_from_model_ansatz,
)


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.17, -0.09)])
@pytest.mark.parametrize("direction", ("x", "y"))
def test_bdg_vector_vertex_from_spec_matches_model_peierls_blocks(q, direction):
    qx, qy = q
    spec = LNO327FourOrbitalSpec()

    actual = bdg_vector_vertex_from_spec(
        spec,
        0.21,
        -0.34,
        qx,
        qy,
        direction,
        "peierls",
    )
    expected = bdg_finite_q_vertex_from_normal_blocks(
        spec.peierls_hamiltonian_vector_vertex(0.21, -0.34, qx, qy, direction),
        spec.peierls_hamiltonian_vector_vertex(-0.21, 0.34, -qx, -qy, direction),
    )

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("directions", (("x", "x"), ("y", "y"), ("x", "y")))
def test_bdg_contact_vertex_from_spec_matches_model_peierls_blocks(directions):
    direction_i, direction_j = directions
    qx, qy = 0.17, -0.09
    spec = LNO327FourOrbitalSpec()

    actual = bdg_contact_vertex_from_spec(
        spec,
        0.21,
        -0.34,
        qx,
        qy,
        direction_i,
        direction_j,
        "peierls",
    )
    expected = bdg_finite_q_vertex_from_normal_blocks(
        spec.peierls_hamiltonian_contact_vertex(0.21, -0.34, qx, qy, direction_i, direction_j),
        spec.peierls_hamiltonian_contact_vertex(-0.21, 0.34, -qx, -qy, direction_i, direction_j),
    )

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("direction", ("x", "y"))
def test_bdg_vector_vertex_from_spec_matches_model_q0_velocity(direction):
    spec = LNO327FourOrbitalSpec()
    actual = bdg_vector_vertex_from_spec(
        spec,
        0.21,
        -0.34,
        0.17,
        -0.09,
        direction,
        "q0_velocity",
    )

    np.testing.assert_allclose(actual, charge_current_vertex_from_model(spec, 0.21, -0.34, direction))


@pytest.mark.parametrize("directions", (("x", "x"), ("y", "y"), ("x", "y")))
def test_bdg_contact_vertex_from_spec_matches_model_q0_velocity(directions):
    direction_i, direction_j = directions
    spec = LNO327FourOrbitalSpec()

    actual = bdg_contact_vertex_from_spec(
        spec,
        0.21,
        -0.34,
        0.17,
        -0.09,
        direction_i,
        direction_j,
        "q0_velocity",
    )

    np.testing.assert_allclose(actual, diamagnetic_vertex_from_model(spec, 0.21, -0.34, direction_i, direction_j))


def test_public_ansatz_adapter_matches_model_driven_core():
    points = uniform_bz_mesh(2)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    q = np.array([0.01, 0.0])
    amp = PairingAmplitudes(delta0_eV=0.04)
    ansatz = build_pairing_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    options = FiniteQEngineOptions()

    public = finite_q_bdg_response_from_ansatz(ansatz, config.omega_eV, q, points, weights, config, amp, options)
    core = finite_q_bdg_response_from_model_ansatz(
        LNO327FourOrbitalSpec(pairing_amplitudes=amp),
        ansatz,
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        options,
    )

    for field in (
        "bare_bubble",
        "direct",
        "bare_total",
        "phase_coupling_left",
        "phase_coupling_right",
        "phase_phase_bubble",
        "phase_phase_direct",
        "phase_phase_total",
        "minus_schur",
        "plus_schur",
        "collective_bubble",
        "collective_counterterm",
        "collective_total",
        "em_collective_left",
        "collective_em_right",
        "amplitude_phase_schur",
        "gauge_restored",
    ):
        np.testing.assert_allclose(getattr(public, field), getattr(core, field))


def test_peierls_current_requires_spec_support():
    class DummySpec:
        def normal_hamiltonian(self, _kx, _ky):
            return np.eye(2, dtype=complex)

    with pytest.raises(ValueError, match="spec must support Peierls finite-q vertices"):
        bdg_vector_vertex_from_spec(DummySpec(), 0.0, 0.0, 0.1, 0.0, "x", "peierls")
