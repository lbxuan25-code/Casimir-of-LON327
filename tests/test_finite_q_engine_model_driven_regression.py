import numpy as np

from lno327.response.config import KuboConfig
from lno327.numerics.weights import k_weights
from lno327.numerics.grids import uniform_bz_mesh
from lno327.workflows.finite_q_engine import (
    FiniteQEngineOptions,
    bdg_finite_q_response_imag_axis,
    finite_q_bdg_response_from_ansatz,
)
from lno327.models.lno327_four_orbital.collective import build_pairing_ansatz
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec
from lno327.response.finite_q_bdg import finite_q_bdg_response_from_model_ansatz


def _inputs():
    points = uniform_bz_mesh(2)
    return (
        np.array([0.01, 0.0]),
        points,
        k_weights(points),
        KuboConfig.from_kelvin(omega_eV=0.01, temperature_K=10.0, eta_eV=1e-8, output_si=False),
        PairingAmplitudes(delta0_eV=0.04),
    )


def test_public_ansatz_adapter_matches_model_driven_core_for_collective_modes():
    q, points, weights, config, amp = _inputs()
    for pairing_name in ("spm", "dwave"):
        ansatz = build_pairing_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
        for mode in ("phase_only", "amplitude_phase"):
            options = FiniteQEngineOptions(collective_mode=mode)
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
                "minus_schur",
                "plus_schur",
                "amplitude_phase_schur",
                "gauge_restored",
            ):
                np.testing.assert_allclose(getattr(public, field), getattr(core, field))
            assert public.metadata["valid_for_casimir_input"] is False


def test_named_public_wrapper_uses_model_driven_core_path():
    q, points, weights, config, amp = _inputs()
    wrapper = bdg_finite_q_response_imag_axis(
        "spm",
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        phase_vertex="bond_endpoint_gauge",
    )
    ansatz = build_pairing_ansatz("spm", phase_vertex="bond_endpoint_gauge")
    core = finite_q_bdg_response_from_model_ansatz(
        LNO327FourOrbitalSpec(pairing_amplitudes=amp),
        ansatz,
        config.omega_eV,
        q,
        points,
        weights,
        config,
        amp,
        FiniteQEngineOptions(),
    )
    np.testing.assert_allclose(wrapper.gauge_restored, core.gauge_restored)
    assert wrapper.metadata["valid_for_casimir_input"] is False
