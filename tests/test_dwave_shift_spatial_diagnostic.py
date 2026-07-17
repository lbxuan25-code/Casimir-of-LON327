from __future__ import annotations

import numpy as np

from lno327.response.effective_kernel import effective_em_kernel_from_components
from validation.lib.dwave_shift_spatial import (
    SpatialDiagnosticConfig,
    block_mass_table,
    components_from_primitive_vector,
    concentration_area,
    evaluate_shift_spatial,
    shift_rule,
)


def _config() -> SpatialDiagnosticConfig:
    return SpatialDiagnosticConfig(
        base_nk=2,
        qx=0.03,
        qy=0.02,
        temperature_K=10.0,
        delta0_eV=0.1,
        eta_eV=1e-8,
    )


def test_shift_rules_are_normalized_complete_lattice_rules():
    midpoint, midpoint_weights = shift_rule("midpoint")
    gauss, gauss_weights = shift_rule("gauss2")
    halton, halton_weights = shift_rule("halton4")
    assert midpoint.shape == (1, 2)
    assert gauss.shape == (4, 2)
    assert halton.shape == (4, 2)
    for shifts, weights in (
        (midpoint, midpoint_weights),
        (gauss, gauss_weights),
        (halton, halton_weights),
    ):
        assert np.all(shifts >= 0.0)
        assert np.all(shifts < 1.0)
        assert np.all(weights > 0.0)
        assert np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1e-14)


def test_pointwise_primitive_sum_reproduces_complete_shift_response():
    result = evaluate_shift_spatial(_config(), np.asarray([0.5, 1.0 / 3.0]), keep_workspace=True)
    vector = np.sum(result["vectors"], axis=0)
    rebuilt, rhs = components_from_primitive_vector(vector, result["workspace"])
    reference = result["components"]
    assert np.allclose(rebuilt.bare_bubble, reference.bare_bubble, rtol=1e-11, atol=1e-12)
    assert np.allclose(rebuilt.direct, reference.direct, rtol=1e-11, atol=1e-12)
    assert np.allclose(
        rebuilt.collective_counterterm,
        reference.collective_counterterm,
        rtol=1e-11,
        atol=1e-12,
    )
    assert np.allclose(
        rebuilt.amplitude_phase_schur,
        reference.amplitude_phase_schur,
        rtol=1e-10,
        atol=1e-11,
    )
    assert np.allclose(rhs.left, result["rhs"].left, rtol=1e-11, atol=1e-12)
    kernel = effective_em_kernel_from_components(rebuilt, q_model=_config().q, xi_eV=0.0)
    assert kernel.schur_inverse_method == "inv"


def test_block_mass_ranking_and_concentration_are_monotone():
    delta = np.zeros((10, 46), dtype=complex)
    delta[:, 0] = np.arange(10, 0, -1)
    delta[:, 26] = np.linspace(0.1, 1.0, 10)
    masses, score = block_mass_table(delta)
    assert score.shape == (10,)
    assert np.isfinite(score).all()
    for values in masses.values():
        assert np.all(values >= 0.0)
    area50 = concentration_area(masses["k_ss"], 0.5)
    area80 = concentration_area(masses["k_ss"], 0.8)
    area90 = concentration_area(masses["k_ss"], 0.9)
    assert 0.0 < area50 <= area80 <= area90 <= 1.0
