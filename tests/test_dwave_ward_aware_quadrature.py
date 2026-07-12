from __future__ import annotations

import numpy as np

from lno327 import KuboConfig
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.dwave_static_primitives import (
    build_dwave_static_integrand_context,
)
from validation.lib.dwave_ward_aware_quadrature import (
    AUGMENTED_REAL_WIDTH,
    WARD_DIAGNOSTIC_COMPLEX_WIDTH,
    paired_physical_primitive,
    unpack_augmented_complex_vector,
    ward_aware_real_vector,
    ward_diagnostic_channels,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _context():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    context = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        np.asarray([0.11, 0.07]),
        config,
        pairing,
        FiniteQEngineOptions(phase_hessian_policy="q_independent"),
    )
    return context


def test_ward_aware_real_vector_preserves_paired_physical_prefix():
    context = _context()
    kx, ky = 0.37, -0.29

    expected = paired_physical_primitive(context, kx, ky)
    augmented = ward_aware_real_vector(context, kx, ky)
    physical = unpack_augmented_complex_vector(augmented)

    assert augmented.shape == (AUGMENTED_REAL_WIDTH,)
    assert np.all(np.isfinite(augmented))
    assert np.allclose(physical, expected, rtol=0.0, atol=0.0)


def test_ward_diagnostics_have_fixed_finite_width():
    context = _context()
    physical = paired_physical_primitive(context, 0.21, 0.44)
    diagnostics = ward_diagnostic_channels(context, physical)

    assert diagnostics.shape == (WARD_DIAGNOSTIC_COMPLEX_WIDTH,)
    assert np.all(np.isfinite(diagnostics.real))
    assert np.all(np.isfinite(diagnostics.imag))


def test_paired_primitive_is_mean_of_periodically_wrapped_half_q_translates():
    context = _context()
    center = np.asarray([np.pi - 0.01, -np.pi + 0.02])
    q = np.asarray(context.q_model, dtype=float)
    points = np.stack((center - 0.5 * q, center + 0.5 * q), axis=0)
    points = (points + np.pi) % (2.0 * np.pi) - np.pi

    expected = np.mean(context.evaluate_complex(points), axis=0)
    actual = paired_physical_primitive(context, center[0], center[1])

    assert np.allclose(actual, expected, rtol=0.0, atol=0.0)
