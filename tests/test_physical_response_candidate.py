from __future__ import annotations

import inspect

import numpy as np

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV, k_weights, uniform_bz_mesh
from lno327.ward_response import (
    normal_density_current_response_imag_axis,
    normal_physical_density_current_response_imag_axis,
)


def test_physical_response_candidate_has_no_sign_scan_parameters():
    signature = inspect.signature(normal_physical_density_current_response_imag_axis)
    for forbidden in (
        "sign_convention",
        "contact_sign_convention",
        "current_vertex_multiplier",
        "ward_q_sign",
        "vertex_scheme",
        "contact_scheme",
    ):
        assert forbidden not in signature.parameters


def test_physical_response_candidate_matches_manual_code_piece_combination():
    mesh = uniform_bz_mesh(4)
    weights = k_weights(mesh)
    q = np.array([0.01, 0.007], dtype=float)
    omega_eV = bosonic_matsubara_energy_eV(1, 30.0)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=30.0,
        eta_eV=1e-10,
        output_si=False,
    )

    physical = normal_physical_density_current_response_imag_axis(mesh, config, q, weights)
    code_bubble = normal_density_current_response_imag_axis(
        mesh,
        config,
        q,
        weights,
        vertex_scheme="peierls",
        contact_scheme="none",
    )
    code_contact_plus = normal_density_current_response_imag_axis(
        mesh,
        config,
        q,
        weights,
        vertex_scheme="peierls",
        contact_scheme="finite_q_peierls",
        contact_sign_convention="plus",
    )
    code_contact = code_contact_plus - code_bubble

    manual = np.array(code_bubble, copy=True)
    manual[0, 0] = code_bubble[0, 0]
    manual[0, 1:] = code_bubble[0, 1:]
    manual[1:, 0] = -code_bubble[1:, 0]
    manual[1:, 1:] = -code_bubble[1:, 1:] - code_contact[1:, 1:]

    np.testing.assert_allclose(physical, manual, atol=1e-13, rtol=1e-11)


def test_old_convention_scanner_docstring_is_diagnostic_only():
    docstring = normal_density_current_response_imag_axis.__doc__ or ""
    assert "Diagnostic-only" in docstring
    assert "Do not use it as the main response path" in docstring
