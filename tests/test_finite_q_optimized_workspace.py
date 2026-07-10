from __future__ import annotations

import warnings

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.electrodynamics.static_sheet import static_matsubara_kernel_to_sheet_response
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.finite_q_bdg import finite_q_bdg_response_from_model_ansatz
from lno327.response.finite_q_optimized import (
    finite_q_bdg_response_from_q_workspace,
    finite_q_bdg_responses_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327.response.ward_validation import (
    primitive_ward_rhs_from_model_ansatz,
    validate_effective_ward_xy,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

_RESPONSE_NAMES = (
    "bare_bubble",
    "direct",
    "bare_total",
    "phase_coupling_left",
    "phase_coupling_right",
    "collective_bubble",
    "collective_counterterm",
    "collective_total",
    "em_collective_left",
    "collective_em_right",
    "amplitude_phase_schur",
)


def _problem(*, omega_eV: float = 0.01, nk: int = 3):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("spm", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    return model, ansatz, pairing, points, weights, config


def _assert_components_close(left, right):
    for name in _RESPONSE_NAMES:
        np.testing.assert_allclose(
            getattr(left, name),
            getattr(right, name),
            rtol=2e-11,
            atol=2e-12,
            err_msg=f"mismatch for {name}",
        )
    np.testing.assert_allclose(
        left.phase_phase_bubble,
        right.phase_phase_bubble,
        rtol=2e-11,
        atol=2e-12,
    )


def test_vectorized_q_workspace_matches_direct_response_at_positive_and_zero_frequency():
    model, ansatz, pairing, points, weights, config = _problem()
    q = np.asarray([0.03, 0.02])
    options = FiniteQEngineOptions()
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec, ansatz, points, weights, config, pairing, options
    )
    workspace = precompute_finite_q_q_workspace(material, q)

    for xi in (0.01, 0.0):
        eval_config = KuboConfig(
            omega_eV=xi,
            temperature_eV=config.temperature_eV,
            fermi_level_eV=config.fermi_level_eV,
            eta_eV=config.eta_eV,
            output_si=config.output_si,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            direct = finite_q_bdg_response_from_model_ansatz(
                model.spec,
                ansatz,
                xi,
                q,
                points,
                weights,
                eval_config,
                pairing,
                options,
            )
            optimized = finite_q_bdg_response_from_q_workspace(workspace, xi)
        _assert_components_close(direct, optimized)


def test_batched_frequency_evaluation_matches_scalar_workspace_evaluation():
    model, ansatz, pairing, points, weights, config = _problem()
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec, ansatz, points, weights, config, pairing, FiniteQEngineOptions()
    )
    workspace = precompute_finite_q_q_workspace(material, np.asarray([0.03, 0.02]))
    xis = np.asarray([0.0, 1e-4, 1e-3, 1e-2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        batch = finite_q_bdg_responses_from_q_workspace(workspace, xis)
        scalar = tuple(finite_q_bdg_response_from_q_workspace(workspace, xi) for xi in xis)
    for left, right in zip(batch, scalar, strict=True):
        _assert_components_close(left, right)


def test_cached_ward_rhs_matches_independent_legacy_builder():
    model, ansatz, pairing, points, weights, config = _problem(omega_eV=0.0)
    q = np.asarray([0.03, 0.02])
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec, ansatz, points, weights, config, pairing, FiniteQEngineOptions()
    )
    workspace = precompute_finite_q_q_workspace(material, q)
    cached = primitive_ward_rhs_from_q_workspace(workspace, 0.0)
    legacy = primitive_ward_rhs_from_model_ansatz(
        model.spec, ansatz, q, points, weights, config, pairing
    )
    np.testing.assert_allclose(cached.left, legacy.left, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(cached.right, legacy.right, rtol=2e-11, atol=2e-12)
    assert cached.metadata["frequency_independent_rhs_reused"] is True


def test_optimized_workspace_feeds_static_kernel_and_ward_contract():
    model, ansatz, pairing, points, weights, config = _problem(omega_eV=0.0)
    q = np.asarray([0.03, 0.02])
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec, ansatz, points, weights, config, pairing, FiniteQEngineOptions()
    )
    workspace = precompute_finite_q_q_workspace(material, q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        components = finite_q_bdg_response_from_q_workspace(workspace, 0.0)
    kernel = effective_em_kernel_from_components(components, q_model=q, xi_eV=0.0)
    rhs = primitive_ward_rhs_from_q_workspace(workspace, 0.0)
    ward = validate_effective_ward_xy(kernel, rhs, residual_tolerance=1e-6)
    static = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward,
        longitudinal_tolerance=1.0,
        mixing_tolerance=1.0,
        passivity_tolerance=1.0,
    )
    assert ward.passed is True
    assert static.validation.ward_passed is True
    assert np.isfinite(static.chi_bar)
    assert np.isfinite(static.dbar_t)


def test_workspace_metadata_locks_expected_reuse_structure():
    model, ansatz, pairing, points, weights, config = _problem(nk=4)
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec, ansatz, points, weights, config, pairing, FiniteQEngineOptions()
    )
    workspace = precompute_finite_q_q_workspace(material, np.asarray([0.03, 0.02]))
    assert material.metadata["midpoint_eigensystem_count"] == points.shape[0]
    assert material.metadata["goldstone_counterterm_cached_once"] is True
    assert workspace.metadata["shifted_eigensystem_count"] == 2 * points.shape[0]
    assert workspace.metadata["midpoint_eigensystems_reused"] == points.shape[0]
    assert workspace.metadata["ward_rhs_cached"] is True
    assert workspace.metadata["unified_channel_count"] == 5
    assert workspace.metadata["phase_only_derived_from_eta2"] is True
