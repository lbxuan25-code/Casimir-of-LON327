from __future__ import annotations

import warnings

import numpy as np
import pytest

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.bdg.hamiltonian import bdg_hamiltonian_from_model_pairing
from lno327.response.finite_q_optimized import (
    finite_q_bdg_responses_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
)
from lno327.response.finite_q_q_workspace_batched import (
    precompute_finite_q_q_workspace_batched,
    supports_batched_finite_q_q_workspace,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import (
    get_finite_q_validation_model,
)

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


@pytest.mark.parametrize("pairing_name", ["spm", "dwave"])
def test_two_band_batch_model_capabilities_match_scalar_helpers(
    pairing_name: str,
) -> None:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    spec = model.spec
    ansatz = model.build_ansatz(
        pairing_name,
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(0.1)
    points = np.asarray(
        [
            [0.11, -0.23],
            [-1.07, 0.41],
            [2.13, -2.31],
            [-2.71, 1.83],
        ],
        dtype=float,
    )
    q = np.asarray([0.07, -0.04], dtype=float)

    normal_batch = spec.normal_hamiltonian_batch(points)
    pairing_batch = ansatz.mean_pairing_batch(points, pairing)
    bdg_batch = spec.bdg_hamiltonian_from_pairing_batch(
        points,
        pairing_batch,
    )
    collective_batch = ansatz.collective_vertices_batch(
        points,
        q,
        pairing,
    )
    phase_pairing_batch = ansatz.phase_pairing_matrix_batch(
        points,
        q,
        pairing,
    )
    vector_batch, contact_batch = spec.peierls_hamiltonian_vertices_batch(
        points,
        q,
    )

    normal_scalar = np.stack(
        [spec.normal_hamiltonian(float(kx), float(ky)) for kx, ky in points]
    )
    pairing_scalar = np.stack(
        [
            ansatz.mean_pairing(float(kx), float(ky), pairing)
            for kx, ky in points
        ]
    )
    bdg_scalar = np.stack(
        [
            bdg_hamiltonian_from_model_pairing(
                spec,
                float(kx),
                float(ky),
                pairing_scalar[index],
            )
            for index, (kx, ky) in enumerate(points)
        ]
    )
    collective_scalar = np.stack(
        [
            np.stack(
                ansatz.collective_vertices(
                    float(kx),
                    float(ky),
                    float(q[0]),
                    float(q[1]),
                    pairing,
                )
            )
            for kx, ky in points
        ]
    )
    phase_pairing_scalar = np.stack(
        [
            ansatz.phase_pairing_matrix(
                float(kx),
                float(ky),
                float(q[0]),
                float(q[1]),
                pairing,
            )
            for kx, ky in points
        ]
    )
    vector_scalar = np.stack(
        [
            np.stack(
                [
                    spec.peierls_hamiltonian_vector_vertex(
                        float(kx),
                        float(ky),
                        float(q[0]),
                        float(q[1]),
                        direction,
                    )
                    for direction in ("x", "y")
                ]
            )
            for kx, ky in points
        ]
    )
    contact_scalar = np.stack(
        [
            np.stack(
                [
                    np.stack(
                        [
                            spec.peierls_hamiltonian_contact_vertex(
                                float(kx),
                                float(ky),
                                float(q[0]),
                                float(q[1]),
                                direction_i,
                                direction_j,
                            )
                            for direction_j in ("x", "y")
                        ]
                    )
                    for direction_i in ("x", "y")
                ]
            )
            for kx, ky in points
        ]
    )

    np.testing.assert_allclose(normal_batch, normal_scalar, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(pairing_batch, pairing_scalar, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(bdg_batch, bdg_scalar, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        collective_batch,
        collective_scalar,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        phase_pairing_batch,
        phase_pairing_scalar,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(vector_batch, vector_scalar, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(contact_batch, contact_scalar, rtol=2e-13, atol=2e-13)


def _assert_workspace_close(left, right) -> None:
    for name in (
        "energies_minus",
        "energies_plus",
        "occupations_minus",
        "occupations_plus",
        "left_vertices_band",
        "right_vertices_band",
        "direct_contact_contribution",
        "ward_rhs_vector",
    ):
        np.testing.assert_allclose(
            getattr(left, name),
            getattr(right, name),
            rtol=2e-11,
            atol=2e-12,
            err_msg=f"workspace mismatch for {name}",
        )
    assert left.phase_phase_direct_plus == pytest.approx(
        right.phase_phase_direct_plus,
        rel=2e-11,
        abs=2e-12,
    )
    assert left.phase_phase_direct_minus == pytest.approx(
        right.phase_phase_direct_minus,
        rel=2e-11,
        abs=2e-12,
    )
    for key in ("ward_equal_forward", "ward_delta_v_mid", "ward_qM_mid"):
        np.testing.assert_allclose(
            left.metadata[key],
            right.metadata[key],
            rtol=2e-11,
            atol=2e-12,
            err_msg=f"metadata mismatch for {key}",
        )


def _assert_response_close(left, right) -> None:
    for name in _RESPONSE_NAMES:
        np.testing.assert_allclose(
            getattr(left, name),
            getattr(right, name),
            rtol=3e-11,
            atol=3e-12,
            err_msg=f"response mismatch for {name}",
        )
    np.testing.assert_allclose(
        left.phase_phase_bubble,
        right.phase_phase_bubble,
        rtol=3e-11,
        atol=3e-12,
    )


@pytest.mark.parametrize("pairing_name", ["spm", "dwave"])
@pytest.mark.parametrize(
    "q",
    [
        np.asarray([0.03, 0.02]),
        np.asarray([0.0, 0.0]),
    ],
)
def test_batched_q_workspace_matches_scalar_reference(
    pairing_name: str,
    q: np.ndarray,
) -> None:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(
        pairing_name,
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(0.1)
    points = uniform_bz_mesh(3)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.01,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        config,
        pairing,
        FiniteQEngineOptions(),
    )
    assert supports_batched_finite_q_q_workspace(material)

    scalar = precompute_finite_q_q_workspace(material, q)
    batched = precompute_finite_q_q_workspace_batched(material, q)
    _assert_workspace_close(scalar, batched)
    assert batched.metadata["q_workspace_implementation"] == (
        "batched_model_capability"
    )

    xis = np.asarray([0.0, 1e-4, 1e-2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scalar_responses = finite_q_bdg_responses_from_q_workspace(
            scalar,
            xis,
        )
        batched_responses = finite_q_bdg_responses_from_q_workspace(
            batched,
            xis,
        )
    for scalar_response, batched_response in zip(
        scalar_responses,
        batched_responses,
        strict=True,
    ):
        _assert_response_close(scalar_response, batched_response)
