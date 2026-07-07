from __future__ import annotations

import numpy as np
import pytest

from lno327.collective.ward import physical_ward_residuals
from validation.lib.finite_q_collective_ward_blocks import (
    BLOCK_IDENTITY_VERSION,
    collective_generators,
    evaluate_collective_ward_blocks,
    schur_residual_reconstruction,
)


def _random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _invertible_complex(rng, dim):
    matrix = _random_complex(rng, (dim, dim))
    return matrix + (dim + 1.0) * np.eye(dim)


def _payload_vector(payload):
    return np.asarray([entry["real"] + 1j * entry["imag"] for entry in payload["vector"]])


def test_collective_generators_follow_eta2_operator_identity_sign():
    left, right = collective_generators(0.1)

    np.testing.assert_allclose(left, np.asarray([0.0 + 0.0j, 0.0 - 0.2j]))
    np.testing.assert_allclose(right, np.asarray([0.0 + 0.0j, 0.0 + 0.2j]))


def test_schur_reconstruction_is_exact_block_algebra():
    rng = np.random.default_rng(20260709)
    aa_left_error = _random_complex(rng, (3,))
    aeta_left_error = _random_complex(rng, (2,))
    aa_right_error = _random_complex(rng, (3,))
    etaa_right_error = _random_complex(rng, (2,))
    k_aeta = _random_complex(rng, (3, 2))
    k_etaa = _random_complex(rng, (2, 3))
    k_etaeta = _invertible_complex(rng, 2)

    predicted_left, predicted_right, inverse_method, condition_number = schur_residual_reconstruction(
        aa_left_error=aa_left_error,
        aeta_left_error=aeta_left_error,
        aa_right_error=aa_right_error,
        etaa_right_error=etaa_right_error,
        k_aeta=k_aeta,
        k_etaa=k_etaa,
        k_etaeta=k_etaeta,
    )

    inverse = np.linalg.inv(k_etaeta)
    np.testing.assert_allclose(predicted_left, aa_left_error - aeta_left_error @ inverse @ k_etaa)
    np.testing.assert_allclose(predicted_right, aa_right_error - k_aeta @ inverse @ etaa_right_error)
    assert inverse_method == "inv"
    assert condition_number is not None


def test_collective_block_payload_reconstructs_explicit_schur_ward_residual():
    rng = np.random.default_rng(20260710)
    omega = 0.17
    q = np.asarray([0.03, -0.02])
    delta0 = 0.11
    k_aa_full = _random_complex(rng, (3, 3))
    k_aeta = _random_complex(rng, (3, 2))
    k_etaa = _random_complex(rng, (2, 3))
    k_etaeta = _invertible_complex(rng, 2)
    schur = k_aa_full - k_aeta @ np.linalg.inv(k_etaeta) @ k_etaa

    payload = evaluate_collective_ward_blocks(
        pairing_name="dwave",
        q_model=q,
        omega_eV=omega,
        delta0_eV=delta0,
        k_aa_full=k_aa_full,
        k_aeta=k_aeta,
        k_etaa=k_etaa,
        k_etaeta=k_etaeta,
        schur_response=schur,
    )

    actual_left, actual_right = physical_ward_residuals(schur, omega, q)
    predicted_left = _payload_vector(payload["schur_reconstruction"]["predicted_left"])
    predicted_right = _payload_vector(payload["schur_reconstruction"]["predicted_right"])

    assert payload["identity_version"] == BLOCK_IDENTITY_VERSION
    assert payload["diagnostic_role"] == "algebraic_block_identity_localization_not_a_new_criterion"
    assert set(payload["block_residuals"]) == {"aa_left", "aeta_left", "aa_right", "etaa_right"}
    assert set(payload["block_decomposition"]) == {"aa_left", "aeta_left", "aa_right", "etaa_right"}
    assert set(payload["schur_contribution_breakdown"]["contributions"]) == {
        "left_from_aa_identity",
        "left_from_aeta_identity",
        "right_from_aa_identity",
        "right_from_etaa_identity",
    }
    np.testing.assert_allclose(predicted_left, actual_left, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(predicted_right, actual_right, atol=1e-12, rtol=1e-12)
    assert payload["schur_reconstruction"]["max_difference_norm"] == pytest.approx(0.0, abs=1e-12)
    assert payload["valid_for_casimir_input"] is False


def test_aa_full_term_decomposition_reconstructs_bubble_direct_sum():
    rng = np.random.default_rng(20260712)
    omega = 0.09
    q = np.asarray([0.02, 0.01])
    delta0 = 0.07
    k_aa_bubble = _random_complex(rng, (3, 3))
    k_aa_direct = _random_complex(rng, (3, 3))
    k_aa_full = k_aa_bubble + k_aa_direct
    k_aeta = _random_complex(rng, (3, 2))
    k_etaa = _random_complex(rng, (2, 3))
    k_etaeta = _invertible_complex(rng, 2)
    schur = k_aa_full - k_aeta @ np.linalg.inv(k_etaeta) @ k_etaa

    payload = evaluate_collective_ward_blocks(
        pairing_name="dwave",
        q_model=q,
        omega_eV=omega,
        delta0_eV=delta0,
        k_aa_full=k_aa_full,
        k_aeta=k_aeta,
        k_etaa=k_etaa,
        k_etaeta=k_etaeta,
        schur_response=schur,
        k_aa_bubble=k_aa_bubble,
        k_aa_direct=k_aa_direct,
    )

    aa_terms = payload["aa_full_term_decomposition"]
    assert aa_terms is not None
    assert aa_terms["aa_full_minus_bubble_plus_direct_norm"] == pytest.approx(0.0, abs=1e-12)
    for side in ("left", "right"):
        section = aa_terms[side]
        total = _payload_vector(section["sum"])
        full = _payload_vector(section["full_identity_residual"])
        difference = _payload_vector(section["sum_minus_full_identity_residual"])
        np.testing.assert_allclose(total, full, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(difference, np.zeros(3, dtype=complex), atol=1e-12, rtol=1e-12)
        assert section["sum_minus_full_identity_residual_norm"] == pytest.approx(0.0, abs=1e-12)


def test_block_decomposition_residuals_are_sums_of_reported_terms():
    rng = np.random.default_rng(20260711)
    payload = evaluate_collective_ward_blocks(
        pairing_name="spm",
        q_model=np.asarray([0.01, 0.0]),
        omega_eV=0.01,
        delta0_eV=0.1,
        k_aa_full=_random_complex(rng, (3, 3)),
        k_aeta=_random_complex(rng, (3, 2)),
        k_etaa=_random_complex(rng, (2, 3)),
        k_etaeta=_invertible_complex(rng, 2),
        schur_response=_random_complex(rng, (3, 3)),
    )

    for decomposition in payload["block_decomposition"].values():
        terms = list(decomposition["terms"].values())
        summed = _payload_vector(terms[0]) + _payload_vector(terms[1])
        residual = _payload_vector(decomposition["residual"])
        np.testing.assert_allclose(summed, residual)
        assert "cancellation_fraction" in decomposition
        assert "cosine_between_terms" in decomposition


def test_collective_block_payload_validates_shapes():
    with pytest.raises(ValueError, match="k_aeta"):
        evaluate_collective_ward_blocks(
            pairing_name="spm",
            q_model=[0.01, 0.0],
            omega_eV=0.01,
            delta0_eV=0.1,
            k_aa_full=np.zeros((3, 3)),
            k_aeta=np.zeros((2, 3)),
            k_etaa=np.zeros((2, 3)),
            k_etaeta=np.eye(2),
            schur_response=np.zeros((3, 3)),
        )

    with pytest.raises(ValueError, match="provided together"):
        evaluate_collective_ward_blocks(
            pairing_name="spm",
            q_model=[0.01, 0.0],
            omega_eV=0.01,
            delta0_eV=0.1,
            k_aa_full=np.zeros((3, 3)),
            k_aeta=np.zeros((3, 2)),
            k_etaa=np.zeros((2, 3)),
            k_etaeta=np.eye(2),
            schur_response=np.zeros((3, 3)),
            k_aa_bubble=np.zeros((3, 3)),
        )
