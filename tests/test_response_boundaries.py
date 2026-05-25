import numpy as np
import pytest

from lno327 import (
    PairingAmplitudes,
    ResponseUnitConvention,
    local_response_imag_axis,
    local_response_matsubara_index,
    model_response_to_sheet_conductivity,
    nonlocal_response_imag_axis,
    sheet_conductivity_to_dimensionless,
    uniform_bz_mesh,
    k_weights,
)


def test_unit_conversion_preserves_2x2_matrix_structure():
    matrix = np.array([[1.0 + 0.0j, 0.2], [0.3, 2.0 + 0.0j]], dtype=complex)
    conversion = model_response_to_sheet_conductivity(matrix, ResponseUnitConvention())
    dimensionless = sheet_conductivity_to_dimensionless(conversion.tensor)

    assert conversion.tensor.matrix().shape == (2, 2)
    assert dimensionless.matrix().shape == (2, 2)


def test_unit_conversion_without_si_convention_is_not_casimir_valid():
    matrix = np.eye(2, dtype=complex)
    conversion = model_response_to_sheet_conductivity(matrix)

    assert conversion.normalization_status == "dimensionless_model_not_si_sheet"
    assert not conversion.valid_for_casimir_input


def test_bdg_n0_direct_sigma_is_rejected():
    mesh = uniform_bz_mesh(3)

    for kind in ("spm", "dwave"):
        with pytest.raises(ValueError, match="n=0 is unresolved"):
            local_response_imag_axis(
                kind,
                0.0,
                mesh,
                temperature_K=30.0,
                pairing_params=PairingAmplitudes(delta0_eV=0.04),
            )


def test_static_response_policies_are_explicit():
    for policy, status, approximate in (
        ("skip", "skipped", False),
        ("extrapolate_from_lowest_matsubara", "extrapolated", True),
        ("use_static_kernel", "static_kernel", False),
    ):
        result = local_response_matsubara_index(
            "spm",
            0,
            30.0,
            policy=policy,  # type: ignore[arg-type]
            nk=3,
            eta_eV=1e-4,
            pairing_params=PairingAmplitudes(delta0_eV=0.04),
        )

        assert result.status == status
        assert result.approximate is approximate
        assert not result.valid_for_casimir_input
        if policy == "skip":
            assert result.matrix is None
        else:
            assert result.matrix is not None
            assert np.isfinite(result.matrix).all()


def test_nonlocal_q0_local_fallback_equals_local_response():
    mesh = uniform_bz_mesh(3)
    weights = k_weights(mesh)
    params = PairingAmplitudes(delta0_eV=0.04)
    local = local_response_imag_axis(
        "dwave",
        0.1,
        mesh,
        temperature_K=30.0,
        eta_eV=0.02,
        pairing_params=params,
        k_weights=weights,
    )
    nonlocal_response = nonlocal_response_imag_axis(
        "dwave",
        0.1,
        0.0,
        0.2,
        "local_fallback",
        mesh,
        temperature_K=30.0,
        eta_eV=0.02,
        pairing_params=params,
        k_weights=weights,
    )

    np.testing.assert_allclose(nonlocal_response.matrix, local.matrix)
    assert not nonlocal_response.nonlocal_resolved
    assert nonlocal_response.local_fallback_used


def test_finite_q_placeholder_raises_not_implemented():
    mesh = uniform_bz_mesh(3)

    with pytest.raises(NotImplementedError, match="finite-q"):
        nonlocal_response_imag_axis(
            "normal",
            0.1,
            1e6,
            0.2,
            "finite_q_placeholder",
            mesh,
            temperature_K=30.0,
        )
