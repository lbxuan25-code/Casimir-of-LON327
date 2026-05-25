import numpy as np
import pytest

from lno327 import (
    PairingAmplitudes,
    SheetConductivityConvention,
    local_response_imag_axis,
    local_response_matsubara_index,
    model_response_to_sheet_conductivity,
    model_response_to_reflection_dimensionless,
    nonlocal_response_imag_axis,
    require_sheet_conductivity_for_reflection,
    sheet_conductivity_to_reflection_dimensionless,
    uniform_bz_mesh,
    k_weights,
)
from lno327.constants import E2_OVER_HBAR, SIGMA0


def test_unit_conversion_preserves_2x2_matrix_structure():
    matrix = np.array([[1.0 + 0.0j, 0.2], [0.3, 2.0 + 0.0j]], dtype=complex)
    conversion = model_response_to_sheet_conductivity(matrix, SheetConductivityConvention())
    dimensionless = sheet_conductivity_to_reflection_dimensionless(conversion)

    assert conversion.tensor.matrix().shape == (2, 2)
    assert dimensionless.tensor.matrix().shape == (2, 2)


def test_model_response_to_sheet_conductivity_applies_e2_over_hbar():
    matrix = np.array([[1.0, 0.2], [0.2, 3.0]], dtype=complex)
    conversion = model_response_to_sheet_conductivity(matrix)

    np.testing.assert_allclose(conversion.tensor.matrix(), E2_OVER_HBAR * matrix)


def test_sheet_conductivity_to_reflection_dimensionless_divides_by_sigma0():
    matrix = np.array([[1.0, 0.2], [0.2, 3.0]], dtype=complex)
    dimensionless = sheet_conductivity_to_reflection_dimensionless(matrix)

    np.testing.assert_allclose(dimensionless.tensor.matrix(), matrix / SIGMA0)


def test_model_response_to_reflection_dimensionless_combines_factors():
    matrix = np.array([[1.0, 0.2], [0.2, 3.0]], dtype=complex)
    dimensionless = model_response_to_reflection_dimensionless(matrix)

    np.testing.assert_allclose(dimensionless.tensor.matrix(), (E2_OVER_HBAR / SIGMA0) * matrix)


def test_unit_conversion_preserves_anisotropy_delta():
    matrix = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=complex)
    before = (matrix[0, 0] - matrix[1, 1]) / (matrix[0, 0] + matrix[1, 1])
    after_matrix = model_response_to_reflection_dimensionless(matrix).tensor.matrix()
    after = (after_matrix[0, 0] - after_matrix[1, 1]) / (after_matrix[0, 0] + after_matrix[1, 1])

    np.testing.assert_allclose(after, before)


def test_unit_conversion_prevents_double_scaling():
    matrix = np.eye(2, dtype=complex)
    sheet = model_response_to_sheet_conductivity(matrix)

    with pytest.raises(ValueError, match="twice"):
        model_response_to_sheet_conductivity(sheet)

    assert require_sheet_conductivity_for_reflection(sheet) is sheet


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
