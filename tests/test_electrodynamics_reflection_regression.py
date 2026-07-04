import numpy as np
import pytest

import lno327.electrodynamics.reflection as new
import lno327.reflection_input as old


def _assert_dict_matches(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], np.ndarray):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
        elif isinstance(actual[key], dict):
            assert actual[key] == expected[key]
        else:
            assert actual[key] == expected[key]


def test_reflection_scalar_and_matrix_helpers_match_old_reference():
    np.testing.assert_allclose(new.model_q_to_si_wavevector(0.1, -0.2, 3.9e-10, 3.8e-10), old.model_q_to_si_wavevector(0.1, -0.2, 3.9e-10, 3.8e-10))
    np.testing.assert_allclose(new.omega_eV_to_xi_si(0.03), old.omega_eV_to_xi_si(0.03), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new.vacuum_kappa(1.0e8, 2.0e13), old.vacuum_kappa(1.0e8, 2.0e13), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new.xy_to_lt_rotation(1.0, 2.0), old.xy_to_lt_rotation(1.0, 2.0), rtol=1e-12, atol=1e-12)

    sigma = np.array([[0.1 + 0.0j, 0.02], [0.03, 0.2]], dtype=complex)
    np.testing.assert_allclose(new.rotate_sigma_tilde_xy_to_lt(sigma, 1.0, 2.0), old.rotate_sigma_tilde_xy_to_lt(sigma, 1.0, 2.0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new.vacuum_admittance_LT(2.0e13, 1.0e8), old.vacuum_admittance_LT(2.0e13, 1.0e8), rtol=1e-12, atol=1e-12)
    sigma_lt = old.rotate_sigma_tilde_xy_to_lt(sigma, 1.0, 2.0)
    np.testing.assert_allclose(
        new.tangential_electric_reflection_matrix_LT(sigma_lt, 2.0e13, 1.0e8),
        old.tangential_electric_reflection_matrix_LT(sigma_lt, 2.0e13, 1.0e8),
        rtol=1e-12,
        atol=1e-12,
    )
    reflection_lt = old.tangential_electric_reflection_matrix_LT(sigma_lt, 2.0e13, 1.0e8)
    np.testing.assert_allclose(new.tangential_electric_LT_to_TE_TM(reflection_lt), old.tangential_electric_LT_to_TE_TM(reflection_lt), rtol=1e-12, atol=1e-12)


def test_reflection_metadata_and_package_helper_match_old_reference():
    sigma = np.array([[0.1 + 0.0j, 0.02], [0.03, 0.2]], dtype=complex)
    _assert_dict_matches(
        new.sigma_tilde_xy_to_te_tm_reflection_matrix(sigma, 0.1, -0.2, 0.03, 3.9e-10, 3.8e-10),
        old.sigma_tilde_xy_to_te_tm_reflection_matrix(sigma, 0.1, -0.2, 0.03, 3.9e-10, 3.8e-10),
    )
    assert new.te_tm_adapter_metadata() == old.te_tm_adapter_metadata()
    assert new.symmetric_antisymmetric_offdiag(sigma) == old.symmetric_antisymmetric_offdiag(sigma)
    assert new.reflection_input_metadata(q_zero_basis_convention="identity") == old.reflection_input_metadata(q_zero_basis_convention="identity")


def test_reflection_error_behavior_matches_old_reference():
    with pytest.raises(ValueError, match="lattice constants"):
        new.model_q_to_si_wavevector(0.1, 0.2, 0.0, 1.0)
    with pytest.raises(ValueError, match="omega_eV must be positive"):
        new.omega_eV_to_xi_si(0.0)
    with pytest.raises(ValueError, match="Q_m_inv"):
        new.vacuum_kappa(-1.0, 1.0)
    with pytest.raises(ValueError, match="xi_si"):
        new.vacuum_kappa(1.0, 0.0)
    with pytest.raises(ValueError, match="Q must be nonzero"):
        new.xy_to_lt_rotation(0.0, 0.0)
    with pytest.raises(ValueError, match="shape"):
        new.rotate_sigma_tilde_xy_to_lt(np.eye(3), 1.0, 0.0)
