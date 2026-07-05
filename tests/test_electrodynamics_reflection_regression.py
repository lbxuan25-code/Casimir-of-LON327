import numpy as np
import pytest

import lno327.electrodynamics.reflection as new


def test_reflection_scalar_and_matrix_helpers_are_well_formed():
    assert len(new.model_q_to_si_wavevector(0.1, -0.2, 3.9e-10, 3.8e-10)) == 3
    assert new.omega_eV_to_xi_si(0.03) > 0.0
    assert new.vacuum_kappa(1.0e8, 2.0e13) > 0.0
    rotation = new.xy_to_lt_rotation(1.0, 2.0)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(2), rtol=1e-12, atol=1e-12)

    sigma = np.array([[0.1 + 0.0j, 0.02], [0.03, 0.2]], dtype=complex)
    sigma_lt = new.rotate_sigma_tilde_xy_to_lt(sigma, 1.0, 2.0)
    admittance = new.vacuum_admittance_LT(2.0e13, 1.0e8)
    reflection_lt = new.tangential_electric_reflection_matrix_LT(sigma_lt, 2.0e13, 1.0e8)
    reflection_te_tm = new.tangential_electric_LT_to_TE_TM(reflection_lt)
    assert sigma_lt.shape == (2, 2)
    assert admittance.shape == (2, 2)
    assert reflection_lt.shape == (2, 2)
    assert reflection_te_tm.shape == (2, 2)
    assert np.all(np.isfinite(reflection_te_tm))


def test_reflection_metadata_and_package_helper_are_well_formed():
    sigma = np.array([[0.1 + 0.0j, 0.02], [0.03, 0.2]], dtype=complex)
    result = new.sigma_tilde_xy_to_te_tm_reflection_matrix(sigma, 0.1, -0.2, 0.03, 3.9e-10, 3.8e-10)
    assert {
        "Q_m_inv",
        "xi_si_s_inv",
        "sigma_tilde_LT_matrix",
        "reflection_tangential_E_LT",
        "reflection_TE_TM",
    }.issubset(result)
    assert result["reflection_TE_TM"].shape == (2, 2)
    assert np.all(np.isfinite(result["reflection_TE_TM"]))
    metadata = new.te_tm_adapter_metadata()
    assert metadata["no_casimir_energy"] is True
    assert metadata["no_casimir_torque"] is True
    offdiag = new.symmetric_antisymmetric_offdiag(sigma)
    assert {"symmetric_offdiag_abs", "antisymmetric_offdiag_abs"}.issubset(offdiag)
    assert "q_zero_basis_convention" in new.reflection_input_metadata(q_zero_basis_convention="identity")


def test_reflection_error_behavior():
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
