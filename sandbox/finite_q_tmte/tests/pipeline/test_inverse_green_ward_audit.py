from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.inverse_green_ward_audit import (
    SCHEMA_VERSION,
    inverse_green_matrix,
    inverse_green_pair,
    inverse_green_reference_matrices,
    ward_combo_matrix,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_inverse_green_ward_audit.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_inverse_green_ward_audit_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inverse_green_matrix_is_z_identity_minus_hamiltonian():
    h = np.asarray([[1.0, 0.2j], [-0.2j, -0.5]], dtype=complex)
    z = 0.3j
    expected = z * np.eye(2, dtype=complex) - h
    np.testing.assert_allclose(inverse_green_matrix(z, h), expected)


def test_inverse_green_pair_matsubara_transfer():
    h = np.eye(2, dtype=complex)
    gm, gp, meta = inverse_green_pair(
        h_minus=h,
        h_plus=2.0 * h,
        xi_eV=0.5,
        fermionic_energy_eV=0.25,
        frequency_convention="matsubara_i_transfer",
    )
    np.testing.assert_allclose(gm, 0.25j * np.eye(2) - h)
    np.testing.assert_allclose(gp, 0.75j * np.eye(2) - 2.0 * h)
    assert meta["frequency_convention"] == "matsubara_i_transfer"
    assert meta["valid_for_casimir_input"] is False


def test_inverse_green_references_include_tau3_sandwiches():
    gm = np.asarray([[1.0, 0.1], [0.2, 2.0]], dtype=complex)
    gp = np.asarray([[3.0, 0.3], [0.4, 4.0]], dtype=complex)
    tau = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    refs = inverse_green_reference_matrices(gm, gp, tau)
    assert "plain_delta_Ginv_plus_minus_minus" in refs
    assert "nambu_tau_left_Gplus_minus_Gminus_tau_right" in refs
    np.testing.assert_allclose(refs["plain_delta_Ginv_plus_minus_minus"], gp - gm)
    np.testing.assert_allclose(refs["nambu_tau_left_Gplus_minus_Gminus_tau_right"], tau @ gp - gm @ tau)


def test_ward_combo_matrix_uses_candidate_coefficients():
    gamma0 = np.eye(2, dtype=complex)
    gamma_l = 2.0 * np.eye(2, dtype=complex)
    gamma_phase = 3.0 * np.eye(2, dtype=complex)
    coeffs = {"a0": 1.0, "l": 2.0, "phase": -1j}
    expected = gamma0 + 2.0 * gamma_l - 1j * gamma_phase
    np.testing.assert_allclose(ward_combo_matrix(gamma0, gamma_l, gamma_phase, coeffs), expected)


def test_schema_version_is_inverse_green_audit():
    assert SCHEMA_VERSION == "finite_q_tmte_inverse_green_ward_audit_v1"


def test_inverse_green_cli_rejects_nonpositive_nk_for_model(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--matsubara-index",
                "1",
                "--q",
                "0.02",
                "--nk-for-model",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_inverse_green_cli_rejects_negative_matsubara_index(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--matsubara-index",
                "-1",
                "--q",
                "0.02",
                "--output-dir",
                str(tmp_path),
            ]
        )
