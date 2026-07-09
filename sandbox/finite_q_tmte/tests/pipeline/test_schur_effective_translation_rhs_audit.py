from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.schur_effective_translation_rhs_audit import (
    SCHEMA_VERSION,
    _left_schur_payload,
    _solve_etaeta,
    _translation_rhs,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_schur_effective_translation_rhs_audit.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_schur_effective_translation_rhs_audit_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_version():
    assert SCHEMA_VERSION == "finite_q_tmte_schur_effective_translation_rhs_audit_v1"


def test_solve_etaeta_direct_solve():
    eta = np.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=complex)
    rhs = np.asarray([[2.0, 4.0, 6.0], [8.0, 12.0, 16.0]], dtype=complex)
    sol, meta = _solve_etaeta(eta, rhs)
    np.testing.assert_allclose(sol, np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=complex))
    assert meta["solve_method"] == "solve"


def test_translation_rhs_scales_qm_contact_term():
    payload = {
        "equal_forward": np.asarray([1.0, 2.0, 3.0], dtype=complex),
        "delta_v_mid": np.asarray([0.5, 1.0, 1.5], dtype=complex),
        "qM_mid": np.asarray([0.0, 2.0, 0.0], dtype=complex),
    }
    rhs = _translation_rhs(payload, contact_scale=0.25)
    np.testing.assert_allclose(rhs["translation_forward"], np.asarray([0.5, 1.0, 1.5], dtype=complex))
    np.testing.assert_allclose(rhs["qM_mid"], np.asarray([0.0, 0.5, 0.0], dtype=complex))
    np.testing.assert_allclose(rhs["translation_plus_qM"], np.asarray([0.5, 1.5, 1.5], dtype=complex))


def test_left_schur_payload_closes_exact_identity():
    k_ss = np.asarray([[1.0, 0.2, 0.0], [0.1, 0.8, 0.0], [0.0, 0.0, 0.5]], dtype=complex)
    k_seta = np.asarray([[0.2, 0.0], [0.0, 0.3], [0.1, 0.2]], dtype=complex)
    k_etas = np.asarray([[0.4, 0.1, 0.0], [0.0, 0.2, 0.3]], dtype=complex)
    k_etaeta = np.eye(2, dtype=complex)
    action = k_etas.copy()
    k_eff = k_ss - k_seta @ action
    u_left = np.asarray([1.0, -0.5, 0.25], dtype=complex)
    w_left = np.asarray([0.3, -0.2], dtype=complex)
    rhs_s = u_left @ k_ss + w_left @ k_etas
    report = _left_schur_payload(
        k_ss=k_ss,
        k_seta=k_seta,
        k_etas=k_etas,
        k_etaeta=k_etaeta,
        k_eff=k_eff,
        action=action,
        u_left=u_left,
        w_left=w_left,
        rhs_s=rhs_s,
    )
    assert report["s_channel_residual"]["norm"] < 1e-12
    assert report["effective_residual"]["norm"] < 1e-12


def test_cli_rejects_nonpositive_q(tmp_path):
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
                "0.0",
                "--nk",
                "5",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_cli_rejects_nonpositive_nk(tmp_path):
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
                "--nk",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )
