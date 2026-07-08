from __future__ import annotations

import numpy as np

from sandbox.finite_q_tmte.tmte.theory.schur import schur_effective


def test_schur_matches_solve_formula():
    k_ss = np.asarray([[2.0, 0.5], [0.25, 1.5]], dtype=complex)
    k_seta = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    k_etaeta = np.asarray([[5.0, 0.5], [0.25, 6.0]], dtype=complex)
    k_etas = np.asarray([[0.2, 0.3], [0.4, 0.5]], dtype=complex)
    result = schur_effective(k_ss, k_seta, k_etaeta, k_etas)
    expected = k_ss - k_seta @ np.linalg.solve(k_etaeta, k_etas)
    np.testing.assert_allclose(result.effective, expected)
    assert result.solve_method == "solve"

