from __future__ import annotations

import numpy as np

from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions
from sandbox.finite_q_tmte.tmte.theory.vertices import target_vertices


def _mat(value: float) -> np.ndarray:
    return np.asarray([[value, value + 1.0], [value + 2.0, value + 3.0]], dtype=complex)


def test_target_vertices_q_along_x():
    gamma0 = _mat(1.0)
    gammax = _mat(10.0)
    gammay = _mat(20.0)
    conventions = finite_q_conventions(np.asarray([0.2, 0.0]), xi=0.01)
    gamma_g, gamma_tm, gamma_te = target_vertices(gamma0, gammax, gammay, conventions)
    np.testing.assert_allclose(gamma_g, conventions.g0 * gamma0 + conventions.gL * gammax)
    np.testing.assert_allclose(gamma_tm, -conventions.gL * gamma0 + conventions.g0 * gammax)
    np.testing.assert_allclose(gamma_te, gammay)


def test_target_vertices_q_along_y():
    gamma0 = _mat(1.0)
    gammax = _mat(10.0)
    gammay = _mat(20.0)
    conventions = finite_q_conventions(np.asarray([0.0, 0.2]), xi=0.01)
    gamma_g, gamma_tm, gamma_te = target_vertices(gamma0, gammax, gammay, conventions)
    np.testing.assert_allclose(gamma_g, conventions.g0 * gamma0 + conventions.gL * gammay)
    np.testing.assert_allclose(gamma_tm, -conventions.gL * gamma0 + conventions.g0 * gammay)
    np.testing.assert_allclose(gamma_te, -gammax)

