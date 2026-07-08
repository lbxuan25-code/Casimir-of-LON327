from __future__ import annotations

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.theory.conventions import finite_q_conventions


def test_q_along_x_basis_vectors():
    conventions = finite_q_conventions(np.asarray([0.2, 0.0]), xi=0.01)
    np.testing.assert_allclose(conventions.qhat, [1.0, 0.0])
    np.testing.assert_allclose(conventions.that, [0.0, 1.0])


def test_q_along_y_basis_vectors():
    conventions = finite_q_conventions(np.asarray([0.0, 0.2]), xi=0.01)
    np.testing.assert_allclose(conventions.qhat, [0.0, 1.0])
    np.testing.assert_allclose(conventions.that, [-1.0, 0.0])


def test_q_zero_raises():
    with pytest.raises(ValueError, match="undefined"):
        finite_q_conventions(np.asarray([0.0, 0.0]), xi=0.01)

