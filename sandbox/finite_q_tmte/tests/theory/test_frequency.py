from __future__ import annotations

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.theory.frequency import K_B_EV_PER_K, matsubara_xi_eV


def test_zero_matsubara_frequency_is_zero():
    assert matsubara_xi_eV(0, 10.0) == 0.0


def test_first_matsubara_frequency_is_two_pi_kbt():
    expected = 2.0 * np.pi * K_B_EV_PER_K * 10.0
    np.testing.assert_allclose(matsubara_xi_eV(1, 10.0), expected)


def test_negative_matsubara_index_raises():
    with pytest.raises(ValueError, match="non-negative"):
        matsubara_xi_eV(-1, 10.0)


def test_non_integer_matsubara_index_raises():
    with pytest.raises(ValueError, match="integer"):
        matsubara_xi_eV(1.5, 10.0)  # type: ignore[arg-type]


def test_non_positive_temperature_raises():
    with pytest.raises(ValueError, match="positive"):
        matsubara_xi_eV(1, 0.0)
