from __future__ import annotations

import numpy as np

from validation.lib.dwave_commensurate_orbit_gauss import (
    commensurate_orbit_basis,
    complementary_orbit_origins,
    integrate_commensurate_orbit_gauss_vector,
)


def test_commensurate_orbit_basis_is_unimodular_and_aligned():
    primitive, transverse, shift_steps = commensurate_orbit_basis(3, 2)
    assert np.array_equal(primitive, np.asarray([3, 2]))
    assert int(primitive[0] * transverse[1] - primitive[1] * transverse[0]) == 1
    assert shift_steps == 1


def test_complementary_orbit_origins_follow_half_shift_parity():
    assert complementary_orbit_origins(1, 0.5, "auto") == (0.5, 0.0)
    assert complementary_orbit_origins(2, 0.5, "auto") == (0.5,)
    assert complementary_orbit_origins(1, 0.5, "none") == (0.5,)


def test_orbit_gauss_integrates_constant_and_preserves_q_translation():
    nk, mx, my = 12, 3, 2
    step = 2.0 * np.pi / float(nk)
    q = step * np.asarray([mx, my], dtype=float)

    def evaluator(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        shifted = (points + q + np.pi) % (2.0 * np.pi) - np.pi
        base = np.exp(0.2 * np.cos(points[:, 0]) + 0.1 * np.sin(points[:, 1]))
        translated = np.exp(
            0.2 * np.cos(shifted[:, 0]) + 0.1 * np.sin(shifted[:, 1])
        )
        return np.column_stack((np.ones(points.shape[0]), base, translated))

    result = integrate_commensurate_orbit_gauss_vector(
        evaluator,
        nk=nk,
        mx=mx,
        my=my,
        transverse_order=10,
        shift_s=0.5,
        subgrid_average="auto",
        chunk_size=5,
        max_point_evaluations=10_000,
    )
    assert result.point_evaluations == 2 * nk * 10
    assert np.isclose(result.value[0], 1.0, rtol=0.0, atol=2e-14)
    assert np.isclose(result.value[1], result.value[2], rtol=0.0, atol=2e-13)


def test_even_reduced_shift_uses_one_orbit_subgrid():
    result = integrate_commensurate_orbit_gauss_vector(
        lambda points: np.ones((points.shape[0], 1), dtype=complex),
        nk=12,
        mx=2,
        my=0,
        transverse_order=6,
        shift_s=0.5,
        subgrid_average="auto",
        max_point_evaluations=1_000,
    )
    assert result.orbit_shift_steps == 2
    assert result.orbit_origins == (0.5,)
    assert result.point_evaluations == 12 * 6
    assert np.isclose(result.value[0], 1.0, rtol=0.0, atol=2e-14)
