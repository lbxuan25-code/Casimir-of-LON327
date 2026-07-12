from __future__ import annotations

import numpy as np

from validation.commands.static.quadrature_compare import _quadrature_points, _run_one


def test_gauss2_shift4_has_equal_cost_and_normalized_positive_weights():
    base_nk = 3
    midpoint_points, midpoint_weights, midpoint_meta = _quadrature_points(
        2 * base_nk,
        "midpoint",
    )
    gauss_points, gauss_weights, gauss_meta = _quadrature_points(
        base_nk,
        "gauss2_shift4",
    )

    assert len(midpoint_points) == len(gauss_points) == 4 * base_nk**2
    assert np.isclose(np.sum(midpoint_weights), 1.0)
    assert np.isclose(np.sum(gauss_weights), 1.0)
    assert np.all(midpoint_weights > 0.0)
    assert np.all(gauss_weights > 0.0)
    assert midpoint_meta["num_grid_shifts"] == 1
    assert gauss_meta["num_grid_shifts"] == 4
    assert gauss_meta["primitive_merge_before_schur"] is True


def test_gauss2_shift4_is_not_a_disguised_refined_midpoint_grid():
    base_nk = 4
    midpoint_points, _, _ = _quadrature_points(2 * base_nk, "midpoint")
    gauss_points, _, _ = _quadrature_points(base_nk, "gauss2_shift4")

    midpoint_sorted = midpoint_points[np.lexsort((midpoint_points[:, 1], midpoint_points[:, 0]))]
    gauss_sorted = gauss_points[np.lexsort((gauss_points[:, 1], gauss_points[:, 0]))]
    assert not np.allclose(midpoint_sorted, gauss_sorted)


def test_gauss2_shift4_integrates_quadratic_bz_moments_exactly():
    points, weights, _ = _quadrature_points(3, "gauss2_shift4")
    mean_x2 = float(np.sum(weights * points[:, 0] ** 2))
    mean_x2_y2 = float(np.sum(weights * points[:, 0] ** 2 * points[:, 1] ** 2))

    assert np.isclose(mean_x2, np.pi**2 / 3.0, rtol=0.0, atol=2e-14)
    assert np.isclose(mean_x2_y2, np.pi**4 / 9.0, rtol=0.0, atol=2e-13)


def test_quadrature_compare_runs_both_equal_cost_rules_through_one_schur():
    common = {
        "base_nk": 2,
        "pairing": "spm",
        "qx": 0.03,
        "qy": 0.02,
        "temperature_K": 10.0,
        "delta0_eV": 0.1,
        "eta_eV": 1e-8,
        "ward_tolerance": 1e-7,
    }
    midpoint = _run_one({**common, "quadrature_rule": "midpoint"})
    gauss2 = _run_one({**common, "quadrature_rule": "gauss2_shift4"})

    for row in (midpoint, gauss2):
        assert row["num_k_points"] == 16
        assert row["primitive_merge_before_schur"] is True
        assert "scaled_kll_effective_real" in row
        assert "scaled_phase_factorized_correction_real" in row
        assert "ward_left_effective" in row

    assert midpoint["cell_nk"] == 4
    assert midpoint["num_grid_shifts"] == 1
    assert gauss2["cell_nk"] == 2
    assert gauss2["num_grid_shifts"] == 4
